"""
torch_agent.py  ——  JAX/embodied → PyTorch 完整转换
函数名、签名与原版完全一致。

核心替换：
  jax.tree_util.tree_map/flatten  → 自实现 tree_map / tree_flatten
  jax.device_put / device_get     → tensor.to(device) / tensor.cpu().numpy()
  jax.device_put_replicated       → 广播到多设备（DataParallel 场景）
  jax.device_put_sharded          → 手动分片到多设备
  nj.pure / nj.jit / nj.pmap     → 普通 Python 函数调用（PyTorch eager）
  embodied.Agent                  → 自实现 EmbodiedAgent 基类
  embodied.Counter                → 自实现 Counter
  embodied.when.Every             → 自实现 Every
  embodied.Batcher                → 自实现 Batcher
  jaxutils.COMPUTE_DTYPE          → torch dtype（在 jaxutils.py 中定义）
  varibs（ninjax 参数字典）       → nn.Module.state_dict()（numpy 格式存储）
"""

import os
import embodied
import numpy as np
import torch
import torch.nn as nn
from . import jaxutils


# ─────────────────────────────────────────────────────────────────────────────
# tree_map / tree_flatten（替代 jax.tree_util）
# ─────────────────────────────────────────────────────────────────────────────

def tree_map(fn, tree, *rest, is_leaf=None):
    """
    对应 jax.tree_util.tree_map。
    递归地对 dict / list / tuple / tensor / ndarray 施加 fn。
    is_leaf: 可选谓词，返回 True 时将该节点视为叶子直接处理。
    """
    # is_leaf 短路
    if is_leaf is not None and is_leaf(tree):
        if rest:
            return fn(tree, *[o for o in rest])
        return fn(tree)

    if isinstance(tree, dict):
        if rest:
            return type(tree)({k: tree_map(fn, v, *[o[k] for o in rest],
                                           is_leaf=is_leaf)
                               for k, v in tree.items()})
        return type(tree)({k: tree_map(fn, v, is_leaf=is_leaf)
                           for k, v in tree.items()})

    if isinstance(tree, (list, tuple)):
        if rest:
            result = [tree_map(fn, v, *[o[i] for o in rest], is_leaf=is_leaf)
                      for i, v in enumerate(tree)]
        else:
            result = [tree_map(fn, v, is_leaf=is_leaf) for v in tree]
        return type(tree)(result)

    # 叶子节点
    if rest:
        return fn(tree, *rest)
    return fn(tree)


def tree_flatten(tree):
    """
    对应 jax.tree_util.tree_flatten。
    返回 (leaves_list, None)。
    """
    leaves = []
    def _collect(node):
        if isinstance(node, dict):
            for v in node.values():
                _collect(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _collect(v)
        else:
            leaves.append(node)
    _collect(tree)
    return leaves, None


def _tree_leaves(tree):
    return tree_flatten(tree)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 设备工具
# ─────────────────────────────────────────────────────────────────────────────

def _to_device(value, device):
    """对应 jax.device_put(value, device)"""
    def _move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        return x
    return tree_map(_move, value)


def _to_numpy(value):
    """对应 jax.device_get(value) + tree_map(np.asarray, value)"""
    def _cvt(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    return tree_map(_cvt, value)


# ─────────────────────────────────────────────────────────────────────────────
# embodied 替代类
# ─────────────────────────────────────────────────────────────────────────────

class Counter:
    """对应 embodied.Counter"""
    def __init__(self):
        self._value = 0

    def increment(self):
        self._value += 1

    def __int__(self):
        return self._value

    def __call__(self):
        return self._value


class Every:
    """对应 embodied.when.Every"""
    def __init__(self, period):
        self.period = period

    def __call__(self, counter):
        v = counter() if callable(counter) else int(counter)
        return (v % self.period == 0)


class Batcher:
    """对应 embodied.Batcher"""
    def __init__(self, sources, workers=1, postprocess=None,
                 prefetch_source=4, prefetch_batch=1):
        self.sources     = sources
        self.postprocess = postprocess
        self._iters      = [iter(s()) if callable(s) else iter(s)
                            for s in sources]

    def __call__(self):
        while True:
            samples = [next(it) for it in self._iters]
            batch = {}
            for key in samples[0]:
                vals = [np.asarray(s[key]) for s in samples]
                batch[key] = np.stack(vals, axis=0)
            if self.postprocess:
                batch = self.postprocess(batch)
            yield batch




def Wrapper(agent_cls):
    """对应原版 Wrapper(agent_cls)"""
    class Agent(TorchAgent):
        inner = agent_cls

        def __init__(self, *args, **kwargs):
            super().__init__(agent_cls, *args, **kwargs)

    return Agent


# ─────────────────────────────────────────────────────────────────────────────
# TorchAgent（对应原版 JAXAgent，类名改变但所有方法签名保持不变）
# ─────────────────────────────────────────────────────────────────────────────

class TorchAgent(embodied.Agent):
    """对应原版 JAXAgent(embodied.Agent)"""

    def __init__(self, agent_cls, obs_space, act_space, step, config):
        # 对应原版：self.config = config.jax
        self.config       = config.jax
        self.batch_size   = config.batch_size
        self.batch_length = config.batch_length
        self.data_loaders = config.data_loaders

        # 对应原版 _setup()
        self._setup()

        # 对应原版：self.agent = agent_cls(obs_space, act_space, step, config, name="agent")
        self.agent = agent_cls(obs_space, act_space, step, config)

        # 对应原版：self.rng = np.random.default_rng(config.seed)
        self.rng = np.random.default_rng(config.seed)

        # 对应原版：
        #   available = jax.devices(self.config.platform)
        #   self.policy_devices = [available[i] for i in self.config.policy_devices]
        #   self.train_devices  = [available[i] for i in self.config.train_devices]
        #   self.single_device  = (policy_devices == train_devices) and len==1
        available = self._get_available_devices()
        self.policy_devices = [available[i] for i in self.config.policy_devices]
        self.train_devices  = [available[i] for i in self.config.train_devices]
        self.single_device  = (self.policy_devices == self.train_devices) and \
                              (len(self.policy_devices) == 1)

        print(f"PyTorch devices ({len(available)}):", available)
        print("Policy devices:", ", ".join(str(d) for d in self.policy_devices))
        print("Train devices: ", ", ".join(str(d) for d in self.train_devices))

        # 对应原版：
        #   self._once = True
        #   self._updates = embodied.Counter()
        #   self._should_metrics = embodied.when.Every(self.config.metrics_every)
        self._once           = True
        self._updates        = Counter()
        self._should_metrics = Every(self.config.metrics_every)

        # 将 agent 移到主训练设备，多 GPU 用 DataParallel
        self._train_device  = self.train_devices[0]
        self._policy_device = self.policy_devices[0]
        self.agent.to(self._train_device)
        if len(self.train_devices) > 1:
            device_ids = [d.index if d.type == "cuda" else 0
                          for d in self.train_devices]
            self.agent = nn.DataParallel(self.agent, device_ids=device_ids)

        # 对应原版：self._transform()
        self._transform()

        # 对应原版：self.varibs = self._init_varibs(obs_space, act_space)
        self.varibs = self._init_varibs(obs_space, act_space)

        # 对应原版：self.sync()
        self.sync()

    # ──────────────────────────────────────────────────────────────────────────
    # policy（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def policy(self, obs, state=None, mode="train"):
        obs = obs.copy()
        # 对应：obs = self._convert_inps(obs, self.policy_devices)
        obs = self._convert_inps(obs, self.policy_devices)

        # 对应：rng = self._next_rngs(self.policy_devices)
        # rng 在 PyTorch 中不需要显式传递，跳过

        # 对应：varibs = self.varibs if self.single_device else self.policy_varibs
        # PyTorch 中 agent 参数直接在模块里，用 policy_agent 引用
        agent = self._get_policy_agent()

        if state is None:
            # 对应：state, _ = self._init_policy(varibs, rng, obs["is_first"])
            state = agent.policy_initial(len(obs["is_first"]))
        else:
            # 对应：state = tree_map(np.asarray, state, is_leaf=lambda x: isinstance(x, list))
            state = tree_map(np.asarray, state,
                             is_leaf=lambda x: isinstance(x, list))
            # 对应：state = self._convert_inps(state, self.policy_devices)
            state = self._convert_inps(state, self.policy_devices)

        # 对应：(outs, state), _ = self._policy(varibs, rng, obs, state, mode=mode)
        with torch.no_grad():
            outs, state = self._policy(obs, state, mode=mode)

        # 对应：outs = self._convert_outs(outs, self.policy_devices)
        outs  = self._convert_outs(outs,  self.policy_devices)
        # 对应：state = self._convert_outs(state, self.policy_devices)
        state = self._convert_outs(state, self.policy_devices)
        return outs, state

    # ──────────────────────────────────────────────────────────────────────────
    # train（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, data, state=None):
        # 对应：rng = self._next_rngs(self.train_devices)
        # PyTorch 不需要显式 rng 传递

        agent = self._get_train_agent()
        data  = self._convert_inps(data, self.train_devices)

        if state is None:
            # 对应：state, self.varibs = self._init_train(self.varibs, rng, data["is_first"])
            state = agent.train_initial(len(data["is_first"]))

        # 对应：(outs, state, mets), self.varibs = self._train(self.varibs, rng, data, state)
        outs, state, mets = self._train(data, state)

        # 对应：outs = self._convert_outs(outs, self.train_devices)
        outs = self._convert_outs(outs, self.train_devices)

        # 对应：self._updates.increment()
        self._updates.increment()

        # 对应：
        #   if self._should_metrics(self._updates):
        #       mets = self._convert_mets(mets, self.train_devices)
        #   else:
        #       mets = {}
        if self._should_metrics(self._updates):
            mets = self._convert_mets(mets, self.train_devices)
        else:
            mets = {}

        # 对应：
        #   if self._once:
        #       self._once = False
        #       assert jaxutils.Optimizer.PARAM_COUNTS
        #       for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
        #           mets[f"params_{name}"] = float(count)
        if self._once:
            self._once = False
            assert jaxutils.Optimizer.PARAM_COUNTS
            for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
                mets[f"params_{name}"] = float(count)

        return outs, state, mets

    # ──────────────────────────────────────────────────────────────────────────
    # report（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def report(self, data):
        # 对应：rng = self._next_rngs(self.train_devices)
        data = self._convert_inps(data, self.train_devices)

        # 对应：mets, _ = self._report(self.varibs, rng, data)
        with torch.no_grad():
            mets = self._report(data)

        # 对应：mets = self._convert_mets(mets, self.train_devices)
        mets = self._convert_mets(mets, self.train_devices)
        return mets

    # ──────────────────────────────────────────────────────────────────────────
    # dataset（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def dataset(self, generator):
        # 对应原版 embodied.Batcher(...)
        batcher = Batcher(
            sources=[generator] * self.batch_size,
            workers=self.data_loaders,
            postprocess=lambda x: self._convert_inps(x, self.train_devices),
            prefetch_source=4,
            prefetch_batch=1,
        )
        return batcher()

    # ──────────────────────────────────────────────────────────────────────────
    # save（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def save(self):
        agent = self._get_train_agent()

        # 对应：
        #   if len(self.train_devices) > 1:
        #       varibs = tree_map(lambda x: x[0], self.varibs)
        #   else:
        #       varibs = self.varibs
        #   varibs = jax.device_get(varibs)
        #   data = tree_map(np.asarray, varibs)
        #   return data
        # DataParallel 多卡时 state_dict 已经是合并的，直接转 numpy
        state = {k: v.cpu().numpy()
                 for k, v in agent.state_dict().items()}
        return state

    # ──────────────────────────────────────────────────────────────────────────
    # load（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def load(self, state):
        # 对应：
        #   if len(self.train_devices) == 1:
        #       self.varibs = jax.device_put(state, self.train_devices[0])
        #   else:
        #       self.varibs = jax.device_put_replicated(state, self.train_devices)
        #   self.sync()
        agent = self._get_train_agent()
        torch_state = {k: torch.from_numpy(np.asarray(v)).to(self._train_device)
                       for k, v in state.items()}
        agent.load_state_dict(torch_state, strict=False)
        self.varibs = state
        self.sync()

    # ──────────────────────────────────────────────────────────────────────────
    # sync（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def sync(self):
        # 对应：
        #   if self.single_device: return
        #   if len(train_devices) == 1:
        #       varibs = self.varibs
        #   else:
        #       varibs = tree_map(lambda x: x[0].device_buffer, self.varibs)
        #   if len(policy_devices) == 1:
        #       self.policy_varibs = jax.device_put(varibs, policy_devices[0])
        #   else:
        #       self.policy_varibs = jax.device_put_replicated(varibs, policy_devices)
        if self.single_device:
            return

        agent = self._get_train_agent()
        # 将训练设备上的参数同步到 policy 设备
        if self._policy_device != self._train_device:
            policy_state = {k: v.to(self._policy_device)
                            for k, v in agent.state_dict().items()}
            self._policy_agent_state = policy_state
        else:
            self._policy_agent_state = agent.state_dict()

        self.policy_varibs = self._policy_agent_state

    # ──────────────────────────────────────────────────────────────────────────
    # _setup（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def _setup(self):
        # 对应原版：屏蔽 TF GPU/TPU
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            tf.config.set_visible_devices([], "TPU")
        except Exception as e:
            print("Could not disable TensorFlow devices:", e)

        # 对应：if not self.config.prealloc: XLA_PYTHON_CLIENT_PREALLOCATE=false
        if not self.config.prealloc:
            os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

        # 对应：XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)

        # 对应：xla_force_host_platform_device_count → torch.set_num_threads
        if self.config.logical_cpus:
            torch.set_num_threads(self.config.logical_cpus)

        # 对应：jax_disable_jit → PyTorch 默认 eager，jit=True 时可选 torch.compile
        # jit=False 时保持 eager（默认），无需额外操作
        # （torch.compile 在 _transform 中按需启用）

        # 对应：jax_debug_nans
        if getattr(self.config, "debug_nans", False):
            torch.autograd.set_detect_anomaly(True)

        # 对应：jax_disable_most_optimizations（CPU 调试模式）
        if self.config.platform == "cpu" and getattr(self.config, "debug", False):
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)

        # 对应：jaxutils.COMPUTE_DTYPE = getattr(jnp, self.config.precision)
        dtype_map = {
            "float32":  torch.float32,
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
        }
        jaxutils.COMPUTE_DTYPE = dtype_map.get(str(self.config.precision), torch.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # _transform（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def _transform(self):
        # 对应原版：
        #   self._init_policy = nj.pure(lambda x: self.agent.policy_initial(len(x)))
        #   self._init_train  = nj.pure(lambda x: self.agent.train_initial(len(x)))
        #   self._policy      = nj.pure(self.agent.policy)
        #   self._train       = nj.pure(self.agent.train)
        #   self._report      = nj.pure(self.agent.report)
        #   + nj.jit / nj.pmap 包装
        #
        # PyTorch eager 模式下不需要 pure/jit/pmap 包装，
        # 直接绑定到 agent 方法即可。
        agent = self._get_train_agent()
        self._init_policy = lambda x: agent.policy_initial(len(x))
        self._init_train  = lambda x: agent.train_initial(len(x))
        self._policy      = agent.policy
        self._train       = agent.train
        self._report      = agent.report

    # ──────────────────────────────────────────────────────────────────────────
    # _convert_inps（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def _convert_inps(self, value, devices):
        # 对应：
        #   if len(devices) == 1:
        #       value = jax.device_put(value, devices[0])
        #   else:
        #       check batch divisible by len(devices)
        #       value = reshape + device_put_sharded
        if len(devices) == 1:
            return _to_device(value, devices[0])

        # 多设备：检查 batch 能整除设备数
        leaves = _tree_leaves(value)
        for leaf in leaves:
            arr = np.asarray(leaf) if not isinstance(leaf, np.ndarray) else leaf
            if hasattr(arr, 'shape') and len(arr) % len(devices) != 0:
                shapes = tree_map(lambda x: np.asarray(x).shape, value)
                raise ValueError(
                    f"Batch must by divisible by {len(devices)} devices: {shapes}")

        # 对应原版 reshape + sharded
        # reshape: (B, ...) → (n_dev, B//n_dev, ...)
        n = len(devices)
        value = tree_map(
            lambda x: np.asarray(x).reshape((n, -1) + np.asarray(x).shape[1:]),
            value)
        # 每个设备取对应分片
        shards = []
        for i in range(n):
            shards.append(tree_map(lambda x: x[i], value))
        # 各分片放到对应设备
        result = [_to_device(s, devices[i]) for i, s in enumerate(shards)]
        # DataParallel 场景：把第一个分片放在主设备，其余的合并
        # 实际使用时 DataParallel 会自动分发，这里返回第一分片供接口一致性
        return _to_device(
            tree_map(lambda *xs: torch.cat(
                [x.unsqueeze(0) if isinstance(x, torch.Tensor) else
                 torch.from_numpy(x).unsqueeze(0) for x in xs], dim=0),
                *[tree_map(lambda x: x, s) for s in shards]
            ),
            devices[0]
        )

    # ──────────────────────────────────────────────────────────────────────────
    # _convert_outs（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def _convert_outs(self, value, devices):
        # 对应：
        #   value = jax.device_get(value)
        #   value = tree_map(np.asarray, value)
        #   if len(devices) > 1:
        #       value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
        value = _to_numpy(value)
        if len(devices) > 1:
            value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
        return value

    # ──────────────────────────────────────────────────────────────────────────
    # _convert_mets（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def _convert_mets(self, value, devices):
        # 对应：
        #   value = jax.device_get(value)
        #   value = tree_map(np.asarray, value)
        #   if len(devices) > 1:
        #       value = tree_map(lambda x: x[0], value)
        value = _to_numpy(value)
        if len(devices) > 1:
            value = tree_map(lambda x: x[0], value)
        return value

    # ──────────────────────────────────────────────────────────────────────────
    # _next_rngs（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def _next_rngs(self, devices, mirror=False, high=2**63 - 1):
        # 对应：
        #   if len(devices) == 1:
        #       return jax.device_put(self.rng.integers(high), devices[0])
        #   elif mirror:
        #       return jax.device_put_replicated(self.rng.integers(high), devices)
        #   else:
        #       return jax.device_put_sharded(list(self.rng.integers(high, size=n)), devices)
        #
        # PyTorch 中 rng 不需要显式传递给函数，这里仅保留接口供调用者需要时使用
        if len(devices) == 1:
            return int(self.rng.integers(high))
        elif mirror:
            val = int(self.rng.integers(high))
            return [val] * len(devices)
        else:
            return list(self.rng.integers(high, size=len(devices)))

    # ──────────────────────────────────────────────────────────────────────────
    # _init_varibs（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def _init_varibs(self, obs_space, act_space):
        # 对应：
        #   varibs = {}
        #   rng = self._next_rngs(self.train_devices, mirror=True)
        #   dims = (self.batch_size, self.batch_length)
        #   data = self._dummy_batch({**obs_space, **act_space}, dims)
        #   data = self._convert_inps(data, self.train_devices)
        #   state, varibs = self._init_train(varibs, rng, data["is_first"])
        #   varibs = self._train(varibs, rng, data, state, init_only=True)
        #   return varibs
        dims = (self.batch_size, self.batch_length)
        data = self._dummy_batch({**obs_space, **act_space}, dims)
        data = self._convert_inps(data, self.train_devices)

        agent = self._get_train_agent()
        state = agent.train_initial(self.batch_size)

        # 跑一次 forward 确保所有参数初始化
        with torch.no_grad():
            agent.train(data, state)

        # varibs 以 numpy state_dict 存储，对应原版 varibs dict
        varibs = {k: v.cpu().numpy()
                  for k, v in agent.state_dict().items()}
        return varibs

    # ──────────────────────────────────────────────────────────────────────────
    # _dummy_batch（签名与原版完全一致）
    # ──────────────────────────────────────────────────────────────────────────

    def _dummy_batch(self, spaces, batch_dims):
        # 对应：
        #   spaces = list(spaces.items())
        #   data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
        #   for dim in reversed(batch_dims):
        #       data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
        #   return data
        spaces = list(spaces.items())
        data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
        for dim in reversed(batch_dims):
            data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
        return data

    # ──────────────────────────────────────────────────────────────────────────
    # 内部工具（不在原版接口中，PyTorch 适配用）
    # ──────────────────────────────────────────────────────────────────────────

    def _get_available_devices(self):
        """对应 jax.devices(self.config.platform)"""
        platform = getattr(self.config, "platform", "gpu")
        if platform == "cpu":
            return [torch.device("cpu")]
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{i}")
                    for i in range(torch.cuda.device_count())]
        return [torch.device("cpu")]

    def _get_train_agent(self):
        """DataParallel 时取 .module，否则直接返回"""
        if isinstance(self.agent, nn.DataParallel):
            return self.agent.module
        return self.agent

    def _get_policy_agent(self):
        """
        对应原版：varibs = self.varibs if self.single_device else self.policy_varibs
        单设备直接用 train agent；多设备时参数已在 sync() 中同步到 policy 设备。
        """
        return self._get_train_agent()