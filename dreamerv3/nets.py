"""
nets.py  ——  JAX/TFP → PyTorch 完整转换
所有类名、函数名、参数签名与原版完全一致。
"""

import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import jaxutils
# from . import ninjax as nj   # nj.Module 基类由下方 NJModule 替代
from .jaxutils import sg 

f32 = torch.float32

cast = jaxutils.cast_to_compute


# ── nj.Module 基类替代 ────────────────────────────────────────────────────────

# class NJModule(nn.Module):
#     """
#     替代 nj.Module。
#     self.get(name, cls, *args, **kw) 对应原版 ninjax 的懒惰子模块注册。
#     """
#     def __init__(self):
#         super().__init__()
#         self._nj_cache = {}   # name -> 子模块或参数

#     def get(self, name, cls, *args, **kw):
#         """
#         对应原版 self.get(name, cls, *args, **kw)。

#         原版 ninjax 中 self.get(name, cls, *args) 的语义：
#           - cls 是 Initializer 实例 → 调用 cls(shape) 生成权重张量，存为参数
#           - cls 是 jnp.zeros/ones  → 生成零/一张量，存为参数
#           - cls 是 _zeros_param/_ones_param 等工厂函数 → 调用并注册返回的 Parameter
#           - cls 是 nn.Module 子类  → 实例化并注册为子模块

#         PyTorch 映射：
#           Initializer(shape) → nn.Parameter(tensor)
#           _zeros_param(shape) / _ones_param(shape) → 直接返回 nn.Parameter
#           nn.Module 子类 → add_module + 返回实例
#         """
#         if name in self._nj_cache:
#             return self._nj_cache[name]

#         safe_name = name.replace("/", "__").replace(" ", "_")

#         # ── 1. Initializer 实例：cls(shape) → tensor → nn.Parameter ──────────
#         if isinstance(cls, Initializer):
#             val = cls(*args)
#             if not isinstance(val, torch.Tensor):
#                 val = torch.tensor(np.array(val), dtype=f32)
#             param = nn.Parameter(val)
#             self.register_parameter(safe_name, param)
#             self._nj_cache[name] = param
#             return param

#         # ── 2. 工厂函数（_zeros_param / _ones_param）：直接调用，返回 nn.Parameter ──
#         #    这类函数签名为 fn(shape) → nn.Parameter
#         if callable(cls) and not isinstance(cls, type) and                 not isinstance(cls, Initializer):
#             result = cls(*args)
#             if isinstance(result, nn.Parameter):
#                 self.register_parameter(safe_name, result)
#                 self._nj_cache[name] = result
#                 return result
#             # 其他可调用对象（如 lambda）返回普通 tensor → 包装为 Parameter
#             if isinstance(result, torch.Tensor):
#                 param = nn.Parameter(result)
#                 self.register_parameter(safe_name, param)
#                 self._nj_cache[name] = param
#                 return param
#             # 返回 nn.Module（如通过 lambda 构造）
#             if isinstance(result, nn.Module):
#                 self.add_module(safe_name, result)
#                 self._nj_cache[name] = result
#                 return result
#             self._nj_cache[name] = result
#             return result

#         # ── 3. nn.Module 子类：实例化并注册 ──────────────────────────────────
#         obj = cls(*args, **kw)
#         if isinstance(obj, nn.Module):
#             self.add_module(safe_name, obj)
#         self._nj_cache[name] = obj
#         return obj
class NJModule(nn.Module):
    """
    替代 nj.Module。
    self.get(name, cls, *args, **kw) 对应原版 ninjax 的懒惰子模块注册。
    """
    def __init__(self):
        super().__init__()
        self._nj_cache = {}   # name -> 子模块或参数

    def get(self, name, cls, *args, **kw):
        """
        对应原版 self.get(name, cls, *args, **kw)。

        原版 ninjax 中 self.get(name, cls, *args) 的语义：
          - cls 是 Initializer 实例 → 调用 cls(shape) 生成权重张量，存为参数
          - cls 是 jnp.zeros/ones  → 生成零/一张量，存为参数
          - cls 是 _zeros_param/_ones_param 等工厂函数 → 调用并注册返回的 Parameter
          - cls 是 nn.Module 子类  → 实例化并注册为子模块

        PyTorch 映射：
          Initializer(shape) → nn.Parameter(tensor)
          _zeros_param(shape) / _ones_param(shape) → 直接返回 nn.Parameter
          nn.Module 子类 → add_module + 返回实例
        """
        if name in self._nj_cache:
            return self._nj_cache[name]

        # 加前缀避免与类方法/属性名冲突：
        #   p_ 前缀用于 nn.Parameter 注册
        #   m_ 前缀用于 nn.Module 子模块注册
        base_name   = name.replace("/", "__").replace(" ", "_")
        param_name  = "p_" + base_name
        module_name = "m_" + base_name

        # ── 1. Initializer 实例：cls(shape) → tensor → nn.Parameter ──────────
        if isinstance(cls, Initializer):
            val = cls(*args)
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(np.array(val), dtype=f32)
            param = nn.Parameter(val)
            self.register_parameter(param_name, param)
            self._nj_cache[name] = param
            return param

        # ── 2. 工厂函数（_zeros_param / _ones_param）：直接调用，返回 nn.Parameter ──
        #    这类函数签名为 fn(shape) → nn.Parameter
        if callable(cls) and not isinstance(cls, type) and \
                not isinstance(cls, Initializer):
            result = cls(*args)
            if isinstance(result, nn.Parameter):
                self.register_parameter(param_name, result)
                self._nj_cache[name] = result
                return result
            # 其他可调用对象（如 lambda）返回普通 tensor → 包装为 Parameter
            if isinstance(result, torch.Tensor):
                param = nn.Parameter(result)
                self.register_parameter(param_name, param)
                self._nj_cache[name] = param
                return param
            # 返回 nn.Module（如通过 lambda 构造）
            if isinstance(result, nn.Module):
                self.add_module(module_name, result)
                self._nj_cache[name] = result
                return result
            self._nj_cache[name] = result
            return result

        # ── 3. nn.Module 子类：实例化并注册 ──────────────────────────────────
        obj = cls(*args, **kw)
        if isinstance(obj, nn.Module):
            self.add_module(module_name, obj)
        self._nj_cache[name] = obj
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# RSSM
# ─────────────────────────────────────────────────────────────────────────────

class RSSM(NJModule):
    def __init__(
        self,
        deter=1024,
        stoch=32,
        classes=32,
        unroll=False,
        initial="learned",
        unimix=0.01,
        action_clip=1.0,
        **kw,
    ):
        super().__init__()
        self._deter       = deter
        self._stoch       = stoch
        self._classes     = classes
        self._unroll      = unroll
        self._initial     = initial
        self._unimix      = unimix
        self._action_clip = action_clip
        self._kw          = kw

    def initial(self, bs):
        # 确定设备
        params = list(self.parameters())
        dev = params[0].device if params else torch.device("cpu")

        if self._classes:
            state = dict(
                deter=torch.zeros([bs, self._deter], dtype=f32, device=dev),
                logit=torch.zeros([bs, self._stoch, self._classes], dtype=f32, device=dev),
                stoch=torch.zeros([bs, self._stoch, self._classes], dtype=f32, device=dev),
            )
        else:
            state = dict(
                deter=torch.zeros([bs, self._deter], dtype=f32, device=dev),
                mean =torch.zeros([bs, self._stoch], dtype=f32, device=dev),
                std  =torch.ones ([bs, self._stoch], dtype=f32, device=dev),
                stoch=torch.zeros([bs, self._stoch], dtype=f32, device=dev),
            )

        if self._initial == "zeros":
            return cast(state)
        elif self._initial == "learned":
            # 对应：deter = self.get("initial", jnp.zeros, state["deter"][0].shape, f32)
            # 注册一个可学习的初始 deter（单行，shape = (deter,)）
            init_param = self.get("initial", _zeros_param, (self._deter,))
            deter = torch.tanh(init_param).unsqueeze(0).expand(bs, -1)
            state["deter"] = deter
            state["stoch"]  = self.get_stoch(cast(state["deter"]))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        # 对应：def swap(x): return x.transpose([1,0]+list(range(2,len(x.shape))))
        def swap(x):
            return x.permute([1, 0] + list(range(2, x.dim())))

        if state is None:
            state = self.initial(action.shape[0])

        # 对应：
        #   def step(prev, inputs): return self.obs_step(prev[0], *inputs)
        #   inputs = swap(action), swap(embed), swap(is_first)
        #   start = state, state
        #   post, prior = jaxutils.scan(step, inputs, start, self._unroll)
        action_s  = swap(action)
        embed_s   = swap(embed)
        isfirst_s = swap(is_first)
        T = action_s.shape[0]

        posts, priors = [], []
        prev = (state, state)   # start = state, state
        for t in range(T):
            post_t, prior_t = self.obs_step(
                prev[0], action_s[t], embed_s[t], isfirst_s[t])
            posts.append(post_t)
            priors.append(prior_t)
            prev = (post_t, prior_t)

        def stack_swap(lst):
            keys = lst[0].keys()
            stacked = {k: torch.stack([d[k] for d in lst], dim=0) for k in keys}
            return {k: swap(v) for k, v in stacked.items()}

        return stack_swap(posts), stack_swap(priors)

    def imagine(self, action, state=None):
        def swap(x):
            return x.permute([1, 0] + list(range(2, x.dim())))

        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state

        # 对应：
        #   action = swap(action)
        #   prior = jaxutils.scan(self.img_step, action, state, self._unroll)
        action_s = swap(action)
        T = action_s.shape[0]

        priors = []
        prev = state
        for t in range(T):
            prior_t = self.img_step(prev, action_s[t])
            priors.append(prior_t)
            prev = prior_t

        def stack_swap(lst):
            keys = lst[0].keys()
            stacked = {k: torch.stack([d[k] for d in lst], dim=0) for k in keys}
            return {k: swap(v) for k, v in stacked.items()}

        return stack_swap(priors)

    def get_dist(self, state, argmax=False):
        if self._classes:
            logit = state["logit"].float()
            # 对应：tfd.Independent(jaxutils.OneHotDist(logit), 1)
            return jaxutils.OneHotDist(logit, 1)
        else:
            mean = state["mean"].float()
            std  = state["std"].float()
            # 对应：tfp.MultivariateNormalDiag(mean, std)
            return torch.distributions.Independent(
                torch.distributions.Normal(mean, std), 1)

    def obs_step(self, prev_state, prev_action, embed, is_first):
        is_first    = cast(is_first)
        prev_action = cast(prev_action)

        # 对应：
        #   prev_action *= sg(action_clip / jnp.maximum(action_clip, jnp.abs(prev_action)))
        if self._action_clip > 0.0:
            prev_action = prev_action * sg(
                self._action_clip / torch.clamp(
                    torch.abs(prev_action), min=self._action_clip))

        # 对应：
        #   prev_state, prev_action = tree_map(lambda x: self._mask(x, 1-is_first),
        #                                      (prev_state, prev_action))
        prev_state  = {k: self._mask(v, 1.0 - is_first)
                       for k, v in prev_state.items()}
        prev_action = self._mask(prev_action, 1.0 - is_first)

        # 对应：
        #   prev_state = tree_map(lambda x, y: x + self._mask(y, is_first),
        #                         prev_state, self.initial(len(is_first)))
        init = self.initial(is_first.shape[0])
        prev_state = {k: v + self._mask(init[k], is_first)
                      for k, v in prev_state.items()}

        prior = self.img_step(prev_state, prev_action)

        x = torch.cat([prior["deter"], embed], dim=-1)
        x = self.get("obs_out", Linear, **self._kw)(x)
        stats = self._stats("obs_stats", x)
        dist  = self.get_dist(stats)
        stoch = dist.sample()
        post  = {"stoch": stoch, "deter": prior["deter"], **stats}
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
        prev_stoch  = prev_state["stoch"]
        prev_action = cast(prev_action)

        # 对应：prev_action *= sg(action_clip / jnp.maximum(action_clip, jnp.abs(prev_action)))
        if self._action_clip > 0.0:
            prev_action = prev_action * sg(
                self._action_clip / torch.clamp(
                    torch.abs(prev_action), min=self._action_clip))

        # 对应：flatten stoch if categorical
        if self._classes:
            shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
            prev_stoch = prev_stoch.reshape(shape)

        # 对应：flatten 2D actions
        if len(prev_action.shape) > len(prev_stoch.shape):
            shape = prev_action.shape[:-2] + (int(np.prod(prev_action.shape[-2:])),)
            prev_action = prev_action.reshape(shape)

        x = torch.cat([prev_stoch, prev_action], dim=-1)
        x = self.get("img_in", Linear, **self._kw)(x)
        x, deter = self._gru(x, prev_state["deter"])
        x = self.get("img_out", Linear, **self._kw)(x)
        stats = self._stats("img_stats", x)
        dist  = self.get_dist(stats)
        stoch = dist.sample()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return cast(prior)

    def get_stoch(self, deter):
        print(f"_kw in get_stoch: {self._kw}")
        x     = self.get("img_out", Linear, **self._kw)(deter)
        stats = self._stats("img_stats", x)
        dist  = self.get_dist(stats)
        return cast(dist.mode())

    def _gru(self, x, deter):
        # 对应：x = jnp.concatenate([deter, x], -1)
        x  = torch.cat([deter, x], dim=-1)
        kw = {**self._kw, "act": "none", "units": 3 * self._deter}
        x  = self.get("gru", Linear, **kw)(x)
        reset, cand, update = torch.chunk(x, 3, dim=-1)
        reset  = torch.sigmoid(reset)
        cand   = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        deter  = update * cand + (1 - update) * deter
        return deter, deter

    def _stats(self, name, x):
        if self._classes:
            x     = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            if self._unimix:
                probs   = F.softmax(logit, dim=-1)
                uniform = torch.ones_like(probs) / probs.shape[-1]
                probs   = (1 - self._unimix) * probs + self._unimix * uniform
                logit   = torch.log(probs)
            return {"logit": logit}
        else:
            x = self.get(name, Linear, 2 * self._stoch)(x)
            mean, std = torch.chunk(x, 2, dim=-1)
            std = 2 * torch.sigmoid(std / 2) + 0.1
            return {"mean": mean, "std": std}

    def _mask(self, value, mask):
        # 对应：jnp.einsum("b...,b->b...", value, mask.astype(value.dtype))
        mask = mask.to(value.dtype)
        while mask.dim() < value.dim():
            mask = mask.unsqueeze(-1)
        return value * mask

    def dyn_loss(self, post, prior, impl="kl", free=1.0):
        if impl == "kl":
            # 对应：self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
            loss = torch.distributions.kl_divergence(
                self.get_dist(sg(post)), self.get_dist(prior))
        elif impl == "logprob":
            loss = -self.get_dist(prior).log_prob(sg(post["stoch"]))
        else:
            raise NotImplementedError(impl)
        if free:
            loss = torch.clamp(loss, min=free)
        return loss

    def rep_loss(self, post, prior, impl="kl", free=1.0):
        if impl == "kl":
            loss = torch.distributions.kl_divergence(
                self.get_dist(post), self.get_dist(sg(prior)))
        elif impl == "uniform":
            uniform = {k: torch.zeros_like(v) for k, v in prior.items()}
            loss = torch.distributions.kl_divergence(
                self.get_dist(post), self.get_dist(uniform))
        elif impl == "entropy":
            loss = -self.get_dist(post).entropy()
        elif impl == "none":
            loss = torch.zeros(post["deter"].shape[:-1],
                               device=post["deter"].device)
        else:
            raise NotImplementedError(impl)
        if free:
            loss = torch.clamp(loss, min=free)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# MultiEncoder
# ─────────────────────────────────────────────────────────────────────────────

class MultiEncoder(NJModule):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=4,
        mlp_units=512,
        cnn="resize",
        cnn_depth=48,
        cnn_blocks=2,
        resize="stride",
        symlog_inputs=False,
        minres=4,
        **kw,
    ):
        super().__init__()
        excluded = ("is_first", "is_last")
        shapes = {k: v for k, v in shapes.items()
                  if k not in excluded and not k.startswith("log_")}
        self.cnn_shapes = {k: v for k, v in shapes.items()
                           if len(v) == 3 and re.match(cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items()
                           if len(v) in (1, 2) and re.match(mlp_keys, k)}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        # 对应原版传入 name="cnn" / name="mlp"，PyTorch 通过 add_module 命名
        cnn_kw = {**kw, "minres": minres}
        mlp_kw = {**kw, "symlog_inputs": symlog_inputs}

        if cnn == "resnet":
            self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **cnn_kw)
            self.add_module("cnn", self._cnn)
        else:
            raise NotImplementedError(cnn)

        if self.mlp_shapes:
            self._mlp = MLP(None, mlp_layers, mlp_units, dist="none", **mlp_kw)
            self.add_module("mlp", self._mlp)
        else:
            self._mlp = None

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        some_key, some_shape = list(self.shapes.items())[0]
        batch_dims = data[some_key].shape[: -len(some_shape)]

        # 对应：data = {k: v.reshape((-1,) + v.shape[len(batch_dims):]) ...}
        data = {k: v.reshape((-1,) + v.shape[len(batch_dims):])
                for k, v in data.items()}

        outputs = []
        if self.cnn_shapes:
            # 原版：jnp.concatenate([data[k] for k in cnn_shapes], -1)
            # 原版 CNN 输入格式 (N, H, W, C)（channel-last）
            inputs = torch.cat([data[k] for k in self.cnn_shapes], dim=-1)
            output = self._cnn(inputs)
            output = output.reshape((output.shape[0], -1))
            outputs.append(output)

        if self.mlp_shapes:
            # 对应：[data[k][...,None] if len(shape)==0 else data[k] for k in mlp_shapes]
            inputs = [data[k].unsqueeze(-1) if len(self.shapes[k]) == 0
                      else data[k] for k in self.mlp_shapes]
            inputs = torch.cat([x.float() for x in inputs], dim=-1)
            inputs = cast(inputs)
            outputs.append(self._mlp(inputs))

        outputs = torch.cat(outputs, dim=-1)
        outputs = outputs.reshape(batch_dims + outputs.shape[1:])
        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# MultiDecoder
# ─────────────────────────────────────────────────────────────────────────────

class MultiDecoder(NJModule):
    def __init__(
        self,
        shapes,
        inputs=["tensor"],
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=4,
        mlp_units=512,
        cnn="resize",
        cnn_depth=48,
        cnn_blocks=2,
        image_dist="mse",
        vector_dist="mse",
        resize="stride",
        bins=255,
        outscale=1.0,
        minres=4,
        cnn_sigmoid=False,
        **kw,
    ):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {k: v for k, v in shapes.items()
                           if re.match(cnn_keys, k) and len(v) == 3}
        self.mlp_shapes = {k: v for k, v in shapes.items()
                           if re.match(mlp_keys, k) and len(v) == 1}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        cnn_kw = {**kw, "minres": minres, "sigmoid": cnn_sigmoid}
        mlp_kw = {**kw, "dist": vector_dist, "outscale": outscale, "bins": bins}

        if self.cnn_shapes:
            cshapes = list(self.cnn_shapes.values())
            assert all(x[:-1] == cshapes[0][:-1] for x in cshapes)
            shape = cshapes[0][:-1] + (sum(x[-1] for x in cshapes),)
            if cnn == "resnet":
                self._cnn = ImageDecoderResnet(
                    shape, cnn_depth, cnn_blocks, resize, **cnn_kw)
                self.add_module("cnn", self._cnn)
            else:
                raise NotImplementedError(cnn)
        else:
            self._cnn = None

        if self.mlp_shapes:
            self._mlp = MLP(self.mlp_shapes, mlp_layers, mlp_units, **mlp_kw)
            self.add_module("mlp", self._mlp)
        else:
            self._mlp = None

        self._inputs     = Input(inputs, dims="deter")
        self._image_dist = image_dist

    def __call__(self, inputs, drop_loss_indices=None):
        return self.forward(inputs, drop_loss_indices)

    def forward(self, inputs, drop_loss_indices=None):
        features = self._inputs(inputs)
        dists    = {}

        if self.cnn_shapes:
            feat = features
            if drop_loss_indices is not None:
                feat = feat[:, drop_loss_indices]
            flat   = feat.reshape([-1, feat.shape[-1]])
            output = self._cnn(flat)
            output = output.reshape(feat.shape[:-1] + output.shape[1:])

            # 对应：split_indices = np.cumsum([v[-1] for v in cnn_shapes.values()][:-1])
            #        means = jnp.split(output, split_indices, -1)
            split_sizes   = [v[-1] for v in self.cnn_shapes.values()]
            split_indices = list(np.cumsum(split_sizes[:-1]))
            means = torch.split(output, split_sizes, dim=-1)
            dists.update({
                key: self._make_image_dist(key, mean)
                for (key, _shape), mean in zip(self.cnn_shapes.items(), means)
            })

        if self.mlp_shapes:
            dists.update(self._mlp(features))

        return dists

    def _make_image_dist(self, name, mean):
        mean = mean.float()
        if self._image_dist == "normal":
            return torch.distributions.Independent(
                torch.distributions.Normal(mean, torch.ones_like(mean)), 3)
        if self._image_dist == "mse":
            return jaxutils.MSEDist(mean, 3, "sum")
        raise NotImplementedError(self._image_dist)


# ─────────────────────────────────────────────────────────────────────────────
# ImageEncoderResnet
# ─────────────────────────────────────────────────────────────────────────────

class ImageEncoderResnet(NJModule):
    def __init__(self, depth, blocks, resize, minres, **kw):
        super().__init__()
        self._depth  = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._kw     = kw

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # 原版输入格式：(N, H, W, C)  channel-last
        # 对应：stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
        # x.shape[-2] 是原版 channel-last 的 W（第二个空间维）
        stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
        depth  = self._depth
        x = cast(x) - 0.5   # 原版保持 channel-last

        for i in range(stages):
            kw = {**self._kw, "preact": False}
            if self._resize == "stride":
                x = self.get(f"s{i}res", Conv2D, depth, 4, 2, **kw)(x)
            elif self._resize == "stride3":
                s = 2 if i else 3
                k = 5 if i else 4
                x = self.get(f"s{i}res", Conv2D, depth, k, s, **kw)(x)
            elif self._resize == "mean":
                # 原版：N,H,W,D = x.shape; reshape→(N,H//2,W//2,4,D).mean(-2)
                N, H, W, D = x.shape
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
                x = x.reshape(N, H // 2, 2, W // 2, 2, D).mean(dim=(2, 4))
            elif self._resize == "max":
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
                # 原版：jax.lax.reduce_window(...) 3x3 max pool stride 2 SAME
                # channel-last (N,H,W,C) → permute → max_pool2d → permute back
                x = x.permute(0, 3, 1, 2)
                x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
                x = x.permute(0, 2, 3, 1)
            else:
                raise NotImplementedError(self._resize)

            for j in range(self._blocks):
                skip = x
                kw2  = {**self._kw, "preact": True}
                x    = self.get(f"s{i}b{j}conv1", Conv2D, depth, 3, **kw2)(x)
                x    = self.get(f"s{i}b{j}conv2", Conv2D, depth, 3, **kw2)(x)
                x   += skip
            depth *= 2

        if self._blocks:
            x = get_act(self._kw["act"])(x)

        # 原版：x.reshape((x.shape[0], -1))
        x = x.reshape((x.shape[0], -1))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# ImageDecoderResnet
# ─────────────────────────────────────────────────────────────────────────────

class ImageDecoderResnet(NJModule):
    def __init__(self, shape, depth, blocks, resize, minres, sigmoid, **kw):
        super().__init__()
        self._shape   = shape    # (H, W, C) channel-last，与原版一致
        self._depth   = depth
        self._blocks  = blocks
        self._resize  = resize
        self._minres  = minres
        self._sigmoid = sigmoid
        self._kw      = kw

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # 对应：stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))
        stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))
        depth  = self._depth * 2 ** (stages - 1)
        x      = cast(x)

        # 对应：x = self.get("in", Linear, (minres, minres, depth))(x)
        # Linear 输出 (B, minres, minres, depth)，channel-last
        x = self.get("in", Linear, (self._minres, self._minres, depth))(x)

        for i in range(stages):
            for j in range(self._blocks):
                skip = x
                kw2  = {**self._kw, "preact": True}
                x    = self.get(f"s{i}b{j}conv1", Conv2D, depth, 3, **kw2)(x)
                x    = self.get(f"s{i}b{j}conv2", Conv2D, depth, 3, **kw2)(x)
                x   += skip
            depth //= 2
            kw = {**self._kw, "preact": False}
            if i == stages - 1:
                kw    = {}
                depth = self._shape[-1]
            if self._resize == "stride":
                x = self.get(f"s{i}res", Conv2D, depth, 4, 2, transp=True, **kw)(x)
            elif self._resize == "stride3":
                s = 3 if i == stages - 1 else 2
                k = 5 if i == stages - 1 else 4
                x = self.get(f"s{i}res", Conv2D, depth, k, s, transp=True, **kw)(x)
            elif self._resize == "resize":
                # 对应：x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)  (N,H,W,C) repeat
                x = x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
            else:
                raise NotImplementedError(self._resize)

        # 对应：
        #   if max(x.shape[1:-1]) > max(self._shape[:-1]):
        #       padh = (x.shape[1] - self._shape[0]) / 2
        #       padw = (x.shape[2] - self._shape[1]) / 2
        #       x = x[:, ceil(padh):-int(padh), :]
        #       x = x[:, :, ceil(padw):-int(padw)]
        if max(x.shape[1:-1]) > max(self._shape[:-1]):
            padh = (x.shape[1] - self._shape[0]) / 2
            padw = (x.shape[2] - self._shape[1]) / 2
            x = x[:, int(math.ceil(padh)) : x.shape[1] - int(padh), :, :]
            x = x[:, :, int(math.ceil(padw)) : x.shape[2] - int(padw), :]

        assert x.shape[-3:] == torch.Size(self._shape), (x.shape, self._shape)

        if self._sigmoid:
            x = torch.sigmoid(x)
        else:
            x = x + 0.5
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────

class MLP(NJModule):
    def __init__(
        self,
        shape,
        layers,
        units,
        inputs=["tensor"],
        dims=None,
        symlog_inputs=False,
        **kw,
    ):
        super().__init__()
        assert shape is None or isinstance(shape, (int, tuple, dict)), shape
        if isinstance(shape, int):
            shape = (shape,)
        self._shape         = shape
        self._layers        = layers
        self._units         = units
        self._inputs        = Input(inputs, dims=dims)
        self._symlog_inputs = symlog_inputs
        distkeys = ("dist", "outscale", "minstd", "maxstd",
                    "outnorm", "unimix", "bins")
        self._dense = {k: v for k, v in kw.items() if k not in distkeys}
        self._dist  = {k: v for k, v in kw.items() if k in distkeys}

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        feat = self._inputs(inputs)
        if self._symlog_inputs:
            feat = jaxutils.symlog(feat)
        x = cast(feat)
        x = x.reshape([-1, x.shape[-1]])
        for i in range(self._layers):
            x = self.get(f"h{i}", Linear, self._units, **self._dense)(x)
        x = x.reshape(feat.shape[:-1] + (x.shape[-1],))
        if self._shape is None:
            return x
        elif isinstance(self._shape, tuple):
            return self._out("out", self._shape, x)
        elif isinstance(self._shape, dict):
            return {k: self._out(k, v, x) for k, v in self._shape.items()}
        else:
            raise ValueError(self._shape)

    def _out(self, name, shape, x):
        return self.get(f"dist_{name}", Dist, shape, **self._dist)(x)


# ─────────────────────────────────────────────────────────────────────────────
# Dist
# ─────────────────────────────────────────────────────────────────────────────

class Dist(NJModule):
    def __init__(
        self,
        shape,
        dist="mse",
        outscale=0.1,
        outnorm=False,
        minstd=1.0,
        maxstd=1.0,
        unimix=0.0,
        bins=255,
    ):
        super().__init__()
        assert all(isinstance(dim, int) for dim in shape), shape
        self._shape    = shape
        self._dist     = dist
        self._minstd   = minstd
        self._maxstd   = maxstd
        self._unimix   = unimix
        self._outscale = outscale
        self._outnorm  = outnorm
        self._bins     = bins

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        dist = self.inner(inputs)
        assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
            dist.batch_shape, dist.event_shape, inputs.shape)
        return dist

    def inner(self, inputs):
        kw = {}
        kw["outscale"] = self._outscale
        kw["outnorm"]  = self._outnorm
        shape = self._shape
        if self._dist.endswith("_disc"):
            shape = (*self._shape, self._bins)
        out = self.get("out", Linear, int(np.prod(shape)), **kw)(inputs)
        out = out.reshape(inputs.shape[:-1] + shape).float()

        if self._dist in ("normal", "trunc_normal"):
            std = self.get("std", Linear, int(np.prod(self._shape)), **kw)(inputs)
            std = std.reshape(inputs.shape[:-1] + self._shape).float()

        if self._dist == "symlog_mse":
            return jaxutils.SymlogDist(out, len(self._shape), "mse", "sum")
        if self._dist == "symlog_disc":
            return jaxutils.DiscDist(out, len(self._shape), -20, 20,
                                     jaxutils.symlog, jaxutils.symexp)
        if self._dist == "mse":
            return jaxutils.MSEDist(out, len(self._shape), "sum")
        if self._dist == "normal":
            lo, hi = self._minstd, self._maxstd
            std = (hi - lo) * torch.sigmoid(std + 2.0) + lo
            dist = torch.distributions.Normal(torch.tanh(out), std)
            dist = torch.distributions.Independent(dist, len(self._shape))
            dist.minent = float(np.prod(self._shape)) * \
                          torch.distributions.Normal(
                              torch.tensor(0.0), torch.tensor(lo)).entropy().item()
            dist.maxent = float(np.prod(self._shape)) * \
                          torch.distributions.Normal(
                              torch.tensor(0.0), torch.tensor(hi)).entropy().item()
            return dist
        if self._dist == "binary":
            dist = torch.distributions.Bernoulli(logits=out)
            return torch.distributions.Independent(dist, len(self._shape))
        if self._dist == "onehot":
            if self._unimix:
                probs   = F.softmax(out, dim=-1)
                uniform = torch.ones_like(probs) / probs.shape[-1]
                probs   = (1 - self._unimix) * probs + self._unimix * uniform
                out     = torch.log(probs)
            dist = jaxutils.OneHotDist(out)
            if len(self._shape) > 1:
                # 对应：tfd.Independent(dist, len(self._shape) - 1)
                dist = jaxutils.IndependentOneHotDist(dist, len(self._shape) - 1)
            dist.minent = 0.0
            dist.maxent = float(np.prod(self._shape[:-1])) * \
                          math.log(self._shape[-1])
            return dist
        raise NotImplementedError(self._dist)


# ─────────────────────────────────────────────────────────────────────────────
# Conv2D
# ─────────────────────────────────────────────────────────────────────────────

class Conv2D(NJModule):
    """
    原版以 channel-last (N,H,W,C) 格式工作（JAX/NHWC）。
    此 PyTorch 实现内部在 _layer 中转为 channel-first 做卷积，
    输入输出仍保持 channel-last，与原版接口一致。
    """
    def __init__(
        self,
        depth,
        kernel,
        stride=1,
        transp=False,
        act="none",
        norm="none",
        pad="same",
        bias=True,
        preact=False,
        winit="uniform",
        fan="avg",
    ):
        super().__init__()
        self._depth  = depth
        self._kernel = kernel
        self._stride = stride
        self._transp = transp
        self._act    = get_act(act)
        self._norm   = Norm(norm)
        self.add_module("norm", self._norm)
        self._pad    = pad.upper()
        self._bias   = bias and (preact or norm == "none")
        self._preact = preact
        self._winit  = winit
        self._fan    = fan

    def __call__(self, hidden):
        return self.forward(hidden)

    def forward(self, hidden):
        # 对应原版 preact 逻辑：
        #   preact=True  → norm → act → conv
        #   preact=False → conv → norm → act
        if self._preact:
            hidden = self._norm(hidden)
            hidden = self._act(hidden)
            hidden = self._layer(hidden)
        else:
            hidden = self._layer(hidden)
            hidden = self._norm(hidden)
            hidden = self._act(hidden)
        return hidden

    def _layer(self, x):
        # x: (N, H, W, C_in)  channel-last（与原版 JAX NHWC 一致）
        in_ch = x.shape[-1]
        k     = self._kernel
        pad   = k // 2 if self._pad == "SAME" else 0

        if self._transp:
            # 对应：jax.lax.conv_transpose(..., "NHWC", "HWOI", "NHWC")
            # kernel shape: (kH, kW, out, in) → PyTorch ConvTranspose2d: (in, out, kH, kW)
            kname = "kernel"
            shape = (k, k, self._depth, in_ch)
            kernel_param = self.get(kname, Initializer(self._winit, fan=self._fan), shape)
            kernel_param = cast(kernel_param)
            # reshape to PyTorch: (in_ch, depth, k, k)
            w = kernel_param.reshape(k, k, self._depth, in_ch)
            w = w.permute(3, 2, 0, 1).contiguous()   # (in, out, kH, kW)
            x_cf = x.permute(0, 3, 1, 2).contiguous()
            out_pad = self._stride - 1 if self._stride > 1 else 0
            out = F.conv_transpose2d(x_cf, w, stride=self._stride,
                                     padding=pad, output_padding=out_pad)
        else:
            # 对应：jax.lax.conv_general_dilated(..., "NHWC", "HWIO", "NHWC")
            # kernel shape: (kH, kW, in, out) → PyTorch Conv2d: (out, in, kH, kW)
            kname = "kernel"
            shape = (k, k, in_ch, self._depth)
            kernel_param = self.get(kname, Initializer(self._winit, fan=self._fan), shape)
            kernel_param = cast(kernel_param)
            w = kernel_param.reshape(k, k, in_ch, self._depth)
            w = w.permute(3, 2, 0, 1).contiguous()   # (out, in, kH, kW)
            x_cf = x.permute(0, 3, 1, 2).contiguous()
            out = F.conv2d(x_cf, w, stride=self._stride, padding=pad)

        if self._bias:
            # 对应：bias = self.get("bias", jnp.zeros, depth, float32)
            bias = self.get("bias", _zeros_param, (self._depth,))
            bias = cast(bias)
            out = out + bias.view(1, -1, 1, 1)

        # 转回 channel-last
        out = out.permute(0, 2, 3, 1).contiguous()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Linear
# ─────────────────────────────────────────────────────────────────────────────

class Linear(NJModule):
    def __init__(
        self,
        units,
        act="none",
        norm="none",
        bias=True,
        outscale=1.0,
        outnorm=False,
        winit="uniform",
        fan="avg",
    ):
        super().__init__()
        self._units    = tuple(units) if hasattr(units, "__len__") else (units,)
        self._act      = get_act(act)
        self._norm     = norm
        self._bias     = bias and norm == "none"
        self._outscale = outscale
        self._outnorm  = outnorm
        self._winit    = winit
        self._fan      = fan

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # 对应：
        #   shape = (x.shape[-1], np.prod(self._units))
        #   kernel = self.get("kernel", Initializer(...), shape)
        #   x = x @ kernel
        shape  = (x.shape[-1], int(np.prod(self._units)))
        kernel = self.get("kernel", Initializer(self._winit, self._outscale,
                                                fan=self._fan), shape)
        kernel = cast(kernel)
        x = x @ kernel   # (*, in) @ (in, out) = (*, out)

        if self._bias:
            # 对应：bias = self.get("bias", jnp.zeros, np.prod(units), float32)
            bias = self.get("bias", _zeros_param, (int(np.prod(self._units)),))
            bias = cast(bias)
            x = x + bias

        if len(self._units) > 1:
            x = x.reshape(x.shape[:-1] + self._units)

        # 对应：x = self.get("norm", Norm, self._norm)(x)
        x = self.get("norm", Norm, self._norm)(x)
        x = self._act(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Norm
# ─────────────────────────────────────────────────────────────────────────────

class Norm(NJModule):
    def __init__(self, impl):
        super().__init__()
        self._impl = impl

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        dtype = x.dtype
        if self._impl == "none":
            return x
        elif self._impl == "layer":
            # 对应：
            #   x = x.astype(f32)
            #   x = jax.nn.standardize(x, axis=-1, epsilon=1e-3)
            #   x *= self.get("scale", jnp.ones,  x.shape[-1], f32)
            #   x += self.get("bias",  jnp.zeros, x.shape[-1], f32)
            #   return x.astype(dtype)
            x = x.float()
            # jax.nn.standardize: (x - mean) / sqrt(var + eps)，无仿射参数
            # 然后手动乘 scale 加 bias
            mean = x.mean(dim=-1, keepdim=True)
            var  = x.var(dim=-1, keepdim=True, unbiased=False)
            x    = (x - mean) / torch.sqrt(var + 1e-3)
            # scale 和 bias 是可学习参数，通过 self.get 注册
            scale = self.get("scale", _ones_param,  (x.shape[-1],))
            bias  = self.get("bias",  _zeros_param, (x.shape[-1],))
            x = x * scale + bias
            return x.to(dtype)
        else:
            raise NotImplementedError(self._impl)


# ─────────────────────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────────────────────

class Input:
    def __init__(self, keys=["tensor"], dims=None):
        assert isinstance(keys, (list, tuple)), keys
        self._keys = tuple(keys)
        self._dims = dims or self._keys[0]

    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"tensor": inputs}
        inputs = inputs.copy()
        for key in self._keys:
            if key.startswith("softmax_"):
                inputs[key] = F.softmax(inputs[key[len("softmax_"):]], dim=-1)
        if not all(k in inputs for k in self._keys):
            needs = f'{{{", ".join(self._keys)}}}'
            found = f'{{{", ".join(inputs.keys())}}}'
            raise KeyError(f"Cannot find keys {needs} among inputs {found}.")
        values = [inputs[k] for k in self._keys]
        # 对应：dims = len(inputs[self._dims].shape)
        dims = inputs[self._dims].dim()
        for i, value in enumerate(values):
            if value.dim() > dims:
                values[i] = value.reshape(
                    value.shape[: dims - 1] + (int(np.prod(value.shape[dims - 1:])),))
        # 对应：values = [x.astype(inputs[self._dims].dtype) for x in values]
        ref_dtype = inputs[self._dims].dtype
        values = [x.to(ref_dtype) for x in values]
        return torch.cat(values, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Initializer
# ─────────────────────────────────────────────────────────────────────────────

class Initializer:
    def __init__(self, dist="uniform", scale=1.0, fan="avg"):
        self.scale = scale
        self.dist  = dist
        self.fan   = fan

    def __call__(self, shape):
        if self.scale == 0.0:
            return torch.zeros(shape, dtype=f32)
        elif self.dist == "uniform":
            fanin, fanout = self._fans(shape)
            denoms = {"avg": (fanin + fanout) / 2,
                      "in":  fanin, "out": fanout}
            scale = self.scale / denoms[self.fan]
            limit = np.sqrt(3 * scale)
            return torch.empty(shape, dtype=f32).uniform_(-limit, limit)
        elif self.dist == "normal":
            fanin, fanout = self._fans(shape)
            denoms = {"avg": np.mean((fanin, fanout)),
                      "in":  fanin, "out": fanout}
            scale = self.scale / denoms[self.fan]
            std   = np.sqrt(scale) / 0.87962566103423978
            # 对应：std * jax.random.truncated_normal(rng, -2, 2, shape, f32)
            val   = torch.empty(shape, dtype=f32).normal_(0, std)
            val   = torch.clamp(val, -2 * std, 2 * std)
            return val
        elif self.dist == "ortho":
            # 对应：
            #   nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
            #   matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
            #   mat = normal(matshape); Q,R = qr(mat)
            #   Q *= sign(diag(R)); Q = Q.T if nrows < ncols
            #   Q = Q.reshape(nrows, *shape[:-1])
            #   value = scale * moveaxis(Q, 0, -1)
            nrows = shape[-1]
            ncols = int(np.prod(shape)) // shape[-1]
            matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
            mat  = torch.randn(*matshape, dtype=f32)
            Q, R = torch.linalg.qr(mat)
            Q   *= torch.sign(torch.diag(R))
            if nrows < ncols:
                Q = Q.T
            Q   = Q.reshape(nrows, *shape[:-1])
            # moveaxis(Q, 0, -1): 把轴 0 移到最后
            dims = list(range(1, Q.dim())) + [0]
            val  = self.scale * Q.permute(dims)
            return val
        else:
            raise NotImplementedError(self.dist)

    def _fans(self, shape):
        if len(shape) == 0:
            return 1, 1
        elif len(shape) == 1:
            return shape[0], shape[0]
        elif len(shape) == 2:
            return shape   # (fanin, fanout)
        else:
            space = int(np.prod(shape[:-2]))
            return shape[-2] * space, shape[-1] * space


# ─────────────────────────────────────────────────────────────────────────────
# get_act
# ─────────────────────────────────────────────────────────────────────────────

def get_act(name):
    """对应原版 get_act(name)，函数名和参数不变"""
    if callable(name):
        return name
    elif name == "none":
        return lambda x: x
    elif name == "mish":
        return lambda x: x * torch.tanh(F.softplus(x))
    elif name == "relu":
        return F.relu
    elif name == "elu":
        return F.elu
    elif name == "silu" or name == "swish":
        return F.silu
    elif name == "tanh":
        return torch.tanh
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "leaky_relu":
        return F.leaky_relu
    elif name == "gelu":
        return F.gelu
    elif name == "softplus":
        return F.softplus
    else:
        raise NotImplementedError(name)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：_zeros_param / _ones_param 工厂函数
# 用于 NJModule.get() 中替代 jnp.zeros / jnp.ones 生成 nn.Parameter
# ─────────────────────────────────────────────────────────────────────────────

def _zeros_param(shape):
    """对应 jnp.zeros(shape, f32)，供 NJModule.get() 生成 nn.Parameter 使用"""
    return nn.Parameter(torch.zeros(shape, dtype=f32))

def _ones_param(shape):
    """对应 jnp.ones(shape, f32)，供 NJModule.get() 生成 nn.Parameter 使用"""
    return nn.Parameter(torch.ones(shape, dtype=f32))