"""
jaxutils.py  ——  JAX/TFP → PyTorch 完整转换
保持与原版完全相同的对外接口和内部逻辑。
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    OneHotCategorical, Independent, Normal, Categorical
)

# ─────────────────────────────────────────────────────────────────────────────
# 全局常量 & 基础工具
# ─────────────────────────────────────────────────────────────────────────────

COMPUTE_DTYPE = torch.float32


def cast_to_compute(values):
    """将 dict/tensor 转换为 COMPUTE_DTYPE（对应原 cast_to_compute）"""
    if isinstance(values, dict):
        return {k: v.to(COMPUTE_DTYPE) for k, v in values.items()}
    return values.to(COMPUTE_DTYPE)


def sg(x):
    """对应 tree_map(jax.lax.stop_gradient, x)"""
    if isinstance(x, dict):
        return {k: sg(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(sg(v) for v in x)
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x



def parallel():
    """原版用于检测 JAX 多设备，PyTorch 中始终返回 False（单设备路径）"""
    return False

# ─────────────────────────────────────────────────────────────────────────────
# tensorstats / subsample
# ─────────────────────────────────────────────────────────────────────────────

def subsample(values, amount=1024):
    """随机下采样（对应原 subsample，用 torch.randperm 替代 jax.random.permutation）"""
    values = values.flatten()
    if values.numel() > amount:
        idx = torch.randperm(values.numel(), device=values.device)[:amount]
        values = values[idx]
    return values


def tensorstats(tensor, prefix=None):
    """返回张量统计量字典（对应原 tensorstats）"""
    metrics = {
        "mean": tensor.mean(),
        "std":  tensor.std(),
        "mag":  tensor.abs().max(),
        "min":  tensor.min(),
        "max":  tensor.max(),
        "dist": subsample(tensor),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# scan（替代 jax.lax.scan / nj.scan）
# ─────────────────────────────────────────────────────────────────────────────

def scan(fn, inputs, start, unroll=True, modify=False):
    """
    对应原 scan。
    fn(carry, inp) -> (carry, out)
    inputs: dict 或 tensor，第 0 维为时间步
    start:  carry 初始值（dict 或 tensor）
    返回与 start 结构相同的 out stacked 张量。
    """
    def fn2(carry, inp):
        out = fn(carry, inp)
        return out, out   # (new_carry, output) 形式

    # 取第一个叶子节点确定长度
    if isinstance(inputs, dict):
        length = next(iter(inputs.values())).shape[0]
        get_t  = lambda t: {k: v[t] for k, v in inputs.items()}
    else:
        length = inputs.shape[0]
        get_t  = lambda t: inputs[t]

    carry = start
    outs  = []
    for t in range(length):
        inp   = get_t(t)
        carry, out = fn2(carry, inp)
        outs.append(out)

    # stack: list of (dict or tensor) -> dict of stacked tensors / stacked tensor
    if isinstance(outs[0], dict):
        return {k: torch.stack([o[k] for o in outs], dim=0) for k in outs[0]}
    return torch.stack(outs, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# symlog / symexp
# ─────────────────────────────────────────────────────────────────────────────

def symlog(x):
    return x.sign() * (1 + x.abs()).log()


def symexp(x):
    return x.sign() * (x.abs().exp() - 1)

# ─────────────────────────────────────────────────────────────────────────────
# OneHotDist
# ─────────────────────────────────────────────────────────────────────────────

class OneHotDist:
    """
    对应原 OneHotDist(tfd.OneHotCategorical)。
    直通梯度（straight-through estimator）：
        sample = sg(hard) + (probs - sg(probs))
    """
    def __init__(self, logits=None, probs=None, dtype=torch.float32):
        self.dtype = dtype
        if logits is not None:
            self._dist = OneHotCategorical(logits=logits)
            self._logits = logits
        else:
            self._dist = OneHotCategorical(probs=probs)
            self._logits = None
        self.batch_shape = self._dist.batch_shape
        self.event_shape = self._dist.event_shape

    # ---- probs_parameter ----
    def probs_parameter(self):
        return self._dist.probs

    # ---- sample：直通梯度 ----
    def sample(self, sample_shape=(), seed=None):
        with torch.no_grad():
            hard = self._dist.sample(sample_shape)          # (shape, num_classes)
        probs = self._pad(self._dist.probs, hard.shape)
        # straight-through
        return sg(hard) + (probs - sg(probs)).to(hard.dtype)

    def _pad(self, tensor, shape):
        while tensor.dim() < len(shape):
            tensor = tensor.unsqueeze(0)
        return tensor

    def log_prob(self, value):
        return self._dist.log_prob(value)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        idx = self._dist.probs.argmax(-1)
        return F.one_hot(idx, self._dist.probs.shape[-1]).to(self.dtype)

    @property
    def mean(self):
        return self._dist.probs

# ─────────────────────────────────────────────────────────────────────────────
# MultiHotDist
# ─────────────────────────────────────────────────────────────────────────────

class MultiHotDist:
    """对应原 MultiHotDist（多个 OneHotDist 的组合）"""
    def __init__(self, logits_array, dtype=torch.float32):
        self.shapes = [logits.shape[-1] for logits in logits_array]
        # 分割索引（cumsum[:-1]）
        self.split_sizes = self.shapes          # 用于 torch.split
        self.split_indices = []
        idx = 0
        for s in self.shapes[:-1]:
            idx += s
            self.split_indices.append(idx)
        self.dists = [OneHotDist(logits=logits, dtype=dtype) for logits in logits_array]
        self.prior_sample = None

    def get_prior_sample(self):
        return self.dists[-1].sample()

    def set_prior_sample(self, sample):
        self.prior_sample = sample

    def sample(self, sample_shape=(), seed=None):
        samples = [d.sample(sample_shape) for d in self.dists]
        if self.prior_sample is not None:
            samples[-1] = self.prior_sample
        return torch.cat(samples, dim=-1)

    def entropy(self):
        return sum(d.entropy() for d in self.dists)

    def log_prob(self, value):
        parts = torch.split(value, self.shapes, dim=-1)
        return sum(d.log_prob(p) for d, p in zip(self.dists, parts))

    @property
    def batch_shape(self):
        return self.dists[0].batch_shape

    @property
    def event_shape(self):
        return (sum(d.event_shape[0] for d in self.dists),)

# ─────────────────────────────────────────────────────────────────────────────
# MSEDist
# ─────────────────────────────────────────────────────────────────────────────

class MSEDist:
    """对应原 MSEDist"""
    def __init__(self, mode, dims, agg="sum"):
        self._mode = mode
        self._dims = tuple(range(-dims, 0))   # 负索引，等价于原版
        self._agg  = agg
        self.batch_shape = mode.shape[:mode.dim() - dims]
        self.event_shape = mode.shape[mode.dim() - dims:]

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss

# ─────────────────────────────────────────────────────────────────────────────
# SymlogDist
# ─────────────────────────────────────────────────────────────────────────────

class SymlogDist:
    """对应原 SymlogDist"""
    def __init__(self, mode, dims, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dims = tuple(range(-dims, 0))
        self._dist = dist
        self._agg  = agg
        self._tol  = tol
        self.batch_shape = mode.shape[:mode.dim() - dims]
        self.event_shape = mode.shape[mode.dim() - dims:]

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2
            distance = torch.where(distance < self._tol,
                                    torch.zeros_like(distance), distance)
        elif self._dist == "abs":
            distance = (self._mode - symlog(value)).abs()
            distance = torch.where(distance < self._tol,
                                    torch.zeros_like(distance), distance)
        else:
            raise NotImplementedError(self._dist)

        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss

class DiscDist:
    """
    对应原 DiscDist（离散化分布，用于 symlog 变换后的预测）。
    用 torch 重写 log_prob 中的 one_hot 与 logsumexp。
    """
    def __init__(self, logits, dims=0, low=-20, high=20,
                    transfwd=symlog, transbwd=symexp):
        self.logits   = logits
        self.probs    = F.softmax(logits, dim=-1)
        self.dims     = tuple(range(-dims, 0)) if dims > 0 else ()
        self.bins     = torch.linspace(low, high, logits.shape[-1],
                                        device=logits.device, dtype=logits.dtype)
        self.low      = low
        self.high     = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self.batch_shape = logits.shape[:logits.dim() - dims - 1]
        self.event_shape = logits.shape[logits.dim() - dims: -1]

    def mean(self):
        return self.transbwd((self.probs * self.bins).sum(-1))

    def mode(self):
        return self.transbwd((self.probs * self.bins).sum(-1))

    def log_prob(self, x):
        x    = self.transfwd(x)
        bins = self.bins

        # 与原版完全等价的 bin 查找
        below = (bins <= x.unsqueeze(-1)).int().sum(-1) - 1
        above = bins.shape[0] - (bins > x.unsqueeze(-1)).int().sum(-1)
        below = below.clamp(0, bins.shape[0] - 1)
        above = above.clamp(0, bins.shape[0] - 1)

        equal         = (below == above)
        dist_to_below = torch.where(equal, torch.ones_like(x), (bins[below] - x).abs())
        dist_to_above = torch.where(equal, torch.ones_like(x), (bins[above] - x).abs())
        total         = dist_to_below + dist_to_above
        weight_below  = dist_to_above / total
        weight_above  = dist_to_below / total

        target = (F.one_hot(below, bins.shape[0]).float() * weight_below.unsqueeze(-1)
                + F.one_hot(above, bins.shape[0]).float() * weight_above.unsqueeze(-1))

        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        lp = (target * log_pred).sum(-1)
        if self.dims:
            lp = lp.sum(self.dims)
        return lp


def video_grid(video):
    """
    对应原 video_grid。
    输入: (B, T, H, W, C)  →  输出: (T, H, B*W, C)
    原版: video.transpose((1,2,0,3,4)).reshape((T, H, B*W, C))
    """
    B, T, H, W, C = video.shape
    # transpose: (B,T,H,W,C) -> (T,H,B,W,C)
    video = video.permute(1, 2, 0, 3, 4).contiguous()
    return video.view(T, H, B * W, C)


def balance_stats(dist, target, thres):
    """
    对应原 balance_stats。
    pos_acc / neg_acc / rate / avg / pred / pos_loss / neg_loss。
    NaN 在无正/负样本时保留（与原版一致，用 np.nanmean 聚合）。
    """
    target = target.float()
    pos    = (target > thres).float()
    neg    = (target <= thres).float()
    pred   = (dist.mean.float() > thres).float() if hasattr(dist, 'mean') \
                else (dist.mean().float() > thres).float()

    loss = -dist.log_prob(target)
    pos_sum = pos.sum()
    neg_sum = neg.sum()

    pos_loss = (loss * pos).sum() / pos_sum   # NaN when pos_sum==0
    neg_loss = (loss * neg).sum() / neg_sum
    pos_acc  = (pred * pos).sum() / pos_sum
    neg_acc  = ((1 - pred) * neg).sum() / neg_sum

    return dict(
        pos_loss=pos_loss,
        neg_loss=neg_loss,
        pos_acc=pos_acc,
        neg_acc=neg_acc,
        rate=pos.mean(),
        avg=target.mean(),
        pred=dist.mean.float().mean() if hasattr(dist, 'mean') and isinstance(dist.mean, torch.Tensor)
                else dist.mean().float().mean(),
    )

class Moments(nn.Module):
    """
    对应原 Moments(nj.Module)。
    支持 impl: off / mean_std / min_max / perc_ema / perc_ema_corr /
               mean_mag / max_mag
    所有状态用 nn.Parameter(requires_grad=False) 存储，等价于原版 nj.Variable。
    """

    def __init__(self, impl="mean_std", decay=0.99, max=1e8, eps=0.0,
                 perclo=5, perchi=95):
        super().__init__()
        self.impl   = impl
        self.decay  = decay
        self.max    = max
        self.eps    = eps
        self.perclo = perclo
        self.perchi = perchi

        def buf(val=0.0):
            return nn.Parameter(torch.tensor(val, dtype=torch.float32),
                                 requires_grad=False)

        if impl == "off":
            pass
        elif impl == "mean_std":
            self.step = buf(0.0)   # 用 float 存，避免 int 运算问题
            self.mean = buf(0.0)
            self.sqrs = buf(0.0)
        elif impl == "min_max":
            self.low  = buf(0.0)
            self.high = buf(0.0)
        elif impl == "perc_ema":
            self.low  = buf(0.0)
            self.high = buf(0.0)
        elif impl == "perc_ema_corr":
            self.step = buf(0.0)
            self.low  = buf(0.0)
            self.high = buf(0.0)
        elif impl == "mean_mag":
            self.mag  = buf(0.0)
        elif impl == "max_mag":
            self.mag  = buf(0.0)
        else:
            raise NotImplementedError(impl)

    def forward(self, x):
        self.update(x)
        return self.stats()

    def update(self, x):
        x = sg(x.float())
        m = self.decay

        if self.impl == "off":
            return
        elif self.impl == "mean_std":
            self.step.data += 1
            self.mean.data.mul_(m).add_((1 - m) * x.mean())
            self.sqrs.data.mul_(m).add_((1 - m) * (x * x).mean())
        elif self.impl == "min_max":
            lo, hi = x.min(), x.max()
            self.low.data.mul_(m).add_((1 - m) * lo).clip_(max=lo)
            self.high.data.mul_(m).add_((1 - m) * hi).clip_(min=hi)
            # 原版：low = m*min(low,lo) + (1-m)*lo
            self.low.data.copy_(m * torch.minimum(self.low.data, lo) + (1 - m) * lo)
            self.high.data.copy_(m * torch.maximum(self.high.data, hi) + (1 - m) * hi)
        elif self.impl == "perc_ema":
            lo = torch.quantile(x, self.perclo / 100.0)
            hi = torch.quantile(x, self.perchi / 100.0)
            self.low.data.mul_(m).add_((1 - m) * lo)
            self.high.data.mul_(m).add_((1 - m) * hi)
        elif self.impl == "perc_ema_corr":
            self.step.data += 1
            lo = torch.quantile(x, self.perclo / 100.0)
            hi = torch.quantile(x, self.perchi / 100.0)
            self.low.data.mul_(m).add_((1 - m) * lo)
            self.high.data.mul_(m).add_((1 - m) * hi)
        elif self.impl == "mean_mag":
            curr = x.abs().mean()
            self.mag.data.mul_(m).add_((1 - m) * curr)
        elif self.impl == "max_mag":
            curr = x.abs().max()
            self.mag.data.copy_(
                m * torch.maximum(self.mag.data, curr) + (1 - m) * curr)
        else:
            raise NotImplementedError(self.impl)

    def stats(self):
        if self.impl == "off":
            return torch.tensor(0.0), torch.tensor(1.0)
        elif self.impl == "mean_std":
            corr   = 1 - self.decay ** self.step.item()
            mean   = self.mean / corr
            var    = self.sqrs / corr - self.mean ** 2
            std    = (var.clamp(min=1 / self.max ** 2) + self.eps).sqrt()
            return sg(mean), sg(std)
        elif self.impl in ("min_max", "perc_ema"):
            offset   = self.low.data
            invscale = torch.clamp(self.high.data - self.low.data,
                                    min=1 / self.max)
            return sg(offset), sg(invscale)
        elif self.impl == "perc_ema_corr":
            corr     = 1 - self.decay ** self.step.item()
            lo       = self.low  / corr
            hi       = self.high / corr
            invscale = torch.clamp(hi - lo, min=1 / self.max)
            return sg(lo), sg(invscale)
        elif self.impl in ("mean_mag", "max_mag"):
            offset   = torch.tensor(0.0, device=self.mag.device)
            invscale = torch.clamp(self.mag.data, min=1 / self.max)
            return sg(offset), sg(invscale)
        else:
            raise NotImplementedError(self.impl)

class Optimizer:
    """
    与原版 Optimizer(nj.Module) 逻辑严格对齐的 PyTorch 实现。

    __init__ 签名与原版完全一致，额外增加 name 参数替代 nj.Module 基类注入。
    """

    PARAM_COUNTS: dict = {}

    def __init__(
        self,
        lr,
        opt="adam",
        eps=1e-5,
        clip=100.0,
        warmup=0,
        wd=0.0,
        wd_pattern=r"/(w|kernel)$",
        lateclip=0.0,
        name="optimizer",        # 替代 nj.Module 的 self.name / self.path
    ):
        assert opt in ("adam", "belief", "yogi")
        assert wd_pattern[0] not in ("0", "1")

        self.name     = name
        self.lr       = lr
        self.eps      = eps
        self.clip     = clip
        self.warmup   = warmup
        self.wd       = wd
        self.wd_re    = re.compile(wd_pattern)   # 对应原版 wd_pattern = re.compile(wd_pattern)
        self.lateclip = lateclip

        # 对应原版 self.PARAM_COUNTS[self.path] = None
        self.PARAM_COUNTS[self.name] = None

        # 对应原版 self.step = nj.Variable(jnp.array, 0, jnp.int32, name="step")
        self._step = 0

        # 对应原版 self.scaling = COMPUTE_DTYPE == jnp.float16
        self.scaling = (COMPUTE_DTYPE == torch.float16)
        if self.scaling:
            # 对应原版：
            #   self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
            #   self.grad_scale = nj.Variable(jnp.array, 1e4, jnp.float32)
            #   self.good_steps = nj.Variable(jnp.array, 0,   jnp.int32)
            self._grad_scale             = 1e4
            self._good_steps             = 0
            self._consecutive_errors     = 0      # apply_if_finite 内部计数
            self._MAX_CONSECUTIVE_ERRORS = 1000   # max_consecutive_errors=1000

        # 对应原版 optstate（通过 self.get/put 延迟初始化）
        self._opt_state: dict = {}
        self._initialized = False

    def __call__(self, modules, lossfn, *args, has_aux=False, **kwargs):
        """
        签名与原版完全一致：
            (self, modules, lossfn, *args, has_aux=False, **kwargs)
        """
        modules_list = modules if isinstance(modules, (list, tuple)) else [modules]

        # ── 收集参数（对应 nj.grad 拿到的 params dict）───────────────────────
        # key 格式："/layer/weight"，模拟 ninjax 路径，供 wd_pattern 匹配
        named_params: dict = {}
        for mod in modules_list:
            for pname, param in mod.named_parameters():
                if param.requires_grad:
                    key = "/" + pname.replace(".", "/")
                    named_params[key] = param

        # ── 对应原版 wrapped 函数 ─────────────────────────────────────────────
        # def wrapped(*args, **kwargs):
        #     outs = lossfn(...)
        #     loss, aux = ...
        #     assert loss.dtype == jnp.float32
        #     assert loss.shape == ()
        #     if self.scaling:
        #         loss *= sg(self.grad_scale.read())
        #     return loss, aux
        for mod in modules_list:
            mod.zero_grad()

        outs = lossfn(*args, **kwargs)
        loss, aux = outs if has_aux else (outs, None)
        assert loss.dtype == torch.float32, (self.name, loss.dtype)
        assert loss.shape == (),            (self.name, loss.shape)

        if self.scaling:
            # loss *= sg(self.grad_scale.read())，sg 即 detach
            (loss * self._grad_scale).backward()
        else:
            loss.backward()

        # ── 对应 nj.grad：拿到 params 和 grads ───────────────────────────────
        # loss, params, grads, aux = nj.grad(wrapped, modules, has_aux=True)(...)
        grads: dict = {}
        for key, param in named_params.items():
            grads[key] = param.grad.clone() if param.grad is not None \
                         else torch.zeros_like(param.data)

        # ── 对应原版参数量打印 ────────────────────────────────────────────────
        # if not self.PARAM_COUNTS[self.path]:
        #     count = sum([np.prod(x.shape) for x in params.values()])
        #     print(f"Optimizer {self.name} has {count:,} variables.")
        #     self.PARAM_COUNTS[self.path] = count
        if not self.PARAM_COUNTS[self.name]:
            count = sum(p.numel() for p in named_params.values())
            print(f"Optimizer {self.name} has {count:,} variables.")
            self.PARAM_COUNTS[self.name] = count

        # ── 对应原版 parallel() pmean（单设备跳过）────────────────────────────
        # if parallel():
        #     grads = tree_map(lambda x: jax.lax.pmean(x, "i"), grads)
        # PyTorch 单设备，跳过

        # ── FP16：unscale + _update_scale ─────────────────────────────────────
        # if self.scaling:
        #     grads = tree_map(lambda x: x / self.grad_scale.read(), grads)
        #     finite = self._update_scale(grads)
        #     metrics[...] = self.grad_scale.read()
        #     metrics[...] = (~finite).astype(float)
        metrics: dict = {}
        if self.scaling:
            grads = {k: g / self._grad_scale for k, g in grads.items()}
            finite = self._update_scale(grads)   # 传入 grads，内部判断 finite，与原版一致
            metrics[f"grad_scale"]    = self._grad_scale
            metrics[f"grad_overflow"] = float(not finite)

        # ── optstate 延迟初始化（对应 self.get("state", self.opt.init, params)）
        if not self._initialized:
            for key, param in named_params.items():
                self._opt_state[key] = {
                    "exp_avg":    torch.zeros_like(param.data),
                    "exp_avg_sq": torch.zeros_like(param.data),
                    "step":       0,
                }
            self._initialized = True

        # ── updates, optstate = self.opt.update(grads, optstate, params) ─────
        # 对应 optax chain：
        #   1. clip_by_global_norm(clip)
        #   2. scale_by_adam(eps)
        #   3. late_grad_clip(lateclip)
        #   4. additive_weight_decay(wd, mask)
        #   5. scale(-lr) 或 linear_warmup + scale(-lr)

        # chain 第 1 步：clip_by_global_norm
        if self.clip:
            total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads.values()))
            clip_coef  = self.clip / (total_norm + 1e-6)
            if clip_coef < 1.0:
                grads = {k: g * clip_coef for k, g in grads.items()}

        # chain 第 2 步：scale_by_adam（只做动量估计，不含 lr）
        beta1, beta2 = 0.9, 0.999
        updates: dict = {}
        for key, g in grads.items():
            st = self._opt_state[key]
            st["step"] += 1
            t = st["step"]
            st["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
            st["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            bias_corr1 = 1 - beta1 ** t
            bias_corr2 = 1 - beta2 ** t
            m_hat = st["exp_avg"]    / bias_corr1
            v_hat = st["exp_avg_sq"] / bias_corr2
            updates[key] = m_hat / (v_hat.sqrt() + self.eps)

        # chain 第 3 步：late_grad_clip（作用在 adam 输出的 updates 上）
        if self.lateclip:
            updates = {k: u.clamp(-self.lateclip, self.lateclip)
                       for k, u in updates.items()}

        # chain 第 4 步：additive_weight_decay（updates += wd * param）
        if self.wd:
            for key in updates:
                if self.wd_re.search(key):
                    updates[key] = updates[key] + self.wd * named_params[key].data

        # chain 第 5 步：scale(-lr) 或 linear_warmup + scale(-lr)
        if self.warmup and self._step < self.warmup:
            # optax.linear_schedule(0.0, -lr, warmup)：从 0 线性增到 -lr
            current_lr = self.lr * (self._step + 1) / self.warmup
        else:
            current_lr = self.lr
        updates = {k: -current_lr * u for k, u in updates.items()}

        # ── nj.context().update(apply_updates(params, updates)) ──────────────
        # apply_if_finite 语义（FP16 时）：
        #   finite                           → 执行更新，errors=0
        #   ~finite, errors <  1000          → 跳过更新，errors++
        #   ~finite, errors >= 1000（兜底）  → 强制更新，errors=0
        do_update = True
        if self.scaling:
            if not finite:
                if self._consecutive_errors < self._MAX_CONSECUTIVE_ERRORS:
                    self._consecutive_errors += 1
                    do_update = False
                else:
                    self._consecutive_errors = 0
                    do_update = True
            else:
                self._consecutive_errors = 0

        if do_update:
            for key, param in named_params.items():
                param.data.add_(updates[key])

        # ── norm = optax.global_norm(grads) ──────────────────────────────────
        # 原版在 apply_updates 之后、对裁剪后的 grads 计算 global_norm
        norm = torch.sqrt(sum(g.norm() ** 2 for g in grads.values()))

        # ── FP16：norm = where(isfinite(norm), norm, nan) ────────────────────
        if self.scaling:
            if not torch.isfinite(norm):
                norm = torch.tensor(float("nan"))

        # ── self.step.write(self.step.read() + isfinite(norm)) ───────────────
        self._step += int(torch.isfinite(norm).item()
                          if isinstance(norm, torch.Tensor)
                          else math.isfinite(float(norm)))

        # ── metrics（原版最后统一加 self.name_ 前缀）─────────────────────────
        # metrics["loss"]       = loss.mean()
        # metrics["grad_norm"]  = norm
        # metrics["grad_steps"] = self.step.read()
        # metrics = {f"{self.name}_{k}": v for k, v in metrics.items()}
        metrics["loss"]       = loss.detach()   # shape==()，.mean() 等价
        metrics["grad_norm"]  = norm
        metrics["grad_steps"] = torch.tensor(self._step)
        metrics = {f"{self.name}_{k}": v for k, v in metrics.items()}

        return (metrics, aux) if has_aux else metrics

    # ──────────────────────────────────────────────────────────────────────────
    # _update_scale
    # ──────────────────────────────────────────────────────────────────────────

    def _update_scale(self, grads) -> bool:
        # 对应原版：
        #   finite = jnp.array([jnp.isfinite(x).all()
        #                        for x in jax.tree_util.tree_leaves(grads)]).all()
        finite = all(torch.isfinite(g).all().item() for g in grads.values())

        keep = finite and (self._good_steps < 1000)
        incr = finite and (self._good_steps >= 1000)
        decr = not finite

        # good_steps = keep * (good_steps + 1)
        self._good_steps = int(keep) * (self._good_steps + 1)

        # grad_scale = clip(keep*s + incr*s*2 + decr*s/2, 1e-4, 1e4)
        self._grad_scale = max(1e-4, min(1e4,
            int(keep) * self._grad_scale
          + int(incr) * self._grad_scale * 2
          + int(decr) * self._grad_scale / 2
        ))

        return finite

class SlowUpdater:
    def __init__(self, src, dst, fraction=1.0, period=1):
        self.src      = src
        self.dst      = dst
        self.fraction = fraction
        self.period   = period
        self._updates = 0

    def __call__(self):
        updates     = self._updates
        need_init   = (updates == 0)
        need_update = (updates % self.period == 0)

        # 对应原版：mix = clip(1.0*need_init + fraction*need_update, 0, 1)
        mix = float(need_init) + self.fraction * float(need_update)
        mix = min(max(mix, 0.0), 1.0)

        src_state = self.src.state_dict()
        dst_state = self.dst.state_dict()

        new_state = {}
        for key in dst_state:
            s = src_state.get(key, dst_state[key])
            d = dst_state[key]
            if s.dtype.is_floating_point:
                # 对应原版：tree_map(lambda s, d: mix * s + (1 - mix) * d, source, dst)
                new_state[key] = mix * s + (1 - mix) * d
            else:
                # int/bool 参数：mix=1时取src，mix=0时取dst
                new_state[key] = s if mix >= 1.0 else d

        self.dst.load_state_dict(new_state)
        self._updates += 1
