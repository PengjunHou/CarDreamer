"""
expl.py  ——  JAX → PyTorch 转换
函数名、签名与原版完全一致。
"""

import torch
import torch.nn as nn

from . import jaxutils
from . import nets
from .nets import Input, MLP, NJModule
from .jaxutils import sg


class Disag(NJModule):
    def __init__(self, wm, act_space, config):
        super().__init__()

        # 对应：self.config = config.update({"disag_head.inputs": ["tensor"]})
        self.config = config.update({"disag_head.inputs": ["tensor"]})

        # 对应：self.opt = jaxutils.Optimizer(name="disag_opt", **config.expl_opt)
        self.opt = jaxutils.Optimizer(name="disag_opt", **config.expl_opt)

        # 对应：self.inputs = nets.Input(config.disag_head.inputs, dims="deter")
        self.inputs = Input(self.config.disag_head.inputs, dims="deter")

        # 对应：self.target = nets.Input(self.config.disag_target, dims="deter")
        self.target = Input(self.config.disag_target, dims="deter")

        # 对应：self.nets = [nets.MLP(shape=None, **self.config.disag_head, name=f"disag{i}")
        #                    for i in range(self.config.disag_models)]
        # 原版用 list，PyTorch 需用 nn.ModuleList 才能正确追踪参数
        self.nets = nn.ModuleList([
            MLP(shape=None, **self.config.disag_head)
            for i in range(self.config.disag_models)
        ])

    def __call__(self, traj):
        return self.forward(traj)

    def forward(self, traj):
        inp = self.inputs(traj)
        # 对应：preds = jnp.array([net(inp).mode() for net in self.nets])
        preds = torch.stack([net(inp).mode() for net in self.nets], dim=0)
        # 对应：return preds.std(0).mean(-1)[1:]
        return preds.std(0).mean(-1)[1:]

    def train(self, data):
        # 对应：return self.opt(self.nets, self.loss, data)
        # list(self.nets) 与 Optimizer.__call__ 的 modules 参数接口一致
        return self.opt(list(self.nets), self.loss, data)

    def loss(self, data):
        # 对应：
        #   inp = sg(self.inputs(data)[:, :-1])
        #   tar = sg(self.target(data)[:, 1:])
        inp = sg(self.inputs(data)[:, :-1])
        tar = sg(self.target(data)[:, 1:])

        losses = []
        for net in self.nets:
            # 对应：net._shape = tar.shape[2:]
            net._shape = tar.shape[2:]
            # 对应：losses.append(-net(inp).log_prob(tar).mean())
            losses.append(-net(inp).log_prob(tar).mean())

        # 对应：return jnp.array(losses).sum()
        return torch.stack(losses).sum()