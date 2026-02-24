"""
behaviors.py  ——  JAX/TFP → PyTorch 转换
函数名、签名与原版完全一致。
"""

import torch
import torch.nn as nn

from . import my_agent as agent
from . import expl, jaxutils
from .nets import NJModule, sg


class Greedy(NJModule):
    def __init__(self, wm, act_space, config):
        super().__init__()

        # 对应：rewfn = lambda s: wm.heads["reward"](s).mean()[1:]
        rewfn = lambda s: wm.heads["reward"](s).mean()[1:]

        if config.critic_type == "vfunction":
            # 对应：critics = {"extr": agent.VFunction(rewfn, config, name="critic")}
            critics = {"extr": agent.VFunction(rewfn, config, name="critic")}
        else:
            raise NotImplementedError(config.critic_type)

        # 对应：self.ac = agent.ImagActorCritic(critics, {"extr": 1.0}, act_space, config, name="ac")
        self.ac = agent.ImagActorCritic(
            critics, {"extr": 1.0}, act_space, config, name="ac")

    def initial(self, batch_size):
        # 对应：return self.ac.initial(batch_size)
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        # 对应：return self.ac.policy(latent, state)
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        # 对应：return self.ac.train(imagine, start, data)
        return self.ac.train(imagine, start, data)

    def report(self, data):
        return {}


class Random(NJModule):
    def __init__(self, wm, act_space, config):
        super().__init__()
        self.config    = config
        self.act_space = act_space

    def initial(self, batch_size):
        # 对应：return jnp.zeros(batch_size)
        return torch.zeros(batch_size)

    def policy(self, latent, state):
        # 对应：
        #   batch_size = len(state)
        #   shape = (batch_size,) + self.act_space.shape
        #   if self.act_space.discrete:
        #       dist = jaxutils.OneHotDist(jnp.zeros(shape))
        #   else:
        #       dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
        #       dist = tfd.Independent(dist, 1)
        #   return {"action": dist}, state
        batch_size = len(state)
        shape = (batch_size,) + self.act_space.shape

        if self.act_space.discrete:
            # 对应：jaxutils.OneHotDist(jnp.zeros(shape))
            dist = jaxutils.OneHotDist(torch.zeros(shape))
        else:
            # 对应：tfd.Independent(tfd.Uniform(-ones, ones), 1)
            low  = -torch.ones(shape)
            high =  torch.ones(shape)
            dist = torch.distributions.Independent(
                torch.distributions.Uniform(low, high), 1)

        return {"action": dist}, state

    def train(self, imagine, start, data):
        # 对应：return None, {}
        return None, {}

    def report(self, data):
        return {}


class Explore(NJModule):
    # 对应：REWARDS = {"disag": expl.Disag}
    REWARDS = {
        "disag": expl.Disag,
    }

    def __init__(self, wm, act_space, config):
        super().__init__()
        self.config  = config
        self.rewards = {}
        critics      = {}

        for key, scale in config.expl_rewards.items():
            # 对应：if not scale: continue
            if not scale:
                continue
            if key == "extr":
                # 对应：rewfn = lambda s: wm.heads["reward"](s).mean()[1:]
                #        critics[key] = agent.VFunction(rewfn, config, name=key)
                rewfn       = lambda s: wm.heads["reward"](s).mean()[1:]
                critics[key] = agent.VFunction(rewfn, config, name=key)
            else:
                # 对应：rewfn = self.REWARDS[key](wm, act_space, config, name=key+"_reward")
                #        critics[key] = agent.VFunction(rewfn, config, name=key)
                #        self.rewards[key] = rewfn
                rewfn        = self.REWARDS[key](wm, act_space, config, name=key + "_reward")
                critics[key] = agent.VFunction(rewfn, config, name=key)
                self.rewards[key] = rewfn

        # 对应：scales = {k: v for k, v in config.expl_rewards.items() if v}
        scales = {k: v for k, v in config.expl_rewards.items() if v}

        # 对应：self.ac = agent.ImagActorCritic(critics, scales, act_space, config, name="ac")
        self.ac = agent.ImagActorCritic(
            critics, scales, act_space, config, name="ac")

        # rewards 里的子模块需要注册到 PyTorch 参数树
        # 原版 ninjax 通过全局参数字典自动追踪，PyTorch 需显式注册
        for key, rewfn in self.rewards.items():
            if isinstance(rewfn, nn.Module):
                self.add_module(f"reward_{key}", rewfn)

    def initial(self, batch_size):
        # 对应：return self.ac.initial(batch_size)
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        # 对应：return self.ac.policy(latent, state)
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        # 对应：
        #   metrics = {}
        #   for key, rewfn in self.rewards.items():
        #       mets = rewfn.train(data)
        #       metrics.update({f"{key}_k": v for k, v in mets.items()})
        #   traj, mets = self.ac.train(imagine, start, data)
        #   metrics.update(mets)
        #   return traj, metrics
        metrics = {}
        for key, rewfn in self.rewards.items():
            mets = rewfn.train(data)
            metrics.update({f"{key}_{k}": v for k, v in mets.items()})
        traj, mets = self.ac.train(imagine, start, data)
        metrics.update(mets)
        return traj, metrics

    def report(self, data):
        return {}