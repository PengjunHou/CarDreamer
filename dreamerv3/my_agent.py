"""
agent.py  ——  JAX → PyTorch 完整转换
函数名、签名与原版完全一致。
"""

import logging
import numpy as np
import torch
import torch.nn as nn

# tree_map = lambda fn, tree, **kw: (
#     {k: tree_map(fn, v, **kw) for k, v in tree.items()} if isinstance(tree, dict)
#     else [tree_map(fn, v, **kw) for v in tree] if isinstance(tree, list)
#     else fn(tree)
# )
# sg = lambda x: tree_map(lambda v: v.detach() if isinstance(v, torch.Tensor) else v, x)

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from . import behaviors, jaxutils, jaxagent, nets
from .nets import NJModule
from .jaxagent import tree_map
from .jaxutils import sg

# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────
@jaxagent.Wrapper
class Agent(NJModule):
    """
    对应原版 @jaxagent.Wrapper class Agent(nj.Module)。
    @jaxagent.Wrapper 由 torch_agent.py 的 TorchAgent 承担。
    """

    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        self.config    = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step      = step

        self.wm = WorldModel(obs_space, act_space, config, name="wm")
        self.add_module("wm", self.wm)

        self.task_behavior = getattr(behaviors, config.task_behavior)(
            self.wm, self.act_space, self.config)
        self.add_module("task_behavior", self.task_behavior)

        # 对应：
        #   if config.expl_behavior == "None":
        #       self.expl_behavior = self.task_behavior
        #   else:
        #       self.expl_behavior = getattr(behaviors, config.expl_behavior)(...)
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(
                self.wm, self.act_space, self.config)
            self.add_module("expl_behavior", self.expl_behavior)

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    def policy(self, obs, state, mode="train"):
        # 对应：self.config.jax.jit and print("Tracing policy function.")
        self.config.jax.jit and print("Tracing policy function.")
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = state

        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(
            prev_latent, prev_action, embed, obs["is_first"])

        # 对应：self.expl_behavior.policy(latent, expl_state)  ← 无赋值，副作用调用
        self.expl_behavior.policy(latent, expl_state)

        task_outs, task_state = self.task_behavior.policy(latent, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)

        if mode == "eval":
            outs = task_outs
            # 对应：outs["action"].sample(seed=nj.rng())
            outs["action"] = outs["action"].sample()
            # 对应：jnp.zeros(outs["action"].shape[:1])
            outs["log_entropy"] = torch.zeros(outs["action"].shape[:1],
                                               device=outs["action"].device)
        elif mode == "explore":
            outs = expl_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"]      = outs["action"].sample()
        elif mode == "train":
            outs = task_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"]      = outs["action"].sample()

        state = ((latent, outs["action"]), task_state, expl_state)
        return outs, state

    def train(self, data, state):
        # 对应：self.config.jax.jit and print("Tracing train function.")
        self.config.jax.jit and print("Tracing train function.")
        metrics = {}
        data = self.preprocess(data)

        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)

        # 对应：context = {**data, **wm_outs["post"]}
        context = {**data, **wm_outs["post"]}

        # 对应：start = tree_map(lambda x: x.reshape([-1]+list(x.shape[2:])), context)
        start = {k: v.reshape([-1] + list(v.shape[2:]))
                 for k, v in context.items()
                 if isinstance(v, torch.Tensor)}

        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)

        if self.config.expl_behavior != "None":
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            # 对应：metrics.update({"expl_" + key: value for key, value in mets.items()})
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        # 对应：
        #   if "keyA" in data.keys():
        #       outs = {"key": ..., "env_step": ...,
        #               "model_loss": metrics["model_loss_raw"].copy(),
        #               "td_error":   metrics["td_error"].copy()}
        #   else:
        #       outs = {}
        if "keyA" in data.keys():
            outs = {
                "key":        data["key"],
                "env_step":   data["env_step"],
                "model_loss": metrics["model_loss_raw"].clone(),
                "td_error":   metrics["td_error"].clone(),
            }
        else:
            outs = {}

        # 对应：
        #   metrics.update({"model_loss_raw": metrics["model_loss_raw"].mean()})
        #   metrics.update({"td_error": metrics["td_error"].mean()})
        metrics.update({"model_loss_raw": metrics["model_loss_raw"].mean()})
        metrics.update({"td_error":       metrics["td_error"].mean()})

        return outs, state, metrics

    def report(self, data):
        # 对应：self.config.jax.jit and print("Tracing report function.")
        self.config.jax.jit and print("Tracing report function.")
        data   = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            # 对应：if key.startswith("log_") or key in ("key","env_step"): continue
            if key.startswith("log_") or key in ("key", "env_step"):
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(np.array(value))
            # 对应：
            #   if len(value.shape) > 3 and value.dtype == jnp.uint8:
            #       value = cast_to_compute(value) / 255.0
            #   else:
            #       value = value.astype(jnp.float32)
            if value.dim() > 3 and value.dtype == torch.uint8:
                value = jaxutils.cast_to_compute(value) / 255.0
            else:
                value = value.float()
            obs[key] = value
        # 对应：obs["cont"] = 1.0 - obs["is_terminal"].astype(jnp.float32)
        obs["cont"] = 1.0 - obs["is_terminal"].float()
        return obs


# ─────────────────────────────────────────────────────────────────────────────
# WorldModel
# ─────────────────────────────────────────────────────────────────────────────

class WorldModel(NJModule):
    def __init__(self, obs_space, act_space, config, name="wm"):
        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.config    = config

        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}

        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name="enc")
        self.add_module("encoder", self.encoder)

        self.rssm = nets.RSSM(**config.rssm) #, name="rssm")
        self.add_module("rssm", self.rssm)

        # 对应：self.heads = {"decoder": ..., "reward": ..., "cont": ...}
        # 原版用普通 dict，PyTorch 用 nn.ModuleDict 保证参数被追踪
        self.heads = nn.ModuleDict({
            "decoder": nets.MultiDecoder(shapes, **config.decoder, name="dec"),
            "reward":  nets.MLP((), **config.reward_head, name="rew"),
            "cont":    nets.MLP((), **config.cont_head,   name="cont"),
        })

        self.opt = jaxutils.Optimizer(name="model_opt", **config.model_opt)

        # 对应：
        #   scales = self.config.loss_scales.copy()
        #   image, vector = scales.pop("image"), scales.pop("vector")
        #   scales.update({k: image  for k in decoder.cnn_shapes})
        #   scales.update({k: vector for k in decoder.mlp_shapes})
        scales = dict(self.config.loss_scales)
        image  = scales.pop("image")
        vector = scales.pop("vector")
        scales.update({k: image  for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        # 对应：prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        prev_action = torch.zeros((batch_size,) + self.act_space.shape)
        return prev_latent, prev_action

    def train(self, data, state):
        # 对应：
        #   modules = [self.encoder, self.rssm, *self.heads.values()]
        #   mets, (state, outs, metrics) = self.opt(modules, self.loss, data, state, has_aux=True)
        #   metrics.update(mets)
        modules = [self.encoder, self.rssm, *self.heads.values()]
        mets, (state, outs, metrics) = self.opt(
            modules, self.loss, data, state, has_aux=True)
        metrics.update(mets)
        return state, outs, metrics

    def loss(self, data, state):
        embed = self.encoder(data)
        prev_latent, prev_action = state

        # 对应：prev_actions = jnp.concatenate([prev_action[:,None], data["action"][:,:-1]], 1)
        prev_actions = torch.cat(
            [prev_action.unsqueeze(1), data["action"][:, :-1]], dim=1)

        post, prior = self.rssm.observe(
            embed, prev_actions, data["is_first"], prev_latent)

        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            # 对应：out = head(feats if name in self.config.grad_heads else sg(feats))
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)

        losses = {}
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        for key, dist in dists.items():
            # 对应：loss = -dist.log_prob(data[key].astype(jnp.float32))
            loss = -dist.log_prob(data[key].float())
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss

        # 对应：scaled = {k: v * self.scales[k] for k, v in losses.items()}
        scaled     = {k: v * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())

        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})

        # 对应：
        #   last_latent = {k: v[:, -1] for k, v in post.items()}
        #   last_action = data["action"][:, -1]
        #   state = last_latent, last_action
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        state       = last_latent, last_action

        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        # 对应：metrics["model_loss_raw"] = model_loss
        metrics["model_loss_raw"] = model_loss

        return model_loss.mean(), (state, out, metrics)

    def imagine(self, policy, start, horizon):
        # 对应：first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        first_cont = (1.0 - start["is_terminal"]).float()

        keys  = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}

        # 对应：start["action"] = policy(start)
        start["action"] = policy(start)

        # 对应：
        #   def step(prev, _):
        #       prev = prev.copy()
        #       state = self.rssm.img_step(prev, prev.pop("action"))
        #       return {**state, "action": policy(state)}
        #   traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj_steps = []
        prev = start
        for _ in range(horizon):
            prev  = prev.copy()
            state = self.rssm.img_step(prev, prev.pop("action"))
            step  = {**state, "action": policy(state)}
            traj_steps.append(step)
            prev = step

        # 对应：traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        traj = {}
        for k in traj_steps[0].keys():
            stacked = torch.stack([s[k] for s in traj_steps], dim=0)
            traj[k] = torch.cat([start[k].unsqueeze(0), stacked], dim=0)

        # 对应：
        #   cont = self.heads["cont"](traj).mode()
        #   traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = torch.cat([first_cont.unsqueeze(0), cont[1:]], dim=0)

        # 对应：
        #   discount = 1 - 1 / self.config.horizon
        #   traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        discount       = 1 - 1 / self.config.horizon
        traj["weight"] = torch.cumprod(discount * traj["cont"], dim=0) / discount

        return traj

    def imagine_carry(self, policy, start, horizon, carry):
        # 对应：first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        first_cont = (1.0 - start["is_terminal"]).float()

        keys  = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}

        # 对应：
        #   outs, carry = policy(start, carry)
        #   start["action"] = outs
        #   start["carry"]  = carry
        outs, carry     = policy(start, carry)
        start["action"] = outs
        start["carry"]  = carry

        # 对应：
        #   def step(prev, _):
        #       prev = prev.copy(); carry = prev.pop("carry")
        #       state = self.rssm.img_step(prev, prev.pop("action"))
        #       outs, carry = policy(state, carry)
        #       return {**state, "action": outs, "carry": carry}
        #   traj = jaxutils.scan(step, jnp.arange(horizon), start, ...)
        traj_steps = []
        prev = start
        for _ in range(horizon):
            prev  = prev.copy()
            carry = prev.pop("carry")
            state = self.rssm.img_step(prev, prev.pop("action"))
            outs, carry = policy(state, carry)
            step  = {**state, "action": outs, "carry": carry}
            traj_steps.append(step)
            prev = step

        # 对应：traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()
        #               if k != "carry"}
        traj = {}
        for k in traj_steps[0].keys():
            if k == "carry":
                continue
            stacked = torch.stack([s[k] for s in traj_steps], dim=0)
            traj[k] = torch.cat([start[k].unsqueeze(0), stacked], dim=0)

        cont = self.heads["cont"](traj).mode()
        traj["cont"] = torch.cat([first_cont.unsqueeze(0), cont[1:]], dim=0)

        discount       = 1 - 1 / self.config.horizon
        traj["weight"] = torch.cumprod(discount * traj["cont"], dim=0) / discount

        return traj

    def report(self, data):
        state  = self.initial(len(data["is_first"]))
        report = {}
        # 对应：report.update(self.loss(data, state)[-1][-1])
        # loss 返回 (model_loss_mean, (state, out, metrics))
        # [-1] = (state, out, metrics)，[-1][-1] = metrics
        report.update(self.loss(data, state)[1][2])

        # 对应：
        #   context, _ = self.rssm.observe(encoder(data)[:6,:5],
        #                                   data["action"][:6,:5],
        #                                   data["is_first"][:6,:5])
        context, _ = self.rssm.observe(
            self.encoder(data)[:6, :5],
            data["action"][:6, :5],
            data["is_first"][:6, :5],
        )

        start = {k: v[:, -1] for k, v in context.items()}

        recon = self.heads["decoder"](context)
        openl = self.heads["decoder"](
            self.rssm.imagine(data["action"][:6, 5:], start))

        for key in self.heads["decoder"].cnn_shapes.keys():
            # 对应：truth = data[key][:6].astype(jnp.float32)
            truth = data[key][:6].float()
            # 对应：model = jnp.concatenate([recon[key].mode()[:,:5], openl[key].mode()], 1)
            model = torch.cat([recon[key].mode()[:, :5], openl[key].mode()], dim=1)
            # 对应：error = (model - truth + 1) / 2
            error = (model - truth + 1) / 2
            # 对应：video = jnp.concatenate([truth, model, error], 2)
            video = torch.cat([truth, model, error], dim=2)
            report[f"openl_{key}"] = jaxutils.video_grid(video)

        return report

    def _metrics(self, data, dists, post, prior, losses, model_loss):
        entropy = lambda feat: self.rssm.get_dist(feat).entropy()
        metrics = {}
        metrics.update(jaxutils.tensorstats(entropy(prior), "prior_ent"))
        metrics.update(jaxutils.tensorstats(entropy(post),  "post_ent"))
        metrics.update({f"{k}_loss_mean": v.mean() for k, v in losses.items()})
        metrics.update({f"{k}_loss_std":  v.std()  for k, v in losses.items()})
        metrics["model_loss_mean"] = model_loss.mean()
        metrics["model_loss_std"]  = model_loss.std()
        # 对应：jnp.abs(data["reward"]).max()
        metrics["reward_max_data"] = data["reward"].abs().max()
        # 对应：jnp.abs(dists["reward"].mean()).max()
        metrics["reward_max_pred"] = dists["reward"].mean().abs().max()
        if "reward" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["reward"], data["reward"], 0.1)
            metrics.update({f"reward_{k}": v for k, v in stats.items()})
        if "cont" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ImagActorCritic
# ─────────────────────────────────────────────────────────────────────────────

class ImagActorCritic(NJModule):
    def __init__(self, critics, scales, act_space, config, name="ac"):
        super().__init__()
        # 对应：critics = {k: v for k,v in critics.items() if scales[k]}
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        # 原版用普通 dict，PyTorch 用 nn.ModuleDict 保证参数被追踪
        self.critics   = nn.ModuleDict(critics)
        self.scales    = scales
        self.act_space = act_space
        self.config    = config

        disc      = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont

        # 对应：self.actor = nets.MLP(name="actor", dims="deter", shape=act_space.shape, ...)
        self.actor = nets.MLP(
            shape=act_space.shape,
            dims="deter",
            **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        )
        self.add_module("actor", self.actor)

        # 对应：self.retnorms = {k: jaxutils.Moments(...) for k in critics}
        # nn.ModuleDict 保证参数被追踪
        self.retnorms = nn.ModuleDict({
            k: jaxutils.Moments(**config.retnorm)
            for k in critics
        })

        self.opt = jaxutils.Optimizer(name="actor_opt", **config.actor_opt)

    def initial(self, batch_size):
        return {}

    def policy(self, state, carry):
        # 对应：return {"action": self.actor(state)}, carry
        return {"action": self.actor(state)}, carry

    def train(self, imagine, start, context):
        # 对应：
        #   def loss(start):
        #       policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
        #       traj = imagine(policy, start, self.config.imag_horizon)
        #       loss, metrics = self.loss(traj)
        #       return loss, (traj, metrics)
        #   mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        def loss(start):
            policy = lambda s: self.actor(sg(s)).sample()
            traj   = imagine(policy, start, self.config.imag_horizon)
            l, metrics = self.loss(traj)
            return l, (traj, metrics)

        mets, (traj, metrics) = self.opt(
            self.actor, loss, start, has_aux=True)
        metrics.update(mets)

        # 对应：
        #   for key, critic in self.critics.items():
        #       mets = critic.train(traj, self.actor)
        #       metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})

        return traj, metrics

    def loss(self, traj):
        metrics = {}
        advs    = []
        total   = sum(self.scales[k] for k in self.critics)

        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret  = (ret  - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(jaxutils.tensorstats(rew,        f"{key}_reward"))
            metrics.update(jaxutils.tensorstats(ret,        f"{key}_return_raw"))
            metrics.update(jaxutils.tensorstats(normed_ret, f"{key}_return_normed"))
            # 对应：(jnp.abs(ret) >= 0.5).mean()
            metrics[f"{key}_return_rate"] = (ret.abs() >= 0.5).float().mean()

        # 对应：
        #   r    = jnp.reshape(rew[0], (batch_size, batch_length))
        #   v    = jnp.reshape(base[0], (batch_size, batch_length))
        #   disc = jnp.reshape(traj["cont"][0], ...) * (1 - 1/horizon)
        #   td_error = r[:,:-1] + disc[:,1:]*v[:,1:] - v[:,:-1]
        r    = rew[0].reshape(self.config.batch_size,  self.config.batch_length)
        v    = base[0].reshape(self.config.batch_size, self.config.batch_length)
        disc = traj["cont"][0].reshape(
            self.config.batch_size, self.config.batch_length
        ) * (1 - 1 / self.config.horizon)
        td_error = r[:, :-1] + disc[:, 1:] * v[:, 1:] - v[:, :-1]
        # 对应：metrics["td_error"] = td_error
        metrics["td_error"] = td_error

        # 对应：adv = jnp.stack(advs).sum(0)
        adv    = torch.stack(advs).sum(0)
        policy = self.actor(sg(traj))

        # 对应：logpi = policy.log_prob(sg(traj["action"]))[:-1]
        logpi = policy.log_prob(sg(traj["action"]))[:-1]

        # 对应：loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]

        # 对应：
        #   ent = policy.entropy()[:-1]
        #   loss -= self.config.actent * ent
        ent   = policy.entropy()[:-1]
        loss -= self.config.actent * ent

        # 对应：
        #   loss *= sg(traj["weight"])[:-1]
        #   loss *= self.config.loss_scales.actor
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor

        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        ent  = policy.entropy()[:-1]
        # 对应：rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        # 对应：rand = rand.mean(range(2, len(rand.shape)))
        dims = list(range(2, rand.dim()))
        rand = rand.mean(dims) if dims else rand

        act = traj["action"]
        # 对应：act = jnp.argmax(act, -1) if self.act_space.discrete else act
        act = act.argmax(-1) if self.act_space.discrete else act

        metrics.update(jaxutils.tensorstats(act,   "action"))
        metrics.update(jaxutils.tensorstats(rand,  "policy_randomness"))
        metrics.update(jaxutils.tensorstats(ent,   "policy_entropy"))
        metrics.update(jaxutils.tensorstats(logpi, "policy_logprob"))
        metrics.update(jaxutils.tensorstats(adv,   "adv"))
        # 对应：metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])
        metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# VFunction
# ─────────────────────────────────────────────────────────────────────────────

class VFunction(NJModule):
    def __init__(self, rewfn, config, name="critic"):
        super().__init__()
        self.rewfn  = rewfn
        self.config = config

        # 对应：self.net  = nets.MLP((), name="net",  dims="deter", **config.critic)
        #        self.slow = nets.MLP((), name="slow", dims="deter", **config.critic)
        self.net  = nets.MLP((), dims="deter", **config.critic)
        self.slow = nets.MLP((), dims="deter", **config.critic)
        self.add_module("net",  self.net)
        self.add_module("slow", self.slow)

        # 对应：self.updater = jaxutils.SlowUpdater(net, slow, fraction, period)
        self.updater = jaxutils.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
        )

        self.opt = jaxutils.Optimizer(name="critic_opt", **config.critic_opt)

    def train(self, traj, actor):
        # 对应：
        #   target = sg(self.score(traj)[1])
        #   mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        #   metrics.update(mets)
        #   self.updater()
        target = sg(self.score(traj)[1])
        mets, metrics = self.opt(
            self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics

    def loss(self, traj, target):
        metrics = {}
        # 对应：traj = {k: v[:-1] for k, v in traj.items()}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)

        # 对应：loss = -dist.log_prob(sg(target))
        loss = -dist.log_prob(sg(target))

        # 对应：
        #   if critic_slowreg == "logprob":
        #       reg = -dist.log_prob(sg(self.slow(traj).mean()))
        #   elif critic_slowreg == "xent":
        #       reg = -jnp.einsum("...i,...i->...", sg(slow.probs), jnp.log(dist.probs))
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            # 对应：einsum("...i,...i->...", sg(slow.probs), log(dist.probs))
            reg = -(sg(self.slow(traj).probs) * torch.log(dist.probs)).sum(-1)
        else:
            raise NotImplementedError(self.config.critic_slowreg)

        # 对应：loss += loss_scales.slowreg * reg
        loss += self.config.loss_scales.slowreg * reg
        # 对应：loss = (loss * sg(traj["weight"])).mean()
        loss  = (loss * sg(traj["weight"])).mean()
        # 对应：loss *= loss_scales.critic
        loss *= self.config.loss_scales.critic

        # 对应：metrics = jaxutils.tensorstats(dist.mean())
        metrics = jaxutils.tensorstats(dist.mean())
        return loss, metrics

    def score(self, traj, actor=None):
        # 对应：rew = self.rewfn(traj)
        rew = self.rewfn(traj)
        assert len(rew) == len(traj["action"]) - 1, \
            "should provide rewards for all but last action"

        # 对应：
        #   discount = 1 - 1 / self.config.horizon
        #   disc = traj["cont"][1:] * discount
        #   value = self.net(traj).mean()
        discount = 1 - 1 / self.config.horizon
        disc     = traj["cont"][1:] * discount
        value    = self.net(traj).mean()

        # 对应：
        #   vals = [value[-1]]
        #   interm = rew + disc * value[1:] * (1 - return_lambda)
        #   for t in reversed(range(len(disc))):
        #       vals.append(interm[t] + disc[t] * return_lambda * vals[-1])
        #   ret = jnp.stack(list(reversed(vals))[:-1])
        vals   = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = torch.stack(list(reversed(vals))[:-1])

        return rew, ret, value[:-1]