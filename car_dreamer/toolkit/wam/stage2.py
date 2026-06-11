"""WAM Stage-2 training pipeline (WAM Design Update §7-§8): train the BS-centric Diffusion UWM.

Stage 2 learns ``p_θ(π_{q,t:t+H-1}, z^bev_{q,t+1:t+H} | C^BS_{q,t-K:t})`` by minimizing the flow-matching
loss ``L = w_π‖û_π−u_π‖² + w_z‖û_z−u_z‖²``. This module is the **offline** training side:

  * :class:`WAMFlowDataset` + :func:`collate_flow_samples` -- load recorded ``.pt`` samples (per-vehicle
    graphs + GT future policy chunk + request notable ids + BEV placeholders) and batch them.
  * :class:`WAMStage2Trainer` -- the loop: build BS condition tokens (per-sample graph encoding via
    :meth:`WAMUnifiedWorldModel.condition_tokens_batch`), sample decoupled diffusion times, forward the
    vector field, ``L``, backward, Adam, with checkpointing.

Data is produced by :class:`car_dreamer.toolkit.wam.flow_recorder.WAMFlowDataRecorder`. The dataset +
trainer are CARLA-free and unit-tested on synthetic samples.

Simplifications (v1, faithful to the update's "defer" notes): the request-vehicle BEV latent
(``bev_history`` condition + ``bev_future`` target) is a zero placeholder; object-state ``X`` is no longer
a generated variable (it lives only in the driving-task condition tokens); samples carry a single
request-vehicle graph (per-vehicle local graphs for all coverage vehicles are deferred); graphs are
encoded per-sample in a Python loop (the context pool must pool *within* a graph).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .flow_matching import WAMFlowMatchingConfig, WAMUnifiedWorldModel
from .graph_model import WAMGraphModelConfig


# =====================================================================
# Sample / dataset / collate
# =====================================================================


def make_flow_sample(
    request_graph,
    policy_chunk: torch.Tensor,
    member_mask: torch.Tensor,
    policy_step_mask: torch.Tensor,
    *,
    vehicle_graphs: Optional[Sequence] = None,
    request_index: int = 0,
    notable_object_ids: Sequence[int] = (),
    bev_history: Optional[torch.Tensor] = None,
    bev_future: Optional[torch.Tensor] = None,
    bev_step_mask: Optional[torch.Tensor] = None,
    bev_latent_dim: Optional[int] = None,
    history_window: int = 0,
) -> Dict[str, object]:
    """Assemble one BS-centric training sample (one = request vehicle q at time t).

    ``policy_chunk [H,M,P]`` is the GT future policy chunk; ``vehicle_graphs`` defaults to
    ``[request_graph]`` (per-vehicle local graphs for all coverage vehicles are deferred). The BEV
    history/future default to zero placeholders.
    """
    h = int(policy_chunk.shape[0])
    dz = int(bev_latent_dim if bev_latent_dim is not None else 0)
    if vehicle_graphs is None:
        vehicle_graphs = [request_graph]
    if bev_history is None:
        bev_history = torch.zeros(int(history_window) + 1, dz, dtype=torch.float32)
    if bev_future is None:
        bev_future = torch.zeros(h, dz, dtype=torch.float32)
    if bev_step_mask is None:
        bev_step_mask = torch.ones(h, dtype=torch.float32)
    return {
        "vehicle_graphs": list(vehicle_graphs),
        "request_index": int(request_index),
        "notable_object_ids": [int(i) for i in notable_object_ids],
        "policy_chunk": policy_chunk.float(),
        "member_mask": member_mask.float(),
        "policy_step_mask": policy_step_mask.float(),
        "bev_history": bev_history.float(),
        "bev_future": bev_future.float(),
        "bev_step_mask": bev_step_mask.float(),
    }


class WAMFlowDataset(Dataset):
    """Flow-matching samples from a directory of ``.pt`` files, or an in-memory list (for tests)."""

    def __init__(self, source: Union[str, Path, Sequence[Dict[str, object]]]):
        if isinstance(source, (str, Path)):
            self.root: Optional[Path] = Path(source)
            self.files: List[Path] = sorted(self.root.glob("*.pt"))
            self.samples: Optional[List[Dict[str, object]]] = None
            if not self.files:
                raise FileNotFoundError(f"no .pt samples found under {self.root}")
        else:
            self.root = None
            self.files = []
            self.samples = list(source)

    def __len__(self) -> int:
        return len(self.samples) if self.samples is not None else len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if self.samples is not None:
            return self.samples[idx]
        return torch.load(self.files[idx], weights_only=False)


def collate_flow_samples(
    batch: Sequence[Dict[str, object]],
    flow_config: WAMFlowMatchingConfig,
) -> Dict[str, object]:
    """Batch samples: keep per-sample graph context as a list (the BS encoder pools within a graph);
    pad/stack the generated-variable tensors to config maxes ``[B, H, M/Dz, ...]``."""
    b = len(batch)
    m, p = flow_config.max_members, flow_config.policy_width
    h = flow_config.horizon
    dz = int(flow_config.bev_latent_dim)

    samples = [
        {
            "vehicle_graphs": s["vehicle_graphs"],
            "request_index": int(s.get("request_index", 0)),
            "notable_object_ids": s.get("notable_object_ids", ()),
            "bev_history": s.get("bev_history"),
        }
        for s in batch
    ]
    policy_chunk = torch.zeros(b, h, m, p)
    member_mask = torch.zeros(b, m)
    policy_step_mask = torch.zeros(b, h)
    bev_future = torch.zeros(b, h, dz)
    bev_step_mask = torch.zeros(b, h)

    for i, s in enumerate(batch):
        nm = min(int(s["policy_chunk"].shape[1]), m)
        nh = min(int(s["policy_chunk"].shape[0]), h)
        policy_chunk[i, :nh, :nm] = s["policy_chunk"][:nh, :nm]
        member_mask[i, :nm] = s["member_mask"][:nm]
        policy_step_mask[i, :nh] = s["policy_step_mask"][:nh]
        if dz > 0 and int(s["bev_future"].shape[-1]) == dz:
            nhb = min(int(s["bev_future"].shape[0]), h)
            bev_future[i, :nhb] = s["bev_future"][:nhb]
        bev_step_mask[i, :h] = s["bev_step_mask"][:h]

    return {
        "samples": samples,
        "policy_chunk": policy_chunk,
        "member_mask": member_mask,
        "policy_step_mask": policy_step_mask,
        "bev_future": bev_future,
        "bev_step_mask": bev_step_mask,
    }


# =====================================================================
# Trainer
# =====================================================================


@dataclass
class WAMStage2Config:
    lr: float = 3e-4
    batch_size: int = 16
    max_steps: int = 2000
    w_policy: float = 1.0
    w_bev: float = 1.0
    grad_clip: float = 1.0
    log_interval: int = 50
    ckpt_interval: int = 500
    ckpt_dir: str = "outputs/wam_stage2"
    val_fraction: float = 0.0
    device: str = "cpu"
    seed: int = 0
    num_workers: int = 0
    freeze_encoder: bool = False
    shuffle: bool = True


class WAMStage2Trainer:
    """§16.2 training loop for the BS-centric Diffusion UWM (:class:`WAMUnifiedWorldModel`)."""

    def __init__(self, model: WAMUnifiedWorldModel, config: WAMStage2Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        if config.freeze_encoder:
            for param in self.model.context_encoder.parameters():
                param.requires_grad_(False)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=config.lr)
        self.step = 0

    def _samples_to_device(self, samples: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for s in samples:
            graphs = [g.to(self.device) for g in s["vehicle_graphs"]]
            bev_hist = s.get("bev_history")
            out.append(
                {
                    "vehicle_graphs": graphs,
                    "request_index": int(s.get("request_index", 0)),
                    "notable_object_ids": s.get("notable_object_ids", ()),
                    "bev_history": None if bev_hist is None else bev_hist.to(self.device),
                }
            )
        return out

    def loss_on_batch(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        samples = self._samples_to_device(batch["samples"])
        cond, tids, mask = self.model.condition_tokens_batch(samples)

        policy_1 = batch["policy_chunk"].to(self.device)
        bev_1 = batch["bev_future"].to(self.device)
        member_mask = batch["member_mask"].to(self.device)
        policy_step_mask = batch["policy_step_mask"].to(self.device)
        bev_step_mask = batch["bev_step_mask"].to(self.device)
        # supervise only present (member, step) policy entries.
        policy_loss_mask = policy_step_mask.unsqueeze(-1) * member_mask.unsqueeze(1)

        return self.model.training_step(
            cond, tids, mask, policy_1, bev_1,
            member_mask=member_mask, policy_step_mask=policy_step_mask, bev_step_mask=bev_step_mask,
            policy_loss_mask=policy_loss_mask,
        )

    def _loader(self, dataset: WAMFlowDataset, shuffle: bool) -> DataLoader:
        collate = functools.partial(collate_flow_samples, flow_config=self.model.flow.config)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate,
            drop_last=False,
        )

    def train(
        self,
        dataset: WAMFlowDataset,
        val_dataset: Optional[WAMFlowDataset] = None,
        *,
        logger=None,
    ) -> Dict[str, float]:
        torch.manual_seed(self.config.seed)
        loader = self._loader(dataset, shuffle=self.config.shuffle)
        self.model.train()
        history: List[float] = []
        done = False
        while not done:
            for batch in loader:
                self.optimizer.zero_grad()
                losses = self.loss_on_batch(batch)
                loss = losses["total"]
                loss.backward()
                if self.config.grad_clip and self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.params, self.config.grad_clip)
                self.optimizer.step()
                self.step += 1
                history.append(float(loss.detach()))

                if self.config.log_interval and self.step % self.config.log_interval == 0:
                    msg = (
                        f"[wam-stage2] step={self.step} L={float(loss.detach()):.4f} "
                        f"policy={float(losses['policy'].detach()):.4f} bev={float(losses['bev'].detach()):.4f}"
                    )
                    (logger.info(msg) if logger is not None else print(msg, flush=True))
                if self.config.ckpt_interval and self.step % self.config.ckpt_interval == 0:
                    self.save_checkpoint()
                if self.step >= self.config.max_steps:
                    done = True
                    break

        self.save_checkpoint()
        out = {"final_loss": history[-1] if history else float("nan"), "steps": float(self.step)}
        if val_dataset is not None:
            out["val_loss"] = self.evaluate(val_dataset)
        return out

    @torch.no_grad()
    def evaluate(self, dataset: WAMFlowDataset) -> float:
        loader = self._loader(dataset, shuffle=False)
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            total += float(self.loss_on_batch(batch)["total"])
            n += 1
        self.model.train()
        return total / max(n, 1)

    # ---- checkpoint ----
    def save_checkpoint(self, path: Optional[Union[str, Path]] = None) -> Path:
        ckpt_dir = Path(self.config.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = Path(path) if path is not None else ckpt_dir / f"stage2_step{self.step}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "graph_config": self.model.context_encoder.graph_net.config,
                "flow_config": self.model.flow.config,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = int(ckpt.get("step", 0))


# =====================================================================
# Config plumbing (env.wam.* -> dataclasses)
# =====================================================================


def _cfg_get(node, key: str, default):
    if node is None:
        return default
    try:
        if key in node:
            value = node[key]
            return value if value is not None else default
    except TypeError:
        pass
    return getattr(node, key, default)


def wam_configs_from_env(config) -> Tuple[WAMGraphModelConfig, WAMFlowMatchingConfig, WAMStage2Config]:
    """Read ``env.wam.graph.* / env.wam.flow.* / env.wam.stage2.*`` into the three dataclasses.

    ``hidden_dim`` / ``route_waypoints`` are shared (graph block) so the trainer's encoder matches the
    recorded graphs; ``flow.bev_latent_dim`` defaults to ``hidden_dim``.
    """
    env = getattr(config, "env", config)
    wam = _cfg_get(env, "wam", None)
    graph = _cfg_get(wam, "graph", None)
    flow = _cfg_get(wam, "flow", None)
    stage2 = _cfg_get(wam, "stage2", None)

    hidden_dim = int(_cfg_get(graph, "hidden_dim", 256))
    route_waypoints = int(_cfg_get(graph, "route_waypoints", 6))

    graph_cfg = WAMGraphModelConfig(
        route_waypoints=route_waypoints,
        hidden_dim=hidden_dim,
        num_layers=int(_cfg_get(graph, "num_layers", 3)),
        num_heads=int(_cfg_get(graph, "num_heads", 8)),
        bev_channels=int(_cfg_get(graph, "bev_channels", 8)),
        bev_size=int(_cfg_get(graph, "bev_size", 128)),
    )
    flow_cfg = WAMFlowMatchingConfig(
        hidden_dim=hidden_dim,
        num_layers=int(_cfg_get(flow, "num_layers", 4)),
        num_heads=int(_cfg_get(flow, "num_heads", 8)),
        time_embed_dim=int(_cfg_get(flow, "time_embed_dim", 128)),
        max_members=int(_cfg_get(flow, "max_members", 8)),
        num_formats=int(_cfg_get(flow, "num_formats", 2)),
        horizon=int(_cfg_get(flow, "horizon", 6)),
        history_window=int(_cfg_get(flow, "history_window", 0)),
        num_register_tokens=int(_cfg_get(flow, "num_register_tokens", 4)),
        bev_latent_dim=hidden_dim,
        enable_bev=bool(_cfg_get(flow, "enable_bev", True)),
        n_inference_steps=int(_cfg_get(flow, "n_inference_steps", 10)),
    )
    stage2_cfg = WAMStage2Config(
        lr=float(_cfg_get(stage2, "lr", 3e-4)),
        batch_size=int(_cfg_get(stage2, "batch_size", 16)),
        max_steps=int(_cfg_get(stage2, "steps", _cfg_get(stage2, "max_steps", 2000))),
        w_policy=float(_cfg_get(stage2, "w_policy", 1.0)),
        w_bev=float(_cfg_get(stage2, "w_bev", _cfg_get(stage2, "w_obs", 1.0))),
        grad_clip=float(_cfg_get(stage2, "grad_clip", 1.0)),
        log_interval=int(_cfg_get(stage2, "log_interval", 50)),
        ckpt_interval=int(_cfg_get(stage2, "ckpt_interval", 500)),
    )
    return graph_cfg, flow_cfg, stage2_cfg
