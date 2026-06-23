"""WAM Stage-1 training pipeline (WAM Design §16.1): pretrain encoder + deterministic heads.

Stage 1 trains the graph encoder + temporal encoder + Notable Object Head (§11.1) + Gaussian Trajectory
Head (§11.2) on real GT labels, so the model learns task-aware environmental understanding and a reliable
predictive uncertainty ``U^π`` (the policy-search reward, §11.2). It is the prerequisite for §17 / §16.3.

This is the **offline** training side (mirrors ``stage2.py``):
  * :func:`make_stage1_sample` -- one sample = a window of ``K+1`` policy-conditioned graphs + the GT
    future trajectory target (perception labels ride on the last graph's object nodes).
  * :class:`WAMStage1Dataset` + :func:`collate_stage1_samples` -- load recorded ``.pt`` windows / batch.
  * :class:`WAMStage1Trainer` -- the loop: per window run :class:`WAMPerceptionModel`, perception BCE
    (§15.1) + notable-weighted Gaussian NLL (§15.2), backward, Adam, checkpoint, ``evaluate`` (§20).
  * :func:`init_encoder_from_stage1` -- warm-start a Stage-2 ``WAMUnifiedWorldModel``'s graph encoder.

Data is produced by :class:`car_dreamer.toolkit.wam.stage1_recorder.WAMStage1DataRecorder`. The dataset +
trainer are CARLA-free and unit-tested on synthetic windows.

Simplifications (v1): occluding (§8.3) unimplemented -> ``occ`` label 0, ``λ_occ = 0``; windows are encoded
per-sample in a Python loop (the temporal encoder aligns objects within one window); samples carry a single
request-vehicle graph per step (per-vehicle graphs are a Stage-2-side concern, deferred).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .heads import (
    WAMPerceptionConfig,
    WAMPerceptionModel,
    gaussian_trajectory_nll,
    perception_metrics,
    perception_loss,
    policy_uncertainty,
    trajectory_ade_fde,
)
from .bev import BEV_NUM_CHANNELS
from .stage2 import _cfg_get

EgoPose = Tuple[float, float, float]


# =====================================================================
# Sample / dataset / collate
# =====================================================================


def make_stage1_sample(
    window: Sequence,
    target_xy: torch.Tensor,
    valid: torch.Tensor,
    object_node_ids: Sequence[int],
    perception_labels: Optional[Dict[str, torch.Tensor]] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Assemble one Stage-1 sample.

    ``window`` is the list of ``K+1`` graphs (oldest->newest); ``object_node_ids`` is the **union of
    valid object nodes across the window** (so collaborator-only objects absent from the ego-only last
    frame are still supervised). ``target_xy [Q,H,2]`` / ``valid [Q,H]`` are the GT future positions in
    that id order. ``perception_labels`` (optional) holds the t-time GT ``notable/visible/invisible``
    tensors (length ``Q``, same id order); when absent the trainer falls back to the per-graph node
    labels carried inside ``window`` (back-compat with older recordings).
    """
    sample: Dict[str, object] = {
        "window": list(window),
        "target_xy": target_xy.float(),
        "valid": valid.float(),
        "object_node_ids": [int(i) for i in object_node_ids],
    }
    if perception_labels is not None:
        sample["perception_labels"] = {k: v.float() for k, v in perception_labels.items()}
    if metadata is not None:
        sample["metadata"] = dict(metadata)
    return sample


class WAMStage1Dataset(Dataset):
    """Stage-1 window samples from a directory of ``.pt`` files, or an in-memory list (for tests)."""

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


def collate_stage1_samples(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Keep per-sample windows + ragged targets as lists (the perception model consumes one window)."""
    return {"samples": list(batch)}


# =====================================================================
# Trainer
# =====================================================================


@dataclass
class WAMStage1Config:
    lr: float = 3e-4
    batch_size: int = 8
    max_steps: int = 2000
    lambda_perc: float = 1.0
    lambda_traj: float = 1.0
    lambda_vis: float = 1.0
    lambda_inv: float = 1.0
    grad_clip: float = 1.0
    log_interval: int = 50
    ckpt_interval: int = 500
    ckpt_dir: str = "outputs/wam_stage1"
    history_window: int = 4
    sample_period_s: float = 0.1
    val_fraction: float = 0.0
    device: str = "cpu"
    seed: int = 0
    num_workers: int = 0
    shuffle: bool = True

    @property
    def perception_weights(self) -> Dict[str, float]:
        return {
            "notable": 1.0,
            "visible": float(self.lambda_vis),
            "invisible": float(self.lambda_inv),
            "occluding": 0.0,
        }


def _align_target(
    out_ids: torch.Tensor,
    rec_ids: Sequence[int],
    target_xy: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reorder recorded ``target_xy``/``valid`` (keyed by ``rec_ids``) to the model's ``out_ids`` order."""
    device = target_xy.device
    h = int(target_xy.shape[1]) if target_xy.dim() == 3 else 0
    out_list = [int(v) for v in out_ids.tolist()]
    id_to_row = {int(v): r for r, v in enumerate(rec_ids)}
    q = len(out_list)
    tgt = torch.zeros(q, h, 2, device=device)
    val = torch.zeros(q, h, device=device)
    for i, oid in enumerate(out_list):
        r = id_to_row.get(oid)
        if r is not None and r < target_xy.shape[0]:
            tgt[i] = target_xy[r].to(device)
            val[i] = valid[r].to(device)
    return tgt, val


def _align_labels(
    out_ids: torch.Tensor,
    rec_ids: Sequence[int],
    labels: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Reorder recorded perception ``labels`` (keyed by ``rec_ids``) to the model's ``out_ids`` order.

    Missing ids default to 0 (e.g. an object absent at t). Robust to ordering differences between the
    recorder's union and the model's union since alignment is by node_id.
    """
    out_list = [int(v) for v in out_ids.tolist()]
    id_to_row = {int(v): r for r, v in enumerate(rec_ids)}
    aligned: Dict[str, torch.Tensor] = {}
    for key, vec in labels.items():
        vec = vec.to(device)
        out_vec = torch.zeros(len(out_list), device=device)
        for i, oid in enumerate(out_list):
            r = id_to_row.get(oid)
            if r is not None and r < vec.shape[0]:
                out_vec[i] = vec[r]
        aligned[key] = out_vec
    return aligned


class WAMStage1Trainer:
    """§16.1 training loop for the deterministic perception model (`WAMPerceptionModel`)."""

    def __init__(self, model: WAMPerceptionModel, config: WAMStage1Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=config.lr)
        self.step = 0

    def _gt_labels(self, out: Dict[str, object], sample: Dict[str, object]) -> Dict[str, torch.Tensor]:
        """t-time GT perception labels aligned to the model's ``object_node_ids``.

        Prefers the recorded ``perception_labels`` (labels at the prediction step t); falls back to the
        per-graph node labels the model surfaces in ``out["labels"]`` for older recordings.
        """
        rec_labels = sample.get("perception_labels")
        if rec_labels is not None:
            return _align_labels(out["object_node_ids"], sample["object_node_ids"], rec_labels, self.device)
        return out["labels"]

    def _sample_loss(self, sample: Dict[str, object]) -> Optional[Dict[str, torch.Tensor]]:
        window = [g.to(self.device) for g in sample["window"]]
        out = self.model(window)
        if int(out["object_node_ids"].numel()) == 0:
            return None
        target_xy, valid = _align_target(
            out["object_node_ids"],
            sample["object_node_ids"],
            sample["target_xy"].to(self.device),
            sample["valid"].to(self.device),
        )
        gt_labels = self._gt_labels(out, sample)
        perc = perception_loss(out["perception_logits"], gt_labels, weights=self.config.perception_weights)
        notable_weight = gt_labels.get("notable")
        if notable_weight is None:
            notable_weight = torch.ones(out["object_node_ids"].shape[0], device=self.device)
        nll = gaussian_trajectory_nll(
            out["traj_mu"], out["traj_log_var"], target_xy,
            notable_weight=notable_weight.to(self.device), valid_mask=valid,
        )
        total = self.config.lambda_perc * perc["total"] + self.config.lambda_traj * nll
        return {"total": total, "perception": perc["total"], "traj": nll}

    def loss_on_batch(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        acc: Dict[str, torch.Tensor] = {}
        n = 0
        for sample in batch["samples"]:
            losses = self._sample_loss(sample)
            if losses is None:
                continue
            n += 1
            for key, value in losses.items():
                acc[key] = value if key not in acc else acc[key] + value
        if n == 0:
            zero = next(self.model.parameters()).sum() * 0.0
            return {"total": zero, "perception": zero.detach(), "traj": zero.detach()}
        return {key: value / n for key, value in acc.items()}

    def _loader(self, dataset: WAMStage1Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_stage1_samples,
            drop_last=False,
        )

    def train(
        self,
        dataset: WAMStage1Dataset,
        val_dataset: Optional[WAMStage1Dataset] = None,
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
                        f"[wam-stage1] step={self.step} L={float(loss.detach()):.4f} "
                        f"perc={float(losses['perception']):.4f} traj={float(losses['traj']):.4f}"
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
            out.update(self.evaluate(val_dataset))
        return out

    @torch.no_grad()
    def evaluate(self, dataset: WAMStage1Dataset) -> Dict[str, float]:
        """§20.1/§20.2 quick metrics: notable F1, invisible recall, ADE/FDE, mean ``U^π``."""
        loader = self._loader(dataset, shuffle=False)
        self.model.eval()
        notable_p, notable_l, inv_p, inv_l = [], [], [], []
        mus, tgts, vals, nws, logvars = [], [], [], [], []
        for batch in loader:
            for sample in batch["samples"]:
                window = [g.to(self.device) for g in sample["window"]]
                out = self.model(window)
                if int(out["object_node_ids"].numel()) == 0:
                    continue
                tgt, val = _align_target(
                    out["object_node_ids"], sample["object_node_ids"],
                    sample["target_xy"].to(self.device), sample["valid"].to(self.device),
                )
                gt_labels = self._gt_labels(out, sample)
                nw = gt_labels.get("notable")
                nw = torch.ones(out["object_node_ids"].shape[0], device=self.device) if nw is None else nw
                notable_p.append(out["notable_prob"]); notable_l.append(gt_labels.get("notable", torch.zeros_like(out["notable_prob"])))
                inv_p.append(out["invisible_prob"]); inv_l.append(gt_labels.get("invisible", torch.zeros_like(out["invisible_prob"])))
                mus.append(out["traj_mu"]); tgts.append(tgt); vals.append(val)
                nws.append(nw); logvars.append(out["traj_log_var"])
        self.model.train()
        if not mus:
            return {"val_notable_f1": 0.0, "val_invisible_recall": 0.0, "val_ade": 0.0, "val_fde": 0.0, "val_mean_uncertainty": 0.0}
        notable_f1 = perception_metrics(torch.cat(notable_p), torch.cat(notable_l))["f1"]
        inv_recall = perception_metrics(torch.cat(inv_p), torch.cat(inv_l))["recall"]
        mu, tgt, val, nw = torch.cat(mus), torch.cat(tgts), torch.cat(vals), torch.cat(nws)
        ade_fde = trajectory_ade_fde(mu, tgt, valid_mask=val, notable_weight=nw)
        u_pi = float(policy_uncertainty(torch.cat(notable_p), torch.cat(logvars), valid_mask=val))
        return {
            "val_notable_f1": notable_f1,
            "val_invisible_recall": inv_recall,
            "val_ade": ade_fde["ade"],
            "val_fde": ade_fde["fde"],
            "val_mean_uncertainty": u_pi,
        }

    # ---- checkpoint ----
    def save_checkpoint(self, path: Optional[Union[str, Path]] = None) -> Path:
        ckpt_dir = Path(self.config.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = Path(path) if path is not None else ckpt_dir / f"stage1_step{self.step}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "perception_config": self.model.config,
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
# Warm-start: Stage-1 encoder -> Stage-2 WAMUnifiedWorldModel
# =====================================================================


def init_encoder_from_stage1(stage2_model, stage1_ckpt: Union[str, Path, Dict[str, object]]) -> Tuple[List[str], List[str]]:
    """Load the Stage-1 graph encoder weights into a Stage-2 ``WAMUnifiedWorldModel``.

    Both encoders are a :class:`WAMHeteroGraphNet` with the same config, so the ``graph_net.*`` sub-state
    transfers directly. Returns ``(missing, unexpected)`` keys (expected empty when configs match).
    """
    if isinstance(stage1_ckpt, (str, Path)):
        ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    else:
        ckpt = stage1_ckpt
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    prefix = "graph_net."
    enc_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    result = stage2_model.context_encoder.graph_net.load_state_dict(enc_state, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


# =====================================================================
# Config plumbing (env.wam.* -> dataclasses)
# =====================================================================


def wam_stage1_configs_from_env(config) -> Tuple[WAMPerceptionConfig, WAMStage1Config]:
    """Read ``env.wam.graph.*`` (shared) + ``env.wam.stage1.*`` into the perception + trainer configs."""
    env = getattr(config, "env", config)
    wam = _cfg_get(env, "wam", None)
    graph = _cfg_get(wam, "graph", None)
    stage1 = _cfg_get(wam, "stage1", None)

    perc_cfg = WAMPerceptionConfig(
        route_waypoints=int(_cfg_get(graph, "route_waypoints", 6)),
        hidden_dim=int(_cfg_get(graph, "hidden_dim", 256)),
        num_layers=int(_cfg_get(graph, "num_layers", 3)),
        num_heads=int(_cfg_get(graph, "num_heads", 8)),
        bev_channels=int(_cfg_get(graph, "bev_channels", BEV_NUM_CHANNELS)),
        bev_size=int(_cfg_get(graph, "bev_size", 64)),
        temporal_hidden_dim=int(_cfg_get(stage1, "temporal_hidden_dim", 256)),
        head_hidden_dim=int(_cfg_get(stage1, "head_hidden_dim", 256)),
        traj_horizon_s=float(_cfg_get(stage1, "traj_horizon_s", 3.0)),
        traj_samples=int(_cfg_get(stage1, "traj_samples", 6)),
        fixed_dt=float(_cfg_get(stage1, "fixed_dt", 0.1)),
    )
    stage1_cfg = WAMStage1Config(
        lr=float(_cfg_get(stage1, "lr", 3e-4)),
        batch_size=int(_cfg_get(stage1, "batch_size", 8)),
        max_steps=int(_cfg_get(stage1, "steps", _cfg_get(stage1, "max_steps", 2000))),
        lambda_perc=float(_cfg_get(stage1, "lambda_perc", 1.0)),
        lambda_traj=float(_cfg_get(stage1, "lambda_traj", 1.0)),
        lambda_vis=float(_cfg_get(stage1, "lambda_vis", 1.0)),
        lambda_inv=float(_cfg_get(stage1, "lambda_inv", 1.0)),
        grad_clip=float(_cfg_get(stage1, "grad_clip", 1.0)),
        log_interval=int(_cfg_get(stage1, "log_interval", 50)),
        ckpt_interval=int(_cfg_get(stage1, "ckpt_interval", 500)),
        history_window=int(_cfg_get(stage1, "history_window", 4)),
        sample_period_s=float(_cfg_get(stage1, "sample_period_s", _cfg_get(stage1, "fixed_dt", 0.1))),
    )
    return perc_cfg, stage1_cfg
