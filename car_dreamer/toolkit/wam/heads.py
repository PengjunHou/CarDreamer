"""Temporal encoder (§10) + deterministic task heads (§11) on top of ``H_t``.

This completes the deterministic perception chain ``x^obj -> h^{obj,0} -> h^obj -> z_o -> heads``
(WAM Design §10). Given a history window of policy-conditioned hetero graphs ``G_{t-K:t}``
(each a :class:`torch_geometric.data.HeteroData` from
:func:`car_dreamer.toolkit.wam.graph.build_wam_hetero_graph`), the model:

  1. runs the §5 embedding + §9 HGT encoder (:class:`WAMHeteroGraphNet`) on every graph -> ``H_τ``;
  2. gathers each persistent object node's graph-contextual embeddings across the window
     (aligned by ``node_id``) and aggregates them with a GRU temporal encoder -> ``z_{o,t}`` (§10);
  3. applies the Notable Object Head (§11.1) and Gaussian Trajectory Head (§11.2) on ``z_{o,t}``.

Loss / reward helpers implement §15.1 (perception BCE), §15.2 (notable-weighted Gaussian NLL) and
§11.2 (the policy-evaluation uncertainty ``U^π_e(t)``).

This is a standalone, training-side model (like the §9 encoder): the env produces graphs, this model
consumes windows. It is not wired into the live env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bev import BEV_NUM_CHANNELS
from .graph import OBJECT
from .graph_model import WAMGraphModelConfig, WAMHeteroGraphNet

_PERCEPTION_KEYS = ("notable", "visible", "invisible", "occluding")


@dataclass
class WAMPerceptionConfig:
    # graph embedding + HGT encoder (§5 / §9)
    route_waypoints: int = 6
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    num_agent_slots: int = 8
    num_object_classes: int = 4
    bev_channels: int = BEV_NUM_CHANNELS
    bev_size: int = 64
    # temporal encoder (§10) + heads (§11)
    temporal_hidden_dim: int = 256
    head_hidden_dim: int = 256
    traj_horizon_s: float = 3.0
    traj_samples: int = 6
    fixed_dt: float = 0.1
    log_var_min: float = -10.0
    log_var_max: float = 10.0

    @property
    def traj_horizon(self) -> int:
        return int(self.traj_samples)

    def graph_config(self) -> WAMGraphModelConfig:
        return WAMGraphModelConfig(
            route_waypoints=self.route_waypoints,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_agent_slots=self.num_agent_slots,
            num_object_classes=self.num_object_classes,
            bev_channels=self.bev_channels,
            bev_size=self.bev_size,
        )


def align_object_history(
    embeddings: Sequence[torch.Tensor],
    node_ids: Sequence[torch.Tensor],
    query_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``[Q, L, d]`` per-object sequences + ``[Q, L]`` presence mask aligned by ``node_id``.

    ``embeddings[τ]`` is the object embedding matrix at window step τ (oldest->newest) and
    ``node_ids[τ]`` its parallel actor ids. Objects absent at a step get a zero row + mask 0.
    """
    L = len(embeddings)
    Q = int(query_ids.shape[0])
    device = query_ids.device
    d = int(embeddings[-1].shape[1]) if L > 0 and embeddings[-1].numel() > 0 else 0
    seq = torch.zeros(Q, L, d, device=device)
    mask = torch.zeros(Q, L, device=device)
    if Q == 0 or d == 0:
        return seq, mask
    query_list = [int(v) for v in query_ids.tolist()]
    for tau in range(L):
        ids_tau = [int(v) for v in node_ids[tau].tolist()]
        id_to_idx = {vid: idx for idx, vid in enumerate(ids_tau) if vid >= 0}
        h_tau = embeddings[tau]
        for q, qid in enumerate(query_list):
            idx = id_to_idx.get(qid)
            if idx is not None:
                seq[q, tau] = h_tau[idx]
                mask[q, tau] = 1.0
    return seq, mask


class WAMTemporalEncoder(nn.Module):
    """§10 temporal aggregation: GRU over a per-node ``[Q, L, d]`` sequence + presence mask.

    A presence bit is appended per step so the GRU can distinguish a genuine zero embedding from an
    absent step. ``z`` is read at the current step (the query node is always present at ``t``).
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim + 1, hidden_dim, batch_first=True)
        self.hidden_dim = int(hidden_dim)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if seq.shape[0] == 0:
            return seq.new_zeros((0, self.hidden_dim))
        presence = mask.unsqueeze(-1).to(seq.dtype)  # [Q, L, 1]
        x = torch.cat([seq * presence, presence], dim=-1)  # zero absent steps + presence flag
        out, _ = self.gru(x)  # [Q, L, hidden]
        return out[:, -1, :]


class NotableObjectHead(nn.Module):
    """§11.1: ``ŷ^notable = σ(MLP(z))`` plus auxiliary ``vis / inv / occ`` logits."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True))
        self.out = nn.Linear(hidden_dim, len(_PERCEPTION_KEYS))

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.out(self.trunk(z))  # [Q, 4]
        return {key: logits[:, i] for i, key in enumerate(_PERCEPTION_KEYS)}


class GaussianTrajectoryHead(nn.Module):
    """§11.2: ``(μ, log σ²)`` over ``H`` future steps; diagonal covariance ``diag(σ_x², σ_y²)``."""

    def __init__(self, in_dim: int, hidden_dim: int, horizon: int, log_var_min: float, log_var_max: float):
        super().__init__()
        self.horizon = int(horizon)
        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True))
        self.mu = nn.Linear(hidden_dim, self.horizon * 2)
        self.log_var = nn.Linear(hidden_dim, self.horizon * 2)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        q = z.shape[0]
        mu = self.mu(h).view(q, self.horizon, 2)
        log_var = self.log_var(h).view(q, self.horizon, 2).clamp(self.log_var_min, self.log_var_max)
        return mu, log_var


class WAMPerceptionModel(nn.Module):
    """§9 encoder -> §10 temporal -> §11 heads. ``forward`` takes a graph history window."""

    def __init__(self, config: WAMPerceptionConfig):
        super().__init__()
        self.config = config
        self.graph_net = WAMHeteroGraphNet(config.graph_config())
        self.temporal = WAMTemporalEncoder(config.hidden_dim, config.temporal_hidden_dim)
        self.notable_head = NotableObjectHead(config.temporal_hidden_dim, config.head_hidden_dim)
        self.trajectory_head = GaussianTrajectoryHead(
            config.temporal_hidden_dim,
            config.head_hidden_dim,
            config.traj_horizon,
            config.log_var_min,
            config.log_var_max,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, window: Sequence["torch.Tensor"]) -> Dict[str, object]:
        graphs = list(window)
        if not graphs:
            raise ValueError("window must contain at least one graph")
        device = self.device

        obj_embeddings: List[torch.Tensor] = []
        obj_ids: List[torch.Tensor] = []
        valid_ids_per_frame: List[torch.Tensor] = []
        for graph in graphs:
            h = self.graph_net(graph)
            obj_embeddings.append(h[OBJECT])
            ids = graph[OBJECT].node_id.to(device)
            obj_ids.append(ids)
            if hasattr(graph[OBJECT], "node_mask"):
                frame_mask = graph[OBJECT].node_mask.to(device) > 0.5
            else:
                frame_mask = torch.ones_like(ids, dtype=torch.bool)
            valid_ids_per_frame.append(ids[frame_mask & (ids >= 0)])

        # Query set = union of valid object ids across the *whole* window (not just the last frame).
        # Due to V2V latency the last frame is ego-only, so collaborator-only (invisible) objects only
        # appear in earlier frames; align_object_history + the GRU still produce a prediction for them
        # via their history (presence is 0 at absent frames). torch.unique returns sorted-ascending ids,
        # matching ``stage1_recorder.union_object_ids`` so recorded targets/labels align by node_id.
        if valid_ids_per_frame and any(int(t.numel()) for t in valid_ids_per_frame):
            query_ids = torch.unique(torch.cat(valid_ids_per_frame))
        else:
            query_ids = torch.empty(0, dtype=torch.long, device=device)

        seq, presence = align_object_history(obj_embeddings, obj_ids, query_ids)
        z = self.temporal(seq, presence)

        perception_logits = self.notable_head(z)
        traj_mu, traj_log_var = self.trajectory_head(z)

        # Best-effort labels aligned to ``query_ids`` (newest frame containing the object wins).
        # These are for live inference / debugging only; Stage-1 training overrides them with the
        # t-time ground-truth labels recorded in the sample (see stage1.WAMStage1Trainer).
        query_list = [int(v) for v in query_ids.tolist()]
        labels = {}
        for key in ("notable", "visible", "invisible"):
            if not any(hasattr(g[OBJECT], key) for g in graphs):
                continue
            id_to_val: Dict[int, torch.Tensor] = {}
            for graph in graphs:  # oldest -> newest, so the newest occurrence overwrites
                if not hasattr(graph[OBJECT], key):
                    continue
                gids = [int(v) for v in graph[OBJECT].node_id.tolist()]
                gval = getattr(graph[OBJECT], key).to(device)
                for idx, vid in enumerate(gids):
                    if vid >= 0:
                        id_to_val[vid] = gval[idx]
            vals = torch.zeros(len(query_list), device=device)
            for i, qid in enumerate(query_list):
                if qid in id_to_val:
                    vals[i] = id_to_val[qid]
            labels[key] = vals

        out: Dict[str, object] = {
            "z_object": z,
            "object_node_ids": query_ids,
            "object_mask": torch.ones_like(query_ids, dtype=torch.bool),
            "traj_mu": traj_mu,
            "traj_log_var": traj_log_var,
            "labels": labels,
            "perception_logits": perception_logits,
        }
        for key, logit in perception_logits.items():
            out[f"{key}_logits"] = logit
            out[f"{key}_prob"] = torch.sigmoid(logit)
        return out


# =====================================================================
# Losses / reward (faithful to §15.1, §15.2, §11.2)
# =====================================================================

_DEFAULT_PERCEPTION_WEIGHTS = {"notable": 1.0, "visible": 1.0, "invisible": 1.0, "occluding": 0.0}


def perception_loss(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """§15.1 perception loss: BCE over notable/vis/inv (+ optional occ; default ``λ_occ = 0``)."""
    weights = {**_DEFAULT_PERCEPTION_WEIGHTS, **(weights or {})}
    out: Dict[str, torch.Tensor] = {}
    total = None
    for key in _PERCEPTION_KEYS:
        if key not in logits:
            continue
        logit = logits[key]
        if logit.numel() == 0:
            loss = logit.new_zeros(())
        else:
            target = labels.get(key)
            if target is None:
                target = torch.zeros_like(logit)
            loss = F.binary_cross_entropy_with_logits(logit, target.to(logit.dtype))
        out[key] = loss
        contribution = float(weights.get(key, 0.0)) * loss
        total = contribution if total is None else total + contribution
    out["total"] = total if total is not None else torch.zeros(())
    return out


def gaussian_trajectory_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target_xy: torch.Tensor,
    *,
    notable_weight: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """§15.2 notable-weighted Gaussian NLL over future positions.

    ``mu``/``log_var``/``target_xy``: ``[Q, H, 2]``; ``notable_weight``: ``[Q]``;
    ``valid_mask``: ``[Q, H]`` (1 where the GT future position is available).
    """
    if mu.numel() == 0:
        return mu.new_zeros(())
    var = torch.exp(log_var)
    per_coord = 0.5 * ((target_xy - mu) ** 2 / (var + eps) + log_var)  # [Q, H, 2]
    per_step = per_coord.sum(dim=-1)  # [Q, H]
    weight = notable_weight.unsqueeze(-1)
    if valid_mask is not None:
        weight = weight * valid_mask
    weight = weight.expand_as(per_step)
    weighted = per_step * weight
    if reduction == "sum":
        return weighted.sum()
    return weighted.sum() / weight.sum().clamp_min(eps)


def notable_soft_gate(notable_prob: torch.Tensor, gate_k: float, threshold: float = 0.5) -> torch.Tensor:
    """Differentiable soft threshold ``σ(k·(p − τ))`` -- a smooth 0/1 gate on the notable probability.

    A temperature sigmoid: ``p < τ`` -> ~0, ``p > τ`` -> ~1, smooth everywhere (so it is safe for
    Stage-2 diffusion gradients, unlike a hard ``p > τ`` step). Larger ``gate_k`` is sharper. Used to
    suppress objects the model is unsure about before the notable-weighted uncertainty average.
    """
    return torch.sigmoid(float(gate_k) * (notable_prob - float(threshold)))


def policy_uncertainty(
    notable_prob: torch.Tensor,
    log_var: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    gate_k: Optional[float] = None,
    gate_threshold: float = 0.5,
    mass_floor: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """§11.2 policy-evaluation uncertainty ``U^π_e(t)`` = notable-weighted mean of ``Tr(Σ)``.

    Per object: ``w_o = notable_prob_o`` (or ``σ(gate_k·(p−gate_threshold))`` when ``gate_k>0``), then
    ``U = Σ_o w_o·TrΣ_o / (Σ_o w_o + mass_floor)``. ``mass_floor`` (units: objects) keeps the denominator
    from collapsing: with no confident-notable object (``Σw ≪ mass_floor``) the value -> ~0 instead of
    the (gate-cancelling) mean trace; with enough notable mass (``Σw ≫ mass_floor``) it is the usual
    weighted average. ``mass_floor=0`` is the plain weighted mean.
    """
    if log_var.numel() == 0:
        return log_var.new_zeros(())
    trace_o = per_object_trace(log_var, valid_mask=valid_mask)  # [Q] mean over (valid) horizon
    w = notable_soft_gate(notable_prob, gate_k, gate_threshold) if (gate_k and gate_k > 0) else notable_prob
    if valid_mask is not None:  # drop objects with no valid future step (as the old weight*mask did)
        w = w * (valid_mask.to(w.dtype).sum(dim=-1) > 0).to(w.dtype)
    return (w * trace_o).sum() / (w.sum() + float(mass_floor)).clamp_min(eps)


def per_object_trace(log_var: torch.Tensor, *, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-object predicted variance ``TrΣ_o = mean_h(σ_x² + σ_y²)`` -> ``[Q]`` (no GT needed).

    Averaged over the horizon (or over valid steps if ``valid_mask`` is given). Building block for the
    saturating [0, 1] uncertainty ``1 - exp(-TrΣ_o / τ)``.
    """
    if log_var.numel() == 0:
        return log_var.new_zeros((0,))
    trace = torch.exp(log_var).sum(dim=-1)  # [Q, H]
    if valid_mask is not None:
        w = valid_mask.to(trace.dtype)
        return (trace * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
    return trace.mean(dim=1)


# =====================================================================
# Stage-1 evaluation metrics (§20.1 perception / §20.2 motion prediction)
# =====================================================================


def perception_metrics(
    prob: torch.Tensor,
    label: torch.Tensor,
    *,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """§20.1 precision / recall / F1 for a binary head (``prob``/``label``: ``[Q]``)."""
    if prob.numel() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred = (prob >= float(threshold)).float()
    target = (label >= 0.5).float()
    tp = float((pred * target).sum())
    fp = float((pred * (1.0 - target)).sum())
    fn = float(((1.0 - pred) * target).sum())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return {"precision": precision, "recall": recall, "f1": f1}


def trajectory_ade_fde(
    mu: torch.Tensor,
    target_xy: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    notable_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """§20.2 ADE / FDE (+ task-weighted) over future positions.

    ``mu``/``target_xy``: ``[Q, H, 2]``; ``valid_mask``: ``[Q, H]``; ``notable_weight``: ``[Q]``.
    ADE averages per-step displacement over valid steps; FDE uses the last valid step per object.
    """
    if mu.numel() == 0:
        return {"ade": 0.0, "fde": 0.0, "ade_notable": 0.0, "fde_notable": 0.0}
    dist = torch.linalg.norm(target_xy - mu, dim=-1)  # [Q, H]
    vmask = torch.ones_like(dist) if valid_mask is None else valid_mask.to(dist.dtype)
    nweight = torch.ones(mu.shape[0], device=mu.device) if notable_weight is None else notable_weight.to(dist.dtype)

    def _ade(w_obj: torch.Tensor) -> float:
        w = vmask * w_obj.unsqueeze(-1)
        return float((dist * w).sum() / w.sum().clamp_min(eps))

    def _fde(w_obj: torch.Tensor) -> float:
        # last valid step per object.
        steps = torch.arange(dist.shape[1], device=dist.device).unsqueeze(0).expand_as(dist)
        masked_steps = torch.where(vmask > 0.5, steps, torch.full_like(steps, -1))
        last = masked_steps.max(dim=1).values  # [Q]; -1 if no valid step
        has = last >= 0
        if not bool(has.any()):
            return 0.0
        idx = last.clamp_min(0)
        fde = dist.gather(1, idx.unsqueeze(1)).squeeze(1)  # [Q]
        w = has.float() * w_obj
        return float((fde * w).sum() / w.sum().clamp_min(eps))

    ones = torch.ones_like(nweight)
    return {
        "ade": _ade(ones),
        "fde": _fde(ones),
        "ade_notable": _ade(nweight),
        "fde_notable": _fde(nweight),
    }
