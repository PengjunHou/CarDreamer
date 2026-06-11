"""BS-centric Graph Diffusion / Flow-Matching Unified World Model (WAM Design Update §1-§8).

The UWM lives at the Base Station. It conditions on the **BS global perception context** -- a set of
clean tokens assembled from every coverage vehicle's perception graph -- and jointly denoises the
request vehicle's **future collaboration-policy chunk** and **future request-vehicle BEV latent**::

    p_θ(π_{q,t:t+H-1}, z^bev_{q,t+1:t+H} | C^BS_{q,t-K:t}).

Condition tokens ``C^BS`` (clean, never noised, §6):
  1. per-vehicle perception graph tokens ``g^cp_v = pool(GraphEncoder(G^cp_v))`` for all coverage vehicles;
  2. the request-vehicle token ``g^req_q = g^cp_q + e^req`` (the request marker is a type embedding);
  3. driving-task tokens ``T^task = {h^obj_o : o ∈ notable(q)}`` -- the request graph's encoder object
     embeddings for its notable objects (the notable set *is* the driving task, §4);
  4. request-vehicle current+historical BEV latent tokens ``{z^bev_{q,τ}}``.

Noised tokens (§8): the policy chunk ``P_{s_π}`` and future BEV latent ``Z_{s_z}`` (decoupled diffusion
times ``s_π`` / ``s_z``), plus learnable register tokens ``R``. A single transformer ``v_θ`` predicts the
flow-matching velocities ``(û_π, û_z)``.

Like the rest of WAM this is a standalone, training-side module: :class:`WAMBSContextEncoder` (reusing the
Part-A edge-aware :class:`WAMHeteroGraphNet`) builds the condition tokens, and :class:`WAMFlowMatchingUWM`
is the vector field network. The offline Stage-2 pipeline (``stage2.py`` / ``flow_recorder.py``) records
samples and trains it. See ``docs/wam_implementation.md`` / ``docs/wam_stage2_training.md``.

Simplifications (v1, faithful to the update's "defer" notes): the request-vehicle BEV latent
(``bev_history`` condition tokens + the noised future ``Z`` target) is a **zero placeholder** until a real
per-vehicle BEV encoder is wired in -- so the generated BEV half is intentionally degenerate for now;
object-state ``X`` is no longer generated (it lives only in the driving-task condition tokens); the live
recorder stores a single request-vehicle graph (per-vehicle local graphs for *all* coverage vehicles are
deferred); ODE integration is fixed-step Euler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .graph import MODALITIES, OBJECT
from .graph_model import WAMGraphModelConfig, WAMHeteroGraphNet
from .runtime import WAMPolicy

# Token-type ids for the unified type-embedding table (§6 condition tokens + §8 noised tokens).
_T_CTX_GRAPH = 0   # per-vehicle perception graph token g^cp_v
_T_REQUEST = 1     # request-vehicle marker (g^cp_q + e^req): e^req == type_emb[_T_REQUEST]
_T_TASK = 2        # driving-task token (notable object embedding)
_T_BEV_HIST = 3    # request BEV latent condition token (current + history)
_T_POLICY = 4      # noised policy-chunk token
_T_BEV_FUT = 5     # noised future-BEV-latent token
_T_REGISTER = 6    # register token
_NUM_TOKEN_TYPES = 7


@dataclass
class WAMFlowMatchingConfig:
    """Hyper-parameters for the §8 diffusion transformer and §13 inference (BS-centric)."""

    # transformer (vector field network)
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.0
    time_embed_dim: int = 128
    # generated-variable shapes (§7-§8)
    max_members: int = 8          # M: candidate collaborators per policy step (== num_agent_slots)
    num_formats: int = len(MODALITIES)  # F: shareable data formats (objlist/bev)
    horizon: int = 6              # H: future policy chunk + future BEV-latent steps
    history_window: int = 0       # K: request BEV-latent history depth carried as condition tokens
    num_register_tokens: int = 4  # |R|: learnable register tokens
    bev_latent_dim: Optional[int] = None  # request-vehicle BEV latent dim; defaults to hidden_dim
    enable_bev: bool = True       # generate the future BEV latent half (the §7 future-observation variable)
    # loss / inference
    w_policy: float = 1.0
    w_bev: float = 1.0
    n_inference_steps: int = 10

    def __post_init__(self) -> None:
        if self.bev_latent_dim is None:
            self.bev_latent_dim = self.hidden_dim

    @property
    def policy_width(self) -> int:
        """``P = sel(1) + fmt(F) + freq(1) + bw(1)`` -- the per-member policy vector width."""
        return int(self.num_formats) + 3


# =====================================================================
# Time embedding + graph context pooling (reused building blocks)
# =====================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Map a scalar diffusion time ``s∈[0,1]`` to ``[B, out_dim]`` (the ``e(s_π)`` / ``e(s_z)`` of §8)."""

    def __init__(self, time_embed_dim: int, out_dim: int):
        super().__init__()
        self.time_embed_dim = int(time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _sinusoid(self, s: torch.Tensor) -> torch.Tensor:
        half = max(self.time_embed_dim // 2, 1)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=s.device, dtype=torch.float32) / half
        )
        args = s.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0  # s in [0,1] -> wide range
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.time_embed_dim:
            emb = torch.nn.functional.pad(emb, (0, self.time_embed_dim - emb.shape[-1]))
        return emb

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.dim() == 0:
            s = s.view(1)
        return self.mlp(self._sinusoid(s))


class WAMGraphContextPool(nn.Module):
    """``g^cp_v = AttentionPooling({z_{i}}_{i∈V})`` over all node embeddings of one graph.

    Consumes the encoder output ``H`` (a ``{node_type: [N_type, d]}`` dict from
    :class:`WAMHeteroGraphNet`) plus an optional matching ``{node_type: [N_type]}`` mask dict, and
    returns a single ``[d]`` graph token. Masked / padded nodes are excluded from the softmax.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_dict: Dict[str, torch.Tensor],
        mask_dict: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        feats, masks = [], []
        for ntype, h in node_dict.items():
            if h is None or h.numel() == 0:
                continue
            feats.append(h)
            m = None if mask_dict is None else mask_dict.get(ntype)
            masks.append(
                torch.ones(h.shape[0], device=h.device) if m is None else m.to(h.device).reshape(-1).float()
            )
        if not feats:
            return self.score.weight.new_zeros(self.hidden_dim)
        x = torch.cat(feats, dim=0)  # [N, d]
        m = torch.cat(masks, dim=0)  # [N]
        score = self.score(x).squeeze(-1).masked_fill(m < 0.5, float("-inf"))  # [N]
        weight = torch.nan_to_num(torch.softmax(score, dim=0), nan=0.0)  # [N]
        return (weight.unsqueeze(-1) * x).sum(dim=0)  # [d]


# =====================================================================
# Flow-matching interpolation + targets (§12.2-style / §15.3)
# =====================================================================


def interpolate(x0: torch.Tensor, x1: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Linear flow-matching interpolation ``x_s = (1 - s)·x0 + s·x1`` (``s`` is per-batch ``[B]``)."""
    while s.dim() < x1.dim():
        s = s.unsqueeze(-1)
    return (1.0 - s) * x0 + s * x1


def sample_training_batch(
    policy_1: torch.Tensor,
    bev_1: Optional[torch.Tensor] = None,
    *,
    enable_bev: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    """Draw a training batch: noise ``x0~N(0,I)``, decoupled diffusion times ``s_π,s_z~U(0,1)``.

    ``policy_1 [B,H,M,P]`` (GT future policy chunk), ``bev_1 [B,H,Dz]`` (GT future request BEV latent).
    Returns noised inputs (``policy_s``, ``bev_s``) + times + target velocities (``u_pi = π_1 - π_0``,
    ``u_bev = z_1 - z_0``).
    """
    b = policy_1.shape[0]
    device = policy_1.device
    s_pi = torch.rand(b, device=device)
    s_z = torch.rand(b, device=device)

    pi_0 = torch.randn_like(policy_1)
    out: Dict[str, Optional[torch.Tensor]] = {
        "policy_s": interpolate(pi_0, policy_1, s_pi),
        "bev_s": None,
        "s_pi": s_pi,
        "s_z": s_z,
        "u_pi": policy_1 - pi_0,
        "u_bev": None,
    }
    if enable_bev and bev_1 is not None:
        z_0 = torch.randn_like(bev_1)
        out["bev_s"] = interpolate(z_0, bev_1, s_z)
        out["u_bev"] = bev_1 - z_0
    return out


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    err = (pred - target) ** 2
    if mask is None:
        return err.mean()
    while mask.dim() < err.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(err)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def flow_matching_loss(
    pred: Dict[str, Optional[torch.Tensor]],
    target: Dict[str, Optional[torch.Tensor]],
    *,
    policy_mask: Optional[torch.Tensor] = None,
    bev_mask: Optional[torch.Tensor] = None,
    w_pi: float = 1.0,
    w_bev: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """``L = w_π·‖û_π − u_π‖² + w_z·‖û_z − u_z‖²`` as masked means (§15.3, BEV-only obs variable).

    ``policy_mask [B,H,M]`` gates policy members/steps; ``bev_mask [B,H]`` gates future BEV steps
    (each broadcast over the trailing feature dim). The BEV term is included only when both ``pred``
    and ``target`` carry ``u_bev``.
    """
    policy_loss = _masked_mse(pred["u_pi"], target["u_pi"], policy_mask)
    total = w_pi * policy_loss
    if pred.get("u_bev") is not None and target.get("u_bev") is not None:
        bev_loss = _masked_mse(pred["u_bev"], target["u_bev"], bev_mask)
        total = total + w_bev * bev_loss
    else:
        bev_loss = policy_loss.new_zeros(())
    return {"total": total, "policy": policy_loss, "bev": bev_loss}


# =====================================================================
# Policy <-> vector codec (interpretation only; flow runs in continuous space)
# =====================================================================

MODALITY_TO_ID_LOCAL = {name: i for i, name in enumerate(MODALITIES)}


def encode_policy(
    policy: WAMPolicy,
    candidate_ids: Sequence[int],
    *,
    max_members: int,
    num_formats: int,
) -> torch.Tensor:
    """Encode one :class:`WAMPolicy` to a ``[M_max, P]`` target (layout ``[sel, fmt..., freq, bw]``)."""
    p = num_formats + 3
    vec = torch.zeros(max_members, p)
    selected = set(int(i) for i in policy.selected_vehicle_ids)
    for m, cid in enumerate(candidate_ids):
        if m >= max_members:
            break
        cid = int(cid)
        vec[m, 0] = 1.0 if cid in selected else 0.0
        modality = policy.modality_by_vehicle.get(cid)
        if modality in MODALITY_TO_ID_LOCAL and MODALITY_TO_ID_LOCAL[modality] < num_formats:
            vec[m, 1 + MODALITY_TO_ID_LOCAL[modality]] = 1.0
        vec[m, 1 + num_formats] = float(policy.frequency_steps)
        vec[m, 2 + num_formats] = float(policy.bandwidth_by_vehicle.get(cid, 0.0))
    return vec


def encode_policy_chunk(
    policies: Sequence[WAMPolicy],
    candidate_ids: Sequence[int],
    *,
    max_members: int,
    num_formats: int,
) -> torch.Tensor:
    """Encode a length-``H`` future policy chunk to ``[H, M_max, P]`` (§7 ``π_{q,t:t+H-1}``)."""
    if len(policies) == 0:
        return torch.zeros(0, max_members, num_formats + 3)
    return torch.stack(
        [encode_policy(p, candidate_ids, max_members=max_members, num_formats=num_formats) for p in policies],
        dim=0,
    )


def decode_policy_vector(vec: torch.Tensor, *, num_formats: int) -> Dict[str, torch.Tensor]:
    """Decode a generated policy vector ``[..., M, P]`` into interpretable components (§12.1 ranges)."""
    sel = torch.sigmoid(vec[..., 0])
    fmt = torch.sigmoid(vec[..., 1 : 1 + num_formats])
    freq = vec[..., 1 + num_formats]
    bw = vec[..., 2 + num_formats]
    return {"sel": sel, "fmt": fmt, "freq": freq, "bw": bw}


# =====================================================================
# BS context encoder (§2-§6): vehicle graphs -> clean condition tokens
# =====================================================================


class WAMBSContextEncoder(nn.Module):
    """Assemble the BS global perception context ``C^BS`` (clean condition tokens) for one request.

    Reuses the Part-A edge-aware :class:`WAMHeteroGraphNet` to encode each vehicle's perception graph,
    :class:`WAMGraphContextPool` to pool each into a graph token, and emits the request marker,
    driving-task (notable object) tokens and request BEV-latent tokens. The unified type embedding is
    added by :class:`WAMFlowMatchingUWM` (so ``e^req`` == ``type_emb[_T_REQUEST]``); this module returns
    raw token vectors plus their type ids.
    """

    def __init__(self, graph_config: WAMGraphModelConfig, bev_latent_dim: int):
        super().__init__()
        self.graph_net = WAMHeteroGraphNet(graph_config)
        self.context_pool = WAMGraphContextPool(graph_config.hidden_dim)
        self.hidden_dim = int(graph_config.hidden_dim)
        self.bev_proj = nn.Linear(int(bev_latent_dim), self.hidden_dim)

    @property
    def device(self) -> torch.device:
        return self.bev_proj.weight.device

    def forward(
        self,
        vehicle_graphs: Sequence,
        request_index: int,
        notable_object_ids: Sequence[int],
        bev_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cond_tokens [T_c, d], cond_type_ids [T_c])`` for one request-vehicle sample."""
        device = self.device
        tokens: List[torch.Tensor] = []
        type_ids: List[int] = []
        req_obj_h: Optional[torch.Tensor] = None
        req_obj_ids: Optional[torch.Tensor] = None

        for i, graph in enumerate(vehicle_graphs):
            h = self.graph_net(graph)
            masks = {nt: getattr(graph[nt], "node_mask", None) for nt in h}
            g_tok = self.context_pool(h, masks)  # [d]
            tokens.append(g_tok)
            type_ids.append(_T_CTX_GRAPH)
            if i == int(request_index):
                tokens.append(g_tok)  # request marker token (type emb adds e^req)
                type_ids.append(_T_REQUEST)
                req_obj_h = h[OBJECT]
                req_obj_ids = graph[OBJECT].node_id.to(device)

        # driving-task tokens: request graph's object embeddings for its notable objects (§4).
        if req_obj_h is not None and req_obj_ids is not None and len(notable_object_ids) > 0:
            id_to_row = {int(v): r for r, v in enumerate(req_obj_ids.tolist()) if int(v) >= 0}
            for oid in notable_object_ids:
                r = id_to_row.get(int(oid))
                if r is not None:
                    tokens.append(req_obj_h[r])
                    type_ids.append(_T_TASK)

        # request BEV-latent condition tokens (current + history); zeros placeholder for now (§5).
        if bev_history is not None and bev_history.numel() > 0:
            proj = self.bev_proj(bev_history.to(device))  # [n_hist, d]
            for k in range(proj.shape[0]):
                tokens.append(proj[k])
                type_ids.append(_T_BEV_HIST)

        if not tokens:  # degenerate guard: at least the request marker should exist
            cond = self.bev_proj.weight.new_zeros((0, self.hidden_dim))
            return cond, torch.zeros(0, dtype=torch.long, device=device)
        cond = torch.stack(tokens, dim=0)  # [T_c, d]
        tid = torch.tensor(type_ids, dtype=torch.long, device=device)
        return cond, tid


def pad_condition_tokens(
    cond_list: Sequence[torch.Tensor],
    type_list: Sequence[torch.Tensor],
    hidden_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad a batch of per-sample condition tokens to ``[B, T_max, d]`` + type ids + valid mask."""
    b = len(cond_list)
    device = cond_list[0].device if b > 0 else torch.device("cpu")
    t_max = max((int(c.shape[0]) for c in cond_list), default=0)
    t_max = max(t_max, 1)
    cond = torch.zeros(b, t_max, hidden_dim, device=device)
    tids = torch.zeros(b, t_max, dtype=torch.long, device=device)
    mask = torch.zeros(b, t_max, device=device)
    for i, (c, t) in enumerate(zip(cond_list, type_list)):
        n = int(c.shape[0])
        if n == 0:
            continue
        cond[i, :n] = c
        tids[i, :n] = t
        mask[i, :n] = 1.0
    return cond, tids, mask


# =====================================================================
# Vector field network v_θ (§8 diffusion transformer)
# =====================================================================


class WAMFlowMatchingUWM(nn.Module):
    """§8 diffusion transformer: ``v_θ(C^BS, P_{s_π}, Z_{s_z}, R, s_π, s_z) = (û_π, û_z)``."""

    def __init__(self, config: WAMFlowMatchingConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        p = config.policy_width

        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim, d)
        self.type_emb = nn.Embedding(_NUM_TOKEN_TYPES, d)
        self.member_emb = nn.Embedding(config.max_members, d)
        self.step_emb = nn.Embedding(config.horizon, d)
        self.register = nn.Parameter(torch.randn(max(config.num_register_tokens, 0), d) * 0.02)

        # projectors: member-level policy chunk + Projector_bev (future request BEV latent).
        self.policy_proj = nn.Linear(p, d)
        self.bev_proj = nn.Linear(config.bev_latent_dim, d) if config.enable_bev else None

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)

        self.head_policy = nn.Linear(d, p)
        self.head_bev = nn.Linear(d, config.bev_latent_dim) if config.enable_bev else None

    @property
    def device(self) -> torch.device:
        return self.type_emb.weight.device

    def forward(
        self,
        cond_tokens: torch.Tensor,
        cond_type_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        policy_s: torch.Tensor,
        bev_s: Optional[torch.Tensor],
        s_pi: torch.Tensor,
        s_z: torch.Tensor,
        *,
        member_mask: Optional[torch.Tensor] = None,
        policy_step_mask: Optional[torch.Tensor] = None,
        bev_step_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        cfg = self.config
        d = cfg.hidden_dim
        device = cond_tokens.device
        b = cond_tokens.shape[0]
        m, h = cfg.max_members, cfg.horizon
        use_bev = cfg.enable_bev and bev_s is not None and self.bev_proj is not None

        if member_mask is None:
            member_mask = torch.ones(b, m, device=device)
        if policy_step_mask is None:
            policy_step_mask = torch.ones(b, h, device=device)
        if bev_step_mask is None:
            bev_step_mask = torch.ones(b, h, device=device)

        e_pi = self.time_embed(s_pi)  # [B, d]
        e_z = self.time_embed(s_z)    # [B, d]
        h_idx = torch.arange(h, device=device)
        m_idx = torch.arange(m, device=device)

        # ---- condition tokens (clean): + type embedding (request marker == type_emb[_T_REQUEST]) ----
        cond = cond_tokens + self.type_emb(cond_type_ids)  # [B, T_c, d]
        seq_parts = [cond]
        valid_parts = [cond_mask]

        # ---- register tokens (learnable scratch) ----
        r = self.register.shape[0]
        if r > 0:
            reg = (self.register + self.type_emb.weight[_T_REGISTER]).unsqueeze(0).expand(b, r, d)
            seq_parts.append(reg)
            valid_parts.append(torch.ones(b, r, device=device))

        # ---- noised policy-chunk tokens (H*M) ----
        pol_tok = (
            self.policy_proj(policy_s)
            + self.type_emb.weight[_T_POLICY]
            + self.step_emb(h_idx).view(1, h, 1, d)
            + self.member_emb(m_idx).view(1, 1, m, d)
            + e_pi.view(b, 1, 1, d)
        ).reshape(b, h * m, d)
        pol_valid = (policy_step_mask.unsqueeze(-1) * member_mask.unsqueeze(1)).reshape(b, h * m)
        seq_parts.append(pol_tok)
        valid_parts.append(pol_valid)

        # ---- noised future-BEV-latent tokens (H) ----
        if use_bev:
            bev_tok = (
                self.bev_proj(bev_s)
                + self.type_emb.weight[_T_BEV_FUT]
                + self.step_emb(h_idx).view(1, h, d)
                + e_z.view(b, 1, d)
            )  # [B, H, d]
            seq_parts.append(bev_tok)
            valid_parts.append(bev_step_mask)

        seq = torch.cat(seq_parts, dim=1)  # [B, T, d]
        key_padding_mask = torch.cat(valid_parts, dim=1) < 0.5  # True == ignore as key
        enc = self.transformer(seq, src_key_padding_mask=key_padding_mask)

        off = cond.shape[1] + r
        pol_out = enc[:, off : off + h * m, :].reshape(b, h, m, d)
        off += h * m
        out: Dict[str, Optional[torch.Tensor]] = {
            "u_pi": self.head_policy(pol_out),
            "u_bev": None,
        }
        if use_bev:
            bev_out = enc[:, off : off + h, :].reshape(b, h, d)
            out["u_bev"] = self.head_bev(bev_out)
        return out

    def sample_training_batch(
        self,
        policy_1: torch.Tensor,
        bev_1: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        return sample_training_batch(policy_1, bev_1, enable_bev=self.config.enable_bev)

    # =================================================================
    # §13 inference modes -- fixed-step Euler ODE integration
    # =================================================================
    def _noise_bev(self, b: int, device) -> Optional[torch.Tensor]:
        cfg = self.config
        if not (cfg.enable_bev and self.bev_proj is not None):
            return None
        return torch.randn(b, cfg.horizon, cfg.bev_latent_dim, device=device)

    @torch.no_grad()
    def rollout_future_bev(
        self,
        cond_tokens: torch.Tensor,
        cond_type_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        policy: torch.Tensor,
        *,
        member_mask: Optional[torch.Tensor] = None,
        policy_step_mask: Optional[torch.Tensor] = None,
        bev_step_mask: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """§13.1: given ``policy`` chunk (``s_π=1``), integrate ``s_z`` 0->1 to roll out ``Ẑ^bev [B,H,Dz]``."""
        b = cond_tokens.shape[0]
        device = cond_tokens.device
        bev = self._noise_bev(b, device)
        if bev is None:
            return None
        n = int(n_steps or self.config.n_inference_steps)
        dt = 1.0 / n
        s_pi = torch.ones(b, device=device)
        for k in range(n):
            s_z = torch.full((b,), k * dt, device=device)
            out = self.forward(
                cond_tokens, cond_type_ids, cond_mask, policy, bev, s_pi, s_z,
                member_mask=member_mask, policy_step_mask=policy_step_mask, bev_step_mask=bev_step_mask,
            )
            bev = bev + dt * out["u_bev"]
        return bev

    @torch.no_grad()
    def propose_policies(
        self,
        cond_tokens: torch.Tensor,
        cond_type_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        *,
        n_candidates: int,
        member_mask: Optional[torch.Tensor] = None,
        policy_step_mask: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """§13.2: marginalize BEV (``s_z=0``), integrate ``s_π`` 0->1 to draw ``n`` candidate policy chunks.

        Returns ``[B, n_candidates, H, M, P]``.
        """
        cfg = self.config
        b = cond_tokens.shape[0]
        device = cond_tokens.device
        n = int(n_steps or cfg.n_inference_steps)
        dt = 1.0 / n
        bn = b * int(n_candidates)
        ct = cond_tokens.repeat_interleave(int(n_candidates), dim=0)
        cti = cond_type_ids.repeat_interleave(int(n_candidates), dim=0)
        cm = cond_mask.repeat_interleave(int(n_candidates), dim=0)
        mm = None if member_mask is None else member_mask.repeat_interleave(int(n_candidates), dim=0)
        pm = None if policy_step_mask is None else policy_step_mask.repeat_interleave(int(n_candidates), dim=0)
        s_z = torch.zeros(bn, device=device)
        bev = self._noise_bev(bn, device)
        pi = torch.randn(bn, cfg.horizon, cfg.max_members, cfg.policy_width, device=device)
        for k in range(n):
            s_pi = torch.full((bn,), k * dt, device=device)
            out = self.forward(ct, cti, cm, pi, bev, s_pi, s_z, member_mask=mm, policy_step_mask=pm)
            pi = pi + dt * out["u_pi"]
        return pi.view(b, int(n_candidates), cfg.horizon, cfg.max_members, cfg.policy_width)

    @torch.no_grad()
    def inverse_policy_search(
        self,
        cond_tokens: torch.Tensor,
        cond_type_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        bev_target: torch.Tensor,
        *,
        member_mask: Optional[torch.Tensor] = None,
        policy_step_mask: Optional[torch.Tensor] = None,
        bev_step_mask: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """§13.3: pin the target future BEV latent (``s_z=1``), integrate ``s_π`` 0->1 -> policy chunk."""
        cfg = self.config
        b = cond_tokens.shape[0]
        device = cond_tokens.device
        n = int(n_steps or cfg.n_inference_steps)
        dt = 1.0 / n
        s_z = torch.ones(b, device=device)
        pi = torch.randn(b, cfg.horizon, cfg.max_members, cfg.policy_width, device=device)
        for k in range(n):
            s_pi = torch.full((b,), k * dt, device=device)
            out = self.forward(
                cond_tokens, cond_type_ids, cond_mask, pi, bev_target, s_pi, s_z,
                member_mask=member_mask, policy_step_mask=policy_step_mask, bev_step_mask=bev_step_mask,
            )
            pi = pi + dt * out["u_pi"]
        return pi

    @torch.no_grad()
    def joint_generate(
        self,
        cond_tokens: torch.Tensor,
        cond_type_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        *,
        member_mask: Optional[torch.Tensor] = None,
        policy_step_mask: Optional[torch.Tensor] = None,
        bev_step_mask: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Integrate both diffusion times together to jointly sample ``(π chunk, Ẑ^bev)``."""
        cfg = self.config
        b = cond_tokens.shape[0]
        device = cond_tokens.device
        n = int(n_steps or cfg.n_inference_steps)
        dt = 1.0 / n
        pi = torch.randn(b, cfg.horizon, cfg.max_members, cfg.policy_width, device=device)
        bev = self._noise_bev(b, device)
        for k in range(n):
            s = torch.full((b,), k * dt, device=device)
            out = self.forward(
                cond_tokens, cond_type_ids, cond_mask, pi, bev, s, s,
                member_mask=member_mask, policy_step_mask=policy_step_mask, bev_step_mask=bev_step_mask,
            )
            pi = pi + dt * out["u_pi"]
            if bev is not None and out["u_bev"] is not None:
                bev = bev + dt * out["u_bev"]
        return {"policy": pi, "bev": bev}


# =====================================================================
# End-to-end composition: vehicle graphs -> C^BS -> diffusion transformer
# =====================================================================


class WAMUnifiedWorldModel(nn.Module):
    """Compose :class:`WAMBSContextEncoder` (§2-§6) + :class:`WAMFlowMatchingUWM` (§7-§8)."""

    def __init__(self, graph_config: WAMGraphModelConfig, flow_config: WAMFlowMatchingConfig):
        super().__init__()
        if int(graph_config.hidden_dim) != int(flow_config.hidden_dim):
            raise ValueError(
                f"hidden_dim must match: graph={graph_config.hidden_dim} flow={flow_config.hidden_dim}"
            )
        self.context_encoder = WAMBSContextEncoder(graph_config, flow_config.bev_latent_dim)
        self.flow = WAMFlowMatchingUWM(flow_config)
        self.flow_config = flow_config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def condition_tokens(
        self,
        vehicle_graphs: Sequence,
        request_index: int,
        notable_object_ids: Sequence[int],
        bev_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode one request sample's vehicle graphs into ``(cond_tokens [T_c,d], type_ids [T_c])``."""
        return self.context_encoder(vehicle_graphs, request_index, notable_object_ids, bev_history)

    def condition_tokens_batch(
        self, samples: Sequence[Dict[str, object]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a padded condition-token batch from samples (grads reach the graph encoder).

        Each sample is a dict with ``vehicle_graphs``, ``request_index``, ``notable_object_ids`` and an
        optional ``bev_history``. Returns ``(cond [B,T,d], type_ids [B,T], mask [B,T])``.
        """
        cond_list, type_list = [], []
        for s in samples:
            c, t = self.condition_tokens(
                s["vehicle_graphs"],
                int(s["request_index"]),
                s.get("notable_object_ids", ()),
                s.get("bev_history"),
            )
            cond_list.append(c)
            type_list.append(t)
        return pad_condition_tokens(cond_list, type_list, self.flow_config.hidden_dim)

    def training_step(
        self,
        cond_tokens: torch.Tensor,
        cond_type_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        policy_1: torch.Tensor,
        bev_1: Optional[torch.Tensor] = None,
        *,
        member_mask: Optional[torch.Tensor] = None,
        policy_step_mask: Optional[torch.Tensor] = None,
        bev_step_mask: Optional[torch.Tensor] = None,
        policy_loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """One flow-matching step given pre-built condition tokens + GT policy chunk / future BEV."""
        batch = self.flow.sample_training_batch(policy_1, bev_1)
        pred = self.flow.forward(
            cond_tokens, cond_type_ids, cond_mask,
            batch["policy_s"], batch["bev_s"], batch["s_pi"], batch["s_z"],
            member_mask=member_mask, policy_step_mask=policy_step_mask, bev_step_mask=bev_step_mask,
        )
        return flow_matching_loss(
            pred, batch,
            policy_mask=policy_loss_mask, bev_mask=bev_step_mask,
            w_pi=self.flow_config.w_policy, w_bev=self.flow_config.w_bev,
        )
