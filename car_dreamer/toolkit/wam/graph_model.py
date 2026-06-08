"""Initial node embeddings (§5) + heterogeneous graph transformer encoder (§9).

``WAMHeteroGraphNet`` consumes a :class:`torch_geometric.data.HeteroData` produced by
:func:`car_dreamer.toolkit.wam.graph.build_wam_hetero_graph` and returns the graph- and
type-contextualized node embeddings ``H_t = {h^veh, h^obs, h^obj}`` (WAM Design §9 output).

Two stages:
    * :class:`WAMHeteroGraphEmbedding` (§5): per-type MLP state encoders plus learned
      type / agent / time / class embeddings, the object-list modality feature
      ``z^objlist = AttentionPooling({MLP_obj(x^obj)}_{o in Obs(v,t)})`` (realized via the
      ``obs_obj`` edges) and a placeholder BEV encoder ``z^bev = E_bev(B^sem)`` (a small CNN
      over a zero ``B^sem`` until a real per-vehicle BEV semantic map is wired in).
    * :class:`WAMHeteroGraphEncoder` (§9): a stack of PyG ``HGTConv`` layers with residual +
      LayerNorm; relation-specific projections and the relation bias are provided by HGTConv,
      matching ``α^r_{ij}`` / ``W_r``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.utils import scatter, softmax

from .graph import (
    MODALITY_TO_ID,
    OBJECT_STATE_DIM,
    OBJECT,
    OBS_OBJ,
    OBS_SCALAR_DIM,
    OBSERVATION,
    VEHICLE,
    WAM_METADATA,
    vehicle_state_dim,
)

# Node-type embedding ids: vehicle / object / observation(objlist) / observation(bev).
_TYPE_VEHICLE = 0
_TYPE_OBJECT = 1
_TYPE_OBS_BASE = 2  # observation type id = _TYPE_OBS_BASE + modality_id
_NUM_TYPE_EMB = _TYPE_OBS_BASE + len(MODALITY_TO_ID)


@dataclass
class WAMGraphModelConfig:
    route_waypoints: int = 6
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    num_agent_slots: int = 8
    num_time_slots: int = 1
    num_object_classes: int = 4
    bev_channels: int = 8
    bev_size: int = 128

    @property
    def veh_state_dim(self) -> int:
        return vehicle_state_dim(self.route_waypoints)


def _mlp(in_dim: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
    )


class _BevEncoder(nn.Module):
    """Lightweight CNN ``E_bev: B^sem -> z^bev`` (§5.3). v1 runs on a zero placeholder."""

    def __init__(self, channels: int, size: int, out_dim: int):
        super().__init__()
        self.channels = int(channels)
        self.size = int(size)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Linear(32, out_dim)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(bev))


class WAMHeteroGraphEmbedding(nn.Module):
    """§5 initial embeddings -> ``h^0`` for every node type."""

    def __init__(self, config: WAMGraphModelConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim

        self.mlp_veh = _mlp(config.veh_state_dim, d)
        self.mlp_obj = _mlp(OBJECT_STATE_DIM, d)
        self.mlp_obs = _mlp(d + OBS_SCALAR_DIM, d)

        self.type_emb = nn.Embedding(_NUM_TYPE_EMB, d)
        self.agent_emb = nn.Embedding(config.num_agent_slots, d)
        self.time_emb = nn.Embedding(max(config.num_time_slots, 1), d)
        self.class_emb = nn.Embedding(config.num_object_classes, d)

        # objlist attention pooling: per-object score over a vehicle's observed objects.
        self.objlist_attn = nn.Linear(d, 1)
        self.bev_encoder = _BevEncoder(config.bev_channels, config.bev_size, d)

    def _clamp(self, idx: torch.Tensor, size: int) -> torch.Tensor:
        return idx.clamp(min=0, max=size - 1).long()

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        cfg = self.config
        device = self.type_emb.weight.device
        d = cfg.hidden_dim

        # ---- objects ----
        obj_x = data[OBJECT].x.to(device)
        g_obj = self.mlp_obj(obj_x)  # [No, d]; reused by both object node and objlist pooling
        class_id = self._clamp(data[OBJECT].class_id.to(device), cfg.num_object_classes)
        time0 = torch.zeros(1, dtype=torch.long, device=device)
        e_time = self.time_emb(time0)  # [1, d]
        h_obj = g_obj + self.type_emb.weight[_TYPE_OBJECT] + self.class_emb(class_id) + e_time

        # ---- vehicles ----
        veh_x = data[VEHICLE].x.to(device)
        veh_slot = self._clamp(data[VEHICLE].agent_slot.to(device), cfg.num_agent_slots)
        h_veh = self.mlp_veh(veh_x) + self.type_emb.weight[_TYPE_VEHICLE] + self.agent_emb(veh_slot) + e_time

        # ---- observations: build modality feature z^r ----
        modality = data[OBSERVATION].modality_id.to(device)
        n_obs = int(modality.shape[0])
        z = torch.zeros(n_obs, d, device=device)

        # objlist: attention-pool MLP_obj(x^obj) over obs_obj neighbors (§5.3).
        edge = data[OBS_OBJ].edge_index.to(device)
        if edge.numel() > 0 and n_obs > 0:
            obs_idx, obj_idx = edge[0], edge[1]
            msg = g_obj[obj_idx]  # [E, d]
            score = self.objlist_attn(msg).squeeze(-1)  # [E]
            weight = softmax(score, obs_idx, num_nodes=n_obs)  # per observation node
            pooled = scatter(msg * weight.unsqueeze(-1), obs_idx, dim=0, dim_size=n_obs, reduce="sum")
            z = z + pooled

        # bev: placeholder CNN over a zero B^sem (swap-in point for real per-vehicle BEV).
        bev_mask = modality == MODALITY_TO_ID.get("bev", -1)
        if bool(bev_mask.any()):
            n_bev = int(bev_mask.sum().item())
            bev_in = torch.zeros(n_bev, cfg.bev_channels, cfg.bev_size, cfg.bev_size, device=device)
            z = z.clone()
            z[bev_mask] = self.bev_encoder(bev_in)

        obs_scalar = data[OBSERVATION].x.to(device)
        obs_slot = self._clamp(data[OBSERVATION].agent_slot.to(device), cfg.num_agent_slots)
        obs_type = self._clamp(modality + _TYPE_OBS_BASE, _NUM_TYPE_EMB)
        h_obs = (
            self.mlp_obs(torch.cat([z, obs_scalar], dim=-1))
            + self.type_emb(obs_type)
            + self.agent_emb(obs_slot)
            + e_time
        )

        return {VEHICLE: h_veh, OBJECT: h_obj, OBSERVATION: h_obs}


class WAMHeteroGraphEncoder(nn.Module):
    """§9 heterogeneous graph transformer: stacked HGTConv + residual + LayerNorm."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, metadata=WAM_METADATA):
        super().__init__()
        self.node_types = list(metadata[0])
        self.convs = nn.ModuleList(
            [HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types}) for _ in range(num_layers)]
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict) -> Dict[str, torch.Tensor]:
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x_dict, edge_index_dict)
            updated: Dict[str, torch.Tensor] = {}
            for nt in x_dict:
                h = out.get(nt, None)
                # node types with no incoming edges this step are carried through unchanged.
                updated[nt] = norm[nt](F.relu(h) + x_dict[nt]) if h is not None else x_dict[nt]
            x_dict = updated
        return x_dict


class WAMHeteroGraphNet(nn.Module):
    """Initial embedding (§5) -> HGT encoder (§9). ``forward`` returns ``H_t``."""

    def __init__(self, config: WAMGraphModelConfig):
        super().__init__()
        self.config = config
        self.embedding = WAMHeteroGraphEmbedding(config)
        self.encoder = WAMHeteroGraphEncoder(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
        )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        h0 = self.embedding(data)
        edge_index_dict = {etype: data[etype].edge_index.to(self.config_device(h0)) for etype in data.edge_types}
        return self.encoder(h0, edge_index_dict)

    @staticmethod
    def config_device(h0: Dict[str, torch.Tensor]) -> torch.device:
        for tensor in h0.values():
            return tensor.device
        return torch.device("cpu")
