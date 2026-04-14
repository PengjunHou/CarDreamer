from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised by runtime guards instead
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def torch_is_available() -> bool:
    return torch is not None


@dataclass
class GraphGRUEmulationConfig:
    node_dim: int
    query_dim: int
    edge_attr_dim: int = 3
    hidden_dim: int = 64
    num_graph_layers: int = 2
    history_len: int = 8
    horizon: int = 5
    dropout: float = 0.0


if torch_is_available():

    class _GraphMessageLayer(nn.Module):
        def __init__(self, in_dim: int, edge_attr_dim: int, out_dim: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.self_linear = nn.Linear(in_dim, out_dim)
            self.msg_linear = nn.Linear(in_dim + edge_attr_dim, out_dim)
            self.norm = nn.LayerNorm(out_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor | None = None,
            edge_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch_size, num_nodes, _ = x.shape
            num_edges = int(edge_index.shape[1])
            residual = self.self_linear(x)
            if num_edges == 0:
                return self.norm(F.relu(residual))

            src = edge_index[0].long()
            dst = edge_index[1].long()
            src_feat = x[:, src, :]
            if edge_attr is None:
                edge_attr = torch.zeros(
                    (batch_size, num_edges, 0),
                    dtype=x.dtype,
                    device=x.device,
                )
            msg_input = torch.cat([src_feat, edge_attr], dim=-1)
            messages = self.msg_linear(msg_input)

            if edge_mask is None:
                edge_mask = torch.ones(
                    (batch_size, num_edges),
                    dtype=x.dtype,
                    device=x.device,
                )
            messages = messages * edge_mask.unsqueeze(-1)

            aggregated = x.new_zeros((batch_size, num_nodes, messages.shape[-1]))
            counts = x.new_zeros((batch_size, num_nodes))
            ones = torch.ones((batch_size, num_edges), dtype=x.dtype, device=x.device) * edge_mask
            for bidx in range(batch_size):
                aggregated[bidx].index_add_(0, dst, messages[bidx])
                counts[bidx].index_add_(0, dst, ones[bidx])
            aggregated = aggregated / counts.clamp_min(1.0).unsqueeze(-1)
            return self.norm(F.relu(self.dropout(residual + aggregated)))


    class GraphGRUEmulationModel(nn.Module):
        def __init__(self, config: GraphGRUEmulationConfig) -> None:
            super().__init__()
            self.config = config
            self.node_input = nn.Linear(config.node_dim, config.hidden_dim)
            self.graph_layers = nn.ModuleList(
                [
                    _GraphMessageLayer(
                        config.hidden_dim,
                        config.edge_attr_dim,
                        config.hidden_dim,
                        dropout=config.dropout,
                    )
                    for _ in range(config.num_graph_layers)
                ]
            )
            self.node_temporal = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                batch_first=True,
            )
            self.global_temporal = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                batch_first=True,
            )
            self.query_encoder = nn.Sequential(
                nn.Linear(config.query_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.node_query_fuser = nn.Sequential(
                nn.Linear(config.hidden_dim * 2 + 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.global_query_fuser = nn.Sequential(
                nn.Linear(config.hidden_dim * 2 + 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.sender_collab_head = nn.Linear(config.hidden_dim, config.horizon)
            self.sender_gain_head = nn.Linear(config.hidden_dim, config.horizon)
            self.ego_sc_head = nn.Linear(config.hidden_dim, config.horizon)

        def forward(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
            node_features = _as_tensor(batch["node_features"]).float()
            node_mask = _as_tensor(batch["node_mask"]).float()
            query_features = _as_tensor(batch["query_features"]).float()
            task_relevance = _as_tensor(batch["task_relevance"]).float()
            edge_attr = _as_tensor(batch["edge_attr"]).float()
            edge_mask = _as_tensor(batch["edge_mask"]).float()
            history_mask = _as_tensor(batch.get("history_mask", torch.ones(node_features.shape[:2]))).float()

            if node_features.dim() == 3:
                node_features = node_features.unsqueeze(0)
                node_mask = node_mask.unsqueeze(0)
                query_features = query_features.unsqueeze(0)
                task_relevance = task_relevance.unsqueeze(0)
                edge_attr = edge_attr.unsqueeze(0)
                edge_mask = edge_mask.unsqueeze(0)
                history_mask = history_mask.unsqueeze(0)

            edge_index = _as_tensor(batch["edge_index"]).long()
            if edge_index.dim() == 3:
                edge_index = edge_index[0]

            batch_size, history_len, num_nodes, _ = node_features.shape
            step_node_embeddings = []

            for tidx in range(history_len):
                x = self.node_input(node_features[:, tidx, :, :])
                x = x * node_mask[:, tidx, :].unsqueeze(-1)
                for layer in self.graph_layers:
                    x = layer(
                        x,
                        edge_index=edge_index,
                        edge_attr=edge_attr[:, tidx, :, :],
                        edge_mask=edge_mask[:, tidx, :],
                    )
                    x = x * node_mask[:, tidx, :].unsqueeze(-1)
                step_node_embeddings.append(x)
            node_history = torch.stack(step_node_embeddings, dim=1)

            node_history_flat = node_history.permute(0, 2, 1, 3).reshape(
                batch_size * num_nodes,
                history_len,
                self.config.hidden_dim,
            )
            _, node_hidden = self.node_temporal(node_history_flat)
            node_hidden = node_hidden[-1].reshape(batch_size, num_nodes, self.config.hidden_dim)

            pooled_steps = []
            for tidx in range(history_len):
                mask = node_mask[:, tidx, :].unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                pooled = (node_history[:, tidx, :, :] * mask).sum(dim=1) / denom
                pooled_steps.append(pooled)
            pooled_history = torch.stack(pooled_steps, dim=1)
            pooled_history = pooled_history * history_mask.unsqueeze(-1)
            _, global_hidden = self.global_temporal(pooled_history)
            global_hidden = global_hidden[-1]

            query_emb = self.query_encoder(query_features)
            hist_denom = history_mask.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
            task_mean = (task_relevance * history_mask[:, :, None, None]).sum(dim=1) / hist_denom
            task_current = task_relevance[:, -1, :, :]

            node_query_input = torch.cat(
                [
                    node_hidden.unsqueeze(2).expand(-1, -1, query_emb.shape[1], -1),
                    query_emb.unsqueeze(1).expand(-1, num_nodes, -1, -1),
                    task_current.unsqueeze(-1),
                    task_mean.unsqueeze(-1),
                ],
                dim=-1,
            )
            fused_node_query = self.node_query_fuser(node_query_input)
            sender_collab = self.sender_collab_head(fused_node_query).permute(0, 3, 1, 2)
            sender_gain = self.sender_gain_head(fused_node_query).permute(0, 3, 1, 2)

            pooled_task_current = (
                (task_current * node_mask[:, -1, :, None]).sum(dim=1)
                / node_mask[:, -1, :].sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            pooled_task_mean = (
                (task_mean * node_mask[:, -1, :, None]).sum(dim=1)
                / node_mask[:, -1, :].sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            global_query_input = torch.cat(
                [
                    global_hidden.unsqueeze(1).expand(-1, query_emb.shape[1], -1),
                    query_emb,
                    pooled_task_current.unsqueeze(-1),
                    pooled_task_mean.unsqueeze(-1),
                ],
                dim=-1,
            )
            fused_global_query = self.global_query_fuser(global_query_input)
            ego_sc = self.ego_sc_head(fused_global_query).permute(0, 2, 1)

            return {
                "sender_collab": sender_collab,
                "sender_gain": sender_gain,
                "ego_sc": ego_sc,
            }


else:

    class GraphGRUEmulationModel:  # pragma: no cover - exercised by import guard tests
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError(
                "GraphGRUEmulationModel requires PyTorch, but torch is not installed."
            ) from _TORCH_IMPORT_ERROR


def compute_emulation_loss(
    predictions: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    loss_type: str = "huber",
    delta: float = 1.0,
    sender_collab_weight: float = 1.0,
    sender_gain_weight: float = 1.0,
    ego_sc_weight: float = 1.0,
) -> Dict[str, Any]:
    if not torch_is_available():
        raise ImportError("compute_emulation_loss requires PyTorch.") from _TORCH_IMPORT_ERROR

    sender_collab_pred = _as_tensor(predictions["sender_collab"]).float()
    sender_gain_pred = _as_tensor(predictions["sender_gain"]).float()
    ego_sc_pred = _as_tensor(predictions["ego_sc"]).float()

    sender_collab_target = _as_tensor(batch["target_sender_collab"]).float()
    sender_gain_target = _as_tensor(batch["target_sender_gain"]).float()
    ego_sc_target = _as_tensor(batch["target_ego_sc"]).float()
    future_mask = _as_tensor(batch["future_mask"]).float()
    query_mask = _as_tensor(batch["query_mask"]).float()
    future_node_mask = _as_tensor(batch.get("future_node_mask", 1.0)).float()

    if sender_collab_target.dim() == 3:
        sender_collab_target = sender_collab_target.unsqueeze(0)
        sender_gain_target = sender_gain_target.unsqueeze(0)
        ego_sc_target = ego_sc_target.unsqueeze(0)
        future_mask = future_mask.unsqueeze(0)
        query_mask = query_mask.unsqueeze(0)
        if not torch.is_tensor(batch.get("future_node_mask")):
            future_node_mask = torch.ones_like(sender_collab_target[:, :, :, 0])
        elif future_node_mask.dim() == 2:
            future_node_mask = future_node_mask.unsqueeze(0)

    sender_mask = future_mask[:, :, None, None] * future_node_mask[:, :, :, None] * query_mask[:, None, None, :]
    ego_mask = future_mask[:, :, None] * query_mask[:, None, :]

    collab_loss = _masked_regression_loss(
        sender_collab_pred,
        sender_collab_target,
        sender_mask,
        loss_type=loss_type,
        delta=delta,
    )
    gain_loss = _masked_regression_loss(
        sender_gain_pred,
        sender_gain_target,
        sender_mask,
        loss_type=loss_type,
        delta=delta,
    )
    ego_loss = _masked_regression_loss(
        ego_sc_pred,
        ego_sc_target,
        ego_mask,
        loss_type=loss_type,
        delta=delta,
    )

    total = (
        float(sender_collab_weight) * collab_loss
        + float(sender_gain_weight) * gain_loss
        + float(ego_sc_weight) * ego_loss
    )
    return {
        "loss": total,
        "sender_collab_loss": collab_loss,
        "sender_gain_loss": gain_loss,
        "ego_sc_loss": ego_loss,
    }


def _masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_type: str,
    delta: float,
) -> torch.Tensor:
    prediction = prediction * mask
    target = target * mask
    if loss_type == "mse":
        losses = (prediction - target) ** 2
    else:
        losses = F.huber_loss(prediction, target, reduction="none", delta=float(delta))
    denom = mask.sum().clamp_min(1.0)
    return (losses * mask).sum() / denom


def _as_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)
