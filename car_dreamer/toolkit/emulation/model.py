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
    action_dim: int = 8
    vehicle_exogenous_dim: int = 7
    ego_state_dim: int = 5
    step_exogenous_dim: int = 2
    raw_state_dim: int = 5
    shared_state_dim: int = 20


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
                edge_attr = torch.zeros((batch_size, num_edges, 0), dtype=x.dtype, device=x.device)
            msg_input = torch.cat([src_feat, edge_attr], dim=-1)
            messages = self.msg_linear(msg_input)

            if edge_mask is None:
                edge_mask = torch.ones((batch_size, num_edges), dtype=x.dtype, device=x.device)
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
        """Policy-conditioned latent world model aligned with Main_V2X Section IV."""

        def __init__(self, config: GraphGRUEmulationConfig) -> None:
            super().__init__()
            self.config = config
            self.task_stat_dim = 2  # mean + max task relevance per vehicle

            self.state_input = nn.Linear(
                config.node_dim + self.task_stat_dim,
                config.hidden_dim,
            )
            self.action_input = nn.Linear(config.action_dim, config.hidden_dim)
            self.vehicle_exogenous_input = nn.Linear(config.vehicle_exogenous_dim, config.hidden_dim)
            self.ego_state_input = nn.Linear(config.ego_state_dim, config.hidden_dim)
            self.step_exogenous_input = nn.Linear(config.step_exogenous_dim, config.hidden_dim)
            self.step_fuser = nn.Sequential(
                nn.Linear(config.hidden_dim * 5, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
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
            self.node_history_encoder = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                batch_first=True,
            )
            self.global_history_encoder = nn.GRU(
                input_size=config.hidden_dim * 3,
                hidden_size=config.hidden_dim,
                batch_first=True,
            )

            self.transition_action = nn.Linear(config.action_dim, config.hidden_dim)
            self.node_transition = nn.GRUCell(config.hidden_dim * 3, config.hidden_dim)
            self.global_transition = nn.GRUCell(config.hidden_dim * 2, config.hidden_dim)

            self.query_encoder = nn.Sequential(
                nn.Linear(config.query_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.raw_decoder = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.raw_state_dim),
            )
            self.shared_decoder = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.shared_state_dim),
            )
            self.complementarity_head = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
            )
            self.accessibility_head = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
            )
            self.task_relevance_head = nn.Sequential(
                nn.Linear(config.hidden_dim * 3, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
            )
            self.sender_gain_head = nn.Sequential(
                nn.Linear(config.hidden_dim * 3, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
            )
            self.ego_sc_head = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
            state_node_features = _as_tensor(batch.get("state_node_features", batch["node_features"])).float()
            action_features = _as_tensor(batch.get("action_features", 0.0)).float()
            vehicle_exogenous_features = _as_tensor(batch.get("vehicle_exogenous_features", 0.0)).float()
            ego_state_features = _as_tensor(batch.get("ego_state_features", 0.0)).float()
            step_exogenous_features = _as_tensor(batch.get("step_exogenous_features", 0.0)).float()
            node_mask = _as_tensor(batch["node_mask"]).float()
            query_features = _as_tensor(batch["query_features"]).float()
            task_relevance = _as_tensor(batch["task_relevance"]).float()
            edge_attr = _as_tensor(batch["edge_attr"]).float()
            edge_mask = _as_tensor(batch["edge_mask"]).float()
            history_mask = _as_tensor(batch.get("history_mask", torch.ones(state_node_features.shape[:2]))).float()
            future_action_features = _as_tensor(batch.get("future_action_features", 0.0)).float()

            if state_node_features.dim() == 3:
                state_node_features = state_node_features.unsqueeze(0)
                action_features = action_features.unsqueeze(0)
                vehicle_exogenous_features = vehicle_exogenous_features.unsqueeze(0)
                ego_state_features = ego_state_features.unsqueeze(0)
                step_exogenous_features = step_exogenous_features.unsqueeze(0)
                node_mask = node_mask.unsqueeze(0)
                query_features = query_features.unsqueeze(0)
                task_relevance = task_relevance.unsqueeze(0)
                edge_attr = edge_attr.unsqueeze(0)
                edge_mask = edge_mask.unsqueeze(0)
                history_mask = history_mask.unsqueeze(0)
                future_action_features = future_action_features.unsqueeze(0)

            edge_index = _as_tensor(batch["edge_index"]).long()
            if edge_index.dim() == 3:
                edge_index = edge_index[0]

            batch_size, history_len, num_nodes, _ = state_node_features.shape
            task_stats = torch.stack(
                [
                    task_relevance.mean(dim=-1),
                    task_relevance.max(dim=-1).values,
                ],
                dim=-1,
            )
            step_node_embeddings = []
            step_global_embeddings = []
            for tidx in range(history_len):
                ego_embed = self.ego_state_input(ego_state_features[:, tidx, :]).unsqueeze(1).expand(-1, num_nodes, -1)
                step_exo_embed = self.step_exogenous_input(step_exogenous_features[:, tidx, :]).unsqueeze(1).expand(
                    -1, num_nodes, -1
                )
                x = self.step_fuser(
                    torch.cat(
                        [
                            self.state_input(
                                torch.cat(
                                    [
                                        state_node_features[:, tidx, :, :],
                                        task_stats[:, tidx, :, :],
                                    ],
                                    dim=-1,
                                )
                            ),
                            self.action_input(action_features[:, tidx, :, :]),
                            self.vehicle_exogenous_input(vehicle_exogenous_features[:, tidx, :, :]),
                            ego_embed,
                            step_exo_embed,
                        ],
                        dim=-1,
                    )
                )
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
                pooled = _masked_mean(x, node_mask[:, tidx, :])
                step_global_embeddings.append(
                    torch.cat(
                        [
                            pooled,
                            self.ego_state_input(ego_state_features[:, tidx, :]),
                            self.step_exogenous_input(step_exogenous_features[:, tidx, :]),
                        ],
                        dim=-1,
                    )
                )
            node_history = torch.stack(step_node_embeddings, dim=1)
            global_history = torch.stack(step_global_embeddings, dim=1) * history_mask.unsqueeze(-1)

            node_history_flat = node_history.permute(0, 2, 1, 3).reshape(
                batch_size * num_nodes,
                history_len,
                self.config.hidden_dim,
            )
            _, node_hidden = self.node_history_encoder(node_history_flat)
            node_latent = node_hidden[-1].reshape(batch_size, num_nodes, self.config.hidden_dim)
            _, global_hidden = self.global_history_encoder(global_history)
            global_latent = global_hidden[-1]

            query_emb = self.query_encoder(query_features)
            last_node_mask = node_mask[:, -1, :]

            raw_predictions = []
            shared_predictions = []
            complementarity_predictions = []
            accessibility_predictions = []
            task_relevance_predictions = []
            sender_collab_predictions = []
            sender_gain_predictions = []
            ego_sc_predictions = []

            horizon = min(int(self.config.horizon), int(future_action_features.shape[1]))
            for offset in range(horizon):
                action_emb = self.transition_action(future_action_features[:, offset, :, :])
                global_expand = global_latent.unsqueeze(1).expand(-1, num_nodes, -1)

                node_transition_in = torch.cat(
                    [node_latent, action_emb, global_expand],
                    dim=-1,
                )
                node_latent = self.node_transition(
                    node_transition_in.reshape(batch_size * num_nodes, -1),
                    node_latent.reshape(batch_size * num_nodes, -1),
                ).reshape(batch_size, num_nodes, -1)

                rollout_mask = ((future_action_features[:, offset, :, :].abs().sum(dim=-1) > 0).float() + last_node_mask).clamp(
                    max=1.0
                )
                pooled_nodes = _masked_mean(node_latent, rollout_mask)
                global_transition_in = torch.cat(
                    [
                        pooled_nodes,
                        _masked_mean(action_emb, rollout_mask),
                    ],
                    dim=-1,
                )
                global_latent = self.global_transition(global_transition_in, global_latent)
                decode_input = torch.cat([node_latent, global_latent.unsqueeze(1).expand(-1, num_nodes, -1)], dim=-1)

                raw_pred = self.raw_decoder(decode_input)
                shared_pred = self.shared_decoder(decode_input)
                comp_pred = torch.sigmoid(self.complementarity_head(decode_input).squeeze(-1))
                access_pred = torch.sigmoid(self.accessibility_head(decode_input).squeeze(-1))

                node_query_input = torch.cat(
                    [
                        node_latent.unsqueeze(2).expand(-1, -1, query_emb.shape[1], -1),
                        global_latent.unsqueeze(1).unsqueeze(2).expand(-1, num_nodes, query_emb.shape[1], -1),
                        query_emb.unsqueeze(1).expand(-1, num_nodes, -1, -1),
                    ],
                    dim=-1,
                )
                relevance_pred = torch.sigmoid(self.task_relevance_head(node_query_input).squeeze(-1))
                sender_gain_pred = self.sender_gain_head(node_query_input).squeeze(-1)
                sender_collab_pred = comp_pred.unsqueeze(-1) * relevance_pred * access_pred.unsqueeze(-1)

                global_query_input = torch.cat(
                    [
                        global_latent.unsqueeze(1).expand(-1, query_emb.shape[1], -1),
                        query_emb,
                    ],
                    dim=-1,
                )
                ego_sc_pred = self.ego_sc_head(global_query_input).squeeze(-1)

                raw_predictions.append(raw_pred)
                shared_predictions.append(shared_pred)
                complementarity_predictions.append(comp_pred)
                accessibility_predictions.append(access_pred)
                task_relevance_predictions.append(relevance_pred)
                sender_collab_predictions.append(sender_collab_pred)
                sender_gain_predictions.append(sender_gain_pred)
                ego_sc_predictions.append(ego_sc_pred)

            return {
                "raw_state": torch.stack(raw_predictions, dim=1),
                "shared_state": torch.stack(shared_predictions, dim=1),
                "derived_complementarity": torch.stack(complementarity_predictions, dim=1),
                "derived_accessibility": torch.stack(accessibility_predictions, dim=1),
                "derived_task_relevance": torch.stack(task_relevance_predictions, dim=1),
                "sender_collab": torch.stack(sender_collab_predictions, dim=1),
                "sender_gain": torch.stack(sender_gain_predictions, dim=1),
                "ego_sc": torch.stack(ego_sc_predictions, dim=1),
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
    raw_state_weight: float = 1.0,
    shared_state_weight: float = 1.0,
    sender_collab_weight: float = 1.0,
    sender_gain_weight: float = 1.0,
    ego_sc_weight: float = 1.0,
    consistency_weight: float = 0.0,
) -> Dict[str, Any]:
    if not torch_is_available():
        raise ImportError("compute_emulation_loss requires PyTorch.") from _TORCH_IMPORT_ERROR

    raw_state_pred = _as_tensor(predictions["raw_state"]).float()
    shared_state_pred = _as_tensor(predictions["shared_state"]).float()
    sender_collab_pred = _as_tensor(predictions["sender_collab"]).float()
    sender_gain_pred = _as_tensor(predictions["sender_gain"]).float()
    ego_sc_pred = _as_tensor(predictions["ego_sc"]).float()

    raw_state_target = _as_tensor(batch["target_raw_state"]).float()
    shared_state_target = _as_tensor(batch["target_shared_state"]).float()
    sender_collab_target = _as_tensor(batch["target_sender_collab"]).float()
    sender_gain_target = _as_tensor(batch["target_sender_gain"]).float()
    ego_sc_target = _as_tensor(batch["target_ego_sc"]).float()
    future_mask = _as_tensor(batch["future_mask"]).float()
    query_mask = _as_tensor(batch["query_mask"]).float()
    future_node_mask = _as_tensor(batch.get("future_node_mask", 1.0)).float()

    if sender_collab_target.dim() == 3:
        raw_state_target = raw_state_target.unsqueeze(0)
        shared_state_target = shared_state_target.unsqueeze(0)
        sender_collab_target = sender_collab_target.unsqueeze(0)
        sender_gain_target = sender_gain_target.unsqueeze(0)
        ego_sc_target = ego_sc_target.unsqueeze(0)
        future_mask = future_mask.unsqueeze(0)
        query_mask = query_mask.unsqueeze(0)
        if not torch.is_tensor(batch.get("future_node_mask")):
            future_node_mask = torch.ones_like(sender_collab_target[:, :, :, 0])
        elif future_node_mask.dim() == 2:
            future_node_mask = future_node_mask.unsqueeze(0)

    node_mask = future_mask[:, :, None, None] * future_node_mask[:, :, :, None]
    shared_state_mask = node_mask
    sender_mask = node_mask * query_mask[:, None, None, :]
    ego_mask = future_mask[:, :, None] * query_mask[:, None, :]

    raw_loss = _masked_regression_loss(
        raw_state_pred,
        raw_state_target,
        node_mask,
        loss_type=loss_type,
        delta=delta,
    )
    shared_loss = _masked_regression_loss(
        shared_state_pred,
        shared_state_target,
        shared_state_mask,
        loss_type=loss_type,
        delta=delta,
    )
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
    consistency_loss = _pairwise_consistency_loss(
        sender_collab_pred,
        sender_gain_pred,
        sender_mask,
    )

    total = (
        float(raw_state_weight) * raw_loss
        + float(shared_state_weight) * shared_loss
        + float(sender_collab_weight) * collab_loss
        + float(sender_gain_weight) * gain_loss
        + float(ego_sc_weight) * ego_loss
        + float(consistency_weight) * consistency_loss
    )
    return {
        "loss": total,
        "raw_state_loss": raw_loss,
        "shared_state_loss": shared_loss,
        "sender_collab_loss": collab_loss,
        "sender_gain_loss": gain_loss,
        "ego_sc_loss": ego_loss,
        "consistency_loss": consistency_loss,
    }


def _masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_type: str,
    delta: float,
) -> torch.Tensor:
    while mask.dim() < prediction.dim():
        mask = mask.unsqueeze(-1)
    prediction = prediction * mask
    target = target * mask
    if loss_type == "mse":
        losses = (prediction - target) ** 2
    else:
        losses = F.huber_loss(prediction, target, reduction="none", delta=float(delta))
    denom = mask.sum().clamp_min(1.0)
    return (losses * mask).sum() / denom


def _pairwise_consistency_loss(
    sender_collab_pred: torch.Tensor,
    sender_gain_pred: torch.Tensor,
    sender_mask: torch.Tensor,
) -> torch.Tensor:
    mask = sender_mask.bool()
    if sender_collab_pred.numel() == 0:
        return sender_collab_pred.new_tensor(0.0)
    diffs_collab = sender_collab_pred.unsqueeze(3) - sender_collab_pred.unsqueeze(2)
    diffs_gain = sender_gain_pred.unsqueeze(3) - sender_gain_pred.unsqueeze(2)
    pair_mask = (
        mask.unsqueeze(3)
        & mask.unsqueeze(2)
        & ~torch.eye(sender_collab_pred.shape[2], device=sender_collab_pred.device, dtype=torch.bool)[None, None, :, :, None]
    )
    signed_product = diffs_collab * diffs_gain
    losses = F.relu(-signed_product)
    if not pair_mask.any():
        return sender_collab_pred.new_tensor(0.0)
    return losses[pair_mask].mean()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (values * mask).sum(dim=1) / denom


def _as_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)
