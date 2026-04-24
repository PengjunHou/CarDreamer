from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .features import (
    build_pairwise_edge_attr,
    get_component_valid_mask_layout,
    pack_component_valid_mask,
    pack_ego_state_features,
    pack_query_features,
    pack_step_exogenous_features,
    pack_vehicle_action_features,
    pack_vehicle_exogenous_features,
    pack_vehicle_node_state,
    pack_vehicle_raw_state_target,
    pack_vehicle_shared_state_target,
    pack_vehicle_state_features,
)
from .schema import CanonicalEpisodeRecord, CanonicalStepRecord


def build_fully_connected_edge_index(num_nodes: int, include_self: bool = False) -> np.ndarray:
    if int(num_nodes) <= 0:
        return np.zeros((2, 0), dtype=np.int64)
    edges: List[List[int]] = []
    for src in range(int(num_nodes)):
        for dst in range(int(num_nodes)):
            if not include_self and src == dst:
                continue
            edges.append([src, dst])
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


@dataclass
class _EpisodeIndex:
    episode: CanonicalEpisodeRecord
    node_ids: List[int]
    query_ids: List[str]
    node_id_to_slot: Dict[int, int]


class CanonicalEmulationDataset:
    def __init__(
        self,
        episodes: Sequence[CanonicalEpisodeRecord],
        *,
        history_len: int = 8,
        horizon: int = 5,
        max_nodes: int | None = None,
        max_queries: int | None = None,
    ) -> None:
        if not episodes:
            raise ValueError("CanonicalEmulationDataset requires at least one episode.")
        self.history_len = max(int(history_len), 1)
        self.horizon = max(int(horizon), 1)
        self.episodes = list(episodes)
        self._require_shared_latent_support()
        self.episode_indices = [self._build_episode_index(episode) for episode in self.episodes]
        self.max_nodes = int(max_nodes or max(len(index.node_ids) for index in self.episode_indices))
        self.max_queries = int(max_queries or max(len(index.query_ids) for index in self.episode_indices))
        self.edge_index = build_fully_connected_edge_index(self.max_nodes, include_self=False)
        self.component_mask_dim = len(get_component_valid_mask_layout())
        self.shared_latent_mask_index = get_component_valid_mask_layout().index("shared_latent")
        (
            self.raw_node_dim,
            self.state_node_dim,
            self.action_dim,
            self.vehicle_exogenous_dim,
            self.shared_state_dim,
            self.raw_state_dim,
            self.ego_state_dim,
            self.step_exogenous_dim,
            self.query_dim,
        ) = self._infer_feature_dims()
        self.samples = self._build_sample_index()

    def _require_shared_latent_support(self) -> None:
        for episode in self.episodes:
            for step in episode.steps:
                for vehicle in step.candidate_vehicles:
                    if vehicle.shared_latent:
                        continue
                    raise ValueError(
                        "Episode "
                        f"{episode.episode_id} step {step.step} vehicle {vehicle.vehicle_id} "
                        "is missing shared_latent. Old compact-summary-only data is not supported "
                        "in strict shared-latent mode. Please regenerate or re-export "
                        "with fixed-width shared_latent vectors."
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_meta = self.samples[index]
        episode_index = self.episode_indices[sample_meta["episode_idx"]]
        episode = episode_index.episode
        end_step = int(sample_meta["step"])

        node_features = np.zeros((self.history_len, self.max_nodes, self.raw_node_dim), dtype=np.float32)
        state_node_features = np.zeros((self.history_len, self.max_nodes, self.state_node_dim), dtype=np.float32)
        action_features = np.zeros((self.history_len, self.max_nodes, self.action_dim), dtype=np.float32)
        vehicle_exogenous_features = np.zeros(
            (self.history_len, self.max_nodes, self.vehicle_exogenous_dim),
            dtype=np.float32,
        )
        component_valid_mask = np.zeros(
            (self.history_len, self.max_nodes, self.component_mask_dim),
            dtype=np.float32,
        )
        history_shared_latent_mask = np.zeros((self.history_len, self.max_nodes), dtype=np.float32)
        node_mask = np.zeros((self.history_len, self.max_nodes), dtype=np.float32)
        task_relevance = np.zeros((self.history_len, self.max_nodes, self.max_queries), dtype=np.float32)
        history_mask = np.zeros((self.history_len,), dtype=np.float32)
        edge_attr = np.zeros((self.history_len, self.edge_index.shape[1], 3), dtype=np.float32)
        edge_mask = np.zeros((self.history_len, self.edge_index.shape[1]), dtype=np.float32)
        ego_state_features = np.zeros((self.history_len, self.ego_state_dim), dtype=np.float32)
        step_exogenous_features = np.zeros((self.history_len, self.step_exogenous_dim), dtype=np.float32)

        query_features = np.zeros((self.max_queries, self.query_dim), dtype=np.float32)
        query_mask = np.zeros((self.max_queries,), dtype=np.float32)
        for qidx, query in enumerate(episode.steps[0].queries):
            if qidx >= self.max_queries:
                break
            query_features[qidx] = pack_query_features(query)
            query_mask[qidx] = 1.0

        hist_start = end_step - self.history_len + 1
        for slot in range(self.history_len):
            step_index = hist_start + slot
            if step_index < 0 or step_index >= len(episode.steps):
                continue
            history_mask[slot] = 1.0
            step = episode.steps[step_index]
            self._fill_step_tensors(
                step,
                episode_index,
                node_features[slot],
                state_node_features[slot],
                action_features[slot],
                vehicle_exogenous_features[slot],
                component_valid_mask[slot],
                history_shared_latent_mask[slot],
                node_mask[slot],
                task_relevance[slot],
            )
            ego_state_features[slot] = pack_ego_state_features(step.ego_state)
            step_exogenous_features[slot] = pack_step_exogenous_features(step)
            positions = node_features[slot, :, 0:2]
            edge_attr[slot] = build_pairwise_edge_attr(positions, self.edge_index)
            if self.edge_index.shape[1] > 0:
                src = self.edge_index[0]
                dst = self.edge_index[1]
                edge_mask[slot] = node_mask[slot, src] * node_mask[slot, dst]

        future_mask = np.zeros((self.horizon,), dtype=np.float32)
        future_node_mask = np.zeros((self.horizon, self.max_nodes), dtype=np.float32)
        future_shared_latent_mask = np.zeros((self.horizon, self.max_nodes), dtype=np.float32)
        future_action_features = np.zeros((self.horizon, self.max_nodes, self.action_dim), dtype=np.float32)
        future_vehicle_exogenous_features = np.zeros(
            (self.horizon, self.max_nodes, self.vehicle_exogenous_dim),
            dtype=np.float32,
        )
        future_step_exogenous_features = np.zeros((self.horizon, self.step_exogenous_dim), dtype=np.float32)
        target_sender_collab = np.zeros((self.horizon, self.max_nodes, self.max_queries), dtype=np.float32)
        target_sender_gain = np.zeros((self.horizon, self.max_nodes, self.max_queries), dtype=np.float32)
        target_ego_sc = np.zeros((self.horizon, self.max_queries), dtype=np.float32)
        target_raw_state = np.zeros((self.horizon, self.max_nodes, self.raw_state_dim), dtype=np.float32)
        target_shared_state = np.zeros((self.horizon, self.max_nodes, self.shared_state_dim), dtype=np.float32)
        node_ids = np.full((self.max_nodes,), fill_value=-1, dtype=np.int64)
        query_ids = [""] * self.max_queries
        for nidx, node_id in enumerate(episode_index.node_ids[: self.max_nodes]):
            node_ids[nidx] = int(node_id)
        for qidx, query_id in enumerate(episode_index.query_ids[: self.max_queries]):
            query_ids[qidx] = str(query_id)

        for offset in range(self.horizon):
            future_index = end_step + offset + 1
            if future_index >= len(episode.steps):
                break
            future_mask[offset] = 1.0
            step = episode.steps[future_index]
            future_step_exogenous_features[offset] = pack_step_exogenous_features(step)
            for qidx, query in enumerate(step.queries[: self.max_queries]):
                target_ego_sc[offset, qidx] = float(step.ego_sc.get(query.query_id, 0.0))
            for vehicle in step.candidate_vehicles:
                slot = episode_index.node_id_to_slot.get(int(vehicle.vehicle_id))
                if slot is None or slot >= self.max_nodes:
                    continue
                future_node_mask[offset, slot] = 1.0
                future_shared_latent_mask[offset, slot] = 1.0 if bool(
                    vehicle.component_valid_mask.get("shared_latent", False)
                ) else 0.0
                future_action_features[offset, slot] = pack_vehicle_action_features(vehicle)
                future_vehicle_exogenous_features[offset, slot] = pack_vehicle_exogenous_features(vehicle)
                target_raw_state[offset, slot] = pack_vehicle_raw_state_target(vehicle)
                target_shared_state[offset, slot] = pack_vehicle_shared_state_target(vehicle)
                for qidx, query in enumerate(step.queries[: self.max_queries]):
                    qid = query.query_id
                    target_sender_collab[offset, slot, qidx] = float(vehicle.sender_collab.get(qid, 0.0))
                    target_sender_gain[offset, slot, qidx] = float(vehicle.sender_gain.get(qid, 0.0))

        # policy_id for this sample — used in policy-aware evaluation splits
        policy_id = str(episode.policy_id or episode.steps[end_step].policy_id or "")

        return {
            "episode_id": episode.episode_id,
            "scene_id": episode.scene_id,
            "scene_type": episode.scene_type,
            "policy_id": policy_id,
            "step": end_step,
            "node_ids": node_ids,
            "query_ids": query_ids,
            "history_mask": history_mask,
            "node_features": node_features,
            "state_node_features": state_node_features,
            "action_features": action_features,
            "vehicle_exogenous_features": vehicle_exogenous_features,
            "ego_state_features": ego_state_features,
            "step_exogenous_features": step_exogenous_features,
            "component_valid_mask": component_valid_mask,
            "history_shared_latent_mask": history_shared_latent_mask,
            "node_mask": node_mask,
            "edge_index": self.edge_index.copy(),
            "edge_attr": edge_attr,
            "edge_mask": edge_mask,
            "query_features": query_features,
            "query_mask": query_mask,
            "task_relevance": task_relevance,
            "future_mask": future_mask,
            "future_node_mask": future_node_mask,
            "future_shared_latent_mask": future_shared_latent_mask,
            "future_action_features": future_action_features,
            "future_vehicle_exogenous_features": future_vehicle_exogenous_features,
            "future_step_exogenous_features": future_step_exogenous_features,
            "target_raw_state": target_raw_state,
            "target_shared_state": target_shared_state,
            "target_sender_collab": target_sender_collab,
            "target_sender_gain": target_sender_gain,
            "target_ego_sc": target_ego_sc,
        }

    def _infer_feature_dims(self) -> tuple[int, int, int, int, int, int, int, int, int]:
        node_dim = 0
        state_node_dim = 0
        action_dim = 0
        vehicle_exogenous_dim = 0
        shared_state_dim = 0
        raw_state_dim = 0
        ego_state_dim = 0
        step_exogenous_dim = 0
        query_dim = 0
        for episode in self.episodes:
            for step in episode.steps:
                if step.queries and query_dim == 0:
                    query_dim = int(pack_query_features(step.queries[0]).shape[0])
                if ego_state_dim == 0:
                    ego_state_dim = int(pack_ego_state_features(step.ego_state).shape[0])
                if step_exogenous_dim == 0:
                    step_exogenous_dim = int(pack_step_exogenous_features(step).shape[0])
                if step.candidate_vehicles and node_dim == 0:
                    node_dim = int(pack_vehicle_node_state(step.candidate_vehicles[0]).shape[0])
                    state_node_dim = int(pack_vehicle_state_features(step.candidate_vehicles[0]).shape[0])
                    action_dim = int(pack_vehicle_action_features(step.candidate_vehicles[0]).shape[0])
                    vehicle_exogenous_dim = int(pack_vehicle_exogenous_features(step.candidate_vehicles[0]).shape[0])
                    shared_state_dim = int(pack_vehicle_shared_state_target(step.candidate_vehicles[0]).shape[0])
                    raw_state_dim = int(pack_vehicle_raw_state_target(step.candidate_vehicles[0]).shape[0])
                if (
                    node_dim > 0
                    and state_node_dim > 0
                    and action_dim > 0
                    and vehicle_exogenous_dim > 0
                    and shared_state_dim > 0
                    and raw_state_dim > 0
                    and ego_state_dim > 0
                    and step_exogenous_dim > 0
                    and query_dim > 0
                ):
                    return (
                        node_dim,
                        state_node_dim,
                        action_dim,
                        vehicle_exogenous_dim,
                        shared_state_dim,
                        raw_state_dim,
                        ego_state_dim,
                        step_exogenous_dim,
                        query_dim,
                    )
        raise ValueError("Could not infer feature dimensions from canonical episodes.")

    def _build_episode_index(self, episode: CanonicalEpisodeRecord) -> _EpisodeIndex:
        node_ids = sorted({int(vehicle.vehicle_id) for step in episode.steps for vehicle in step.candidate_vehicles})
        query_ids = [query.query_id for query in episode.steps[0].queries]
        return _EpisodeIndex(
            episode=episode,
            node_ids=node_ids,
            query_ids=query_ids,
            node_id_to_slot={node_id: idx for idx, node_id in enumerate(node_ids)},
        )

    def _build_sample_index(self) -> List[Dict[str, int]]:
        samples: List[Dict[str, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            for step in range(len(episode.steps)):
                samples.append({"episode_idx": episode_idx, "step": step})
        return samples

    def _fill_step_tensors(
        self,
        step: CanonicalStepRecord,
        episode_index: _EpisodeIndex,
        node_features: np.ndarray,
        state_node_features: np.ndarray,
        action_features: np.ndarray,
        vehicle_exogenous_features: np.ndarray,
        component_valid_mask: np.ndarray,
        history_shared_latent_mask: np.ndarray,
        node_mask: np.ndarray,
        task_relevance: np.ndarray,
    ) -> None:
        for vehicle in step.candidate_vehicles:
            slot = episode_index.node_id_to_slot.get(int(vehicle.vehicle_id))
            if slot is None or slot >= node_features.shape[0]:
                continue
            node_features[slot] = pack_vehicle_node_state(vehicle)
            state_node_features[slot] = pack_vehicle_state_features(vehicle)
            action_features[slot] = pack_vehicle_action_features(vehicle)
            vehicle_exogenous_features[slot] = pack_vehicle_exogenous_features(vehicle)
            component_valid_mask[slot] = pack_component_valid_mask(vehicle)
            history_shared_latent_mask[slot] = component_valid_mask[slot, self.shared_latent_mask_index]
            node_mask[slot] = 1.0
            for qidx, query in enumerate(step.queries[: task_relevance.shape[1]]):
                task_relevance[slot, qidx] = float(vehicle.query_task_relevance.get(query.query_id, 0.0))
