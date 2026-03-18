# coop_gnn_policy.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# torch-geometric (PyG)
from torch_geometric.data import Data


# =========================
# 1) GraphBuilder
# =========================

@dataclass
class GraphBuildConfig:
    window_s: float = 2.0          # 时间窗 (秒)
    Tmax: int = 20                 # 每个车辆最多使用多少条消息
    max_nodes: int = 8             # 图中最多多少个车辆节点（含ego），防止爆
    feat_dim_max: int = 1024       # 统一 padding 到这个维度
    include_sender_state: bool = True  # 是否附加 sender 的速度/yaw 等状态（推荐）
    star_graph: bool = True        # True=星形(ego<->others)，False=全连接


def _pad_feat(feat: np.ndarray, feat_dim_max: int) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    out = np.zeros((feat_dim_max,), dtype=np.float32)
    d = min(len(feat), feat_dim_max)
    if d > 0:
        out[:d] = feat[:d]
    return out


def _rel_pose_ego_frame(ego_tf, sender_tf) -> np.ndarray:
    """
    sender 相对 ego 的位姿（用 ego yaw 把世界坐标旋转到 ego 坐标系）
    输出: [dx_ego, dy_ego, dyaw]  (float32)
    """
    ex, ey = ego_tf.location.x, ego_tf.location.y
    sx, sy = sender_tf.location.x, sender_tf.location.y
    dxw, dyw = (sx - ex), (sy - ey)

    yaw = math.radians(float(ego_tf.rotation.yaw))
    c, s = math.cos(-yaw), math.sin(-yaw)  # world->ego rotation
    dx = c * dxw - s * dyw
    dy = s * dxw + c * dyw

    dyaw = float(sender_tf.rotation.yaw - ego_tf.rotation.yaw)
    # wrap to [-180, 180]
    while dyaw > 180:
        dyaw -= 360
    while dyaw < -180:
        dyaw += 360

    return np.array([dx, dy, dyaw], dtype=np.float32)


class VehicleNodeGraphBuilder:
    """
    从 env 的 comm_received[ego_id] (deque/list of msg) 构建“车辆节点图”：

    - 节点0: ego
    - 节点1..N-1: 过去 window_s 内给 ego 发送过消息的 sender（最多 max_nodes-1 个）
    - 每个 sender 节点的序列 token = 时间窗内该 sender 的所有 message（最多 Tmax 条）
    - token 特征包含:
        [feat_padded (feat_dim_max),
         feat_dim_ratio,
         rel_pose (dx,dy,dyaw),
         age_s,
         latency_s,
         distance_m,
         payload_kb,
         (可选) sender_speed, sender_yaw_rate]
    """

    def __init__(self, cfg: GraphBuildConfig):
        self.cfg = cfg

    @torch.no_grad()
    def build(
        self,
        *,
        ego_actor,                 # carla.Vehicle
        carla_world,               # carla.World
        ego_feat: np.ndarray,      # ego 当前特征 (D,)
        ego_feat_dim: int,         # eg. 1024
        msgs: Sequence[Any],       # env.comm_received[ego_id] 里的 msg
        t_step: int,
        dt: float,
        device: torch.device,
    ) -> Data:
        cfg = self.cfg
        t_now = float(t_step) * float(dt)

        ego_tf = ego_actor.get_transform()
        # ego token：用“伪消息”把 ego 自己当作一个节点（序列长度=1）
        ego_token = self._make_token_from_feat(
            feat=ego_feat,
            feat_dim=ego_feat_dim,
            rel_pose=np.zeros(3, np.float32),
            age_s=0.0,
            latency_s=0.0,
            distance_m=0.0,
            payload_bytes=0,
            sender_actor=ego_actor,
            ego_actor=ego_actor,
        )

        # 1) 选窗口内消息 + 按 sender 分组
        msgs_by_sender: Dict[int, List[Any]] = {}
        for m in msgs:
            # 用“到达时间”判定是否在窗口内
            t_deliver = float(m.deliver_step) * float(dt)
            # print(f"[Build] t_now: {t_now}, t_deliver: {t_deliver}, window_s: {cfg.window_s}, diff: {t_now - t_deliver}")
            if (t_now - t_deliver) <= cfg.window_s:
                sid = int(m.sender_id)
                msgs_by_sender.setdefault(sid, []).append(m)

        # 2) 选最多 max_nodes-1 个 sender（优先：最近到达 or 最近 created）
        sender_ids = list(msgs_by_sender.keys())
        # 打印每个sender id有多少条消息
        # for sid in sender_ids:
        #     print(f"[Build] Sender {sid} has {len(msgs_by_sender[sid])} messages")

        # 排序：按该 sender 最新 deliver_step 降序（最新先）
        sender_ids.sort(
            key=lambda sid: max(int(mm.deliver_step) for mm in msgs_by_sender[sid]),
            reverse=True,
        )
        sender_ids = sender_ids[: max(cfg.max_nodes - 1, 0)]

        node_ids_var: List[int] = [int(ego_actor.id)] + sender_ids
        N_var = len(node_ids_var)

        # 固定输出大小
        Nmax = int(cfg.max_nodes)
        Tmax = int(cfg.Tmax)

        # 先构造变长的 seq_tokens/seq_lens（你的原逻辑不变）
        seq_tokens: List[np.ndarray] = []
        seq_lens: List[int] = []

        # node 0: ego
        seq_tokens.append(np.stack([ego_token], axis=0))  # [1, F]
        seq_lens.append(1)

        for sid in sender_ids:
            ms = msgs_by_sender[sid]
            ms.sort(key=lambda mm: int(mm.created_step))
            if len(ms) > cfg.Tmax:
                ms = ms[-cfg.Tmax:]

            sender_actor = carla_world.get_actor(sid)
            tokens = []
            for m in ms:
                payload = m.payload
                feat = payload.get("feat", None)
                feat_dim = int(payload.get("feat_dim", cfg.feat_dim_max))
                if feat is None:
                    continue

                sender_tf = sender_actor.get_transform()
                rel_pose = _rel_pose_ego_frame(ego_tf, sender_tf)

                age_s = float(t_now - float(m.created_step) * float(dt))
                latency_s = float(m.latency_s)
                distance_m = float(m.distance_m)
                payload_bytes = int(m.payload_bytes) if hasattr(m, "payload_bytes") else 0

                tok = self._make_token_from_feat(
                    feat=feat,
                    feat_dim=feat_dim,
                    rel_pose=rel_pose,
                    age_s=age_s,
                    latency_s=latency_s,
                    distance_m=distance_m,
                    payload_bytes=payload_bytes,
                    sender_actor=sender_actor,
                    ego_actor=ego_actor,
                )
                tokens.append(tok)

            if len(tokens) == 0:
                tokens = [self._make_empty_token()]

            tokens_np = np.stack(tokens, axis=0)  # [Li, F]
            seq_tokens.append(tokens_np)
            seq_lens.append(tokens_np.shape[0])

        # token dim
        F = int(seq_tokens[0].shape[-1])

        # ===== 关键：固定 pad 到 [Nmax, Tmax, F] =====
        X = np.zeros((Nmax, Tmax, F), dtype=np.float32)
        lengths = np.zeros((Nmax,), dtype=np.int64)
        node_ids = -np.ones((Nmax,), dtype=np.int64)

        # 填充已有的节点（最多 Nmax）
        n_fill = min(N_var, Nmax)
        for i in range(n_fill):
            li = min(seq_tokens[i].shape[0], Tmax)
            X[i, :li] = seq_tokens[i][:li]
            lengths[i] = li
            node_ids[i] = node_ids_var[i]

        # pack_padded_sequence 要求 lengths >= 1，所以把空节点长度设为 1（但 token 全 0）
        # 同时你可以额外提供一个 node_mask 表示哪些节点有效
        node_mask = (node_ids != -1).astype(np.float32)  # [Nmax]
        lengths = np.maximum(lengths, 1)
        lengths = np.minimum(lengths, Tmax)

        # ===== 固定边：用 Nmax 构图，保证 edge_index 恒定形状 =====
        edge_index = self._build_edges(Nmax, star=cfg.star_graph)  # [2, E] 固定
        edge_attr = self._build_edge_attr_fixed(
            ego_actor=ego_actor,
            carla_world=carla_world,
            node_ids=node_ids.tolist(),   # 固定长度 list，含 -1
            edge_index=edge_index,
        )  # [E, 3] 固定

        data = {
            "x_seq": X.astype(np.float32),                        # [Nmax, Tmax, F]
            "lengths": lengths.astype(np.int64),                  # [Nmax]
            "edge_index": edge_index.cpu().numpy().astype(np.int64),  # [2, E]
            "edge_attr": edge_attr.cpu().numpy().astype(np.float32),  # [E, 3]
            "node_ids": node_ids.astype(np.int64),                # [Nmax]
            "node_mask": node_mask.astype(np.float32),            # [Nmax]  <-- 强烈推荐加
            "ego_index": np.int64(0),
        }
        return data

    def _build_edge_attr_fixed(self, ego_actor, carla_world, node_ids: List[int], edge_index: torch.Tensor) -> torch.Tensor:
        # 固定 Nmax 的 edge_attr: [dx, dy, dist] in ego frame
        if edge_index.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32)

        ego_tf = ego_actor.get_transform()

        # 预先为每个 node_id 准备 transform；padding(-1) 用 ego_tf 代替（dx=dy=0）
        tfs = {}
        for vid in node_ids:
            if vid == -1:
                tfs[vid] = ego_tf
            elif vid == int(ego_actor.id):
                tfs[vid] = ego_tf
            else:
                act = carla_world.get_actor(int(vid))
                if act is None:
                    tfs[vid] = ego_tf
                else:
                    tfs[vid] = act.get_transform()

        E = edge_index.shape[1]
        out = torch.zeros((E, 3), dtype=torch.float32)
        for e in range(E):
            u = int(edge_index[0, e].item())
            v = int(edge_index[1, e].item())
            vid_v = node_ids[v]

            rel = _rel_pose_ego_frame(ego_tf, tfs.get(vid_v, ego_tf))
            dx, dy = float(rel[0]), float(rel[1])
            dist = float(math.sqrt(dx * dx + dy * dy))
            out[e, 0] = dx
            out[e, 1] = dy
            out[e, 2] = dist
        return out

    def _make_empty_token(self) -> np.ndarray:
        # 与 _make_token_from_feat 的维度保持一致
        feat = np.zeros((self.cfg.feat_dim_max,), dtype=np.float32)
        # [feat, feat_dim_ratio, rel_pose(3), age, latency, dist, payload_kb, speed, yaw_rate]
        extra_dim = 1 + 3 + 4 + (2 if self.cfg.include_sender_state else 0)
        return np.concatenate([feat, np.zeros((extra_dim,), dtype=np.float32)], axis=0)

    def _make_token_from_feat(
        self,
        *,
        feat: np.ndarray,
        feat_dim: int,
        rel_pose: np.ndarray,
        age_s: float,
        latency_s: float,
        distance_m: float,
        payload_bytes: int,
        sender_actor,
        ego_actor,
    ) -> np.ndarray:
        cfg = self.cfg
        feat_pad = _pad_feat(feat, cfg.feat_dim_max)
        feat_dim_ratio = np.array([float(feat_dim) / float(cfg.feat_dim_max)], dtype=np.float32)

        payload_kb = np.array([float(payload_bytes) / 1024.0], dtype=np.float32)
        age = np.array([float(age_s)], dtype=np.float32)
        lat = np.array([float(latency_s)], dtype=np.float32)
        dist = np.array([float(distance_m)], dtype=np.float32)

        # image feature size + extra feature
        parts = [feat_pad, feat_dim_ratio, rel_pose.astype(np.float32), age, lat, dist, payload_kb]

        # if cfg.include_sender_state:
        #     # sender 速度（在 ego 坐标系下的 speed 也行，这里先用标量 speed）
        #     v = sender_actor.get_velocity()
        #     speed = float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))
        #     # yaw_rate 需要角速度传感器，CARLA actor 没直接给，这里用 0 占位（你可替换为 IMU/gyro）
        #     yaw_rate = 0.0
        #     parts.append(np.array([speed, yaw_rate], dtype=np.float32))
            
        if cfg.include_sender_state:
            v = sender_actor.get_velocity()
            # ego yaw
            ego_tf = ego_actor.get_transform()
            yaw = math.radians(float(ego_tf.rotation.yaw))
            c, s = math.cos(-yaw), math.sin(-yaw)
            # 世界速度 -> ego 坐标系
            vx_e = c * v.x - s * v.y
            vy_e = s * v.x + c * v.y

            rel_vel = np.array([vx_e, vy_e], dtype=np.float32)
            parts.append(rel_vel)

        return np.concatenate(parts, axis=0).astype(np.float32)

    def _build_edges(self, N: int, star: bool) -> torch.Tensor:
        if N <= 1:
            return torch.zeros((2, 0), dtype=torch.long)

        if star:
            # 0 <-> i
            src = []
            dst = []
            for i in range(1, N):
                src += [0, i]
                dst += [i, 0]
            return torch.tensor([src, dst], dtype=torch.long)
        else:
            # fully connected (no self-loops), bidirectional
            src = []
            dst = []
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    src.append(i)
                    dst.append(j)
            return torch.tensor([src, dst], dtype=torch.long)

    def _build_edge_attr(self, ego_actor, carla_world, node_ids: List[int], edge_index: torch.Tensor) -> torch.Tensor:
        # edge_attr: [dx, dy, dist] in ego frame
        if edge_index.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32)

        ego_tf = ego_actor.get_transform()

        # cache actors transforms
        tfs = {}
        for vid in node_ids:
            act = ego_actor if vid == int(ego_actor.id) else carla_world.get_actor(int(vid))
            tfs[vid] = act.get_transform()

        E = edge_index.shape[1]
        out = torch.zeros((E, 3), dtype=torch.float32)
        for e in range(E):
            u = int(edge_index[0, e].item())
            v = int(edge_index[1, e].item())
            vid_u = node_ids[u]
            vid_v = node_ids[v]

            rel = _rel_pose_ego_frame(ego_tf, tfs[vid_v])  # v 相对 ego
            dx, dy = float(rel[0]), float(rel[1])
            dist = float(math.sqrt(dx * dx + dy * dy))
            out[e, 0] = dx
            out[e, 1] = dy
            out[e, 2] = dist
        return out

