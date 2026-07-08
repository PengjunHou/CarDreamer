# WAM per-vehicle visibility-aware BEV（对照 `docs/WAM/WAM Design.pdf` §5.3 / §14 / §15.4）

本文记录把 BEV 从「零占位」变成**真实、按车视角、visibility-aware** 的语义地图：某辆车看不见的物体
**不会**出现在它的 BEV 里。逐项说明 **设计怎么写 → 代码怎么实现 → 有哪些简化**。

> 之前 BEV 全是零占位：`_BevEncoder` 跑 `torch.zeros(...)`、Stage-2 `bev_history/bev_future` 全零，
> 导致 diffusion UWM 的 future-observation 那半是退化目标（去噪向 0）。本次实现「full BEV half」。

---

## 0. 核心语义：按车视角 + visibility-aware

一辆车的 BEV `B^sem [C,H,W]` 是 **ego 中心、heading-up** 的语义占据栅格，**只包含该车能看到的物体**。
可见性**复用**已有结果（不重算）：`ObjectState.visible_to_ego` / `visible_to_collaborators`（由
`is_fov_visible` 的 FOV+遮挡判定得到）。栅格化时调用方只传入「可见子集」，**不可见物体天然不会被画进去**。

通道（固定顺序，共 7）：`vehicle, pedestrian, bicycle, other`（可见物体按类占据）+ `ego`（本车足迹）
+ `route`（本车规划路线折线）+ `drivable`（可行驶区，best-effort）。

---

## 1. 组件

### 1.1 栅格化器（[bev.py](car_dreamer/toolkit/wam/bev.py) `rasterize_bev`）
- `BevSpec(size=64, range_m=50, route_width_m, ego_*)`；`pixels_per_meter = size/(2·range_m)`。
- `rasterize_bev(ego_pose, objects, *, route_xy, drivable_polygons=None, spec) -> np.uint8 [C,H,W]`：
  世界多边形 → ego 帧（`_world_to_ego`，与 `graph._EgoFrame` 同约定）→ 像素（ego 居中、前向朝上）→
  **凸多边形填充**（对包围盒内像素做半平面一致符号判定，无 cv2 依赖）。route 用加粗折线段。
- **亚像素鲁棒**：当一个足迹小于 1 像素（远处小目标 / 粗分辨率下的 ego）时，回退到「最近质心格」标 1，
  保证小目标不至于整体消失。
- 纯 numpy、CARLA-free、可单测（可见性排除、ego 居中、heading 旋转、通道路由、route）。

### 1.2 BEV 自编码器：E_bev + D_bev（§5.3 / §14）
- **E_bev**：图嵌入里的 `_BevEncoder`（CNN `[C,H,W]→[d]`，`AdaptiveAvgPool` 适配任意 size），**共享**给
  「图观测 BEV 特征」与「diffusion 的 BEV latent 空间」。`WAMHeteroGraphNet.encode_bev(raster)` 暴露它。
- **D_bev**（[bev.py](car_dreamer/toolkit/wam/bev.py) `WAMBevDecoder`，§14）：`Linear(d→8×8 spatial) →
  ConvTranspose 上采样到 size → 1×1 conv → C 通道 logits`。挂在 `WAMUnifiedWorldModel.bev_decoder`。
- **重建 loss**（§15.4）：`bev_reconstruction_loss`（逐通道 BCE-with-logits 占据）+ `bev_iou`（评测）。

### 1.3 接入图的 BEV 观测节点（§5.3）
- `ObservationNodeInput.bev_raster`（[graph.py](car_dreamer/toolkit/wam/graph.py)）携带 `B^sem`；
  `build_wam_hetero_graph` 把它堆进 `data[OBSERVATION].bev_raster [n_obs,C,H,W]`（仅当存在 bev 节点）。
- [graph_model.py](car_dreamer/toolkit/wam/graph_model.py)：`WAMHeteroGraphEmbedding.forward` 用
  `data[OBSERVATION].bev_raster[bev_mask]` 过 `E_bev`（真实 `z^bev`），无 raster 时才回退零（仅旧 fixture）。
- 环境侧（[v2v_comm_mixin.py](car_dreamer/v2v_comm_mixin.py)）：`modality=='bev'` 分支里，对该协作车
  rasterize 它能看到的物体（`vid in visible_to_collaborators`）+ 自身足迹 → `bev_raster`。

### 1.4 Stage-2 真实 BEV 生成目标（§7-§8 + §15.4）
- 录制器（[flow_recorder.py](car_dreamer/toolkit/wam/flow_recorder.py)）每步 `observe_bev` 栅格化
  **request(ego) 车**的 visibility-aware `B^sem`；flush 时取 `bev_history = B^sem_{t-K..t}`、
  `bev_future = B^sem_{t+1..t+H}`（uint8 栅格栈）+ `bev_step_mask`。
- 训练（[stage2.py](car_dreamer/toolkit/wam/stage2.py) `WAMStage2Trainer.loss_on_batch`）：
  - `encode_bev(history) → 条件 BEV-history latent`（喂 condition tokens）；
  - `bev_1 = encode_bev(future).detach()` → diffusion 的 BEV 目标（**stop-grad 防塌缩**）；
  - 重建项 `w_bev_recon · bev_reconstruction_loss(decode_bev(encode_bev(future)), future)` 训练 E_bev+D_bev；
  - diffusion 仍在 latent 空间跑（`flow_matching.py` 基本不变）。

---

## 2. 数据流

```
 live env (CARLA)                                offline (no CARLA)
 每步: ego_pose + 可见物体 + route ──观测──▶ rasterize_bev → B^sem (uint8 [C,H,W])
       sim._wam_graph (含 bev 观测栅格)          WAMFlowDataRecorder.observe_bev/observe_policy/register
                                                 flush → sample{ ..., bev_history[K+1,C,H,W], bev_future[H,C,H,W] }
                                                 WAMStage2Trainer:
                                                   encode_bev(history)→cond; bev_1=encode_bev(future).detach()
                                                   L = w_π·policy + w_z·bev_diff + w_recon·CE(D_bev(E_bev), raster)
```

---

## 3. 简化 / 未实现

| 项 | 现状 | 影响 |
| --- | --- | --- |
| `drivable` 通道 | 通道已留，但**暂为空**（除非调用方传入地图多边形） | 可行驶区上下文待接地图 |
| AE 训练 | **joint + detach**（diffusion 目标 detach，重建训 E_bev/D_bev） | 比「预训练+冻结」简单；latent 稳定性靠 recon 锚定 |
| 可见性 | GT + FOV/遮挡几何（非真实 detector） | 标签为 oracle |
| 栅格 | uint8 占据，`bev_size` 默认 64 | `.pt` 较省；高分辨率更细但更大 |
| 通道填充 | 二值占据；亚像素回退质心格 | 远处小目标≥1 格，不消失 |

**后续**：地图 drivable/lane 栅格化；预训练+冻结 BEV-AE；真实 detector；解码可视化评测（IoU/mIoU）。

---

## 4. 如何验证

```bash
conda run -n cardreamer_gnn python -m unittest tests.test_wam_bev tests.test_wam_graph tests.test_wam_stage2 -v
conda run -n cardreamer_gnn python -m unittest discover -s tests -p "test_*.py"
```
`test_wam_bev` 覆盖：栅格通道布局/形状、**可见性排除**（只画传入的可见物体）、ego 居中、**heading-up 旋转**、
类路由、route 通道；解码器形状、重建 loss/IoU、**自编码过拟合下降+IoU 上升**；**图 BEV 节点用真实栅格 →
`z^bev` 随栅格变化（≠ 零占位）**。`test_wam_stage2` 覆盖 BEV 栅格样本 schema/collate、trainer 编码栅格 +
重建项、过拟合下降。

```bash
# (需 CARLA) 录制带真实 BEV 的样本再训练
python scripts/record_wam_flow_data.py --task carla_group_right_turn_auto --carla-port 2000 --steps 400 --out-dir data/wam_flow
python scripts/train_wam_stage2.py --data-dir data/wam_flow --task carla_group_right_turn_auto --steps 2000
```

## 5. 关键文件 / 配置
- 代码：`bev.py`（rasterizer + D_bev + recon/iou）、`graph.py`/`graph_model.py`（bev_raster + 真实 E_bev）、
  `v2v_comm_mixin.py`（按车栅格化）、`flow_recorder.py`/`stage2.py`（栅格样本 + encode/decode + recon）。
- 配置：`env.wam.graph.{bev_channels=7, bev_size=64, bev_range_m=50}`、`env.wam.stage2.w_bev_recon`。
- 导出：`BevSpec` / `rasterize_bev` / `WAMBevDecoder` / `bev_reconstruction_loss` / `bev_iou` /
  `BEV_NUM_CHANNELS` / `BEV_CHANNEL_NAMES`（见 `car_dreamer/toolkit/wam/__init__.py`）。
