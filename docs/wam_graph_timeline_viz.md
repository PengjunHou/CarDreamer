# WAM 全局协同图 · 逐时间步可视化

观察 **全局协同图**(ego 视角的 `env._wam_graph`)随时间如何变化:在一个 policy 的生命周期内、
以及在多个 policy 之间。画的就是那张分层结构图——**车辆行**(ego + 协作者)、**观测行**
(每辆车每个 modality 一个 Object-list / BEV 节点)、**对象行**,以及 `veh_obs`(车→观测)、
`obs_obj`(观测→对象)、`veh_veh`(协作者→ego,弧线,带实测延迟 `L_M`)三类边。逐项说明
**设计怎么写 → 代码怎么实现 → 有哪些简化**。

整体沿用仓库「record(在线,需 CARLA)→ offline viz(离线)」的拆分:

```
HeteroData --hetero_graph_to_record--> 纯 JSON record(CARLA/torch-free)
          --layout_layered--> 节点坐标 --render_*--> matplotlib 帧
          --write_graph_frames_png / _gif / _html--> 每步 PNG / 动图 GIF / 滑块 HTML
```

核心模块:[graph_timeline_viz.py](car_dreamer/toolkit/wam/graph_timeline_viz.py)(单一 matplotlib
渲染核;只有抽取那一步碰 torch tensor,其余都吃纯 dict)。

---

## 0. 「全局图」是什么

= 请求车辆(ego)的协同感知图 `env._wam_graph`,由
[v2v_comm_mixin.py](car_dreamer/v2v_comm_mixin.py) `_build_wam_graph` 从**接收队列**重建:无可用消息→
ego-only 的 local 图;有消息→V2V 图,`veh_veh` 边用实测 `L_M`。和用户给的草图一一对应。

---

## 1. 抽取:HeteroData → 纯 JSON record（设计§5-§6 的节点/边都在）

### 代码
[graph_timeline_viz.py](car_dreamer/toolkit/wam/graph_timeline_viz.py) `hetero_graph_to_record(graph, *,
step, policy_label, policy_id=None, extra=None)`:从 [graph.py](car_dreamer/toolkit/wam/graph.py) 的
`build_wam_hetero_graph` 产物里读出:
- `vehicles[{idx,node_id,is_ego,slot}]`(`VEHICLE.node_id/is_ego/agent_slot`)
- `observations[{idx,vehicle_id,modality,payload_kb,latency_s,freshness,quality,sample_age_s}]`
  (`OBSERVATION.node_id/modality_id` + `x[5]` 标量;`MODALITY_TO_ID` 解出 objlist/bev)
- `objects[{idx,node_id,valid,notable,visible,invisible}]`(`OBJECT.*`;`node_mask` 判 valid,过滤占位)
- `edges{veh_obs, obs_obj(+det_conf), veh_veh(+latency_s)}`、`counts`、`is_v2v`
只用 tensor 的 `.tolist()`,产物是 JSON-able,offline 渲染无需 torch。

---

## 2. 布局 + 渲染（对照草图）

### 代码
- `layout_layered(record)`:三行——车辆 `y=2`(ego 居中 `x=0`,协作者按 slot 左右交替展开)、
  观测 `y=1`(归到各自车辆下方)、对象 `y=0`(均匀铺开)。
- `_draw_on_ax / render_graph_matplotlib`:ego=蓝圆(标 `ego`)、协作者=紫圆(标 **`V1/V2`**,小字 `id=`);
  观测=圆角矩形,**按归属×modality 上色**(ego objlist=绿、ego bev=橙、协作者=米黄,标签写
  Object-list/BEV,下方小字 `L=.. f=..`);对象=圆(标 **`O<id>`**),**填充编码 ego-notability:对 ego
  notable→红、不重要→灰**。`veh_obs/obs_obj` 直箭头、`veh_veh` 用 `FancyArrowPatch(connectionstyle=
  "arc3")` 弧线并标注 `L=<L_M>s`。标题写 `step / policy_label / V2V|local / 计数`。
- **拓扑图右边并排一张 BEV**(`_draw_bev_on_ax`):`render_graph_matplotlib` 每个 policy 渲成
  **左拓扑 + 右 BEV** 两栏。默认 `--bev-mode generated`，不读取 `data/birdeye_frames`，而是用录制时生成的
  单张 CARLA map background（道路/车道线/路肩等，来自 `MapRenderer`，颜色/通道约定与 `data/birdeye_frames`
  完全一致）。默认 `--bev-frame map`：把这张地图按 **episode 固定窗口**裁剪一次，**背景不随 ego 移动**，
  ego 框在固定窗口里移动；车辆/对象用与 `BirdeyeRenderer` 相同的 cv2 `fillPoly`+白描边画法叠加（ego
  橙色 ``Color.ORANGE_1``，与浅蓝车道中心线区分明显）。整张裁剪图按 episode 起始 ego yaw 做固定
  ``yaw+90°`` 旋转，前进方向朝上（与 ``data/birdeye_frames`` 一致），背景不随 ego 平移/旋转。
  固定窗口在离线渲染前扫描整个 episode 的 ego / cooperative candidates / graph objects，加 `--bev-margin-m`
  得到同一 episode 共用的裁剪框，所以不会随 timestep 抖动。`--bev-frame birdeye` 改为 ego-centric warp、
  `--bev-frame world/episode_start` 为 matplotlib 调试视图。
- 可选 `--bev-mode auto --birdeye-dir data/birdeye_frames` 或 `--bev-mode birdeye-dir` 时，按 **ego id + step**
  找 `vehicle_<egoid>/birdeye_<step>.png` (`BirdeyeHandler` dump 的那批,带道路/车道/route/ego蓝盒/他车绿盒)
  `imshow` 做背景,再把**该 policy 图里的车辆/对象投影叠加上去**(`_overlay_graph_nodes`):
  - 投影:ego-frame (x=forward, y=right) → birdeye 像素,`ppm=W/obs_range`,
    ego 在 `(W/2, H/2+(obs_range/2-ego_offset)*ppm)`,forward↑、right→+x(`BevOptions(obs_range=64,
    ego_offset=12)`,与 `birdeye_wpt` 标定一致;`--bev-obs-range/--bev-ego-offset` 可调)。
  - **只画图里的节点** = 该 policy 条件下的可见集(ego 可见 ∪ **协作(被选中)车辆**可见);
    某辆车不协作 → 它独有的 object 不在图里 → **不画**。`EGO`、`V1/V2`、`O<id>` 标签与拓扑图一一对应。
  - **标记法(在花花绿绿的 birdeye 上要一眼分清)**:
    - **ego = 亮青色五角星 `★` + 加粗 `EGO`**(birdeye 本身把 ego 画成蓝盒,星标叠在上面,最醒目)。
    - **协作车 = 紫色菱形 `◆` + `V1/V2`**(叠在它对应的他车绿盒上,指明哪辆绿盒是被选中的协作者)。
    - **vehicle 类 object = 朝向矩形框**(用 `class_id` 判类 + `length/width/heading` 画 `Polygon`,
      和 birdeye 的车盒对齐;**不再在车盒上叠圆圈**);pedestrian/bicycle 等非车 = 小三角 `▲`。
    - **轮廓颜色编码 notability**:对 ego notable→**红**,不重要→**蓝**(红蓝比原来的黑/白边好区分)。
    - **线型编码可见性**:实线=ego 看得到,虚线=只有协作者看得到(区分 ego 视角 vs 协作视角)。
    - **标签 = 带白色描边的彩色文字**(`patheffects.withStroke`),直接落在图形上(无像素偏移,不会和
      图形错位),在深色路面和亮色车盒上都清楚;白字看不清的问题解决。
  - **固定视野不抖动(关键修复)**:birdeye-dir 模式下 BEV 坐标轴**恒等于 birdeye 图像边界**
    (`set_xlim(0,w)/set_ylim(h,0)`);generated+birdeye 模式下同一 episode 复用同一个自动 `obs_range`。
  - `--bev-mode scatter` 才使用旧的带标签 ego-centric 散点 BEV；这是 debug fallback，ego 会固定在原点。
  - **可选:想让远处 object 也进图** → 录制时加 `--wide-bev` dump 一张 100m、ego 居中的 `birdeye_wam100`
    (`common.yaml` 定义,obs_range=100/ego_offset=50,**不是模型输入**),渲染时配 `--bev-obs-range 100
    --bev-ego-offset 50`。这是**可选的更大底图**,不是抖动的修复手段;默认 64m `birdeye_wpt` 已经稳定,
    只是视野外的 object 会被裁掉。`birdeye_wpt`(64m,模型 CNN 输入)始终不动。
- **整图固定尺寸**:`render_graph_matplotlib` 用**常量 figsize**(`TOPO_WIDTH+BEV_WIDTH × FIG_HEIGHT`),
  且 `_fig_to_png_bytes` **去掉 `bbox_inches="tight"`**(它会按内容裁剪导致每帧大小不一)——配合上面
  BEV 固定坐标轴,滑块逐帧切换时整张图大小恒定。
- **多 policy = 各自独立的图**:PNG/GIF 仍由 `_frame_png_bytes` 竖直拼接成一帧;但 **HTML 把每个 policy
  作为独立 `<img>` 元素**(`_policy_png_list` + `#panels` 容器,每行一个),不再压成一张大图。

---

## 3. 输出（三种;同一渲染核）

[graph_timeline_viz.py](car_dreamer/toolkit/wam/graph_timeline_viz.py)(三种输出共用 `_frame_png_bytes`
把一帧内各 policy 竖直拼成一张图):
- `write_graph_frames_png(records, dir)` 每步一张 PNG。
- `write_graph_timeline_gif(records, gif, fps)` 帧→PIL 合成动图 GIF。
- `write_graph_timeline_html(records, html, fps)` **自包含交互 HTML**:每帧渲成内嵌 base64 PNG +
  `<input type=range>` **时间步滑块** + ▶/⏸ 播放 + 每步 caption。
`group_records_to_frames` 把扁平 record 列表按 **`(episode, step)`** 分帧(episode 取 `extra.episode`,
默认 0),所以**不同 episode 的同号步不会串到同一帧**;每帧 `panels{label:record}`。
`append_record_jsonl/load_records_jsonl` 做 JSONL 持久化。

---

## 4. 数据来源

- **实际时间线(需 CARLA)** [scripts/record_wam_graph_timeline.py](scripts/record_wam_graph_timeline.py):
  仿 run_env 逐步跑 env,每步把 `sim._wam_graph` + 当前激活 policy(`sim._comm_process.policy`:local /
  coop[ids])抽成 record 追加进 JSONL。**只录单个 episode**:episode 结束(terminated/truncated)即停止,
  不再 reset 续录(避免不同 episode 步号撞车串帧)。policy 随 Td 切换体现为一个 episode 内的变化。
  record 的 `extra` 同时保存 `ego_world`、`candidate_world`、`graph_object_world`，用于 generated BEV
  按 `BirdeyeRenderer` 同款坐标/warp 叠加当前 policy graph；默认还会写一张 `<jsonl_stem>_map.png`
  全局地图背景并记录 world→pixel 标定。旧 JSONL 没有这些字段时会退回简化 scatter。

离线渲染 [scripts/visualize_wam_graph_timeline.py](scripts/visualize_wam_graph_timeline.py)
`--jsonl IN --html/--gif/--png-dir [--policies a,b] [--fps] [--bev-mode generated|auto|birdeye-dir|scatter]`
(CARLA-free)。

---

## 5. 有哪些简化

1. **单一 matplotlib 渲染核**喂 PNG+GIF+HTML;「交互」= 滑块翻 matplotlib 帧,延迟/新鲜度/policy
   直接画进帧,无需 hover。plotly 原生节点-边图(每帧重建箭头注记)对「逐步变结构」太繁琐,暂不做。
2. **车辆友好短标签**用枚举 `V1/V2`(真实 actor id 以小字 `id=` 标注),不强绑 node_id;对象标 `O<id>`。
3. 对象超过 `max_object_nodes` 会被图构建侧截断(可视化只画图里实际有的)。

---

## 6. 后续计划

- HTML 增加 per-policy 复选(多 panel 时动态增删 panel)与按 `policy_id` 着色的时间轴缩略带。
- 帧上叠加 BEV 缩略图(对 bev modality 的观测节点),把 `docs/wam_bev.md` 的栅格和结构图并看。
- 可选 plotly 原生交互(hover 看完整 obs 标量/edge attr),用于精细排查。

---

## 7. 测试

[tests/test_wam_graph_timeline_viz.py](tests/test_wam_graph_timeline_viz.py)(offline、unittest、
CARLA-free):合成 `HeteroData`(复用 test_wam_graph 风格)覆盖——record 抽取(节点/边计数、ego 标志、
modality 标签、`veh_veh` 延迟、`obs_obj` 置信度、JSONL round-trip)、分层布局(行序 + ego 居中)、
matplotlib 渲染返回非空 Figure、**V/O 标签 + notable=红/非notable=灰 填充**(拓扑)、**record 含 ego-frame 位置
+ class_id/朝向/尺寸 且拓扑+BEV 两栏**、**有 birdeye 图时 BEV 用 imshow(按 ego id+step 解析路径)且叠加
`EGO` 星标/`O<id>`**、**vehicle 类 object 叠成 `Polygon` 框(数量==车类对象数)+ BEV 坐标轴恒等于图像边界
(固定不抖)**、**HTML 用独立 per-policy `<img>`**、单 panel / 多 panel 分帧、
**`(episode, step)` 分帧不串帧**、PNG/GIF/HTML 写出非空。

运行:`conda run -n cardreamer_gnn python -m unittest tests.test_wam_graph_timeline_viz`(14 项);
全量离线回归 `python -m unittest discover -s tests`(135 项,全过)。

离线 smoke(无需 CARLA;`--birdeye-dir` 指向 `data/birdeye_frames` 即用真实 birdeye 作 BEV):
```
python scripts/visualize_wam_graph_timeline.py --jsonl <timeline.jsonl> \
    --birdeye-dir data/birdeye_frames --html out.html --gif out.gif --png-dir frames
```
