# WAM Notable Object Debug JSONL Schema

This document explains the JSONL file produced by:

```bash
python scripts/record_wam_notable_debug.py \
  --task carla_group_right_turn_auto \
  --carla-port 2000 \
  --steps 300 \
  --render-every 5
```

The default output file is:

```text
outputs/wam_notable_debug/notable_motion.jsonl
```

The file is JSON Lines format. Each line is one complete JSON object for one
simulation timestep. The recorder uses CARLA ground truth for actor state and
future motion, and uses the current WAM V1 rule-based runtime for notable
object selection, visibility, uncertainty, and placeholder collaboration policy.

## Top-Level Record

Each JSONL line has this structure:

```json
{
  "step": 39,
  "time_s": 3.9,
  "ego": {},
  "wam": {},
  "notable_objects": []
}
```

| Field | Type | Meaning | Source / Rule |
| --- | --- | --- | --- |
| `step` | integer | Environment timestep index. | Read from the environment `_time_step`. |
| `time_s` | float | Simulation time in seconds for this record. | Computed as `step * fixed_delta_seconds`. The default `fixed_delta_seconds` is `0.1`, so step 39 is 3.9s. |
| `ego` | object | Ego vehicle state at this timestep. | CARLA ground-truth actor state. |
| `wam` | object | WAM runtime summary for this timestep. | Current WAM V1 rule-based runtime. |
| `notable_objects` | list | Objects selected as notable for the ego at this timestep. | Current WAM V1 notable object rule. |

## Ego Object

Example:

```json
{
  "id": 1101,
  "type": "vehicle",
  "actor_type": "vehicle.audi.etron",
  "position": [-17.86, -135.12, -0.01],
  "velocity": [5.19, 0.07, 0.0],
  "yaw": 0.88,
  "bbox": [[-20.28, -136.17], [-15.43, -136.10], [-15.46, -134.06], [-20.31, -134.14]]
}
```

| Field | Type | Meaning | Source / Rule |
| --- | --- | --- | --- |
| `id` | integer | CARLA actor id of the ego vehicle. | `ego.id`. |
| `type` | string | Coarse object class. | Derived from `actor_type`; vehicle actors become `"vehicle"`, walker actors become `"pedestrian"`. Ego is normally `"vehicle"`. |
| `actor_type` | string | Full CARLA blueprint type id. | `actor.type_id`, for example `vehicle.audi.etron`. |
| `position` | `[x, y, z]` | Current ego position in CARLA world coordinates, meters. | `actor.get_transform().location`. |
| `velocity` | `[vx, vy, vz]` | Current ego velocity in CARLA world coordinates, meters per second. | `actor.get_velocity()`. |
| `yaw` | float | Ego yaw angle in degrees. | `actor.get_transform().rotation.yaw`. |
| `bbox` | list of 4 `[x, y]` points | Ego BEV footprint polygon in CARLA world coordinates. | Computed from CARLA `actor.bounding_box`, actor transform, and yaw. Only the ground-plane footprint is saved, so it has 4 points. |

The `bbox` is not an image pixel coordinate. It is a world-coordinate polygon in
meters and is used by the BEV debug visualization.

## WAM Runtime Summary

Example:

```json
{
  "coop_triggered": false,
  "uncertainty_max": 0.2,
  "policy_selected_vehicle_ids": [],
  "policy_modality_by_vehicle": {},
  "notable_object_ids": [1110],
  "visible_notable_object_ids": [1110],
  "invisible_notable_object_ids": []
}
```

| Field | Type | Meaning | Source / Rule |
| --- | --- | --- | --- |
| `coop_triggered` | boolean | Whether ego triggered a cooperative perception request at this timestep. | `true` if at least one notable object's uncertainty is greater than `env.wam.uncertainty_threshold`. Default threshold is `1.0`. |
| `uncertainty_max` | float | Maximum uncertainty among all notable objects. | `max(notable_object.uncertainty_score)`, or `0.0` if no notable object exists. |
| `policy_selected_vehicle_ids` | list of integers | Collaborator vehicles selected by the current base-station policy. | Placeholder policy. If `coop_triggered=true`, select all current `coop_participant_ids`; otherwise select none. |
| `policy_modality_by_vehicle` | object | Selected payload modality per collaborator id. | Placeholder policy assigns `env.wam.default_modality` to each selected collaborator. Default is `"objlist"`. |
| `notable_object_ids` | list of integers | Actor ids of all notable objects at this timestep. | Current output of WAM notable object selection. |
| `visible_notable_object_ids` | list of integers | Notable objects visible to ego. | Subset of notable objects with `visible_to_ego=true`. |
| `invisible_notable_object_ids` | list of integers | Notable objects not visible to ego but visible to at least one collaborator. | Subset of notable objects with `visible_to_ego=false` and non-empty `visible_to_collaborators`. |

## Notable Object

Each item in `notable_objects` has this structure:

```json
{
  "id": 1110,
  "type": "vehicle",
  "actor_type": "vehicle.audi.tt",
  "position": [-13.64, -138.56, -0.01],
  "velocity": [-7.95, -0.08, 0.0],
  "yaw": 179.83,
  "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "visible_to_ego": true,
  "invisible_to_ego": false,
  "occluding": false,
  "route_distance": 3.49,
  "visible_to_collaborators": [1102],
  "uncertainty_score": 0.2,
  "predicted": {},
  "ground_truth": {}
}
```

| Field | Type | Meaning | Source / Rule |
| --- | --- | --- | --- |
| `id` | integer | CARLA actor id. | `actor.id`. |
| `type` | string | Coarse object class. | `"vehicle"` for CARLA vehicle actors; `"pedestrian"` for walker actors. |
| `actor_type` | string | Full CARLA blueprint type id. | `actor.type_id`. |
| `position` | `[x, y, z]` | Current object position in CARLA world coordinates, meters. | `actor.get_transform().location`. |
| `velocity` | `[vx, vy, vz]` | Current object velocity in CARLA world coordinates, meters per second. | `actor.get_velocity()`. Current WAM prediction uses `vx` and `vy`. |
| `yaw` | float | Object yaw angle in degrees. | `actor.get_transform().rotation.yaw`. |
| `bbox` | list of 4 `[x, y]` points | Object BEV footprint polygon in CARLA world coordinates. | Current WAM actor polygon helper. For vehicles and walkers this is a 4-point BEV footprint. |
| `visible_to_ego` | boolean | Whether ego can locally observe this object. | WAM FOV + line-of-sight geometry rule. |
| `invisible_to_ego` | boolean | Whether this object is not visible to ego but is visible to at least one collaborator. | `visible_to_ego == false and len(visible_to_collaborators) > 0`. |
| `occluding` | boolean | Whether this object is treated as an occluder in the notable record. | Currently always `false` in WAM V1; occluder tagging is reserved for later versions. |
| `route_distance` | float | Minimum BEV distance from object center to ego's reference route, meters. | Computed as point-to-polyline distance from `(object.x, object.y)` to the selected ego route waypoints. |
| `visible_to_collaborators` | list of integers | Collaborator actor ids that can observe this object. | For each current collaborator, apply the same FOV + line-of-sight visibility rule from that collaborator to this object. |
| `uncertainty_score` | float | Rule-based motion uncertainty for this object. | `env.wam.visible_uncertainty` if visible to ego, otherwise `env.wam.invisible_uncertainty`. Defaults are `0.2` and `2.0`. |
| `predicted` | object | Rule-based predicted future motion. | Constant-velocity prediction from the current state. |
| `ground_truth` | object | True future motion from CARLA. | Filled by stepping the simulator forward and reading future actor positions. |

## Notable Object Selection Rule

WAM V1 currently uses a simple route-distance rule:

1. Build candidate objects from CARLA vehicles and pedestrians, excluding ego.
2. Build a short ego reference route from the current planned route waypoints.
3. For each candidate object, compute:

```text
route_distance = min distance from object center (x, y) to the reference route polyline
```

4. Keep objects with:

```text
route_distance < env.wam.notable_distance_m
```

The default `notable_distance_m` is `10.0`.

5. Sort kept objects by:

```text
(route_distance, actor_id)
```

6. Keep at most:

```text
env.wam.max_notable_objects
```

The default is `3`.

This means the current debug file records the nearest route-relevant objects,
not all objects in the scene.

## Visibility Rule

`visible_to_ego` and `visible_to_collaborators` are not raw CARLA fields. They
are computed by WAM using geometry.

For an observer and a target object, the target is visible if:

1. The target polygon is inside the observer's configured field of view.
2. The target is within the observer's configured sight range.
3. The line of sight is not blocked by other actor polygons according to the
current line-of-sight helper.

The ego visibility uses:

```text
env.wam.local_sight_fov
env.wam.local_sight_range_m
```

Collaborator visibility uses:

```text
env.wam.collaborator_sight_fov
env.wam.collaborator_sight_range_m
```

If these fields are not explicitly set, the runtime falls back to the task's
existing observation/FOV settings and default sight ranges.

## Motion Prediction

The recorder writes 3 seconds of predicted future motion by default, sampled at
6 waypoints:

```text
dt = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
```

Each predicted waypoint uses a constant-velocity model:

```text
pred_x(t) = current_x + current_vx * t
pred_y(t) = current_y + current_vy * t
pred_z(t) = current_z
```

Example:

```json
"predicted": {
  "future_waypoints": [
    {"dt": 0.5, "position": [x, y, z], "uncertainty": 0.2},
    {"dt": 1.0, "position": [x, y, z], "uncertainty": 0.2}
  ]
}
```

| Field | Type | Meaning | Source / Rule |
| --- | --- | --- | --- |
| `dt` | float | Future time offset from the current record, seconds. | Determined by `--future-horizon-s / --future-waypoints`. Defaults to `3.0 / 6 = 0.5s` spacing. |
| `position` | `[x, y, z]` | Predicted future object position in CARLA world coordinates. | Constant-velocity rollout from current CARLA ground-truth state. |
| `uncertainty` | float | Same uncertainty assigned to this object's prediction. | Equals `uncertainty_score`. |

## Ground-Truth Future Motion

The recorder cannot know future motion at the current step directly, so it keeps
stepping the simulator and fills each JSONL row once enough future actor history
is available.

With the default setup:

```text
fixed_delta_seconds = 0.1
future_horizon_s = 3.0
future_waypoints = 6
```

the ground-truth samples are read at future offsets:

```text
0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s
```

which correspond to future step offsets:

```text
5, 10, 15, 20, 25, 30
```

Example:

```json
"ground_truth": {
  "future_waypoints": [
    {"dt": 0.5, "position": [x, y, z], "available": true},
    {"dt": 1.0, "position": [x, y, z], "available": true}
  ]
}
```

| Field | Type | Meaning | Source / Rule |
| --- | --- | --- | --- |
| `dt` | float | Future time offset from the current record, seconds. | Same offsets used by `predicted.future_waypoints`. |
| `position` | `[x, y, z]` or `null` | True future object position in CARLA world coordinates. | Future `actor.get_transform().location` at the matching timestep. |
| `available` | boolean | Whether this future actor position exists. | `false` if the actor disappeared, was destroyed, or the episode ended before this future point. |

When computing prediction error, only compare waypoints with:

```text
ground_truth.future_waypoints[i].available == true
```

## Default Config Values Used by WAM V1

The most relevant defaults are:

```yaml
env:
  world:
    fixed_delta_seconds: 0.1
  wam:
    enabled: true
    notable_distance_m: 10.0
    max_notable_objects: 3
    reference_waypoint_count: 6
    prediction_horizon_steps: 6
    uncertainty_threshold: 1.0
    visible_uncertainty: 0.2
    invisible_uncertainty: 2.0
    default_modality: objlist
    base_station_policy: placeholder_all
```

The debug recorder's 3-second JSON prediction horizon is controlled by script
arguments:

```text
--future-horizon-s 3.0
--future-waypoints 6
```

These debug-recording arguments are separate from the runtime
`env.wam.prediction_horizon_steps`, which is used by the WAM runtime's internal
step-level policy state.

## Field Origin Summary

| Field Group | Ground Truth From CARLA? | Rule-Based? |
| --- | --- | --- |
| `id`, `actor_type`, `position`, `velocity`, `yaw` | Yes | No |
| `bbox` | Uses CARLA bbox and transform | Converted to 4-point BEV footprint |
| `type` | Derived from CARLA `actor_type` | Yes |
| `notable_object_ids` and `notable_objects` membership | No | Route-distance notable selection |
| `visible_to_ego`, `visible_to_collaborators` | No | FOV + line-of-sight geometry |
| `route_distance` | Uses CARLA current object position | Point-to-route distance rule |
| `uncertainty_score` | No | Visible objects get low uncertainty; invisible objects get high uncertainty |
| `predicted.future_waypoints` | Uses current CARLA state as input | Constant-velocity rollout |
| `ground_truth.future_waypoints` | Yes | Future samples are aligned to debug `dt` offsets |
| `coop_triggered`, `policy_selected_vehicle_ids` | No | Placeholder WAM V1 cooperative perception policy |
