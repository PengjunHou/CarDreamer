# Emulation System State, Input, and Output

## 1. System Overview

This document describes the current engineering implementation of the emulation system in CarDreamer. It answers three questions:

1. What is the full emulation state?
2. How is each part of that state constructed in runtime logging, old-log adaptation, and dataset building?
3. What are the predictor model inputs, outputs, supervision targets, and losses?

This document reflects the current codebase, where:

- the shared state is a `21`-dimensional compact summary
- the policy action is per-vehicle `u_t = {alpha, nu, bandwidth, beta}`
- `beta` is encoded as a 5-way payload one-hot vector
- the predictor is the current `GraphGRUEmulationModel`

The main sources of truth are:

- `car_dreamer/toolkit/emulation/schema.py`
- `car_dreamer/toolkit/emulation/features.py`
- `car_dreamer/toolkit/emulation/dataset.py`
- `car_dreamer/toolkit/emulation/model.py`
- `car_dreamer/toolkit/vlm/right_turn_auto_predictor_logging.py`
- `car_dreamer/toolkit/emulation/adapter_vlm.py`

---

## 2. Canonical Schema

The emulation system is organized around three nested records.

### 2.1 `CanonicalEpisodeRecord`

An episode contains:

- `scene_id`
- `episode_id`
- `scene_type`
- `dt`
- `steps: List[CanonicalStepRecord]`
- `metadata`
- `policy_id`

`policy_id` is the episode-level policy label. It may be a fixed policy such as `P3`, or `"mixed"` if multiple policies are used across steps.

### 2.2 `CanonicalStepRecord`

A step contains:

- scene identifiers and `step`
- `ego_state: EgoState`
- `candidate_vehicles: List[CandidateVehicleState]`
- `queries: List[QueryRecord]`
- `ego_sc: Dict[query_id, float]`
- `communication_stats`
- `metadata`
- `policy_id`

`ego_sc` is the ego semantic confidence per query at this step.

### 2.3 `CandidateVehicleState`

Each candidate/member vehicle contains four groups of information:

1. **Raw kinematic state**
   - `delta_pos = (dx, dy)`
   - `delta_vel = (dvx, dvy)`
   - `delta_yaw`

2. **Shared state**
   - `shared_summary_raw: List[float]` with length `8`
   - `shared_summary_semantic: List[float]` with length `8`
   - `intent_summary: List[float]` with length `4`

3. **Derived collaboration state**
   - `complementarity`
   - `accessibility`
   - `query_task_relevance: Dict[query_id, float]`
   - `sender_collab: Dict[query_id, float]`
   - `sender_gain: Dict[query_id, float]`

4. **Action and communication context**
   - `alpha`
   - `nu`
   - `bandwidth`
   - `payload_type`
   - `payload_encoder_id`
   - `communication_stats`
   - `component_valid_mask`

---

## 3. Full State Definition

At the engineering level, the emulation state is split across four layers.

### 3.1 Canonical logical state

This is the human-readable step record:

- ego state
- query list
- candidate vehicles
- per-query ego confidence
- step-level communication stats

This is the state that gets serialized to canonical JSON.

### 3.2 Per-vehicle shared state

The current shared state is a **compact summary**, not an observation latent vector.

It is always packed in this fixed order:

`shared_summary_raw(8) + shared_summary_semantic(8) + intent_summary(4)`

Total:

- `shared_state_dim = 20`

### 3.3 Per-vehicle action state

The per-vehicle action `u_t` is:

- `alpha`
- `nu`
- `bandwidth`
- `beta_one_hot[5]`

The one-hot payload order is defined in `car_dreamer/toolkit/communication/payloads.py`:

1. `object_list`
2. `occupancy`
3. `images`
4. `latent`
5. `tokens`

Total:

- `action_dim = 8`

### 3.4 Exogenous context

The predictor uses two exogenous context levels.

**Vehicle-level exogenous features**:

1. `window_message_count`
2. `selected_message_count`
3. `latest_latency_s`
4. `latest_payload_bytes`
5. `current_distance_m`
6. `shared_source_received_feat`
7. `shared_source_raw`

So:

- `vehicle_exogenous_dim = 7`

**Step-level exogenous features**:

1. `num_candidate_vehicles`
2. `avg_latency_s`

So:

- `step_exogenous_dim = 2`

Note that runtime `communication_stats` may contain more keys such as `env_step` or `num_questions`, but the packed model input currently uses only the keys above.

---

## 4. How Per-Step and Per-Vehicle State Is Constructed

There are two main construction paths.

## 4.1 Runtime path

The main runtime path is:

- `right_turn_auto_scoring.py`
- `build_runtime_emulation_step(...)` in `right_turn_auto_predictor_logging.py`

For each candidate vehicle, the runtime code constructs:

### Raw state

- `delta_pos`: sender pose minus ego pose
- `delta_vel`: sender velocity minus ego velocity
- `delta_yaw`: sender yaw minus ego yaw, wrapped to `[-pi, pi)`

### Observable geometry

- `sender_region` is built from `delta_pos` and `delta_yaw`
- `ego_region` is built from a fixed ego-centered observable box

### Derived state

- `current_distance_m = sqrt(dx^2 + dy^2)`
- `latency_s` comes from the latest window message
- `complementarity = compute_complementarity(sender_region, ego_region)`
- `accessibility = compute_accessibility(current_distance_m, latency_s)`
- `query_task_relevance[q] = compute_task_relevance(sender_region, ego_region, required_region_q)`
- `sender_collab[q] = complementarity * query_task_relevance[q] * accessibility`
- `sender_gain[q]` is extracted from question-result attribution

### 4.1.1 `shared_summary_raw` in runtime

Runtime `shared_summary_raw` has length `8` and is built by `_build_shared_summary_raw(...)`.

Its entries are:

1. `log1p(len(window_messages))`
2. `log1p(len(selected_infos))`
3. mean `received_age_s` over selected infos
4. latest message latency
5. latest payload size in KB
6. mean `feat_dim / feature_size`
7. fraction of selected infos with non-empty `scene_description`
8. fraction of selected infos with non-empty `text`

This block summarizes message freshness, payload size, feature coverage, and whether the shared messages actually contain text/scene information.

### 4.1.2 `shared_summary_semantic` in runtime

Runtime `shared_summary_semantic` has length `8` and is built by `_build_shared_summary_semantic(...)`.

For each query, the code collects per-sensor scores for the current non-ego sender and averages the following:

1. `positive_score`
2. `negative_score`
3. `unknown_score`
4. `confidence`
5. `belief`
6. `evidence`
7. `answerability_score`
8. `visibility_score`

Then it averages these 8 values across all queries with available sender evidence.

`query_task_relevance` now measures the fraction of sampled `sender_region` points that fall inside the query's `required_region`. It does not subtract ego-visible overlap; ego-relative novelty is carried separately by `complementarity`.

### 4.1.3 `intent_summary` in runtime

Runtime `intent_summary` has length `4` and is built by `_infer_intent_summary(...)`.

Its entries are:

1. `lane_follow`
2. `turn_left`
3. `turn_right`
4. `stationary`

The logic is heuristic:

- left turn if `scene_type == left_turn` or `delta_yaw > 0.2`
- right turn if `scene_type == right_turn` or `delta_yaw < -0.2`
- stationary if relative speed `< 0.15`

### 4.1.4 Runtime action and communication fields

Runtime also writes:

- `alpha`
- `nu`
- `bandwidth`
- `payload_type`
- `payload_encoder_id`

These come from the active policy and payload selection logic before canonical serialization.

## 4.2 Old VLM-log adapter path

The compatibility path is:

- `adapt_vlm_records_to_canonical_episode(...)` in `adapter_vlm.py`

This path reconstructs canonical episodes from old `vlm_records*.json` logs.

It still produces the same compact-summary shape:

- `shared_summary_raw(8)`
- `shared_summary_semantic(8)`
- `intent_summary(4)`

But the **semantic meaning is slightly different** from the runtime path.

### 4.2.1 Adapter `shared_summary_raw`

The adapter builds an 8D `raw_summary` from per-query observations:

1. mean `visibility_score`
2. mean `answerability_score`
3. mean `latency`
4. mean `information_term`
5. fraction with `question_answerability == "answerable"`
6. fraction with `visibility_status == "visible"`
7. number of observations
8. mean `num_images`

### 4.2.2 Adapter `shared_summary_semantic`

The adapter builds an 8D `semantic_summary`:

1. mean `positive_score`
2. mean `negative_score`
3. mean `unknown_score`
4. mean `confidence`
5. mean `belief`
6. mean `evidence`
7. max `positive_score`
8. max `negative_score`

### 4.2.3 Important note

Runtime and adapter paths share the same **shape** and field names, but not perfectly identical semantics for every summary dimension. This is important when mixing data from both sources.

---

## 5. Feature Packing and Dataset Sample Structure

The dataset is built by `CanonicalEmulationDataset` in `dataset.py`.

For each sample:

- a history window of length `H = history_len` is collected
- a future rollout horizon of length `T = horizon` is collected
- all vehicles are aligned into `N = max_nodes` slots
- all queries are aligned into `Q = max_queries` slots

### 5.1 Packed per-vehicle features

#### `node_features`

`node_features` is the full packed node vector from `pack_vehicle_node_state(...)`.

Layout:

- `x_raw(5)`
- `x_shared(20)`
- `x_derived(2)`
- `action(8)`

So:

- `raw_node_dim = 35`

#### `state_node_features`

`state_node_features` is the action-free state vector from `pack_vehicle_state_features(...)`.

Layout:

- `x_raw(5)`
- `x_shared(20)`
- `x_derived(2)`

So:

- `node_dim = 27`

This is the node-state tensor that the current model actually consumes.

#### `action_features`

`action_features` comes from `pack_vehicle_action_features(...)`.

Layout:

- `alpha`
- `nu`
- `bandwidth`
- `beta_one_hot[5]`

So:

- `action_dim = 8`

### 5.2 Packed targets

#### `target_raw_state`

From `pack_vehicle_raw_state_target(...)`:

- `dx`
- `dy`
- `dvx`
- `dvy`
- `delta_yaw`

So:

- `raw_state_dim = 5`

#### `target_shared_state`

From `pack_vehicle_shared_state_target(...)`:

- `shared_summary_raw(8)`
- `shared_summary_semantic(8)`
- `intent_summary(4)`

So:

- `shared_state_dim = 20`

### 5.3 Dataset sample keys

Current sample keys are:

**History tensors**

- `node_features`: `[H, N, 35]`
- `state_node_features`: `[H, N, 27]`
- `action_features`: `[H, N, 8]`
- `vehicle_exogenous_features`: `[H, N, 7]`
- `ego_state_features`: `[H, 5]`
- `step_exogenous_features`: `[H, 2]`
- `component_valid_mask`: `[H, N, 9]`
- `node_mask`: `[H, N]`
- `task_relevance`: `[H, N, Q]`
- `edge_attr`: `[H, E, 3]`
- `edge_mask`: `[H, E]`
- `history_mask`: `[H]`

**Static / indexing tensors**

- `edge_index`: `[2, E]`
- `query_features`: `[Q, D_query]`
- `query_mask`: `[Q]`
- `node_ids`: `[N]`
- `query_ids`: length `Q` list of strings

**Future rollout-condition tensors**

- `future_action_features`: `[T, N, 8]`
- `future_mask`: `[T]`
- `future_node_mask`: `[T, N]`

**Supervision targets**

- `target_raw_state`: `[T, N, 5]`
- `target_shared_state`: `[T, N, 20]`
- `target_sender_collab`: `[T, N, Q]`
- `target_sender_gain`: `[T, N, Q]`
- `target_ego_sc`: `[T, Q]`

---

## 6. Model Inputs

The current predictor is `GraphGRUEmulationModel`.

The model primarily consumes:

### 6.1 Historical inputs

- `state_node_features` instead of `node_features`
- `action_features`
- `vehicle_exogenous_features`
- `ego_state_features`
- `step_exogenous_features`
- `task_relevance`
- `query_features`
- `node_mask`
- `edge_index`
- `edge_attr`
- `edge_mask`
- `history_mask`

### 6.2 Rollout conditioning inputs

During future rollout, the model conditions on:

- `future_action_features`

### 6.3 Query features

Each query feature is built by `pack_query_features(...)` and contains:

- the query embedding input from `QueryRecord.query_embedding_input`

So:

- `query_dim = len(query_embedding_input)`

### 6.4 Ego features

`ego_state_features` contains:

1. ego x
2. ego y
3. ego vx
4. ego vy
5. ego yaw

So:

- `ego_state_dim = 5`

### 6.5 Important note about `node_features`

`node_features` is still present in dataset samples, but the current model uses:

- `state_node_features` as the main historical vehicle-state tensor
- `action_features` as a separate input branch

This keeps `x_t` and `u_t` factored even though the compatibility `node_features` tensor still exists.

---

## 7. Model Outputs and Supervision Targets

`GraphGRUEmulationModel.forward()` returns the following keys.

### 7.1 State outputs

- `raw_state`: `[B, T, N, 5]`
- `shared_state`: `[B, T, N, 20]`

These correspond to:

- `target_raw_state`
- `target_shared_state`

### 7.2 Derived collaboration outputs

- `derived_complementarity`: `[B, T, N]`
- `derived_accessibility`: `[B, T, N]`
- `derived_task_relevance`: `[B, T, N, Q]`

These are produced as interpretable intermediate predictions from latent rollout. They are not currently backed by separate target tensors in the dataset.

### 7.3 Task-conditioned collaboration outputs

- `sender_collab`: `[B, T, N, Q]`
- `sender_gain`: `[B, T, N, Q]`
- `ego_sc`: `[B, T, Q]`

These correspond to:

- `target_sender_collab`
- `target_sender_gain`
- `target_ego_sc`

### 7.4 How `sender_collab` is formed inside the model

Inside `forward()`, the model predicts:

- complementarity
- accessibility
- task relevance

Then combines them as:

`sender_collab = complementarity * task_relevance * accessibility`

So `sender_collab` is not decoded as a totally independent scalar head; it is the product of the three derived factors.

---

## 8. Loss and Training Semantics

The loss is computed by `compute_emulation_loss(...)`.

The directly supervised outputs are:

- `raw_state`
- `shared_state`
- `sender_collab`
- `sender_gain`
- `ego_sc`

The target tensors are:

- `target_raw_state`
- `target_shared_state`
- `target_sender_collab`
- `target_sender_gain`
- `target_ego_sc`

Masks used in loss computation:

- `future_mask`
- `future_node_mask`
- `query_mask`

Effective masks:

- raw-state loss uses a node mask over `[B, T, N, 1]`
- shared-state loss uses the same node mask
- sender losses use node-and-query masks over `[B, T, N, Q]`
- ego semantic-confidence loss uses future-and-query masks over `[B, T, Q]`

Available loss weights:

- `raw_state_weight`
- `shared_state_weight`
- `sender_collab_weight`
- `sender_gain_weight`
- `ego_sc_weight`
- `consistency_weight`

Current default dimensions in `GraphGRUEmulationConfig` are:

- `shared_state_dim = 20`
- `action_dim = 8`
- `vehicle_exogenous_dim = 7`
- `step_exogenous_dim = 2`
- `raw_state_dim = 5`

However, the training entry point infers these from a dataset sample and writes them into `GraphGRUEmulationConfig` dynamically.

---

## 9. Paper-Term Mapping

This section maps the current implementation to the Main_V2X-style terminology.

### 9.1 State

Paper-style conceptual state maps approximately to:

- ego state: `ego_state`
- per-vehicle raw state: `delta_pos`, `delta_vel`, `delta_yaw`
- per-vehicle shared state: compact summary `20D`
- derived collaboration terms: `complementarity`, `accessibility`, `query_task_relevance`

### 9.2 Action

The current per-vehicle action is:

- `alpha`
- `nu`
- `bandwidth`
- `beta_one_hot`

This is the engineering realization of policy-conditioned communication control.

### 9.3 Exogenous variables

The current engineering `w_t` equivalent is:

- vehicle-level communication context
- step-level communication context

### 9.4 Important difference from the removed shared-latent design

The current implementation does **not** use:

- image observation latent vectors
- text observation latent vectors
- CLIP-based shared-latent targets

The only active shared-state representation is the compact-summary path described above.

---

## 10. Current Boundaries and Caveats

1. The runtime compact summary and old-log adapter summary have the same shape but not identical per-dimension semantics.
2. `derived_complementarity`, `derived_accessibility`, and `derived_task_relevance` are predicted by the model but are not currently supervised with dedicated dataset targets.
3. `node_features` is retained mainly as a packed compatibility tensor; the active model path uses `state_node_features` plus separate `action_features`.
4. The document describes the current right-turn collaboration pipeline only.
5. `shared_state` now exclusively means the 20D compact summary, not any observation latent representation.
