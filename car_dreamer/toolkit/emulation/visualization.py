from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dataset import CanonicalEmulationDataset
from .model import GraphGRUEmulationConfig, GraphGRUEmulationModel, torch_is_available
from .schema import CanonicalEpisodeRecord, CanonicalStepRecord, episode_from_dict
from .training import load_episode_from_path, resolve_device

try:
    import torch
except ImportError:  # pragma: no cover - guarded at runtime
    torch = None


SUPPORTED_METRICS: Tuple[str, ...] = ("sender_collab", "sender_gain")
DEFAULT_CANVAS_SIZE: Tuple[int, int] = (720, 720)
DEFAULT_EDGE_WIDTH_RANGE: Tuple[float, float] = (2.0, 12.0)
DEFAULT_EPSILON = 1e-6
GT_SUBDIR = "gt"
COMPARE_SUBDIR = "compare"

_BACKGROUND = (250, 250, 252)
_PANEL_BACKGROUND = (255, 255, 255)
_GRID = (226, 229, 235)
_TITLE = (32, 35, 43)
_SUBTITLE = (92, 98, 110)
_EGO_FILL = (196, 59, 51)
_EGO_OUTLINE = (128, 29, 23)
_MEMBER_FILL = (68, 114, 196)
_MEMBER_OUTLINE = (34, 61, 108)
_TEXT = (24, 27, 33)
_EDGE_COLORS = {
    "sender_collab": (55, 108, 214),
    "sender_gain": (232, 128, 44),
}


def render_ground_truth_topology_sequences(
    episode_source: str | Path | CanonicalEpisodeRecord | Mapping[str, Any],
    output_dir: str | Path,
    *,
    metrics: Sequence[str] = SUPPORTED_METRICS,
    queries: Sequence[str] | None = None,
    step_start: int | None = None,
    step_end: int | None = None,
    gif_duration_ms: int = 180,
    canvas_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
    epsilon: float = DEFAULT_EPSILON,
) -> Dict[str, Any]:
    episode = _coerce_episode(episode_source)
    output_dir = Path(output_dir)
    selected_metrics = _resolve_metrics(metrics)
    selected_queries = _resolve_queries(episode, queries)
    selected_steps = _select_step_indices(episode, step_start=step_start, step_end=step_end)

    summary: Dict[str, Any] = {
        "mode": "ground_truth",
        "output_dir": str(output_dir),
        "frame_counts": {},
        "gif_paths": [],
    }
    sequence_positions = _collect_step_positions(episode.steps[index] for index in selected_steps)
    position_layout = _compute_position_layout(sequence_positions, canvas_size)

    for metric in selected_metrics:
        metric_counts: Dict[str, int] = {}
        for query_id in selected_queries:
            values = [
                _extract_step_metric_values(episode.steps[index], metric, query_id, epsilon=epsilon)
                for index in selected_steps
            ]
            width_scale_max = _collect_global_metric_max(values)
            frames: List[Image.Image] = []
            sequence_dir = output_dir / GT_SUBDIR / metric / _slugify(query_id)
            sequence_dir.mkdir(parents=True, exist_ok=True)
            for step_index, step_values in zip(selected_steps, values):
                frame = _render_topology_panel(
                    step=episode.steps[step_index],
                    positions=_step_positions(episode.steps[step_index]),
                    edge_values=step_values,
                    metric=metric,
                    query_id=query_id,
                    canvas_size=canvas_size,
                    position_layout=position_layout,
                    width_scale_max=width_scale_max,
                    title_lines=[
                        f"Ground Truth | step {step_index:03d}",
                        f"{metric} | {query_id} | ego_sc={episode.steps[step_index].ego_sc.get(query_id, 0.0):.3f}",
                    ],
                )
                frame_path = sequence_dir / f"step_{step_index:03d}.png"
                frame.save(frame_path)
                frames.append(frame)
            gif_path = sequence_dir / "sequence.gif"
            _save_gif(frames, gif_path, duration_ms=int(gif_duration_ms))
            summary["gif_paths"].append(str(gif_path))
            metric_counts[query_id] = len(frames)
        summary["frame_counts"][metric] = metric_counts
    return summary


def render_prediction_comparison_sequences(
    episode_source: str | Path | CanonicalEpisodeRecord | Mapping[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    metrics: Sequence[str] = SUPPORTED_METRICS,
    queries: Sequence[str] | None = None,
    step_start: int | None = None,
    step_end: int | None = None,
    history_len: int | None = None,
    horizon: int | None = None,
    device: str = "auto",
    gif_duration_ms: int = 180,
    canvas_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
    epsilon: float = DEFAULT_EPSILON,
) -> Dict[str, Any]:
    if not torch_is_available() or torch is None:
        raise ImportError(
            "Prediction comparison visualization requires PyTorch, but torch is not installed."
        )

    episode = _coerce_episode(episode_source)
    output_dir = Path(output_dir)
    selected_metrics = _resolve_metrics(metrics)
    selected_queries = _resolve_queries(episode, queries)

    checkpoint_payload = torch.load(str(checkpoint_path), map_location=resolve_device(device))
    dataset_history_len = _resolve_history_len(checkpoint_payload, history_len=history_len)
    dataset_horizon = _resolve_horizon(checkpoint_payload, horizon=horizon)
    dataset = CanonicalEmulationDataset(
        [episode],
        history_len=int(dataset_history_len),
        horizon=int(dataset_horizon),
    )
    model = _build_model_from_checkpoint_payload(
        checkpoint_payload,
        sample=dataset[0],
        device=resolve_device(device),
    )

    anchor_steps = _select_step_indices(episode, step_start=step_start, step_end=step_end)
    selected_pairs = _collect_prediction_pairs(
        episode,
        dataset,
        model,
        anchor_steps=anchor_steps,
        max_horizon=int(dataset_horizon),
        device=resolve_device(device),
        epsilon=float(epsilon),
    )

    summary: Dict[str, Any] = {
        "mode": "prediction_compare",
        "output_dir": str(output_dir),
        "frame_counts": {},
        "gif_paths": [],
    }
    position_layout = _compute_position_layout(
        _collect_step_positions(pair["future_step"] for pair in selected_pairs),
        canvas_size,
    )

    for metric in selected_metrics:
        metric_counts: Dict[str, int] = {}
        for query_id in selected_queries:
            gt_values = [
                _extract_step_metric_values(pair["future_step"], metric, query_id, epsilon=epsilon)
                for pair in selected_pairs
            ]
            pred_values = [
                _extract_prediction_metric_values(
                    prediction_pair=pair,
                    metric=metric,
                    query_id=query_id,
                    epsilon=epsilon,
                )
                for pair in selected_pairs
            ]
            width_scale_max = _collect_global_metric_max(gt_values + pred_values)
            frames: List[Image.Image] = []
            sequence_dir = output_dir / COMPARE_SUBDIR / metric / _slugify(query_id)
            sequence_dir.mkdir(parents=True, exist_ok=True)

            for pair, gt_metric_values, pred_metric_values in zip(selected_pairs, gt_values, pred_values):
                frame = _render_prediction_compare_frame(
                    metric=metric,
                    query_id=query_id,
                    future_step=pair["future_step"],
                    gt_edge_values=gt_metric_values,
                    pred_edge_values=pred_metric_values,
                    gt_ego_sc=float(pair["future_step"].ego_sc.get(query_id, 0.0)),
                    pred_ego_sc=float(pair["pred_ego_sc_by_query"].get(query_id, 0.0)),
                    anchor_step=int(pair["anchor_step"]),
                    horizon_offset=int(pair["horizon_offset"]),
                    panel_size=canvas_size,
                    position_layout=position_layout,
                    width_scale_max=width_scale_max,
                )
                frame_path = sequence_dir / (
                    f"anchor_{int(pair['anchor_step']):03d}_h{int(pair['horizon_offset']):02d}.png"
                )
                frame.save(frame_path)
                frames.append(frame)
            gif_path = sequence_dir / "sequence.gif"
            _save_gif(frames, gif_path, duration_ms=int(gif_duration_ms))
            summary["gif_paths"].append(str(gif_path))
            metric_counts[query_id] = len(frames)
        summary["frame_counts"][metric] = metric_counts
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render emulation topology visualizations from canonical episode JSON."
    )
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(SUPPORTED_METRICS),
        choices=list(SUPPORTED_METRICS),
    )
    parser.add_argument("--queries", nargs="+", default=None)
    parser.add_argument("--step-start", type=int, default=None)
    parser.add_argument("--step-end", type=int, default=None)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gif-duration-ms", type=int, default=180)
    parser.add_argument("--canvas-size", nargs=2, type=int, default=list(DEFAULT_CANVAS_SIZE))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    canvas_size = (int(args.canvas_size[0]), int(args.canvas_size[1]))

    gt_summary = render_ground_truth_topology_sequences(
        args.episode,
        args.output_dir,
        metrics=args.metrics,
        queries=args.queries,
        step_start=args.step_start,
        step_end=args.step_end,
        gif_duration_ms=int(args.gif_duration_ms),
        canvas_size=canvas_size,
    )
    print(
        "[emulation][viz] rendered ground-truth sequences "
        f"to {gt_summary['output_dir']}"
    )

    if args.checkpoint:
        compare_summary = render_prediction_comparison_sequences(
            args.episode,
            args.checkpoint,
            args.output_dir,
            metrics=args.metrics,
            queries=args.queries,
            step_start=args.step_start,
            step_end=args.step_end,
            history_len=args.history_len,
            horizon=args.horizon,
            device=args.device,
            gif_duration_ms=int(args.gif_duration_ms),
            canvas_size=canvas_size,
        )
        print(
            "[emulation][viz] rendered prediction comparison sequences "
            f"to {compare_summary['output_dir']}"
        )


def _coerce_episode(
    episode_source: str | Path | CanonicalEpisodeRecord | Mapping[str, Any],
) -> CanonicalEpisodeRecord:
    if isinstance(episode_source, CanonicalEpisodeRecord):
        return episode_source
    if isinstance(episode_source, Mapping):
        return episode_from_dict(dict(episode_source))
    return load_episode_from_path(episode_source)


def _resolve_metrics(metrics: Sequence[str]) -> List[str]:
    selected = [str(metric) for metric in metrics]
    invalid = sorted(set(selected) - set(SUPPORTED_METRICS))
    if invalid:
        raise ValueError(f"Unsupported metrics: {invalid}")
    return selected


def _resolve_queries(episode: CanonicalEpisodeRecord, queries: Sequence[str] | None) -> List[str]:
    available = [query.query_id for query in episode.steps[0].queries]
    if queries is None:
        return available
    selected = [str(query_id) for query_id in queries]
    invalid = sorted(set(selected) - set(available))
    if invalid:
        raise ValueError(f"Unknown queries for episode: {invalid}")
    return selected


def _select_step_indices(
    episode: CanonicalEpisodeRecord,
    *,
    step_start: int | None,
    step_end: int | None,
) -> List[int]:
    start = max(int(step_start or 0), 0)
    end = len(episode.steps) - 1 if step_end is None else min(int(step_end), len(episode.steps) - 1)
    if end < start:
        return []
    return list(range(start, end + 1))


def _resolve_history_len(payload: Mapping[str, Any], *, history_len: int | None) -> int:
    if history_len is not None:
        return max(int(history_len), 1)
    model_config = dict(payload.get("model_config", {}))
    train_config = dict(payload.get("train_config", {}))
    return max(
        int(model_config.get("history_len", train_config.get("history_len", 8))),
        1,
    )


def _resolve_horizon(payload: Mapping[str, Any], *, horizon: int | None) -> int:
    model_config = dict(payload.get("model_config", {}))
    train_config = dict(payload.get("train_config", {}))
    checkpoint_horizon = max(
        int(model_config.get("horizon", train_config.get("horizon", 5))),
        1,
    )
    if horizon is None:
        return checkpoint_horizon
    return max(min(int(horizon), checkpoint_horizon), 1)


def _build_model_from_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
    device: str,
) -> GraphGRUEmulationModel:
    model_config_payload = dict(payload.get("model_config", {}))
    if model_config_payload:
        model_config = GraphGRUEmulationConfig(**model_config_payload)
    else:
        train_config = dict(payload.get("train_config", {}))
        model_config = GraphGRUEmulationConfig(
            node_dim=int(np.asarray(sample["node_features"]).shape[-1]),
            query_dim=int(np.asarray(sample["query_features"]).shape[-1]),
            edge_attr_dim=int(np.asarray(sample["edge_attr"]).shape[-1]),
            hidden_dim=int(train_config.get("hidden_dim", 64)),
            num_graph_layers=int(train_config.get("num_graph_layers", 2)),
            history_len=int(train_config.get("history_len", np.asarray(sample["node_features"]).shape[0])),
            horizon=int(train_config.get("horizon", np.asarray(sample["target_ego_sc"]).shape[0])),
            dropout=float(train_config.get("dropout", 0.0)),
        )

    model = GraphGRUEmulationModel(model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _collect_prediction_pairs(
    episode: CanonicalEpisodeRecord,
    dataset: CanonicalEmulationDataset,
    model: GraphGRUEmulationModel,
    *,
    anchor_steps: Sequence[int],
    max_horizon: int,
    device: str,
    epsilon: float,
) -> List[Dict[str, Any]]:
    del epsilon
    prediction_pairs: List[Dict[str, Any]] = []
    with torch.no_grad():
        for anchor_step in anchor_steps:
            sample = dataset[int(anchor_step)]
            future_mask = np.asarray(sample["future_mask"])
            if float(future_mask.sum()) <= 0.0:
                continue
            tensor_sample = _tensorize_sample_for_model(sample, device=device)
            predictions = model(tensor_sample)
            sender_collab = predictions["sender_collab"][0].detach().cpu().numpy()
            sender_gain = predictions["sender_gain"][0].detach().cpu().numpy()
            ego_sc = predictions["ego_sc"][0].detach().cpu().numpy()

            valid_query_ids = [
                str(query_id)
                for query_id, valid in zip(sample["query_ids"], np.asarray(sample["query_mask"]))
                if str(query_id) and float(valid) > 0.0
            ]
            node_ids = [int(node_id) for node_id in np.asarray(sample["node_ids"]).tolist()]
            valid_horizon = min(int(max_horizon), int(sender_collab.shape[0]))
            for offset in range(valid_horizon):
                if float(future_mask[offset]) <= 0.0:
                    continue
                future_index = int(anchor_step) + offset + 1
                if future_index >= len(episode.steps):
                    continue
                future_step = episode.steps[future_index]
                pred_collab_by_query = _prediction_values_by_query(
                    sender_collab[offset],
                    node_ids=node_ids,
                    query_ids=valid_query_ids,
                )
                pred_gain_by_query = _prediction_values_by_query(
                    sender_gain[offset],
                    node_ids=node_ids,
                    query_ids=valid_query_ids,
                )
                pred_ego_sc_by_query = {
                    query_id: float(ego_sc[offset, query_index])
                    for query_index, query_id in enumerate(valid_query_ids)
                }
                prediction_pairs.append(
                    {
                        "anchor_step": int(anchor_step),
                        "horizon_offset": int(offset + 1),
                        "future_step": future_step,
                        "pred_sender_collab_by_query": pred_collab_by_query,
                        "pred_sender_gain_by_query": pred_gain_by_query,
                        "pred_ego_sc_by_query": pred_ego_sc_by_query,
                    }
                )
    return prediction_pairs


def _prediction_values_by_query(
    prediction_matrix: np.ndarray,
    *,
    node_ids: Sequence[int],
    query_ids: Sequence[str],
) -> Dict[str, Dict[int, float]]:
    values: Dict[str, Dict[int, float]] = {str(query_id): {} for query_id in query_ids}
    for node_slot, node_id in enumerate(node_ids):
        if int(node_id) < 0:
            continue
        for query_index, query_id in enumerate(query_ids):
            values[str(query_id)][int(node_id)] = float(prediction_matrix[node_slot, query_index])
    return values


def _tensorize_sample_for_model(sample: Mapping[str, Any], *, device: str) -> Dict[str, Any]:
    tensorized: Dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            tensorized[key] = torch.as_tensor(value, device=device)
        else:
            tensorized[key] = value
    return tensorized


def _collect_step_positions(steps: Iterable[CanonicalStepRecord]) -> List[Tuple[float, float]]:
    positions: List[Tuple[float, float]] = [(0.0, 0.0)]
    for step in steps:
        positions.extend(_step_positions(step).values())
    return positions


def _step_positions(step: CanonicalStepRecord) -> Dict[int, Tuple[float, float]]:
    return {
        int(vehicle.vehicle_id): (
            float(vehicle.delta_pos[0]),
            float(vehicle.delta_pos[1]),
        )
        for vehicle in step.candidate_vehicles
    }


def _compute_position_layout(
    positions: Sequence[Tuple[float, float]],
    canvas_size: Tuple[int, int],
) -> Dict[str, float]:
    width = int(canvas_size[0])
    height = int(canvas_size[1])
    header_h = 74.0
    footer_h = 24.0
    side_margin = 34.0
    body_height = max(float(height) - header_h - footer_h, 1.0)
    center_x = float(width) * 0.5
    center_y = header_h + body_height * 0.5
    max_abs_x = max(abs(float(x)) for x, _ in positions) if positions else 1.0
    max_abs_y = max(abs(float(y)) for _, y in positions) if positions else 1.0
    max_abs_x = max(max_abs_x, 1.0)
    max_abs_y = max(max_abs_y, 1.0)
    usable_w = max(float(width) - 2.0 * side_margin, 1.0)
    usable_h = max(body_height - 2.0 * side_margin, 1.0)
    scale = min(usable_w / (2.0 * max_abs_x), usable_h / (2.0 * max_abs_y))
    return {
        "width": float(width),
        "height": float(height),
        "center_x": center_x,
        "center_y": center_y,
        "header_h": header_h,
        "footer_h": footer_h,
        "scale": max(scale, 1e-6),
    }


def _extract_step_metric_values(
    step: CanonicalStepRecord,
    metric: str,
    query_id: str,
    *,
    epsilon: float,
) -> Dict[int, float]:
    values: Dict[int, float] = {}
    for vehicle in step.candidate_vehicles:
        source = vehicle.sender_collab if metric == "sender_collab" else vehicle.sender_gain
        values[int(vehicle.vehicle_id)] = max(float(source.get(query_id, 0.0)), 0.0)
    return {
        vehicle_id: value
        for vehicle_id, value in values.items()
        if float(value) > float(epsilon)
    }


def _extract_prediction_metric_values(
    *,
    prediction_pair: Mapping[str, Any],
    metric: str,
    query_id: str,
    epsilon: float,
) -> Dict[int, float]:
    source_key = "pred_sender_collab_by_query" if metric == "sender_collab" else "pred_sender_gain_by_query"
    source = dict(prediction_pair.get(source_key, {})).get(query_id, {})
    values: Dict[int, float] = {}
    for vehicle in prediction_pair["future_step"].candidate_vehicles:
        raw_value = float(dict(source).get(int(vehicle.vehicle_id), 0.0))
        values[int(vehicle.vehicle_id)] = max(raw_value, 0.0)
    return {
        vehicle_id: value
        for vehicle_id, value in values.items()
        if float(value) > float(epsilon)
    }


def _collect_global_metric_max(values_by_frame: Sequence[Mapping[int, float]]) -> float:
    max_value = 0.0
    for values in values_by_frame:
        if values:
            max_value = max(max_value, max(float(value) for value in values.values()))
    return float(max_value)


def _render_prediction_compare_frame(
    *,
    metric: str,
    query_id: str,
    future_step: CanonicalStepRecord,
    gt_edge_values: Mapping[int, float],
    pred_edge_values: Mapping[int, float],
    gt_ego_sc: float,
    pred_ego_sc: float,
    anchor_step: int,
    horizon_offset: int,
    panel_size: Tuple[int, int],
    position_layout: Mapping[str, float],
    width_scale_max: float,
) -> Image.Image:
    panel_w = int(panel_size[0])
    panel_h = int(panel_size[1])
    gap = 20
    header_h = 44
    image = Image.new("RGB", (panel_w * 2 + gap, panel_h + header_h), color=_BACKGROUND)
    draw = ImageDraw.Draw(image)
    title = (
        f"{metric} | {query_id} | anchor {anchor_step:03d} -> step {int(future_step.step):03d} "
        f"(h={horizon_offset})"
    )
    draw.text((18, 12), title, fill=_TITLE, font=_load_font())

    positions = _step_positions(future_step)
    gt_panel = _render_topology_panel(
        step=future_step,
        positions=positions,
        edge_values=gt_edge_values,
        metric=metric,
        query_id=query_id,
        canvas_size=panel_size,
        position_layout=position_layout,
        width_scale_max=width_scale_max,
        title_lines=[
            "Ground Truth",
            f"ego_sc={gt_ego_sc:.3f}",
        ],
    )
    pred_panel = _render_topology_panel(
        step=future_step,
        positions=positions,
        edge_values=pred_edge_values,
        metric=metric,
        query_id=query_id,
        canvas_size=panel_size,
        position_layout=position_layout,
        width_scale_max=width_scale_max,
        title_lines=[
            "Prediction",
            f"ego_sc={pred_ego_sc:.3f}",
        ],
    )
    image.paste(gt_panel, (0, header_h))
    image.paste(pred_panel, (panel_w + gap, header_h))
    return image


def _render_topology_panel(
    *,
    step: CanonicalStepRecord,
    positions: Mapping[int, Tuple[float, float]],
    edge_values: Mapping[int, float],
    metric: str,
    query_id: str,
    canvas_size: Tuple[int, int],
    position_layout: Mapping[str, float],
    width_scale_max: float,
    title_lines: Sequence[str],
) -> Image.Image:
    del query_id
    panel = Image.new("RGB", (int(canvas_size[0]), int(canvas_size[1])), color=_PANEL_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    font = _load_font()

    header_h = int(position_layout["header_h"])
    draw.rectangle((0, 0, int(canvas_size[0]) - 1, int(canvas_size[1]) - 1), outline=_GRID, width=1)
    for index, line in enumerate(title_lines):
        fill = _TITLE if index == 0 else _SUBTITLE
        draw.text((18, 12 + index * 18), str(line), fill=fill, font=font)

    ego_center = (float(position_layout["center_x"]), float(position_layout["center_y"]))
    _draw_reference_axes(draw, ego_center, canvas_size, header_h)
    edge_color = _EDGE_COLORS[str(metric)]

    for vehicle in step.candidate_vehicles:
        vehicle_id = int(vehicle.vehicle_id)
        pos = positions.get(vehicle_id, (0.0, 0.0))
        member_center = _project_position(pos, position_layout)
        value = max(float(edge_values.get(vehicle_id, 0.0)), 0.0)
        if value > 0.0:
            draw.line(
                (ego_center[0], ego_center[1], member_center[0], member_center[1]),
                fill=edge_color,
                width=int(round(_edge_width(value, width_scale_max))),
            )

    _draw_node(draw, ego_center, radius=18, fill=_EGO_FILL, outline=_EGO_OUTLINE, label="ego")
    for vehicle in step.candidate_vehicles:
        vehicle_id = int(vehicle.vehicle_id)
        member_center = _project_position(positions.get(vehicle_id, (0.0, 0.0)), position_layout)
        draw_value = max(float(edge_values.get(vehicle_id, 0.0)), 0.0)
        _draw_node(
            draw,
            member_center,
            radius=14,
            fill=_MEMBER_FILL,
            outline=_MEMBER_OUTLINE,
            label=str(vehicle_id),
            value=draw_value,
        )
    return panel


def _draw_reference_axes(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    canvas_size: Tuple[int, int],
    header_h: int,
) -> None:
    width = int(canvas_size[0])
    height = int(canvas_size[1])
    draw.line((24, center[1], width - 24, center[1]), fill=_GRID, width=1)
    draw.line((center[0], header_h + 8, center[0], height - 24), fill=_GRID, width=1)


def _draw_node(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    *,
    radius: int,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    label: str,
    value: float | None = None,
) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
    font = _load_font()
    label_bbox = draw.textbbox((0, 0), label, font=font)
    label_w = label_bbox[2] - label_bbox[0]
    label_h = label_bbox[3] - label_bbox[1]
    draw.text((x - label_w * 0.5, y - label_h * 0.5), label, fill=(255, 255, 255), font=font)
    if value is not None:
        value_text = f"{value:.3f}"
        value_bbox = draw.textbbox((0, 0), value_text, font=font)
        value_w = value_bbox[2] - value_bbox[0]
        draw.text((x - value_w * 0.5, y + radius + 4), value_text, fill=_TEXT, font=font)


def _project_position(
    position: Tuple[float, float],
    layout: Mapping[str, float],
) -> Tuple[float, float]:
    x, y = float(position[0]), float(position[1])
    return (
        float(layout["center_x"]) + x * float(layout["scale"]),
        float(layout["center_y"]) - y * float(layout["scale"]),
    )


def _edge_width(value: float, scale_max: float) -> float:
    min_w, max_w = DEFAULT_EDGE_WIDTH_RANGE
    if scale_max <= 0.0:
        return min_w
    ratio = max(min(float(value) / float(scale_max), 1.0), 0.0)
    return min_w + ratio * (max_w - min_w)


def _save_gif(frames: Sequence[Image.Image], path: str | Path, *, duration_ms: int) -> None:
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=max(int(duration_ms), 1),
        loop=0,
    )


def _slugify(name: str) -> str:
    safe = [
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(name)
    ]
    return "".join(safe).strip("_") or "item"


def _load_font():
    return ImageFont.load_default()


__all__ = [
    "SUPPORTED_METRICS",
    "build_arg_parser",
    "main",
    "render_ground_truth_topology_sequences",
    "render_prediction_comparison_sequences",
]
