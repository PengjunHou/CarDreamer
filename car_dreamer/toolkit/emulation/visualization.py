from __future__ import annotations

import argparse
import glob
import functools
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dataset import CanonicalEmulationDataset
from .model import GraphGRUEmulationConfig, GraphGRUEmulationModel, torch_is_available
from .schema import CanonicalEpisodeRecord, CanonicalStepRecord, QueryRecord, RegionBox, episode_from_dict
from .training import load_episode_from_path, resolve_device

try:
    import torch
except ImportError:  # pragma: no cover - guarded at runtime
    torch = None


SUPPORTED_METRICS: Tuple[str, ...] = ("sender_collab", "sender_gain")
DEFAULT_CANVAS_SIZE: Tuple[int, int] = (720, 720)
DEFAULT_REGION_CANVAS_SIZE: Tuple[int, int] = (1800, 1200)
DEFAULT_EDGE_WIDTH_RANGE: Tuple[float, float] = (2.0, 12.0)
DEFAULT_EPSILON = 1e-6
GT_SUBDIR = "gt"
COMPARE_SUBDIR = "compare"
REGIONS_SUBDIR = "regions_overview"
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_POLICY_DIR_RE = re.compile(r"^P\d+$", re.IGNORECASE)

_BACKGROUND = (250, 250, 252)
_PANEL_BACKGROUND = (255, 255, 255)
_GRID = (226, 229, 235)
_TITLE = (32, 35, 43)
_SUBTITLE = (92, 98, 110)
_MUTED = (130, 138, 150)
_EGO_FILL = (196, 59, 51)
_EGO_OUTLINE = (128, 29, 23)
_MEMBER_FILL = (68, 114, 196)
_MEMBER_OUTLINE = (34, 61, 108)
_TEXT = (24, 27, 33)
_EDGE_COLORS = {
    "sender_collab": (55, 108, 214),
    "sender_gain": (232, 128, 44),
}
_MAIN_PANEL_BACKGROUND = (251, 252, 255, 255)
_CARD_BACKGROUND = (255, 255, 255, 255)
_CARD_BORDER = (221, 226, 235, 255)
_EGO_REGION_FILL = (196, 59, 51, 48)
_EGO_REGION_OUTLINE = (128, 29, 23, 255)
_MEMBER_REGION_FILL = (68, 114, 196, 34)
_MEMBER_REGION_OUTLINE = (49, 85, 158, 190)
_MEMBER_REGION_HIGHLIGHT_FILL = (55, 108, 214, 54)
_MEMBER_REGION_HIGHLIGHT_OUTLINE = (34, 61, 108, 255)
_REQUIRED_REGION_FILL = (231, 177, 72, 56)
_REQUIRED_REGION_OUTLINE = (176, 121, 24, 255)
_TABLE_HEADER_FILL = (242, 245, 251, 255)
_TOP_BADGE_FILL = (237, 242, 252, 255)
_TOP_BADGE_OUTLINE = (205, 215, 234, 255)


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


def render_region_overview_sequences(
    episode_source: str | Path | CanonicalEpisodeRecord | Mapping[str, Any],
    output_dir: str | Path,
    *,
    queries: Sequence[str] | None = None,
    step_start: int | None = None,
    step_end: int | None = None,
    gif_duration_ms: int = 180,
    canvas_size: Tuple[int, int] = DEFAULT_REGION_CANVAS_SIZE,
    highlight_top_k: int = 3,
) -> Dict[str, Any]:
    episode = _coerce_episode(episode_source)
    output_dir = Path(output_dir)
    selected_queries = _resolve_queries(episode, queries)
    selected_steps = _select_step_indices(episode, step_start=step_start, step_end=step_end)
    summary: Dict[str, Any] = {
        "mode": "regions_overview",
        "output_dir": str(output_dir),
        "frame_counts": {"regions_overview": len(selected_steps)},
        "gif_paths": [],
    }
    sequence_dir = output_dir / REGIONS_SUBDIR
    sequence_dir.mkdir(parents=True, exist_ok=True)
    world_bounds = _collect_region_world_bounds(
        [episode.steps[index] for index in selected_steps],
        selected_queries,
    )
    frames: List[Image.Image] = []
    for step_index in selected_steps:
        frame = _render_region_overview_frame(
            step=episode.steps[step_index],
            selected_queries=selected_queries,
            canvas_size=canvas_size,
            world_bounds=world_bounds,
            highlight_top_k=max(int(highlight_top_k), 1),
        )
        frame_path = sequence_dir / f"step_{step_index:03d}.png"
        frame.save(frame_path)
        frames.append(frame)
    gif_path = sequence_dir / "sequence.gif"
    _save_gif(frames, gif_path, duration_ms=int(gif_duration_ms))
    summary["gif_paths"].append(str(gif_path))
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
    parser.add_argument("--episode", default="")
    parser.add_argument(
        "--episode-glob",
        nargs="+",
        default=None,
        help="Glob pattern(s) that expand to episode JSON files for batch rendering.",
    )
    parser.add_argument(
        "--policy-ids",
        nargs="+",
        default=None,
        help="Policy IDs whose episode JSON files should be rendered in batch.",
    )
    parser.add_argument(
        "--policy-root",
        default="data",
        help="Root directory searched when resolving --policy-ids.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view", choices=("topology", "regions"), default="topology")
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
    parser.add_argument("--canvas-size", nargs=2, type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    episode_paths = _resolve_cli_episode_paths(
        episode=args.episode,
        episode_globs=args.episode_glob,
        policy_ids=args.policy_ids,
        policy_root=args.policy_root,
    )
    batch_mode = len(episode_paths) > 1 or bool(args.episode_glob) or bool(args.policy_ids)
    if args.canvas_size is None:
        canvas_size = DEFAULT_REGION_CANVAS_SIZE if args.view == "regions" else DEFAULT_CANVAS_SIZE
    else:
        canvas_size = (int(args.canvas_size[0]), int(args.canvas_size[1]))

    for episode_path in episode_paths:
        render_output_dir = (
            _build_batch_output_dir(args.output_dir, episode_path) if batch_mode else Path(args.output_dir)
        )

        if args.view == "regions":
            region_summary = render_region_overview_sequences(
                episode_path,
                render_output_dir,
                queries=args.queries,
                step_start=args.step_start,
                step_end=args.step_end,
                gif_duration_ms=int(args.gif_duration_ms),
                canvas_size=canvas_size,
            )
            print(
                "[emulation][viz] rendered region overview sequences "
                f"for {episode_path} to {region_summary['output_dir']}"
            )
            continue

        gt_summary = render_ground_truth_topology_sequences(
            episode_path,
            render_output_dir,
            metrics=args.metrics,
            queries=args.queries,
            step_start=args.step_start,
            step_end=args.step_end,
            gif_duration_ms=int(args.gif_duration_ms),
            canvas_size=canvas_size,
        )
        print(
            "[emulation][viz] rendered ground-truth sequences "
            f"for {episode_path} to {gt_summary['output_dir']}"
        )

        if args.checkpoint:
            compare_summary = render_prediction_comparison_sequences(
                episode_path,
                args.checkpoint,
                render_output_dir,
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
                f"for {episode_path} to {compare_summary['output_dir']}"
            )


def _resolve_cli_episode_paths(
    *,
    episode: str,
    episode_globs: Sequence[str] | None,
    policy_ids: Sequence[str] | None,
    policy_root: str | Path,
) -> List[Path]:
    has_episode = bool(str(episode).strip())
    has_glob = bool(episode_globs)
    has_policy_ids = bool(policy_ids)
    selected_modes = int(has_episode) + int(has_glob) + int(has_policy_ids)
    if selected_modes != 1:
        raise ValueError("Specify exactly one of --episode, --episode-glob, or --policy-ids.")

    if has_episode:
        episode_path = Path(str(episode)).expanduser()
        if not episode_path.exists():
            raise FileNotFoundError(f"Episode file not found: {episode_path}")
        return [episode_path.resolve()]

    if has_glob:
        matches: List[Path] = []
        for pattern in episode_globs or []:
            matches.extend(Path(path) for path in glob.glob(str(pattern), recursive=True))
        unique_matches = _dedupe_and_sort_paths(matches)
        if not unique_matches:
            raise FileNotFoundError(f"No episode files matched --episode-glob: {list(episode_globs or [])}")
        return unique_matches

    return _resolve_policy_episode_paths(policy_ids or [], policy_root=policy_root)


def _resolve_policy_episode_paths(
    policy_ids: Sequence[str],
    *,
    policy_root: str | Path,
) -> List[Path]:
    root = Path(policy_root).expanduser()
    matches: List[Path] = []
    missing_policy_ids: List[str] = []
    for policy_id in policy_ids:
        normalized_policy_id = str(policy_id).strip()
        policy_matches = sorted(root.glob(f"**/{normalized_policy_id}/emulation_episode_*.json"))
        if not policy_matches:
            missing_policy_ids.append(normalized_policy_id)
            continue
        matches.extend(policy_matches)
    if missing_policy_ids:
        missing = ", ".join(missing_policy_ids)
        raise FileNotFoundError(f"No episode files found for policy ids: {missing}")
    return _dedupe_and_sort_paths(matches)


def _dedupe_and_sort_paths(paths: Sequence[Path]) -> List[Path]:
    return sorted({path.resolve() for path in paths})


def _build_batch_output_dir(base_output_dir: str | Path, episode_path: str | Path) -> Path:
    base = Path(base_output_dir)
    episode = Path(episode_path)
    policy_id = _extract_policy_id_from_path(episode)
    if policy_id:
        return base / policy_id / episode.stem
    return base / episode.stem


def _extract_policy_id_from_path(path: str | Path) -> str:
    for parent in Path(path).parents:
        if _POLICY_DIR_RE.match(parent.name):
            return parent.name
    return ""


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
            node_dim=int(np.asarray(sample["state_node_features"]).shape[-1]),
            query_dim=int(np.asarray(sample["query_features"]).shape[-1]),
            edge_attr_dim=int(np.asarray(sample["edge_attr"]).shape[-1]),
            hidden_dim=int(train_config.get("hidden_dim", 64)),
            num_graph_layers=int(train_config.get("num_graph_layers", 2)),
            history_len=int(train_config.get("history_len", np.asarray(sample["node_features"]).shape[0])),
            horizon=int(train_config.get("horizon", np.asarray(sample["target_ego_sc"]).shape[0])),
            dropout=float(train_config.get("dropout", 0.0)),
            raw_state_dim=int(np.asarray(sample["target_raw_state"]).shape[-1]),
            shared_state_dim=int(np.asarray(sample["target_shared_state"]).shape[-1]),
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
    # positions are (forward, left); forward maps to vertical, left maps to horizontal
    max_forward = max(abs(float(f)) for f, _ in positions) if positions else 1.0
    max_lateral = max(abs(float(l)) for _, l in positions) if positions else 1.0
    max_forward = max(max_forward, 1.0)
    max_lateral = max(max_lateral, 1.0)
    usable_w = max(float(width) - 2.0 * side_margin, 1.0)
    usable_h = max(body_height - 2.0 * side_margin, 1.0)
    scale = min(usable_w / (2.0 * max_lateral), usable_h / (2.0 * max_forward))
    return {
        "width": float(width),
        "height": float(height),
        "center_x": center_x,
        "center_y": center_y,
        "header_h": header_h,
        "footer_h": footer_h,
        "scale": max(scale, 1e-6),
    }


def _collect_region_world_bounds(
    steps: Sequence[CanonicalStepRecord],
    query_ids: Sequence[str],
) -> Dict[str, float]:
    points: List[Tuple[float, float]] = [(0.0, 0.0)]
    selected = {str(query_id) for query_id in query_ids}
    for step in steps:
        points.extend(_region_corners(step.ego_state.observable_region))
        for vehicle in step.candidate_vehicles:
            points.extend(_region_corners(vehicle.observable_region))
        for query in step.queries:
            if str(query.query_id) in selected:
                points.extend(_region_corners(query.required_region))
    forwards = [float(point[0]) for point in points]
    rights = [float(point[1]) for point in points]
    min_forward = min(forwards) - 2.0
    max_forward = max(forwards) + 2.0
    min_right = min(rights) - 2.0
    max_right = max(rights) + 2.0
    if max_forward - min_forward < 1.0:
        max_forward += 0.5
        min_forward -= 0.5
    if max_right - min_right < 1.0:
        max_right += 0.5
        min_right -= 0.5
    return {
        "min_forward": float(min_forward),
        "max_forward": float(max_forward),
        "min_right": float(min_right),
        "max_right": float(max_right),
    }


def _make_world_layout(
    rect: Tuple[int, int, int, int],
    world_bounds: Mapping[str, float],
    *,
    margin: float = 28.0,
) -> Dict[str, float]:
    x0, y0, width, height = [float(value) for value in rect]
    inner_w = max(width - 2.0 * margin, 1.0)
    inner_h = max(height - 2.0 * margin, 1.0)
    lateral_span = max(float(world_bounds["max_right"]) - float(world_bounds["min_right"]), 1e-6)
    forward_span = max(float(world_bounds["max_forward"]) - float(world_bounds["min_forward"]), 1e-6)
    scale = min(inner_w / lateral_span, inner_h / forward_span)
    used_w = lateral_span * scale
    used_h = forward_span * scale
    offset_x = x0 + margin + 0.5 * (inner_w - used_w)
    offset_y = y0 + margin + 0.5 * (inner_h - used_h)
    return {
        "x0": x0,
        "y0": y0,
        "width": width,
        "height": height,
        "margin": margin,
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "min_forward": float(world_bounds["min_forward"]),
        "max_forward": float(world_bounds["max_forward"]),
        "min_right": float(world_bounds["min_right"]),
        "max_right": float(world_bounds["max_right"]),
    }


def _project_world_point(
    point: Tuple[float, float],
    layout: Mapping[str, float],
) -> Tuple[float, float]:
    forward, right = float(point[0]), float(point[1])
    return (
        float(layout["offset_x"]) + (right - float(layout["min_right"])) * float(layout["scale"]),
        float(layout["offset_y"]) + (float(layout["max_forward"]) - forward) * float(layout["scale"]),
    )


def _region_corners(region: RegionBox) -> List[Tuple[float, float]]:
    half_forward = 0.5 * float(region.size[0])
    half_right = 0.5 * float(region.size[1])
    local = [
        (-half_forward, -half_right),
        (half_forward, -half_right),
        (half_forward, half_right),
        (-half_forward, half_right),
    ]
    c = math.cos(float(region.yaw))
    s = math.sin(float(region.yaw))
    corners: List[Tuple[float, float]] = []
    for forward, right in local:
        corners.append(
            (
                float(region.center[0]) + forward * c - right * s,
                float(region.center[1]) + forward * s + right * c,
            )
        )
    return corners


def _draw_region_box(
    draw: ImageDraw.ImageDraw,
    *,
    region: RegionBox,
    layout: Mapping[str, float],
    fill: Tuple[int, int, int, int],
    outline: Tuple[int, int, int, int],
    width: int,
) -> None:
    polygon = [_project_world_point(point, layout) for point in _region_corners(region)]
    draw.polygon(polygon, fill=fill, outline=outline)
    draw.line([*polygon, polygon[0]], fill=outline, width=max(int(width), 1), joint="curve")


def _draw_heading_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    region: RegionBox,
    layout: Mapping[str, float],
    color: Tuple[int, int, int, int],
    width: int,
) -> None:
    length = max(float(region.size[0]) * 0.38, 1.8)
    tip = (
        float(region.center[0]) + length * math.cos(float(region.yaw)),
        float(region.center[1]) + length * math.sin(float(region.yaw)),
    )
    start = _project_world_point(tuple(region.center), layout)
    end = _project_world_point(tip, layout)
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=max(int(width), 1))
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = max(7.0, 0.018 * float(layout["width"]))
    left = (
        end[0] - head_len * math.cos(angle - math.pi / 7.0),
        end[1] - head_len * math.sin(angle - math.pi / 7.0),
    )
    right = (
        end[0] - head_len * math.cos(angle + math.pi / 7.0),
        end[1] - head_len * math.sin(angle + math.pi / 7.0),
    )
    draw.polygon([end, left, right], fill=color)


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


def _render_region_overview_frame(
    *,
    step: CanonicalStepRecord,
    selected_queries: Sequence[str],
    canvas_size: Tuple[int, int],
    world_bounds: Mapping[str, float],
    highlight_top_k: int,
) -> Image.Image:
    width = int(canvas_size[0])
    height = int(canvas_size[1])
    image = Image.new("RGBA", (width, height), color=_BACKGROUND + (255,))
    draw = ImageDraw.Draw(image)

    padding = 28
    header_h = 86
    footer_h = 22
    body_y = padding + header_h
    body_h = max(height - body_y - padding - footer_h, 1)
    gap = 24
    right_w = min(max(int(width * 0.36), 520), width - 360)
    left_w = max(width - 2 * padding - gap - right_w, 320)
    left_rect = (padding, body_y, left_w, body_h)
    right_rect = (padding + left_w + gap, body_y, right_w, body_h)
    main_layout = _make_world_layout(left_rect, world_bounds, margin=36.0)

    draw.text(
        (padding, padding),
        f"Region Overview | step {int(step.step):03d} | scene={step.scene_type}",
        fill=_TITLE,
        font=_load_font(28),
    )
    draw.text(
        (padding, padding + 36),
        f"episode={step.episode_id} | vehicles={len(step.candidate_vehicles)} | queries={len(selected_queries)}",
        fill=_SUBTITLE,
        font=_load_font(16),
    )

    _draw_panel_background(draw, left_rect, radius=20, fill=_MAIN_PANEL_BACKGROUND, outline=_CARD_BORDER)
    _draw_reference_grid(draw, main_layout, show_scale=True)
    _draw_main_region_scene(
        draw,
        step=step,
        layout=main_layout,
        selected_queries=selected_queries,
        highlight_top_k=highlight_top_k,
    )

    _draw_query_cards(
        image,
        right_rect=right_rect,
        step=step,
        selected_queries=selected_queries,
        world_bounds=world_bounds,
    )

    draw.text(
        (padding, height - padding - 14),
        "All geometry is shown in the ego-centered BEV frame. Query cards share the same world extent.",
        fill=_MUTED,
        font=_load_font(13),
    )
    return image.convert("RGB")


def _draw_main_region_scene(
    draw: ImageDraw.ImageDraw,
    *,
    step: CanonicalStepRecord,
    layout: Mapping[str, float],
    selected_queries: Sequence[str],
    highlight_top_k: int,
) -> None:
    x0 = int(layout["x0"])
    y0 = int(layout["y0"])
    draw.text((x0 + 18, y0 + 14), "Observable Regions", fill=_TITLE, font=_load_font(22))
    draw.text(
        (x0 + 18, y0 + 44),
        "Ego region in red, member regions in blue, top collaborative members emphasized.",
        fill=_SUBTITLE,
        font=_load_font(15),
    )
    _draw_region_box(
        draw,
        region=step.ego_state.observable_region,
        layout=layout,
        fill=_EGO_REGION_FILL,
        outline=_EGO_REGION_OUTLINE,
        width=4,
    )
    _draw_heading_arrow(
        draw,
        region=step.ego_state.observable_region,
        layout=layout,
        color=_EGO_REGION_OUTLINE,
        width=4,
    )
    ego_center = _project_world_point((0.0, 0.0), layout)
    _draw_node(
        draw,
        ego_center,
        radius=12,
        fill=_EGO_FILL,
        outline=_EGO_OUTLINE,
        label="ego",
    )

    scores = _mean_sender_collab_by_vehicle(step, selected_queries)
    highlight_ids = {
        vehicle_id
        for vehicle_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[: max(int(highlight_top_k), 0)]
    }
    legend_x = x0 + 18
    legend_y = y0 + int(layout["height"]) - 72
    _draw_region_legend(draw, (legend_x, legend_y))

    for vehicle in step.candidate_vehicles:
        vehicle_id = int(vehicle.vehicle_id)
        is_highlight = vehicle_id in highlight_ids
        _draw_region_box(
            draw,
            region=vehicle.observable_region,
            layout=layout,
            fill=_MEMBER_REGION_HIGHLIGHT_FILL if is_highlight else _MEMBER_REGION_FILL,
            outline=_MEMBER_REGION_HIGHLIGHT_OUTLINE if is_highlight else _MEMBER_REGION_OUTLINE,
            width=4 if is_highlight else 2,
        )
        _draw_heading_arrow(
            draw,
            region=vehicle.observable_region,
            layout=layout,
            color=_MEMBER_REGION_HIGHLIGHT_OUTLINE if is_highlight else _MEMBER_REGION_OUTLINE,
            width=3 if is_highlight else 2,
        )
        center = _project_world_point(tuple(vehicle.delta_pos), layout)
        _draw_node(
            draw,
            center,
            radius=10,
            fill=_MEMBER_FILL,
            outline=_MEMBER_OUTLINE,
            label=f"v{vehicle_id}",
        )
        if is_highlight:
            badge_text = f"{scores.get(vehicle_id, 0.0):.3f}"
            _draw_query_badge(
                draw,
                rect=(int(center[0] + 10), int(center[1] - 10), 56, 22),
                text=badge_text,
            )


def _draw_query_cards(
    image: Image.Image,
    *,
    right_rect: Tuple[int, int, int, int],
    step: CanonicalStepRecord,
    selected_queries: Sequence[str],
    world_bounds: Mapping[str, float],
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, width, height = right_rect
    draw.text((x0, y0 - 34), "Query-conditioned Required Regions", fill=_TITLE, font=_load_font(22))
    draw.text(
        (x0, y0 - 10),
        "Each card highlights one required region and lists per-member metrics.",
        fill=_SUBTITLE,
        font=_load_font(15),
    )

    query_map = {str(query.query_id): query for query in step.queries}
    query_records = [query_map[query_id] for query_id in selected_queries if query_id in query_map]
    if not query_records:
        return
    cols = 2 if len(query_records) > 1 else 1
    rows = int(math.ceil(len(query_records) / float(cols)))
    gap = 14
    card_w = int((width - gap * (cols - 1)) / cols)
    card_h = int((height - gap * (rows - 1)) / rows)
    for index, query in enumerate(query_records):
        row = index // cols
        col = index % cols
        card_rect = (
            x0 + col * (card_w + gap),
            y0 + row * (card_h + gap),
            card_w,
            card_h,
        )
        _draw_query_card(image, step=step, query=query, card_rect=card_rect, world_bounds=world_bounds)


def _draw_query_card(
    image: Image.Image,
    *,
    step: CanonicalStepRecord,
    query: QueryRecord,
    card_rect: Tuple[int, int, int, int],
    world_bounds: Mapping[str, float],
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, width, height = card_rect
    _draw_panel_background(draw, card_rect, radius=18, fill=_CARD_BACKGROUND, outline=_CARD_BORDER)
    title_font = _load_font(16)
    body_font = _load_font(13)
    draw.text((x0 + 14, y0 + 12), str(query.query_id), fill=_TITLE, font=title_font)
    draw.text(
        (x0 + 14, y0 + 34),
        f"required center=({query.required_region.center[0]:.1f}, {query.required_region.center[1]:.1f})",
        fill=_SUBTITLE,
        font=body_font,
    )

    inner_y = y0 + 58
    inner_h = max(height - 72, 1)
    mini_w = int(width * 0.42)
    mini_rect = (x0 + 12, inner_y, mini_w, inner_h)
    table_rect = (x0 + mini_w + 20, inner_y, width - mini_w - 32, inner_h)
    mini_layout = _make_world_layout(mini_rect, world_bounds, margin=18.0)

    _draw_reference_grid(draw, mini_layout, show_scale=False, draw_labels=False)
    _draw_region_box(
        draw,
        region=step.ego_state.observable_region,
        layout=mini_layout,
        fill=(196, 59, 51, 28),
        outline=(128, 29, 23, 180),
        width=2,
    )
    _draw_region_box(
        draw,
        region=query.required_region,
        layout=mini_layout,
        fill=_REQUIRED_REGION_FILL,
        outline=_REQUIRED_REGION_OUTLINE,
        width=3,
    )
    _draw_heading_arrow(
        draw,
        region=step.ego_state.observable_region,
        layout=mini_layout,
        color=(128, 29, 23, 180),
        width=2,
    )
    top_vehicle_id = _top_sender_for_query(step, str(query.query_id))
    for vehicle in step.candidate_vehicles:
        vehicle_id = int(vehicle.vehicle_id)
        is_top = vehicle_id == top_vehicle_id
        _draw_region_box(
            draw,
            region=vehicle.observable_region,
            layout=mini_layout,
            fill=_MEMBER_REGION_HIGHLIGHT_FILL if is_top else (112, 139, 191, 18),
            outline=_MEMBER_REGION_HIGHLIGHT_OUTLINE if is_top else (102, 129, 176, 120),
            width=3 if is_top else 1,
        )
        center = _project_world_point(tuple(vehicle.delta_pos), mini_layout)
        draw.ellipse((center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4), fill=_MEMBER_FILL)
    _draw_query_badge(draw, rect=(mini_rect[0] + 10, mini_rect[1] + 10, 74, 24), text="need")
    _draw_query_metrics_table(draw, rect=table_rect, step=step, query_id=str(query.query_id))


def _draw_query_metrics_table(
    draw: ImageDraw.ImageDraw,
    *,
    rect: Tuple[int, int, int, int],
    step: CanonicalStepRecord,
    query_id: str,
) -> None:
    x0, y0, width, height = rect
    rows = sorted(
        step.candidate_vehicles,
        key=lambda vehicle: float(vehicle.sender_collab.get(query_id, 0.0)),
        reverse=True,
    )
    if not rows:
        draw.text((x0, y0 + 8), "No candidates", fill=_SUBTITLE, font=_load_font(13))
        return

    header_h = 26
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=12, outline=(233, 236, 242, 255), width=1)
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + header_h), radius=12, fill=_TABLE_HEADER_FILL)
    col_fracs = [0.16, 0.16, 0.16, 0.16, 0.36]
    headers = ["veh", "comp", "acc", "rel", "collab"]
    col_x = [x0]
    cursor = float(x0)
    for frac in col_fracs[:-1]:
        cursor += float(width) * frac
        col_x.append(int(round(cursor)))
    header_font = _load_font(11)
    text_font = _load_font(11)
    for index, label in enumerate(headers):
        cell_x = col_x[index] + 8
        draw.text((cell_x, y0 + 7), label, fill=_TITLE, font=header_font)

    row_h = max(int((height - header_h - 8) / max(len(rows), 1)), 18)
    for row_index, vehicle in enumerate(rows):
        y = y0 + header_h + row_index * row_h
        if y + row_h > y0 + height:
            break
        if row_index % 2 == 0:
            draw.rectangle((x0 + 1, y, x0 + width - 1, y + row_h), fill=(250, 251, 254, 255))
        values = [
            f"v{int(vehicle.vehicle_id)}",
            f"{float(vehicle.complementarity):.3f}",
            f"{float(vehicle.accessibility):.3f}",
            f"{float(vehicle.query_task_relevance.get(query_id, 0.0)):.3f}",
            f"{float(vehicle.sender_collab.get(query_id, 0.0)):.3f}",
        ]
        for index, value in enumerate(values):
            draw.text((col_x[index] + 8, y + 4), value, fill=_TEXT, font=text_font)


def _draw_panel_background(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    *,
    radius: int,
    fill: Tuple[int, int, int, int],
    outline: Tuple[int, int, int, int],
) -> None:
    x0, y0, width, height = rect
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=radius, fill=fill, outline=outline, width=1)


def _draw_reference_grid(
    draw: ImageDraw.ImageDraw,
    layout: Mapping[str, float],
    *,
    show_scale: bool,
    draw_labels: bool = True,
) -> None:
    x0 = float(layout["x0"])
    y0 = float(layout["y0"])
    width = float(layout["width"])
    height = float(layout["height"])
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=18, outline=_CARD_BORDER, width=1)
    right_ticks = _grid_ticks(float(layout["min_right"]), float(layout["max_right"]), spacing=5.0)
    forward_ticks = _grid_ticks(float(layout["min_forward"]), float(layout["max_forward"]), spacing=5.0)
    for right in right_ticks:
        px = _project_world_point((0.0, right), layout)[0]
        draw.line((px, y0 + 1, px, y0 + height - 1), fill=(237, 240, 246, 255), width=1)
        if draw_labels:
            draw.text((px + 2, y0 + height - 18), f"{int(right)}", fill=_MUTED, font=_load_font(11))
    for forward in forward_ticks:
        py = _project_world_point((forward, 0.0), layout)[1]
        draw.line((x0 + 1, py, x0 + width - 1, py), fill=(237, 240, 246, 255), width=1)
        if draw_labels:
            draw.text((x0 + 6, py - 12), f"{int(forward)}", fill=_MUTED, font=_load_font(11))
    px0, py0 = _project_world_point((0.0, 0.0), layout)
    draw.line((x0 + 1, py0, x0 + width - 1, py0), fill=(200, 207, 220, 255), width=2)
    draw.line((px0, y0 + 1, px0, y0 + height - 1), fill=(200, 207, 220, 255), width=2)
    if draw_labels:
        draw.text((px0 + 8, y0 + 8), "ego axes", fill=_SUBTITLE, font=_load_font(12))
    if show_scale:
        bar_w = 5.0 * float(layout["scale"])
        bar_x = x0 + 18
        bar_y = y0 + height - 24
        draw.line((bar_x, bar_y, bar_x + bar_w, bar_y), fill=_TEXT, width=3)
        draw.line((bar_x, bar_y - 4, bar_x, bar_y + 4), fill=_TEXT, width=2)
        draw.line((bar_x + bar_w, bar_y - 4, bar_x + bar_w, bar_y + 4), fill=_TEXT, width=2)
        draw.text((bar_x + bar_w + 10, bar_y - 9), "5 m", fill=_TEXT, font=_load_font(12))


def _draw_region_legend(
    draw: ImageDraw.ImageDraw,
    origin: Tuple[int, int],
) -> None:
    x0, y0 = int(origin[0]), int(origin[1])
    items = [
        ("ego region", _EGO_REGION_OUTLINE),
        ("member region", _MEMBER_REGION_OUTLINE),
        ("highlighted member", _MEMBER_REGION_HIGHLIGHT_OUTLINE),
    ]
    for index, (label, color) in enumerate(items):
        y = y0 + index * 18
        draw.line((x0, y + 7, x0 + 18, y + 7), fill=color, width=3)
        draw.text((x0 + 26, y), label, fill=_TEXT, font=_load_font(12))


def _draw_query_badge(
    draw: ImageDraw.ImageDraw,
    *,
    rect: Tuple[int, int, int, int],
    text: str,
) -> None:
    x0, y0, width, height = rect
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=8, fill=_TOP_BADGE_FILL, outline=_TOP_BADGE_OUTLINE)
    bbox = draw.textbbox((0, 0), text, font=_load_font(12))
    draw.text(
        (x0 + 0.5 * (width - (bbox[2] - bbox[0])), y0 + 0.5 * (height - (bbox[3] - bbox[1])) - 1),
        text,
        fill=_TEXT,
        font=_load_font(12),
    )


def _mean_sender_collab_by_vehicle(
    step: CanonicalStepRecord,
    query_ids: Sequence[str],
) -> Dict[int, float]:
    means: Dict[int, float] = {}
    for vehicle in step.candidate_vehicles:
        values = [float(vehicle.sender_collab.get(query_id, 0.0)) for query_id in query_ids]
        means[int(vehicle.vehicle_id)] = float(sum(values) / len(values)) if values else 0.0
    return means


def _top_sender_for_query(step: CanonicalStepRecord, query_id: str) -> int | None:
    if not step.candidate_vehicles:
        return None
    top = max(step.candidate_vehicles, key=lambda vehicle: float(vehicle.sender_collab.get(query_id, 0.0)))
    return int(top.vehicle_id)


def _grid_ticks(start: float, end: float, *, spacing: float) -> List[float]:
    tick_start = math.floor(start / spacing) * spacing
    tick_end = math.ceil(end / spacing) * spacing
    values: List[float] = []
    current = tick_start
    while current <= tick_end + 1e-6:
        values.append(float(current))
        current += spacing
    return values


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
    # delta_pos[0] = forward (+x = up), delta_pos[1] = right (+y = screen right)
    forward, right = float(position[0]), float(position[1])
    return (
        float(layout["center_x"]) + right * float(layout["scale"]),
        float(layout["center_y"]) - forward * float(layout["scale"]),
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


@functools.lru_cache(maxsize=32)
def _load_font(size: int = 12):
    try:
        return ImageFont.truetype(DEFAULT_FONT_PATH, size=max(int(size), 8))
    except OSError:
        return ImageFont.load_default()


__all__ = [
    "SUPPORTED_METRICS",
    "build_arg_parser",
    "main",
    "render_ground_truth_topology_sequences",
    "render_region_overview_sequences",
    "render_prediction_comparison_sequences",
]
