from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .adapter_vlm import adapt_vlm_records_to_canonical_episode
from .dataset import CanonicalEmulationDataset
from .model import (
    GraphGRUEmulationConfig,
    GraphGRUEmulationModel,
    compute_emulation_loss,
    torch_is_available,
)
from .schema import CanonicalEpisodeRecord, episode_from_dict, validate_episode_record
from .synthetic import generate_synthetic_canonical_episode

try:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Subset
except ImportError as exc:  # pragma: no cover - guarded at runtime
    torch = None
    AdamW = None
    DataLoader = None
    Subset = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class EpisodeSource:
    path: str
    scene_type: str = "right_turn"
    dt: float = 0.1


@dataclass
class EmulationTrainingConfig:
    data: List[str] = field(default_factory=list)
    scene_type: str = "right_turn"
    dt: float = 0.1
    history_len: int = 8
    horizon: int = 5
    batch_size: int = 8
    max_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    hidden_dim: int = 64
    num_graph_layers: int = 2
    dropout: float = 0.0
    val_ratio: float = 0.2
    seed: int = 0
    device: str = "auto"
    save_dir: str = "logdir/emulation"
    num_workers: int = 0
    gradient_clip_norm: float = 5.0
    scheduler: str = "cosine"
    min_learning_rate: float = 1e-5
    loss_type: str = "huber"
    huber_delta: float = 1.0
    sender_collab_weight: float = 1.0
    sender_gain_weight: float = 1.0
    ego_sc_weight: float = 1.0
    shuffle: bool = True
    report_every: int = 10
    from_checkpoint: str = ""
    synthetic_episodes: int = 0
    synthetic_scene_type: str = "right_turn"
    synthetic_num_steps: int = 24
    synthetic_num_vehicles: int = 4
    max_nodes: int | None = None
    max_queries: int | None = None


def parse_episode_source_spec(
    spec: str,
    *,
    default_scene_type: str = "right_turn",
    default_dt: float = 0.1,
) -> EpisodeSource:
    parts = [part.strip() for part in str(spec).split("::")]
    if len(parts) == 1:
        return EpisodeSource(path=parts[0], scene_type=default_scene_type, dt=float(default_dt))
    if len(parts) == 2:
        return EpisodeSource(path=parts[0], scene_type=parts[1] or default_scene_type, dt=float(default_dt))
    if len(parts) == 3:
        return EpisodeSource(
            path=parts[0],
            scene_type=parts[1] or default_scene_type,
            dt=float(parts[2]) if parts[2] else float(default_dt),
        )
    raise ValueError(
        "Episode source specs must be 'path', 'path::scene_type', or 'path::scene_type::dt'."
    )


def resolve_device(device: str = "auto") -> str:
    if not torch_is_available():
        return "cpu"
    requested = str(device).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def load_episode_from_path(path: str | Path, *, scene_type: str = "right_turn", dt: float = 0.1) -> CanonicalEpisodeRecord:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if _looks_like_canonical_episode(payload):
        episode = episode_from_dict(payload)
    else:
        episode = adapt_vlm_records_to_canonical_episode(path, scene_type=scene_type, dt=dt)
    validate_episode_record(episode)
    return episode


def load_episodes_from_sources(
    sources: Sequence[EpisodeSource],
    *,
    synthetic_episodes: int = 0,
    synthetic_scene_type: str = "right_turn",
    synthetic_num_steps: int = 24,
    synthetic_num_vehicles: int = 4,
    seed: int = 0,
) -> List[CanonicalEpisodeRecord]:
    episodes = [
        load_episode_from_path(source.path, scene_type=source.scene_type, dt=source.dt)
        for source in sources
    ]
    for index in range(int(synthetic_episodes)):
        episodes.append(
            generate_synthetic_canonical_episode(
                scene_type=synthetic_scene_type,
                scene_id=f"synthetic_scene_{index}",
                episode_id=f"synthetic_episode_{index}",
                num_steps=int(synthetic_num_steps),
                num_vehicles=int(synthetic_num_vehicles),
                dt=0.1,
                seed=int(seed) + index,
            )
        )
    if not episodes:
        raise ValueError("No training episodes were loaded.")
    return episodes


def split_episode_indices(num_episodes: int, val_ratio: float = 0.2, seed: int = 0) -> Tuple[List[int], List[int]]:
    if int(num_episodes) <= 0:
        raise ValueError("num_episodes must be positive.")
    indices = list(range(int(num_episodes)))
    random.Random(int(seed)).shuffle(indices)
    if len(indices) == 1:
        return indices, []
    desired_val = int(round(float(val_ratio) * len(indices))) if float(val_ratio) > 0 else 0
    desired_val = max(desired_val, 1) if float(val_ratio) > 0 else 0
    val_count = min(desired_val, len(indices) - 1)
    if val_count <= 0:
        return indices, []
    return indices[val_count:], indices[:val_count]


def build_dataset_splits(
    episodes: Sequence[CanonicalEpisodeRecord],
    config: EmulationTrainingConfig,
) -> Tuple[CanonicalEmulationDataset, CanonicalEmulationDataset | None, List[int], List[int]]:
    if not episodes:
        raise ValueError("At least one episode is required to build dataset splits.")
    train_indices, val_indices = split_episode_indices(
        len(episodes),
        val_ratio=float(config.val_ratio),
        seed=int(config.seed),
    )
    max_nodes = int(config.max_nodes or max(len({vehicle.vehicle_id for step in ep.steps for vehicle in step.candidate_vehicles}) for ep in episodes))
    max_queries = int(config.max_queries or max(len(ep.steps[0].queries) for ep in episodes))
    train_dataset = CanonicalEmulationDataset(
        [episodes[idx] for idx in train_indices],
        history_len=int(config.history_len),
        horizon=int(config.horizon),
        max_nodes=max_nodes,
        max_queries=max_queries,
    )
    val_dataset = None
    if val_indices:
        val_dataset = CanonicalEmulationDataset(
            [episodes[idx] for idx in val_indices],
            history_len=int(config.history_len),
            horizon=int(config.horizon),
            max_nodes=max_nodes,
            max_queries=max_queries,
        )
    return train_dataset, val_dataset, train_indices, val_indices


def emulation_collate_fn(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _require_torch("emulation_collate_fn")
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    tensor_keys = (
        "node_ids",
        "history_mask",
        "node_features",
        "component_valid_mask",
        "node_mask",
        "edge_index",
        "edge_attr",
        "edge_mask",
        "query_features",
        "query_mask",
        "task_relevance",
        "future_mask",
        "future_node_mask",
        "target_sender_collab",
        "target_sender_gain",
        "target_ego_sc",
    )
    batch: Dict[str, Any] = {
        key: torch.stack([torch.as_tensor(sample[key]) for sample in samples], dim=0)
        for key in tensor_keys
    }
    batch["step"] = torch.as_tensor([int(sample["step"]) for sample in samples], dtype=torch.long)
    batch["episode_id"] = [str(sample["episode_id"]) for sample in samples]
    batch["scene_id"] = [str(sample["scene_id"]) for sample in samples]
    batch["scene_type"] = [str(sample["scene_type"]) for sample in samples]
    batch["query_ids"] = [list(sample["query_ids"]) for sample in samples]
    return batch


def build_dataloaders(
    train_dataset: CanonicalEmulationDataset,
    val_dataset: CanonicalEmulationDataset | None,
    config: EmulationTrainingConfig,
):
    _require_torch("build_dataloaders")
    train_indices = _nonempty_future_indices(train_dataset)
    if not train_indices:
        raise ValueError("Training dataset has no steps with future supervision.")
    train_subset = Subset(train_dataset, train_indices)
    pin_memory = resolve_device(config.device) == "cuda"
    train_loader = DataLoader(
        train_subset,
        batch_size=int(config.batch_size),
        shuffle=bool(config.shuffle),
        num_workers=int(config.num_workers),
        pin_memory=pin_memory,
        collate_fn=emulation_collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_indices = _nonempty_future_indices(val_dataset)
        if val_indices:
            val_loader = DataLoader(
                Subset(val_dataset, val_indices),
                batch_size=int(config.batch_size),
                shuffle=False,
                num_workers=int(config.num_workers),
                pin_memory=pin_memory,
                collate_fn=emulation_collate_fn,
            )
    return train_loader, val_loader


def make_model_from_dataset(
    dataset: CanonicalEmulationDataset,
    config: EmulationTrainingConfig,
) -> Tuple[GraphGRUEmulationModel, GraphGRUEmulationConfig]:
    _require_torch("make_model_from_dataset")
    sample = dataset[0]
    model_config = GraphGRUEmulationConfig(
        node_dim=int(sample["node_features"].shape[-1]),
        query_dim=int(sample["query_features"].shape[-1]),
        edge_attr_dim=int(sample["edge_attr"].shape[-1]),
        hidden_dim=int(config.hidden_dim),
        num_graph_layers=int(config.num_graph_layers),
        history_len=int(config.history_len),
        horizon=int(config.horizon),
        dropout=float(config.dropout),
    )
    return GraphGRUEmulationModel(model_config), model_config


def fit_emulation_model(
    config: EmulationTrainingConfig,
    *,
    episodes: Sequence[CanonicalEpisodeRecord] | None = None,
) -> Dict[str, Any]:
    _require_torch("fit_emulation_model")
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "train_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if episodes is None:
        sources = [
            parse_episode_source_spec(item, default_scene_type=config.scene_type, default_dt=config.dt)
            for item in config.data
        ]
        episodes = load_episodes_from_sources(
            sources,
            synthetic_episodes=int(config.synthetic_episodes),
            synthetic_scene_type=str(config.synthetic_scene_type),
            synthetic_num_steps=int(config.synthetic_num_steps),
            synthetic_num_vehicles=int(config.synthetic_num_vehicles),
            seed=int(config.seed),
        )
    else:
        episodes = list(episodes)

    train_dataset, val_dataset, train_episode_indices, val_episode_indices = build_dataset_splits(
        episodes,
        config,
    )
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, config)

    model, model_config = make_model_from_dataset(train_dataset, config)
    device = resolve_device(config.device)
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = _build_scheduler(optimizer, config)

    latest_path = save_dir / "checkpoint_latest.pt"
    best_path = save_dir / "checkpoint_best.pt"
    history_path = save_dir / "history.jsonl"
    start_epoch = 0
    global_step = 0
    best_metric = math.inf
    history: List[Dict[str, Any]] = []

    if config.from_checkpoint:
        payload = load_training_checkpoint(
            config.from_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        best_metric = float(payload.get("best_metric", math.inf))
        history = list(payload.get("history", []))

    model_config_path = save_dir / "model_config.json"
    model_config_path.write_text(
        json.dumps(asdict(model_config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for epoch in range(start_epoch, int(config.max_epochs)):
        train_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            config=config,
            global_step=global_step,
        )
        if scheduler is not None:
            scheduler.step()
        val_metrics = (
            evaluate_emulation_model(model, val_loader, device=device, config=config)
            if val_loader is not None
            else {}
        )
        epoch_record = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")

        monitored_metric = float(
            val_metrics.get("loss", train_metrics.get("loss", math.inf))
        )
        save_training_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_metric=min(best_metric, monitored_metric),
            train_config=config,
            model_config=model_config,
            history=history,
        )
        if monitored_metric < best_metric:
            best_metric = monitored_metric
            save_training_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                train_config=config,
                model_config=model_config,
                history=history,
            )

        print(
            f"[emulation][epoch {epoch:03d}] "
            f"train_loss={train_metrics.get('loss', math.nan):.6f} "
            f"val_loss={val_metrics.get('loss', math.nan):.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )

    summary = {
        "save_dir": str(save_dir),
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "history_path": str(history_path),
        "best_metric": float(best_metric),
        "num_episodes": len(episodes),
        "train_episode_indices": train_episode_indices,
        "val_episode_indices": val_episode_indices,
        "device": device,
    }
    (save_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def train_one_epoch(
    model: GraphGRUEmulationModel,
    loader,
    optimizer,
    *,
    device: str,
    config: EmulationTrainingConfig,
    global_step: int = 0,
) -> Tuple[Dict[str, float], int]:
    _require_torch("train_one_epoch")
    model.train()
    metric_sums: Dict[str, float] = {}
    num_batches = 0
    for batch_index, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch)
        losses = compute_emulation_loss(
            predictions,
            batch,
            loss_type=str(config.loss_type),
            delta=float(config.huber_delta),
            sender_collab_weight=float(config.sender_collab_weight),
            sender_gain_weight=float(config.sender_gain_weight),
            ego_sc_weight=float(config.ego_sc_weight),
        )
        losses["loss"].backward()
        if float(config.gradient_clip_norm) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(config.gradient_clip_norm),
            )
            metric_sums["grad_norm"] = metric_sums.get("grad_norm", 0.0) + float(grad_norm)
        optimizer.step()

        batch_metrics = {key: float(value.detach().cpu().item()) for key, value in losses.items()}
        for key, value in batch_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value
        num_batches += 1
        global_step += 1

        if int(config.report_every) > 0 and (batch_index + 1) % int(config.report_every) == 0:
            print(
                f"[emulation][train] batch={batch_index + 1} "
                f"loss={batch_metrics['loss']:.6f} "
                f"collab={batch_metrics['sender_collab_loss']:.6f} "
                f"gain={batch_metrics['sender_gain_loss']:.6f} "
                f"ego={batch_metrics['ego_sc_loss']:.6f}"
            )

    return _finalize_metrics(metric_sums, num_batches), global_step


def evaluate_emulation_model(
    model: GraphGRUEmulationModel,
    loader,
    *,
    device: str,
    config: EmulationTrainingConfig,
) -> Dict[str, float]:
    _require_torch("evaluate_emulation_model")
    model.eval()
    metric_sums: Dict[str, float] = {}
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            predictions = model(batch)
            losses = compute_emulation_loss(
                predictions,
                batch,
                loss_type=str(config.loss_type),
                delta=float(config.huber_delta),
                sender_collab_weight=float(config.sender_collab_weight),
                sender_gain_weight=float(config.sender_gain_weight),
                ego_sc_weight=float(config.ego_sc_weight),
            )
            batch_metrics = {key: float(value.detach().cpu().item()) for key, value in losses.items()}
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value
            num_batches += 1
    return _finalize_metrics(metric_sums, num_batches)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: GraphGRUEmulationModel,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_metric: float,
    train_config: EmulationTrainingConfig,
    model_config: GraphGRUEmulationConfig,
    history: Sequence[Mapping[str, Any]],
) -> None:
    _require_torch("save_training_checkpoint")
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "train_config": asdict(train_config),
        "model_config": asdict(model_config),
        "history": list(history),
    }
    torch.save(payload, str(path))


def load_training_checkpoint(
    path: str | Path,
    *,
    model: GraphGRUEmulationModel,
    optimizer=None,
    scheduler=None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    _require_torch("load_training_checkpoint")
    payload = torch.load(str(path), map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload


def move_batch_to_device(batch: Mapping[str, Any], device: str) -> Dict[str, Any]:
    _require_torch("move_batch_to_device")
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=(device == "cuda"))
        else:
            moved[key] = value
    return moved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Graph+GRU emulation predictor.")
    parser.add_argument(
        "--data",
        nargs="*",
        default=[],
        help="Episode sources: path, path::scene_type, or path::scene_type::dt",
    )
    parser.add_argument("--scene-type", default="right_turn")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="logdir/emulation")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--loss-type", default="huber", choices=["huber", "mse"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--sender-collab-weight", type=float, default=1.0)
    parser.add_argument("--sender-gain-weight", type=float, default=1.0)
    parser.add_argument("--ego-sc-weight", type=float, default=1.0)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--from-checkpoint", default="")
    parser.add_argument("--synthetic-episodes", type=int, default=0)
    parser.add_argument("--synthetic-scene-type", default="right_turn")
    parser.add_argument("--synthetic-num-steps", type=int, default=24)
    parser.add_argument("--synthetic-num-vehicles", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = EmulationTrainingConfig(
        data=list(args.data),
        scene_type=str(args.scene_type),
        dt=float(args.dt),
        history_len=int(args.history_len),
        horizon=int(args.horizon),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        num_graph_layers=int(args.num_graph_layers),
        dropout=float(args.dropout),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        device=str(args.device),
        save_dir=str(args.save_dir),
        num_workers=int(args.num_workers),
        gradient_clip_norm=float(args.gradient_clip_norm),
        scheduler=str(args.scheduler),
        min_learning_rate=float(args.min_learning_rate),
        loss_type=str(args.loss_type),
        huber_delta=float(args.huber_delta),
        sender_collab_weight=float(args.sender_collab_weight),
        sender_gain_weight=float(args.sender_gain_weight),
        ego_sc_weight=float(args.ego_sc_weight),
        report_every=int(args.report_every),
        from_checkpoint=str(args.from_checkpoint),
        synthetic_episodes=int(args.synthetic_episodes),
        synthetic_scene_type=str(args.synthetic_scene_type),
        synthetic_num_steps=int(args.synthetic_num_steps),
        synthetic_num_vehicles=int(args.synthetic_num_vehicles),
    )
    return fit_emulation_model(config)


def _build_scheduler(optimizer, config: EmulationTrainingConfig):
    _require_torch("_build_scheduler")
    if str(config.scheduler).lower() == "none":
        return None
    t_max = max(int(config.max_epochs), 1)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=float(config.min_learning_rate),
    )


def _finalize_metrics(metric_sums: Mapping[str, float], num_batches: int) -> Dict[str, float]:
    if num_batches <= 0:
        return {}
    return {key: float(value) / float(num_batches) for key, value in metric_sums.items()}


def _looks_like_canonical_episode(payload: Any) -> bool:
    return isinstance(payload, dict) and "steps" in payload and "scene_id" in payload and "episode_id" in payload


def _nonempty_future_indices(dataset: CanonicalEmulationDataset) -> List[int]:
    indices: List[int] = []
    cursor = 0
    for episode in dataset.episodes:
        usable_steps = max(len(episode.steps) - 1, 0)
        indices.extend(range(cursor, cursor + usable_steps))
        cursor += len(episode.steps)
    return indices


def _require_torch(function_name: str) -> None:
    if not torch_is_available():
        raise ImportError(
            f"{function_name} requires PyTorch, but torch is not installed in this environment."
        ) from _TORCH_IMPORT_ERROR


if __name__ == "__main__":  # pragma: no cover
    main()
