from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

DEFAULT_RUNTIME_LOGGING_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "level": "INFO",
    "console_level": "INFO",
    "file_level": "INFO",
    "console": True,
    "file": True,
    "console_key_only": True,
    "step_debug_interval": 100,
    "driver_debug_shapes": False,
}

_RUNTIME_LOGGING_CONFIG: Dict[str, Any] = DEFAULT_RUNTIME_LOGGING_CONFIG.copy()
_HANDLER_MARKER = "_cardreamer_runtime_handler"


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def normalize_runtime_logging_config(cfg: Any = None) -> Dict[str, Any]:
    return {
        "enabled": bool(_cfg_get(cfg, "enabled", DEFAULT_RUNTIME_LOGGING_CONFIG["enabled"])),
        "level": str(_cfg_get(cfg, "level", DEFAULT_RUNTIME_LOGGING_CONFIG["level"])).upper(),
        "console_level": str(
            _cfg_get(cfg, "console_level", _cfg_get(cfg, "level", DEFAULT_RUNTIME_LOGGING_CONFIG["console_level"]))
        ).upper(),
        "file_level": str(
            _cfg_get(cfg, "file_level", _cfg_get(cfg, "level", DEFAULT_RUNTIME_LOGGING_CONFIG["file_level"]))
        ).upper(),
        "console": bool(_cfg_get(cfg, "console", DEFAULT_RUNTIME_LOGGING_CONFIG["console"])),
        "file": bool(_cfg_get(cfg, "file", DEFAULT_RUNTIME_LOGGING_CONFIG["file"])),
        "console_key_only": bool(
            _cfg_get(cfg, "console_key_only", DEFAULT_RUNTIME_LOGGING_CONFIG["console_key_only"])
        ),
        "step_debug_interval": int(
            _cfg_get(
                cfg,
                "step_debug_interval",
                DEFAULT_RUNTIME_LOGGING_CONFIG["step_debug_interval"],
            )
        ),
        "driver_debug_shapes": bool(
            _cfg_get(
                cfg,
                "driver_debug_shapes",
                DEFAULT_RUNTIME_LOGGING_CONFIG["driver_debug_shapes"],
            )
        ),
    }


def get_runtime_logging_config() -> Dict[str, Any]:
    return dict(_RUNTIME_LOGGING_CONFIG)


def get_runtime_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class _ConsoleKeyFilter(logging.Filter):
    def __init__(self, key_only: bool):
        super().__init__()
        self._key_only = key_only

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._key_only:
            return True
        if record.levelno >= logging.WARNING:
            return True
        return bool(getattr(record, "console", False))


def configure_runtime_logging(cfg: Any = None, logdir: Optional[str] = None) -> Dict[str, Any]:
    global _RUNTIME_LOGGING_CONFIG

    settings = normalize_runtime_logging_config(cfg)
    print(f"Configuring runtime logging: {settings}")
    _RUNTIME_LOGGING_CONFIG = settings

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)
            handler.close()

    if not settings["enabled"]:
        return settings

    console_level = getattr(logging, settings["console_level"], logging.INFO)
    file_level = getattr(logging, settings["file_level"], logging.INFO)
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if settings["console"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(_ConsoleKeyFilter(settings["console_key_only"]))
        setattr(console_handler, _HANDLER_MARKER, True)
        root.addHandler(console_handler)

    if settings["file"] and logdir:
        runtime_log_path = Path(str(logdir)) / "runtime.log"
        runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(runtime_log_path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARKER, True)
        root.addHandler(file_handler)

    logging.captureWarnings(True)
    return settings


def should_log_periodic(step: int, interval: int, logger: Optional[logging.Logger] = None, level: int = logging.DEBUG) -> bool:
    if interval <= 0:
        return False
    if logger is not None and not logger.isEnabledFor(level):
        return False
    if step < 0:
        return False
    return step == 0 or (step % interval == 0)


def log_key_event(logger: logging.Logger, level: int, message: str, *args: Any, **kwargs: Any) -> None:
    extra = dict(kwargs.pop("extra", {}) or {})
    extra["console"] = True
    logger.log(level, message, *args, extra=extra, **kwargs)


def summarize_keys(mapping: Mapping[str, Any], limit: int = 12) -> str:
    keys = sorted(str(key) for key in mapping.keys())
    if len(keys) <= limit:
        return ", ".join(keys)
    head = ", ".join(keys[:limit])
    return f"{head}, ... (+{len(keys) - limit} more)"


def summarize_array_mapping(mapping: Mapping[str, Any], limit: int = 8) -> str:
    parts = []
    for index, key in enumerate(sorted(mapping.keys())):
        if index >= limit:
            parts.append(f"... (+{len(mapping) - limit} more)")
            break
        value = mapping[key]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", type(value).__name__)
        parts.append(f"{key}:shape={shape},dtype={dtype}")
    return "; ".join(parts)
