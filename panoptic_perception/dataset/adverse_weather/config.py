"""Configuration loader for synthetic adverse augmentation.

All tunable parameters live in config. Use default_config.json as template.
Override with --config /path/to/custom.json from CLI.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.json"

_DEFAULTS: dict = {
    "heuristic_depth": {
        "vertical_weight": 0.7,
        "intensity_weight": 0.2,
        "edge_weight": 0.1,
        "edge_epsilon": 1e-8,
    },
    "depth_anything": {
        "model_name": "LiheYoung/depth-anything-small-hf",
        "device": "cpu",
        "normalization_epsilon": 1e-8,
    },
    "fog": {
        "max_depth_meters": 120.0,
        "betas": [0.008, 0.015, 0.025, 0.040, 0.060],
        "atmospheric_light_quantile": 0.9,
        "atmospheric_light_min_pixels": 16,
    },
    "low_light": {
        "gamma_min": 1.5,
        "gamma_max": 3.0,
        "gamma_min_threshold": 1.0,
        "gammas": [1.5, 1.9, 2.3, 2.7, 3.0],
    },
    "dataset": {
        "compound_order": "dark_then_fog",
        "fog_betas": [0.008, 0.015, 0.025, 0.040, 0.060],
        "darkness_gammas": [1.5, 1.9, 2.3, 2.7, 3.0],
    },
    "io": {
        "uint8_max": 255.0,
        "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        "json_indent": 2,
    },
    "analysis": {
        "num_samples": 8,
        "seed": 7,
        "dpi": 150,
        "figsize_width": 10,
        "figsize_height_per_sample": 4,
        "luma_coefficients": [0.299, 0.587, 0.114],
        "saturation_max": 255.0,
        "epsilon": 1e-8,
    },
}

VALID_COMPOUND_ORDERS = frozenset({"dark_then_fog", "fog_then_dark"})


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into a deep copy of base. Does not mutate base."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v) if isinstance(v, (dict, list)) else v
    return out


def _validate_config(cfg: dict) -> None:
    """Raise ValueError if config has invalid values."""
    ll = cfg["low_light"]
    gamma_min, gamma_max = ll["gamma_min"], ll["gamma_max"]
    threshold = ll["gamma_min_threshold"]
    if not isinstance(gamma_min, (int, float)) or not isinstance(gamma_max, (int, float)):
        raise ValueError("low_light.gamma_min and gamma_max must be numbers")
    if gamma_min <= threshold:
        raise ValueError(f"low_light.gamma_min must be > {threshold} for darkening")
    if gamma_max < gamma_min:
        raise ValueError("low_light.gamma_max must be >= gamma_min")

    order = cfg["dataset"]["compound_order"]
    if order not in VALID_COMPOUND_ORDERS:
        raise ValueError(f"dataset.compound_order must be one of {VALID_COMPOUND_ORDERS}")

    betas = cfg["fog"].get("betas") or cfg["dataset"].get("fog_betas")
    if not betas or not all(isinstance(b, (int, float)) and b > 0 for b in betas):
        raise ValueError("fog.betas / dataset.fog_betas must be a list of positive numbers")

    gammas = cfg["dataset"].get("darkness_gammas") or ll.get("gammas")
    if not gammas or not all(isinstance(g, (int, float)) for g in gammas):
        raise ValueError("dataset.darkness_gammas must be a list of numbers")


def load_config(path: str | Path | None = None) -> dict:
    """Load config from JSON. Missing keys fall back to defaults.

    When path is explicitly provided and the file does not exist, raises FileNotFoundError.
    Returns a deep copy so callers can mutate without affecting defaults.
    """
    explicit_path = path is not None
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)

    if not path.exists():
        if explicit_path:
            raise FileNotFoundError(f"Config file not found: {path}")
        return copy.deepcopy(_DEFAULTS)

    with path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    cfg = _deep_merge(_DEFAULTS, loaded)
    _validate_config(cfg)
    return cfg
