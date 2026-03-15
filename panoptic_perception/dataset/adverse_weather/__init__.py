"""Synthetic adverse weather augmentation package.

This package focuses on:
1) Depth-based synthetic fog (Koschmieder ASM)
2) Gamma-based low-light synthesis
3) Compound nighttime fog synthesis
4) Paired data export helpers
5) Analysis hooks for synthetic-vs-real comparison
"""

from .analysis import (
    compute_distribution_features,
    summarize_feature_distributions,
    visualize_random_triplets,
)
from .augmentors import (
    ApplyOrder,
    FogParameters,
    SyntheticFogGenerator,
    SyntheticLowLightGenerator,
    apply_nighttime_fog,
)
from .config import DEFAULT_CONFIG_PATH, load_config
from .dataset_builder import (
    ImagePathDataset,
    build_depth_dataset,
    build_depth_dataset_batch,
    build_paired_dataset_grid,
    default_darkness_gammas,
    default_fog_betas,
)
from .depth_estimators import (
    DepthAnythingEstimator,
    DepthEstimator,
    HeuristicDepthEstimator,
    ONNXDepthEstimator,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "load_config",
    "ApplyOrder",
    "DepthEstimator",
    "DepthAnythingEstimator",
    "FogParameters",
    "HeuristicDepthEstimator",
    "ONNXDepthEstimator",
    "SyntheticFogGenerator",
    "SyntheticLowLightGenerator",
    "ImagePathDataset",
    "apply_nighttime_fog",
    "build_depth_dataset",
    "build_depth_dataset_batch",
    "build_paired_dataset_grid",
    "compute_distribution_features",
    "default_darkness_gammas",
    "default_fog_betas",
    "summarize_feature_distributions",
    "visualize_random_triplets",
]
