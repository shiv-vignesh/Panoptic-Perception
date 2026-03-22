# Synthetic Adverse Augmentation Module

Depth-based fog, gamma low-light, and compound nighttime fog synthesis for training robust perception models.

## Features

- **Synthetic fog** — Koschmieder ASM with depth (heuristic or Depth Anything)
- **Synthetic low-light** — gamma transform in configurable range
- **Compound nighttime fog** — `dark_then_fog` or `fog_then_dark` order
- **Paired grid** — clean/degraded pairs with CSV + JSON manifests
- **Analysis** — distribution comparison and visualization

## Config

All parameters live in `default_config.json`. Override with `--config /path/to/custom.json`.

Key sections: `heuristic_depth`, `depth_anything`, `fog`, `low_light`, `dataset`, `io`, `analysis`. See `default_config.json` for the full schema.

## CLI

### Build paired grid

```bash
python -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module \
  --config /path/to/custom.json \
  build-grid \
  --input-images-dir /path/to/images \
  --output-dir /path/to/output \
  --depth-backend heuristic
```

Options: `--depth-backend` (heuristic | depth_anything), `--device`, `--compound-order`, `--gamma-min`, `--gamma-max`.

### Analyze distributions

```bash
python -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module \
  analyze \
  --synthetic-dir /path/to/synthetic \
  --real-dir /path/to/real
```

## Output layout

```
output/
├── clean/<image>
├── degraded/fog_{i}_dark_{j}/<image>
└── manifests/
    ├── paired_manifest.csv
    ├── paired_manifest.json
    └── summary.json
```

## Programmatic usage

```python
from panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module import (
    build_paired_dataset_grid,
    compute_distribution_features,
    visualize_random_triplets,
)
```
