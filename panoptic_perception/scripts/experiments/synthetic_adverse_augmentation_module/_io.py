from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .config import load_config


def ensure_images_dir(input_dir: Path) -> None:
    """Raise if path does not exist or is not a directory."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")


def iter_images(input_dir: Path, config: dict | None = None) -> Iterator[Path]:
    """Yield sorted image paths from a directory."""
    cfg = config if config is not None else load_config()
    extensions = frozenset(ext.lower() for ext in cfg["io"]["image_extensions"])
    ensure_images_dir(input_dir)
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def read_rgb(path: Path) -> np.ndarray:
    """Read an image file as uint8 RGB (H, W, 3)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    """Write an RGB uint8 image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise OSError(f"Failed to write image: {path}")
