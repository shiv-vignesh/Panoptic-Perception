from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import torch


class DepthEstimator(Protocol):
    """Interface for all depth backends.

    Implementations must return a float32 array in [0, 1] with shape (H, W),
    where 0 = near and 1 = far.
    """

    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return normalized relative depth in [0, 1], shape (H, W).

        Args:
            image_rgb: uint8 RGB image, shape (H, W, 3).
        """
        ...


@dataclass
class HeuristicDepthEstimator:
    """Fast fallback: vertical prior + inverse-intensity + edge attenuation."""

    vertical_weight: float
    intensity_weight: float
    edge_weight: float
    edge_epsilon: float
    uint8_max: float

    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        gray = (
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            / self.uint8_max
        )

        vertical = np.linspace(1.0, 0.0, num=h, dtype=np.float32)[:, None]
        vertical = np.broadcast_to(vertical, (h, w))

        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(sx * sx + sy * sy)
        edge /= edge.max() + self.edge_epsilon

        depth = (
            self.vertical_weight * vertical
            + self.intensity_weight * (1.0 - gray)
            + self.edge_weight * (1.0 - edge)
        )
        return np.clip(depth, 0.0, 1.0).astype(np.float32)


class DepthAnythingEstimator:
    """Depth backend powered by Depth Anything (HF transformers).

    Model is lazy-loaded on first call. GPU inference uses inference_mode
    and autocast (float16) to minimize VRAM.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        normalization_epsilon: float,
    ) -> None:
        self.model_name = model_name
        self._normalization_epsilon = normalization_epsilon
        self._processor = None
        self._model = None

        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            resolved = torch.device("cpu")
        self._torch_device = resolved

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise ImportError(
                "DepthAnythingEstimator requires `transformers`."
            ) from exc

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self._model.to(self._torch_device).eval()

    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        self._ensure_loaded()

        h, w = image_rgb.shape[:2]
        inputs = self._processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}

        use_amp = self._torch_device.type == "cuda"
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self._torch_device.type, enabled=use_amp
        ):
            outputs = self._model(**inputs)

        post = self._processor.post_process_depth_estimation(
            outputs, target_sizes=[(h, w)]
        )
        depth = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)

        del inputs, outputs, post
        if use_amp:
            torch.cuda.empty_cache()

        dmin, dmax = depth.min(), depth.max()
        depth = 1.0 - (depth - dmin) / (dmax - dmin + self._normalization_epsilon)
        return np.clip(depth, 0.0, 1.0)
