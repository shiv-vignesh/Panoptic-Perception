from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import VALID_COMPOUND_ORDERS, load_config
from .depth_estimators import DepthEstimator


def _to_float01(image: np.ndarray, uint8_max: float) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / uint8_max
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def _to_uint8(image: np.ndarray, uint8_max: float) -> np.ndarray:
    return np.clip(image * uint8_max, 0.0, uint8_max).astype(np.uint8)


@dataclass
class FogParameters:
    """Koschmieder atmospheric scattering model parameters."""

    beta: float
    max_depth_meters: float
    atmospheric_light_quantile: float
    atmospheric_light_min_pixels: int
    atmospheric_light: np.ndarray | None = None


class SyntheticFogGenerator:
    """Depth-based synthetic fog using Koschmieder ASM.

    I_fog(x) = I(x) * t(x) + A * (1 - t(x)),  t(x) = exp(-beta * d(x))
    """

    def __init__(self, depth_estimator: DepthEstimator, config: dict | None = None) -> None:
        self.depth_estimator = depth_estimator
        self._config = config if config is not None else load_config()
        self._uint8_max = float(self._config["io"]["uint8_max"])

    def estimate_atmospheric_light(
        self, image_float: np.ndarray, depth: np.ndarray, params: FogParameters
    ) -> np.ndarray:
        mask = depth >= np.quantile(depth, params.atmospheric_light_quantile)
        if mask.sum() < params.atmospheric_light_min_pixels:
            return image_float.reshape(-1, 3).mean(axis=0)
        return image_float[mask].mean(axis=0)

    def generate(
        self,
        image_rgb: np.ndarray,
        params: FogParameters,
        precomputed_depth: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = _to_float01(image_rgb, self._uint8_max)

        if precomputed_depth is not None:
            depth = precomputed_depth.astype(np.float32)
        else:
            img_u8 = image_rgb if image_rgb.dtype == np.uint8 else _to_uint8(image, self._uint8_max)
            depth = self.depth_estimator.estimate(img_u8)
        depth = np.clip(depth, 0.0, 1.0)

        transmission = np.exp(
            -params.beta * depth * params.max_depth_meters
        ).astype(np.float32)
        t = np.clip(transmission, 0.0, 1.0)[..., None]

        if params.atmospheric_light is not None:
            A = np.asarray(params.atmospheric_light, dtype=np.float32)
        else:
            A = self.estimate_atmospheric_light(image, depth, params)
        A = np.clip(A, 0.0, 1.0).reshape(1, 1, 3)

        foggy = image * t + A * (1.0 - t)
        return _to_uint8(np.clip(foggy, 0.0, 1.0), self._uint8_max), depth, transmission


class SyntheticLowLightGenerator:
    """Gamma darkening in bounded range [gamma_min, gamma_max]."""

    def __init__(
        self,
        gamma_min: float,
        gamma_max: float,
        gamma_min_threshold: float,
        config: dict | None = None,
    ) -> None:
        self._config = config if config is not None else load_config()
        self._uint8_max = float(self._config["io"]["uint8_max"])
        if gamma_min <= gamma_min_threshold:
            raise ValueError(
                f"gamma_min must be > {gamma_min_threshold} for darkening."
            )
        if gamma_max < gamma_min:
            raise ValueError("gamma_max must be >= gamma_min.")
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def apply(self, image_rgb: np.ndarray, gamma: float) -> np.ndarray:
        if gamma < self.gamma_min or gamma > self.gamma_max:
            raise ValueError(
                f"gamma={gamma} outside [{self.gamma_min}, {self.gamma_max}]"
            )
        return _to_uint8(
            np.power(_to_float01(image_rgb, self._uint8_max), gamma),
            self._uint8_max,
        )


ApplyOrder = Literal["dark_then_fog", "fog_then_dark"]


def apply_nighttime_fog(
    image_rgb: np.ndarray,
    fog_generator: SyntheticFogGenerator,
    fog_params: FogParameters,
    gamma_generator: SyntheticLowLightGenerator,
    gamma: float,
    apply_order: ApplyOrder = "dark_then_fog",
    precomputed_depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compound nighttime fog with configurable operation order."""
    if apply_order not in VALID_COMPOUND_ORDERS:
        raise ValueError(f"apply_order must be one of {VALID_COMPOUND_ORDERS}")

    if apply_order == "dark_then_fog":
        dark = gamma_generator.apply(image_rgb, gamma)
        out, depth, t = fog_generator.generate(dark, fog_params, precomputed_depth)
    else:
        foggy, depth, t = fog_generator.generate(image_rgb, fog_params, precomputed_depth)
        out = gamma_generator.apply(foggy, gamma)
    return out, depth, t
