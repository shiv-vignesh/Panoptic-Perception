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

    def estimate_batch(self, images_rgb: list[np.ndarray]) -> list[np.ndarray]:
        """Return per-image normalized relative depth.

        Args:
            images_rgb: list of B uint8 RGB images, shapes (H_i, W_i, 3) — sizes
                may differ across images.

        Returns:
            list of B float32 arrays in [0, 1], each matching its input size,
            where 0 = near and 1 = far.
        """
        ...

    def __call__(self):
        raise NotImplementedError

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
    
    def estimate_batch(self, images_rgb: list[np.ndarray], return_tensors:bool = False) -> list[np.ndarray]:
        """Heuristic depth is pure numpy/cv2 — no GPU saturation gain from
        vectorizing across the batch. Loop is correct and fast enough.
        """

        if not return_tensors:
            return [self.estimate(img) for img in images_rgb]
        
        return torch.from_numpy(
            np.stack([self.estimate(img) for img in images_rgb])
        )

class ONNXDepthEstimator:
    """Depth estimation via ONNX Runtime. Supports CUDA, TensorRT, and CPU providers.

    Expects an ONNX model exported from Depth Anything (input: pixel_values NCHW float32,
    output: predicted_depth NHW float32). Preprocessing replicates the HF image processor
    using numpy/cv2 — no transformers dependency at inference time.
    """

    # ImageNet normalization (same as Depth Anything's HF processor)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        onnx_path: str,
        device: str = "cuda",
        input_size: int = 518,
        normalization_epsilon: float = 1e-8,
        quantized: bool = False,
    ) -> None:
        self.onnx_path = onnx_path
        self._input_size = input_size
        self._normalization_epsilon = normalization_epsilon
        self._device = device
        self._quantized = quantized
        self._session = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNXDepthEstimator requires `onnxruntime-gpu` or `onnxruntime`."
            ) from exc

        providers = []
        provider_options = []
        if self._device.startswith("cuda"):
            device_id = 0
            if ":" in self._device:
                device_id = int(self._device.split(":")[1])

            available = ort.get_available_providers()
            if "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
                provider_options.append({"device_id": str(device_id)})
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
                provider_options.append({"device_id": str(device_id)})
        providers.append("CPUExecutionProvider")
        provider_options.append({})

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            self.onnx_path, sess_options=sess_opts,
            providers=providers, provider_options=provider_options,
        )
        active = self._session.get_providers()
        print(f"[ONNXDepthEstimator] Loaded {self.onnx_path} | providers: {active}")

    def _preprocess_single(self, image_rgb: np.ndarray) -> np.ndarray:
        """HWC uint8 → (3, S, S) float32. Shared by single and batch paths."""
        img = cv2.resize(
            image_rgb, (self._input_size, self._input_size),
            interpolation=cv2.INTER_CUBIC,
        )
        img = img.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        return np.transpose(img, (2, 0, 1))  # HWC → CHW

    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """Single-image entry: add the batch dim → (1, 3, S, S)."""
        return np.expand_dims(self._preprocess_single(image_rgb), 0).astype(np.float32)

    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        self._ensure_loaded()

        h, w = image_rgb.shape[:2]
        input_tensor = self._preprocess(image_rgb)

        outputs = self._session.run(None, {"pixel_values": input_tensor})
        depth = outputs[0].squeeze().astype(np.float32)

        # Resize back to original image dimensions
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1], invert: far=1, near=0
        dmin, dmax = depth.min(), depth.max()
        depth = 1.0 - (depth - dmin) / (dmax - dmin + self._normalization_epsilon)
        return np.clip(depth, 0.0, 1.0).astype(np.float32)

    def estimate_batch(self, images_rgb: list[np.ndarray]) -> list[np.ndarray]:
        """Batched depth estimation in a single session.run call.

        Requires the ONNX model to have been exported with a dynamic batch
        dimension on `pixel_values`. Standard Depth Anything exports do; if
        yours doesn't, re-export or fall back to looping `estimate` per image.
        """
        self._ensure_loaded()

        target_sizes = [(img.shape[0], img.shape[1]) for img in images_rgb]

        # Preprocess each image, stack into (B, 3, S, S)
        batch = np.stack(
            [self._preprocess_single(img) for img in images_rgb],
            axis=0,
        ).astype(np.float32)
        batch = np.ascontiguousarray(batch)   # ORT is strict about layout

        outputs = self._session.run(None, {"pixel_values": batch})
        depth_batch = outputs[0].astype(np.float32)   # (B, S, S)

        depths = []
        for i, (h, w) in enumerate(target_sizes):
            depth = depth_batch[i]
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            dmin, dmax = depth.min(), depth.max()
            depth = 1.0 - (depth - dmin) / (dmax - dmin + self._normalization_epsilon)
            depths.append(np.clip(depth, 0.0, 1.0))

        return depths


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

        self._processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
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

        dmin, dmax = depth.min(), depth.max()
        depth = 1.0 - (depth - dmin) / (dmax - dmin + self._normalization_epsilon)
        return np.clip(depth, 0.0, 1.0)

    def estimate_batch(self, images_rgb: list[np.ndarray], return_tensors:bool = False) -> list[np.ndarray]:
        """Batched depth estimation. Returns list of float32 arrays in [0, 1], shape (H, W)."""
        self._ensure_loaded()

        target_sizes = [(img.shape[0], img.shape[1]) for img in images_rgb]

        inputs = self._processor(images=images_rgb, return_tensors="pt")
        inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}

        use_amp = self._torch_device.type == "cuda"
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self._torch_device.type, enabled=use_amp
        ):
            outputs = self._model(**inputs)

        post = self._processor.post_process_depth_estimation(
            outputs, target_sizes=target_sizes
        )

        del inputs, outputs

        preds = torch.stack([item["predicted_depth"] for item in post]).float()
        preds = 1.0 - (preds - preds.amin(dim=(-1,-2), keepdim=True)) / \
                        (preds.amax(dim=(-1,-2), keepdim=True) - preds.amin(dim=(-1,-2), keepdim=True) + 1e-6)

        if not return_tensors:
            return preds.clamp(0, 1).cpu().numpy()

        return preds.clamp(0, 1)

class TorchCompiledDepthEstimator:
    """Wraps DepthAnythingEstimator with torch.compile for faster inference.

    Compilation happens lazily on first call. The first inference is slow
    (compilation overhead), subsequent calls benefit from compiled kernels.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        normalization_epsilon: float,
        compile_mode: str = "reduce-overhead",
    ) -> None:
        self._inner = DepthAnythingEstimator(model_name, device, normalization_epsilon)
        self._compile_mode = compile_mode
        self._compiled = False

    def _ensure_compiled(self) -> None:
        self._inner._ensure_loaded()
        if self._compiled:
            return
        self._inner._model = torch.compile(
            self._inner._model, mode=self._compile_mode
        )
        self._compiled = True
        print(
            f"[TorchCompiledDepthEstimator] Compiled with mode={self._compile_mode}"
        )

    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        self._ensure_compiled()
        return self._inner.estimate(image_rgb)

    def estimate_batch(self, images_rgb: list[np.ndarray]) -> list[np.ndarray]:
        self._ensure_compiled()
        return self._inner.estimate_batch(images_rgb)


class TensorRTDepthEstimator:
    """Depth estimation via a pre-compiled TensorRT engine.

    Expects an engine built from the Depth Anything ONNX export.
    Input: pixel_values (1, 3, input_size, input_size) float32.
    Output: predicted_depth (1, input_size, input_size) float32.

    Requires: tensorrt, pycuda.
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        engine_path: str,
        input_size: int = 518,
        normalization_epsilon: float = 1e-8,
    ) -> None:
        self.engine_path = engine_path
        self._input_size = input_size
        self._normalization_epsilon = normalization_epsilon
        self._context = None
        self._d_input = None
        self._d_output = None
        self._stream = None

    def _ensure_loaded(self) -> None:
        if self._context is not None:
            return

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TensorRTDepthEstimator requires `tensorrt` and `pycuda`."
            ) from exc

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        self._context = engine.create_execution_context()

        s = self._input_size
        input_nbytes = 1 * 3 * s * s * np.dtype(np.float32).itemsize
        output_nbytes = 1 * s * s * np.dtype(np.float32).itemsize

        self._d_input = cuda.mem_alloc(input_nbytes)
        self._d_output = cuda.mem_alloc(output_nbytes)
        self._stream = cuda.Stream()
        self._cuda = cuda

        print(f"[TensorRTDepthEstimator] Loaded {self.engine_path}")

    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(
            image_rgb, (self._input_size, self._input_size),
            interpolation=cv2.INTER_CUBIC,
        )
        img = img.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return np.expand_dims(img, 0).astype(np.float32)  # (1, 3, H, W)

    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        self._ensure_loaded()

        h, w = image_rgb.shape[:2]
        input_tensor = np.ascontiguousarray(self._preprocess(image_rgb))

        self._cuda.memcpy_htod_async(self._d_input, input_tensor, self._stream)
        self._context.execute_async_v2(
            bindings=[int(self._d_input), int(self._d_output)],
            stream_handle=self._stream.handle,
        )
        output = np.empty((1, self._input_size, self._input_size), dtype=np.float32)
        self._cuda.memcpy_dtoh_async(output, self._d_output, self._stream)
        self._stream.synchronize()

        depth = output.squeeze()
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        dmin, dmax = depth.min(), depth.max()
        depth = 1.0 - (depth - dmin) / (dmax - dmin + self._normalization_epsilon)
        return np.clip(depth, 0.0, 1.0).astype(np.float32)

@dataclass
class RadialDistance:

    radial_depth_decay_rate: float = -0.04

    def estimate(self, image_rgb: np.ndarray):
        h, w = image_rgb.shape[:2]

        x = np.linspace(0, w-1, w)
        y = np.linspace(0, h-1, h)
        xx, yy = np.meshgrid(x, y)
        x_c, y_c = w // 2, h // 2

        d = RadialDistance.radial_depth_decay_rate * \
                np.sqrt((yy - y_c)**2 + (xx - x_c)**2) + \
                    np.sqrt(np.maximum(h, w))

        d_min, d_max = d.min(), d.max()
        d = ((d - d_min) / (d_max - d_min + 1e-8)).astype(np.float32)

        return d