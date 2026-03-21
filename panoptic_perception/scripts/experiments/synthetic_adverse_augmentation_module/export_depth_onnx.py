"""Export Depth Anything Small to ONNX format with optional quantization.

Usage:
    # Full precision (float32)
    python3 -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.export_depth_onnx \
        --output /workspace/depth_anything_small.onnx --quantize none

    # FP16 (GPU-optimized, half the size)
    python3 -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.export_depth_onnx \
        --output /workspace/depth_anything_small_fp16.onnx --quantize fp16

    # INT8 dynamic (CPU-optimized, smallest model)
    python3 -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.export_depth_onnx \
        --output /workspace/depth_anything_small_int8_cpu.onnx --quantize int8-dynamic

    # INT8 static (GPU+CPU optimized, requires calibration images)
    python3 -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.export_depth_onnx \
        --output /workspace/depth_anything_small_int8_static.onnx --quantize int8-static \
        --calibration-dir /workspace/data/100k/100k/val

The exported model expects input named "pixel_values" with shape (1, 3, 518, 518)
and outputs "predicted_depth" with shape (1, 518, 518).
"""
import argparse
import os
import glob
import numpy as np
import cv2
import torch


# ---------------------------------------------------------------------------
# Calibration data reader for INT8 static quantization
# ---------------------------------------------------------------------------

class DepthCalibrationDataReader:
    """Feeds calibration images to onnxruntime static quantizer."""

    def __init__(self, image_dir: str, input_size: int = 518, max_samples: int = 100):
        self._input_size = input_size
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        exts = ("*.jpg", "*.jpeg", "*.png")
        self._paths = []
        for ext in exts:
            self._paths.extend(glob.glob(os.path.join(image_dir, ext)))
        self._paths = self._paths[:max_samples]
        self._idx = 0

        assert len(self._paths) > 0, f"No images found in {image_dir}"
        print(f"[Calibration] Using {len(self._paths)} images from {image_dir}")

    def get_next(self):
        if self._idx >= len(self._paths):
            return None
        img = cv2.imread(self._paths[self._idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._input_size, self._input_size), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        img = np.transpose(img, (2, 0, 1))
        tensor = np.expand_dims(img, 0).astype(np.float32)
        self._idx += 1
        return {"pixel_values": tensor}

    def rewind(self):
        self._idx = 0


# ---------------------------------------------------------------------------
# Quantization methods
# ---------------------------------------------------------------------------

def _quantize_int8_dynamic(onnx_path: str, output_path: str) -> None:
    """CPU-optimized: quantizes weights only, no calibration needed."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )


def _quantize_int8_static(onnx_path: str, output_path: str,
                           calibration_dir: str, input_size: int,
                           max_calibration_samples: int) -> None:
    """GPU+CPU optimized: quantizes weights AND activations using calibration data."""
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod

    calibration_reader = DepthCalibrationDataReader(
        calibration_dir, input_size=input_size, max_samples=max_calibration_samples,
    )

    quantize_static(
        model_input=onnx_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=3,  # QDQ format — compatible with both GPU and CPU EPs
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )


def _quantize_fp16(onnx_path: str, output_path: str) -> None:
    """GPU-optimized: converts all ops to float16."""
    from onnxruntime.transformers.float16 import convert_float_to_float16
    import onnx

    model = onnx.load(onnx_path)
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_path)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(
    model_name: str,
    output_path: str,
    input_size: int = 518,
    opset: int = 17,
    quantize: str = "none",
    calibration_dir: str | None = None,
    max_calibration_samples: int = 100,
):
    from transformers import AutoModelForDepthEstimation

    print(f"Loading {model_name}...")
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)

    # Export full-precision ONNX first
    if quantize != "none":
        fp32_path = output_path.replace(".onnx", "_fp32_tmp.onnx")
    else:
        fp32_path = output_path

    print(f"Exporting to {fp32_path} (opset {opset})...")
    torch.onnx.export(
        model,
        (dummy,),
        fp32_path,
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "predicted_depth": {0: "batch"},
        },
        opset_version=opset,
    )

    if quantize == "int8-dynamic":
        print(f"Quantizing to INT8 dynamic (CPU-optimized) → {output_path}")
        _quantize_int8_dynamic(fp32_path, output_path)
        os.remove(fp32_path)

    elif quantize == "int8-static":
        assert calibration_dir is not None, \
            "--calibration-dir required for int8-static quantization"
        print(f"Quantizing to INT8 static (GPU+CPU optimized) → {output_path}")
        _quantize_int8_static(fp32_path, output_path, calibration_dir,
                               input_size, max_calibration_samples)
        os.remove(fp32_path)

    elif quantize == "fp16":
        print(f"Quantizing to FP16 (GPU-optimized) → {output_path}")
        _quantize_fp16(fp32_path, output_path)
        os.remove(fp32_path)

    print(f"Done. ONNX model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Depth Anything to ONNX")
    parser.add_argument(
        "--model-name",
        default="LiheYoung/depth-anything-small-hf",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output",
        default="depth_anything_small.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--quantize",
        choices=["none", "fp16", "int8-dynamic", "int8-static"],
        default="fp16" if torch.cuda.is_available() else "int8-dynamic",
        help=(
            "none: FP32 (baseline). "
            "fp16: GPU-optimized half precision. "
            "int8-dynamic: CPU-optimized, no calibration. "
            "int8-static: GPU+CPU optimized, requires --calibration-dir."
        ),
    )
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help="Directory of calibration images (required for int8-static)",
    )
    parser.add_argument(
        "--max-calibration-samples",
        type=int,
        default=100,
        help="Max calibration images for int8-static (default: 100)",
    )
    args = parser.parse_args()

    export(
        args.model_name, args.output, args.input_size, args.opset,
        args.quantize, args.calibration_dir, args.max_calibration_samples,
    )
