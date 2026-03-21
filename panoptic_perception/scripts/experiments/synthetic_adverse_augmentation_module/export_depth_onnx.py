"""Export Depth Anything Small to ONNX format with optional quantization.

Usage:
    # Full precision (float32)
    python3 -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.export_depth_onnx \
        --output /workspace/depth_anything_small.onnx

    # Dynamic INT8 quantization (CPU-optimized, smallest model)
    python3 -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.export_depth_onnx \
        --output /workspace/depth_anything_small_int8.onnx --quantize int8

    # Float16 quantization (GPU-optimized, half the size)
    python3 -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.export_depth_onnx \
        --output /workspace/depth_anything_small_fp16.onnx --quantize fp16

The exported model expects input named "pixel_values" with shape (1, 3, 518, 518)
and outputs "predicted_depth" with shape (1, 518, 518).
"""
import argparse
import os
import torch


def _quantize_int8(onnx_path: str, output_path: str) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )


def _quantize_fp16(onnx_path: str, output_path: str) -> None:
    from onnxruntime.transformers.float16 import convert_float_to_float16
    import onnx

    model = onnx.load(onnx_path)
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_path)


def export(
    model_name: str,
    output_path: str,
    input_size: int = 518,
    opset: int = 17,
    quantize: str | None = None,
):
    from transformers import AutoModelForDepthEstimation

    print(f"Loading {model_name}...")
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)

    # Export full-precision ONNX first
    if quantize is not None:
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

    if quantize == "int8":
        print(f"Quantizing to INT8 → {output_path}")
        _quantize_int8(fp32_path, output_path)
        os.remove(fp32_path)
    elif quantize == "fp16":
        print(f"Quantizing to FP16 → {output_path}")
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
        choices=["int8", "fp16"],
        default="fp16" if torch.cuda.is_available() else "int8",
        help="Quantization mode: int8 (dynamic, CPU-optimized) or fp16 (GPU-optimized)",
    )
    args = parser.parse_args()

    export(args.model_name, args.output, args.input_size, args.opset, args.quantize)
