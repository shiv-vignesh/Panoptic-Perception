"""Export Depth Anything Small to ONNX format.

Usage:
    python3 -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.export_depth_onnx \
        --output /workspace/depth_anything_small.onnx

The exported model expects input named "pixel_values" with shape (1, 3, 518, 518)
and outputs "predicted_depth" with shape (1, 518, 518).
"""
import argparse
import torch


def export(model_name: str, output_path: str, input_size: int = 518, opset: int = 17):
    from transformers import AutoModelForDepthEstimation

    print(f"Loading {model_name}...")
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)

    print(f"Exporting to {output_path} (opset {opset})...")
    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "predicted_depth": {0: "batch"},
        },
        opset_version=opset,
    )
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
    args = parser.parse_args()

    export(args.model_name, args.output, args.input_size, args.opset)
