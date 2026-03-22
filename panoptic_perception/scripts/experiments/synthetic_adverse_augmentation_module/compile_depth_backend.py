"""Compile / export Depth Anything to an optimized inference backend.

Backends:
    onnx          Export HF model to ONNX. Load with ONNXDepthEstimator.
    tensorrt      Export to ONNX then compile to TensorRT .engine.
                  Falls back to onnx if CUDA or tensorrt unavailable.
    torch_compile Verify torch.compile works on current hardware (runtime only).

Usage:
    python -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.compile_depth_backend \
        --backend onnx --output /workspace/depth.onnx

    python -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.compile_depth_backend \
        --backend onnx --output /workspace/depth.onnx --quantize

    python -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.compile_depth_backend \
        --backend tensorrt --output /workspace/depth.engine

    python -m panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.compile_depth_backend \
        --backend torch_compile --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_cuda() -> bool:
    return torch.cuda.is_available()


def _check_tensorrt() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_backend(backend: str) -> str:
    """Override tensorrt → onnx if requirements not met."""
    if backend == "tensorrt":
        if not _check_cuda():
            print("[WARN] CUDA not available. Overriding backend: tensorrt → onnx")
            return "onnx"
        if not _check_tensorrt():
            print("[WARN] tensorrt package not installed. Overriding backend: tensorrt → onnx")
            return "onnx"
    return backend


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _export_onnx(model_name: str, output_path: str, input_size: int, opset: int) -> str:
    """Export HF Depth Anything model to ONNX. Returns path to .onnx file."""
    from .export_depth_onnx import export
    export(model_name, output_path, input_size, opset)
    return output_path


def _quantize_onnx(onnx_path: str) -> str:
    """Apply dynamic INT8 quantization to an ONNX model."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as exc:
        raise ImportError(
            "ONNX quantization requires `onnxruntime`. "
            "Install with: pip install onnxruntime"
        ) from exc

    p = Path(onnx_path)
    quantized_path = str(p.parent / f"{p.stem}_quantized{p.suffix}")

    print(f"Quantizing {onnx_path} → {quantized_path} (INT8 dynamic)...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8,
    )
    print(f"Done. Quantized model saved to {quantized_path}")
    return quantized_path


# ---------------------------------------------------------------------------
# TensorRT compilation
# ---------------------------------------------------------------------------

def _compile_tensorrt(
    model_name: str,
    output_path: str,
    input_size: int,
    opset: int,
    quantize: bool,
) -> str:
    """Export to ONNX, then compile to TensorRT engine."""
    import tensorrt as trt

    # Step 1: Export to temp ONNX
    onnx_path = str(Path(output_path).with_suffix(".onnx"))
    _export_onnx(model_name, onnx_path, input_size, opset)

    # Step 2: Build TensorRT engine
    print(f"Building TensorRT engine from {onnx_path}...")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  TRT parse error: {parser.get_error(i)}")
            raise RuntimeError("TensorRT ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if quantize:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  INT8 quantization enabled")
        else:
            print("[WARN] Platform does not support fast INT8. Building without quantization.")
    else:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled")

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(output_path, "wb") as f:
        f.write(engine_bytes)

    print(f"Done. TensorRT engine saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# torch.compile verification
# ---------------------------------------------------------------------------

def _verify_torch_compile(
    model_name: str,
    device: str,
    compile_mode: str,
    input_size: int,
    n_warmup: int = 3,
    n_bench: int = 10,
) -> None:
    """Load model, apply torch.compile, run warmup and benchmark."""
    from panoptic_perception.dataset.adverse_weather.depth_estimators import (
        TorchCompiledDepthEstimator,
    )

    print(f"Building TorchCompiledDepthEstimator (mode={compile_mode}, device={device})...")
    estimator = TorchCompiledDepthEstimator(
        model_name=model_name,
        device=device,
        normalization_epsilon=1e-8,
        compile_mode=compile_mode,
    )

    dummy = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

    print(f"Warmup ({n_warmup} iterations)...")
    for i in range(n_warmup):
        t0 = time.perf_counter()
        estimator.estimate(dummy)
        dt = time.perf_counter() - t0
        print(f"  warmup {i + 1}: {dt:.3f}s")

    print(f"Benchmark ({n_bench} iterations)...")
    times = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        estimator.estimate(dummy)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  iter {i + 1}: {dt:.3f}s")

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg * 1000:.1f}ms/image ({1 / avg:.1f} img/s)")
    print("torch.compile verification complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile Depth Anything to an optimized backend",
    )
    parser.add_argument(
        "--backend",
        choices=["onnx", "tensorrt", "torch_compile"],
        required=True,
    )
    parser.add_argument(
        "--model-name",
        default="LiheYoung/depth-anything-small-hf",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (required for onnx/tensorrt)",
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 quantization (onnx: dynamic, tensorrt: builder flag)",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="torch.compile mode (only for torch_compile backend)",
    )

    args = parser.parse_args()
    backend = _resolve_backend(args.backend)

    if backend in ("onnx", "tensorrt") and args.output is None:
        parser.error(f"--output is required for backend={backend}")

    if backend == "onnx":
        path = _export_onnx(args.model_name, args.output, args.input_size, args.opset)
        if args.quantize:
            _quantize_onnx(path)

    elif backend == "tensorrt":
        _compile_tensorrt(
            args.model_name, args.output, args.input_size, args.opset, args.quantize,
        )

    elif backend == "torch_compile":
        if args.quantize:
            print("[INFO] --quantize ignored for torch_compile (AMP handles precision)")
        _verify_torch_compile(
            args.model_name, args.device, args.compile_mode, args.input_size,
        )


if __name__ == "__main__":
    main()
