from __future__ import annotations

import argparse
import json

from panoptic_perception.dataset.adverse_weather.analysis import compute_distribution_features, summarize_feature_distributions
from panoptic_perception.dataset.adverse_weather.augmentors import SyntheticFogGenerator, SyntheticLowLightGenerator
from panoptic_perception.dataset.adverse_weather.config import load_config
from panoptic_perception.dataset.adverse_weather.dataset_builder import build_paired_dataset_grid, build_depth_dataset, build_depth_dataset_batch
from panoptic_perception.dataset.adverse_weather.depth_estimators import (
    DepthAnythingEstimator,
    DepthEstimator,
    HeuristicDepthEstimator,
    ONNXDepthEstimator,
    TensorRTDepthEstimator,
    TorchCompiledDepthEstimator,
)


def _build_depth_estimator(backend: str, cfg: dict) -> DepthEstimator:
    if backend == "heuristic":
        h, io_cfg = cfg["heuristic_depth"], cfg["io"]
        return HeuristicDepthEstimator(
            vertical_weight=h["vertical_weight"],
            intensity_weight=h["intensity_weight"],
            edge_weight=h["edge_weight"],
            edge_epsilon=h["edge_epsilon"],
            uint8_max=float(io_cfg["uint8_max"]),
        )
    if backend == "depth_anything":
        d = cfg["depth_anything"]
        return DepthAnythingEstimator(
            model_name=d["model_name"],
            device=d["device"],
            normalization_epsilon=d["normalization_epsilon"],
        )
    if backend == "torch_compile":
        d = cfg["depth_anything"]
        return TorchCompiledDepthEstimator(
            model_name=d["model_name"],
            device=d["device"],
            normalization_epsilon=d["normalization_epsilon"],
            compile_mode=d.get("compile_mode", "reduce-overhead"),
        )
    if backend == "onnx":
        return ONNXDepthEstimator(
            onnx_path=cfg["onnx"]["model_path"],
            device=cfg["onnx"].get("device", "cuda"),
            input_size=cfg["onnx"].get("input_size", 518),
            normalization_epsilon=cfg["onnx"].get("normalization_epsilon", 1e-8),
        )
    if backend == "tensorrt":
        return TensorRTDepthEstimator(
            engine_path=cfg["tensorrt"]["engine_path"],
            input_size=cfg["tensorrt"].get("input_size", 518),
            normalization_epsilon=cfg["tensorrt"].get("normalization_epsilon", 1e-8),
        )
    raise ValueError(f"Unknown depth backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic adverse augmentation module (fog + low-light + compound)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config. Default: module's default_config.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-grid", help="Build paired 5x5 dataset grid")
    build_parser.add_argument("--input-images-dir", required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument(
        "--depth-backend",
        choices=["heuristic", "depth_anything", "torch_compile", "onnx", "tensorrt"],
        default=None,
        help="Override config. Default: heuristic",
    )
    build_parser.add_argument("--depth-model-name", default=None)
    build_parser.add_argument("--device", default=None)
    build_parser.add_argument("--compound-order", choices=["dark_then_fog", "fog_then_dark"], default=None)
    build_parser.add_argument("--gamma-min", type=float, default=None)
    build_parser.add_argument("--gamma-max", type=float, default=None)

    depth_parser = subparsers.add_parser("depth-only", help="Pre-compute and cache depth maps as .npy")
    depth_parser.add_argument("--input-images-dir", required=True)
    depth_parser.add_argument("--output-dir", required=True)
    depth_parser.add_argument(
        "--depth-backend",
        choices=["heuristic", "depth_anything", "torch_compile", "onnx", "tensorrt"],
        default=None,
    )
    depth_parser.add_argument("--depth-model-name", default=None)
    depth_parser.add_argument("--device", default=None)

    batch_parser = subparsers.add_parser("depth-only-batch", help="Batched depth map generation (faster, requires depth_anything)")
    batch_parser.add_argument("--input-images-dir", required=True)
    batch_parser.add_argument("--output-dir", required=True)
    batch_parser.add_argument("--depth-model-name", default=None)
    batch_parser.add_argument("--device", default=None)
    batch_parser.add_argument("--batch-size", type=int, default=16)
    batch_parser.add_argument("--num-workers", type=int, default=4)
    batch_parser.add_argument("--write-threads", type=int, default=4)

    analyze_parser = subparsers.add_parser("analyze", help="Compare synthetic vs real distributions")
    analyze_parser.add_argument("--synthetic-dir", required=True)
    analyze_parser.add_argument("--real-dir", required=True)

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "build-grid":
        backend = args.depth_backend or "heuristic"
        if args.depth_model_name is not None:
            cfg = {**cfg, "depth_anything": {**cfg.get("depth_anything", {}), "model_name": args.depth_model_name}}
        if args.device is not None:
            cfg = {**cfg, "depth_anything": {**cfg.get("depth_anything", {}), "device": args.device}}

        estimator = _build_depth_estimator(backend, cfg)
        fog = SyntheticFogGenerator(depth_estimator=estimator, config=cfg)

        ll = cfg["low_light"]
        gamma_min = args.gamma_min if args.gamma_min is not None else ll["gamma_min"]
        gamma_max = args.gamma_max if args.gamma_max is not None else ll["gamma_max"]
        low_light = SyntheticLowLightGenerator(
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            gamma_min_threshold=ll["gamma_min_threshold"],
            config=cfg,
        )

        ds = cfg["dataset"]
        compound_order = args.compound_order or ds["compound_order"]

        summary = build_paired_dataset_grid(
            input_images_dir=args.input_images_dir,
            output_dir=args.output_dir,
            fog_generator=fog,
            low_light_generator=low_light,
            compound_order=compound_order,
            config_path=args.config,
        )
        print(json.dumps(summary, indent=cfg["io"]["json_indent"]))
    
    elif args.command == "depth-only":
        backend = args.depth_backend or "heuristic"
        if args.depth_model_name is not None:
            cfg = {**cfg, "depth_anything": {**cfg.get("depth_anything", {}), "model_name": args.depth_model_name}}
        if args.device is not None:
            cfg = {**cfg, "depth_anything": {**cfg.get("depth_anything", {}), "device": args.device}}

        estimator = _build_depth_estimator(backend, cfg)
        build_depth_dataset(
            input_images_dir=args.input_images_dir,
            output_dir=args.output_dir,
            depth_estimator=estimator,
            config_path=args.config,
        )
    
    elif args.command == "depth-only-batch":
        if args.depth_model_name is not None:
            cfg = {**cfg, "depth_anything": {**cfg.get("depth_anything", {}), "model_name": args.depth_model_name}}
        if args.device is not None:
            cfg = {**cfg, "depth_anything": {**cfg.get("depth_anything", {}), "device": args.device}}

        estimator = _build_depth_estimator("depth_anything", cfg)
        build_depth_dataset_batch(
            input_images_dir=args.input_images_dir,
            output_dir=args.output_dir,
            depth_estimator=estimator,
            config_path=args.config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            write_threads=args.write_threads,
        )

    elif args.command == "analyze":
        synthetic = compute_distribution_features(
            args.synthetic_dir, config_path=args.config
        )
        real = compute_distribution_features(
            args.real_dir, config_path=args.config
        )
        report = summarize_feature_distributions(
            synthetic, real, config_path=args.config
        )
        print(json.dumps(report, indent=cfg["io"]["json_indent"]))


if __name__ == "__main__":
    main()
