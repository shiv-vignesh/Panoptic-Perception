import argparse
import os
import torch
import sys

from panoptic_perception.models import ModelFactory, BaseTaskModel, BaseEnhancementModel
from panoptic_perception.models.utils import WeightsManager

from .profile import Profiler

def load_model(model_kwargs:dict, checkpoint_path:str):

    device = model_kwargs.get("device", "cuda")

    use_gdip = model_kwargs.get("use_gdip", False)
    use_denet = model_kwargs.get("use_denet", False)

    assert not (use_gdip and use_denet), "use_gdip and use_denet cannot both be True"
    if use_gdip:
        assert "gdip_kwargs" in model_kwargs and model_kwargs["gdip_kwargs"], \
            f'Key Error: gdip_kwargs missing'    

        model_kwargs["enhancement"] = "gdip-yolo"

    if use_denet:
        assert "denet_kwargs" in model_kwargs and model_kwargs["denet_kwargs"], \
            f'Key Error: denet_kwargs missing'

        model_kwargs["enhancement"] = "denet-yolo"

    model = ModelFactory.from_config(model_kwargs)

    device = torch.device(device) if torch.cuda.is_available() and "cuda" in device else torch.device("cpu")
    model.to(device)

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f'Error loading checkpoint: {checkpoint_path} - {e}')

        key_prefix = None
        if isinstance(model, BaseEnhancementModel):
            ckpt_state = checkpoint.get("model_state", checkpoint)
            sample_key = next(iter(ckpt_state), "")
            if not sample_key.startswith("task_network."):
                key_prefix = "task_network"

        try:
            missing, unexpected, loaded_keys = WeightsManager().load(model, 
                                                                checkpoint_path, 
                                                                key_prefix=key_prefix)
            print("=== Weights Loaded ===")
            print(f"Loaded     : {len(loaded_keys)} keys")
            print(f"Missing    : {len(missing)} keys")
            print(f"Unexpected : {len(unexpected)} keys")

        except RuntimeError as e:
            raise RuntimeError(
                f"Failed loading model from {checkpoint_path}"
            ) from e

    return model, device    

def main(args:argparse.Namespace):

    model_kwargs = {
        "model_type":args.model_type,
        "cfg_path":args.cfg_path,
        "device":args.device
    }
    
    model, device = load_model(
        model_kwargs, args.checkpoint_path
    )

    profiler = Profiler(
        warmup_iters=args.warmup_iters,
        profile_iters=args.profile_iters,
        export_trace=args.export_trace,
        torch_compile_logs=args.torch_compile_logs,
        memory_snapshot=args.memory_snapshot,
        group_by_input_shape=args.group_by_input_shape,
        record_shapes=args.record_shapes,
        with_stack=args.with_stack
    )

    profiler.run(
        model=model,
        input_shape=args.input_shape,
        device=device,
        dtype=torch.float32
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Profiler Wrapper")

    parser.add_argument(
        "--model_type",
        type=str,
        default="yolop"
    )

    parser.add_argument(
        "--cfg_path",
        type=str        
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=""
    )    

    parser.add_argument(
        "--input_shape",
        type=tuple,
        default=[4, 3, 768, 1280]
    )

    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=5
    )

    parser.add_argument(
        "--profile_iters",
        type=int,
        default=10
    )

    parser.add_argument(
        "--export_trace",
        type=bool,
        default=False
    )    

    parser.add_argument(
        "--torch_compile_logs",
        type=bool,
        default=False
    )

    parser.add_argument(
        "--memory_snapshot",
        type=bool,
        default=False
    )

    parser.add_argument(
        "--record_shapes",
        type=bool,
        default=True
    )

    parser.add_argument(
        "--group_by_input_shape",
        type=bool,
        default=True
    )        

    parser.add_argument(
        "--profile_memory",
        type=bool,
        default=True
    )

    parser.add_argument(
        "--with_stack",
        type=bool,
        default=False
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiler-dev-dir"
    )

    args = parser.parse_args()

    main(args)