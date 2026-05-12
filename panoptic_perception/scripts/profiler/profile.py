import torch
import pickle
from pathlib import Path

from typing import List, Any
from collections import defaultdict

from .trace_models import GPUSpecs, KernelRecord, ForwardError, ProfilerInfo

class Profiler:
    def __init__(self, 
                warmup_iters:int = 5, 
                profile_iters:int = 10,
                export_trace:bool = False, 
                torch_compile_logs:bool = False,                
                memory_snapshot:bool = False,
                output_dir:str = None,
                record_shapes = True,
                group_by_input_shape = True,
                profile_memory = True,
                with_stack = False,
                ):
        
        r"""
        
        Args:
            warmup_iters: Number of warmup iterations (not measured for profile)
            profile_iters: Number of measured iterations
            export_trace: Number of measured iterations (Export Chrome trace JSON for HTA/trace-blame analysis)
            memory_snapshot: Capture CUDA memory snapshot for mosaic analysis
            torch_compile_logs: Save torch.compile logs for tlparse analysis
        """

        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters
        self.export_trace = export_trace
        self.torch_compile_logs = torch_compile_logs
        self.memory_snapshot = memory_snapshot

        # ---- Torch Profiler Args ----
        self.record_shapes = record_shapes
        self.group_by_input_shape = group_by_input_shape
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path.joinpath(Path.cwd(), "profiler_outputs")
        )

    def _detect_device(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return device   

    def _gpu_specs(self, device:torch.device):

        if device is None or not isinstance(device, torch.device):
            raise TypeError(f"Expected device=torch.Device, got {device}")

        if device.type != "cuda":
            raise ValueError(f"{device.type}, CUDA Device required")
        
        device_props = torch.cuda.get_device_properties(device)
        return GPUSpecs(
            name=device_props.name,
            sm_count=device_props.multi_processor_count,
            memory_gb=device_props.total_memory / 1024**3,
            compute_capability=(device_props.major, device_props.minor)         
        )
    
    def _generate_inputs(self, input_shape:List[int],
                        dtype:torch.dtype, device:torch.device):
        
        return torch.randn(*input_shape, device=device, dtype=dtype)

    def _validate_model_inputs(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor):

        model_device = next(model.parameters()).device
        if model_device != inputs.device:
            raise ValueError(
                f"Inputs are on {inputs.device}, "
                f"but model is on {model_device}. "
                f"Ensure `run()` or `_prepare_inputs()` "
                f"has been called first."
            )

        model_dtype = next(model.parameters()).dtype
        if model_dtype != inputs.dtype:
            raise ValueError(
                f"Inputs use dtype={inputs.dtype}, "
                f"but model uses dtype={model_dtype}."
            )        

    def _run_forward(self, model:torch.nn.Module, inputs:torch.Tensor):
        model(inputs)

    def _prepare_inputs(self, model:torch.nn.Module, 
                        inputs:torch.Tensor, 
                        input_shape:List[int],
                        device:torch.device,
                        dtype:torch.dtype):

        if inputs is None and input_shape is None:
            raise ValueError("Either inputs tensor or input_shape must be specified")

        if inputs is not None:            
            if inputs.device != device or inputs.dtype != dtype:
                inputs = inputs.to(device=device, dtype=dtype)

            original_batch = inputs.shape[0]
        else:
            original_batch = input_shape[0] if len(input_shape) >= 1 else 1
        
        half_batch = max(1, original_batch // 2)
        batch_attempts = [original_batch, half_batch, 1]

        model_device = next(model.parameters()).device
        if model_device.type != device.type:
            model.to(device)

        model_dtype = next(model.parameters()).dtype
        if model_dtype != dtype:
            model.to(dtype)

        for batch_attempt in batch_attempts:
            if inputs is not None:
                _input = inputs[:batch_attempt]
            else:
                current_shape = [batch_attempt] + input_shape[1:]
                _input = self._generate_inputs(current_shape, dtype, device)

            try:
                with torch.no_grad():
                    self._run_forward(model, _input)
                
                if batch_attempt != original_batch:
                    print(
                        f'Reduced Batch Size from {original_batch} to {batch_attempt} to fit in GPU memory'
                    )
                return _input

            except Exception as e:
                error = ForwardError(e)

                if error.oom_error:
                    if batch_attempt == 1:
                        error_msg = f"Model does not fit in GPU with batch_size={batch_attempt} "
                        if device.type == "cuda":
                            error_msg += f"- Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB "

                        raise RuntimeError(error_msg)

                    print(
                        f"OOM with batch_size={batch_attempt}, trying smaller..."
                    )
                
                #runtime error: shape error, device mismatch, dtype mismatch, indexing error .....
                else:
                    raise ValueError(f"Forward/input error: {e}\n"
                                     f"Check --input_shape or --input are correct"
                                     f"Check positional arguments are correct") from e

        raise RuntimeError("Could not run forward pass with any batch size.")
   
    def profiler_activity(self, device:torch.device):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities += [torch.profiler.ProfilerActivity.CUDA]
        if device.type == "xpu":
            activities += [torch.profiler.ProfilerActivity.XPU]

        return activities

    def profile(self, model:torch.nn.Module, inputs:torch.Tensor, device:torch.device):

        """
        Profile and return KernelRecords sorted by GPU time desc
        """

        trace_path = Path.joinpath(self.output_dir, "trace.json")
        snapshot_path = Path.joinpath(self.output_dir, "memory_snapshot.pickle")

        self._validate_model_inputs(model, inputs)
        profiler_info = ProfilerInfo()

        with torch.no_grad():
            for _ in range(self.warmup_iters):
                try:
                    self._run_forward(model, inputs)

                except Exception as e:
                    error = ForwardError(e)
                    if error.oom_error:
                        raise RuntimeError(
                            f"Model does not fit in GPU with batch_size={inputs.shape[0]}",
                            f"Call `_prepare_inputs()` before warmup to determine a device-compatible batch size."
                        )
                    else:
                        raise ValueError(f"Forward/input error: {e}\n"
                                    f"Check --inputs are model compatible") from e
                
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            if self.memory_snapshot:
                try:
                    torch.cuda.memory._record_memory_history(max_entries=100_000)
                except:
                    print(f"  WARNING: Could not start memory history recording: {e}")
                    self.memory_snapshot = False

        # --- Torch Profiler
        with torch.no_grad():
            with torch.profiler.profile(activities=self.profiler_activity(device),
                                        record_shapes=self.record_shapes,
                                        with_stack=self.with_stack,
                                        profile_memory=self.profile_memory,
                                        with_flops=True) as prof:

                for _ in range(self.profile_iters):
                    self._run_forward(model, inputs)

                    if device.type == "cuda":
                        torch.cuda.synchronize()

        if self.export_trace:
            try:
                prof.export_chrome_trace(trace_path)
                profiler_info.trace_path = trace_path
            except Exception as e:
                print(f"  WARNING: Export Chrome trace failed: {e}")

        if self.memory_snapshot and device.type == "cuda":
            try:
                snapshot = torch.cuda.memory_snapshot()
                with open(snapshot_path, "wb") as snap_f:
                    pickle.dump(snapshot, snap_f)

                profiler_info.memory_snapshot_path = snapshot_path
            except Exception as e:
                print(f"  WARNING: Memory Snapshot failed: {e}")

            finally:
                try:
                    torch.cuda.memory._record_memory_history(enabled=None)
                except Exception:
                    pass                

        event_lists : List[KernelRecord] = []

        key_averages = prof.key_averages(group_by_input_shape=self.group_by_input_shape, group_by_stack_n=True)
        for evnt in key_averages:
            kernel_record = KernelRecord(
                name=evnt.key,
                gpu_time=getattr(evnt, "self_cuda_time_total", 0.0),
                cpu_time=getattr(evnt, "self_cpu_time_total", 0.0),
                cpu_mem_usage=evnt.self_cpu_memory_usage,
                gpu_mem_usage=evnt.device_memory_usage,
                call_count=evnt.count,
                input_shapes=evnt.input_shapes,
                total_kflops=evnt.flops
            )
            event_lists.append(kernel_record)

    def run(self, model:torch.nn.Module, inputs:torch.Tensor = None, 
            input_shape:List[int] = None,
            device:torch.device = None,
            dtype:torch.dtype = torch.float16):

        """
        Args:
            Model - nn.Module object, currently only supports models within this codebase catalog

        """

        if model is None:
            raise ValueError("model=None option cannot run profiler")

        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        
        if device is None:
            device = self._detect_device()
            model.to(device)

        if dtype in (torch.float16, torch.bfloat16):
            model = model.to(dtype=dtype)

        model.eval()

        gpu_specs = None
        try:
            gpu_specs = self._gpu_specs(device)
        except Exception as e:
            print(f"[WARN] failed to get GPU specs: {e}")
        
        _input = self._prepare_inputs(model, inputs=inputs, 
                                    input_shape=input_shape, 
                                    device=device, dtype=dtype)

        print(f"Profiling..."
              f"{self.warmup_iters} warmup + "
              f"{self.profile_iters} measured iters "
              f"Device: {device} "
              f"input shape: {_input.shape} ")

        self.profile(
            model, 
            _input,
            device
        )
    