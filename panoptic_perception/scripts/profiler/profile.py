import torch
import pickle
from pathlib import Path

from typing import List, Any
from collections import defaultdict

from .trace_models import GPUSpecs, KernelRecord, ForwardError, ProfilerInfo, DTYPE_MAP
from .bytes_estimators import estimate_bytes
from .utils import get_pynvml_gpu_specs, get_cores_per_sm, render_kernel_table

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
        
        specs : GPUSpecs
        try:
            specs = get_pynvml_gpu_specs(device)
        except ImportError:
            print(f'pynvml not installed, fallback to torch API device properties ')
            
            specs = GPUSpecs()
            device_props = torch.cuda.get_device_properties(device)
            cc_major, cc_minor = device_props.major, device_props.minor

            clock_rate_mhz  = device_props.clock_rate / 1000

            cores_per_sm = get_cores_per_sm(cc_major, cc_minor)
            total_cores = specs.sm_count * cores_per_sm

            fp32_tflops = (total_cores * 2 * clock_rate_mhz) / 1e12
            specs.peak_tflops_by_dtype["fp32"] = round(fp32_tflops, 2)
            specs.peak_tflops_by_dtype["fp16"] = round(fp32_tflops * 2, 2)    

        return specs            

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
                total_flops=evnt.flops
            )
            event_lists.append(kernel_record)

        return event_lists

    def compute_roofline(self, event_lists:List[KernelRecord], device:torch.device, dtype:torch.dtype, gpu_specs:GPUSpecs):

        is_cpu_profile = device.type == "cpu" and gpu_specs is None
        dtype_bytes = torch.tensor([], dtype=dtype).element_size()

        if gpu_specs is None:
            print(f'[WARN] cannot compute ceil performance for kernels on non-GPU device')

        for record in event_lists:
            if record.call_count <= 0:
                continue

            bytes_est = estimate_bytes(record, dtype_bytes)
            if bytes_est is None:
                record.roofline = "no_estimator"
                continue

            flops_per_call = (record.total_flops or 0) / record.call_count
            #arthematic intensity
            ai = flops_per_call / bytes_est if bytes_est > 0 else 0.0
            record.bytes_estimate = bytes_est
            record.arithmetic_intensity = ai

            cpu_time_per_call_s = (record.cpu_time * 1e-6) / record.call_count
            record.cpu_time_per_call_s = cpu_time_per_call_s

            if is_cpu_profile and record.gpu_time <= 0:
                record.roofline = "cpu_only_profile"
                continue

            gpu_time_per_call_s = (record.gpu_time * 1e-6) / record.call_count            
            achieved_tflops = (flops_per_call / 1e12) / gpu_time_per_call_s if flops_per_call else 0.0
            achieved_bw_gbps = (bytes_est / 1e9) / gpu_time_per_call_s if bytes_est > 0 else 0.0

            record.gpu_time_per_call_s = gpu_time_per_call_s
            record.achieved_bandwidth_gbps = achieved_bw_gbps
            record.achieved_tflops = achieved_tflops

            ceiling = None
            ridge = None
            if gpu_specs:
                _dtype_str = DTYPE_MAP[dtype]
                if dtype not in gpu_specs.peak_tflops_by_dtype:
                    print(f'[WARN] cannot compute ceil for dtype: {_dtype_str}'\
                          f'Supported dtypes: {gpu_specs.peak_tflops_by_dtype.keys()}')
                    continue

                ceiling = min(gpu_specs.peak_tflops_by_dtype[_dtype_str], 
                              gpu_specs.peak_memory_bandwidth * ai / 1000)

                ridge = gpu_specs.peak_tflops_by_dtype[_dtype_str] * 1000 / gpu_specs.peak_memory_bandwidth  # in FLOPs/byte

            record.roofline_ratio = (achieved_tflops / ceiling) if ceiling and ceiling > 0 else 0.0
            record.roofline_ceiling_tflops = ceiling if ceiling and ceiling > 0 else 0.0
            if ridge:            
                if ai == 0:
                    record.roofline = "no_flops"
                elif ai < ridge * 0.5:
                    record.roofline = "memory_bound"
                elif ai > ridge * 2:
                    record.roofline = "compute_bound"
                else:
                    record.roofline = "balanced"


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

        event_lists = self.profile(
            model, 
            _input,
            device
        )

        self.compute_roofline(event_lists, device, dtype, gpu_specs)
        render_table = render_kernel_table(event_lists)
        print(render_table)
    