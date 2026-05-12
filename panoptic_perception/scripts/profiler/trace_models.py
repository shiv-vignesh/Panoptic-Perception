from typing import Tuple
from dataclasses import dataclass

import torch

@dataclass
class GPUSpecs:
    name : str = "unknown"
    sm_count: int = 0
    memory_gb : float = 0.0
    l2_cache_mb : float = 0.0 #NOT covered in torch.cuda.get_device_properties 
    peak_memory_bandwidth : float = 0.0 #NOT covered in torch.cuda.get_device_properties 
    compute_capability: Tuple[int, int] = (0, 0) 

@dataclass
class KernelRecord:
    name: str
    
    op_type: str = None

    gpu_time: float = None
    cpu_time: float = None

    cpu_mem_usage: int = None
    gpu_mem_usage: int = None

    call_count: int = None
    input_shapes: list = None
    total_kflops: float = None

    roofline: str = ""

    bytes_estimate: float = None
    arithmetic_intensity: float = None
    achieved_tflops: float = None
    achieved_bandwidth_gbps: float = None
    roofline_ratio: float = None    

    def __str__(self):
        return \
            f"Kernel: {self.name} " \
            f"gpu_time: {self.gpu_time} " \
            f"cpu_time: {self.cpu_time} " \
            f"cpu_mem_usage: {self.cpu_mem_usage} " \
            f"gpu_mem_usage: {self.gpu_mem_usage} " \
            f"call_count: {self.call_count} " \
            f"input_shapes: {self.input_shapes} " \
            f"total_kflops: {self.total_kflops} " \
            

@dataclass
class ProfilerInfo:
    trace_path:str = None
    memory_snapshot_path:str = None


class ForwardError:
    def __init__(self, exception:Exception):
        self._exception = exception

    @property
    def oom_error(self):
        return (
            isinstance(self._exception, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(self._exception).lower()
        )     