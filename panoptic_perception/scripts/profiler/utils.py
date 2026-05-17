
from .trace_models import GPUSpecs, KernelRecord

from typing import List      
from tabulate import tabulate         

import torch
import ctypes

def get_cores_per_sm(major: int, minor: int) -> int:
    mapping = {
        7: {0: 64, 2: 64, 5: 64},       # Volta, Turing
        8: {0: 64, 6: 128, 9: 128},     # Ampere, Ada Lovelace
        9: {0: 128},                    # Hopper
        10: {0: 128},                   # Blackwell Enterprise (10.0)
        12: {0: 128}                    # Blackwell Consumer/Mobile (12.0)
    }
    return mapping.get(major, {}).get(minor, 128)

def get_pynvml_gpu_specs(device:torch.device) -> GPUSpecs:

    specs = GPUSpecs()
    try:
        import pynvml
    except ImportError:
        raise

    # torch.device("cuda") leaves index as None; resolve to a concrete int
    # before handing to NVML / CUDA Runtime (both expect uint device IDs).
    device_index = device.index if device.index is not None else torch.cuda.current_device()

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(handle)
        specs.name = name.decode("utf-8") if isinstance(name, bytes) else name

        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        specs.memory_gb = round(mem_info.total / (1024**3), 2)

        # Calculate Peak Memory Bandwidth (requires max memory clock & bus width)
        # Formula: (Memory Clock * 2 [for DDR] * Bus Width in bits) / 8 bits-per-byte / 1e9
        mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        bus_width_bits = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
        specs.peak_memory_bandwidth = round((mem_clock_mhz * 1e6 * 2 * bus_width_bits) / 8 / 1e9, 2)

        # Max Graphics Clock needed for TFLOPS calculation
        max_gfx_clock_hz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS) * 1e6

    finally:
        pynvml.nvmlShutdown()

    # 3. Gather L2 Cache size via native CUDA Runtime API (not exposed by PyTorch/NVML)
    try:
        cuda = ctypes.CDLL('libcudart.so' if torch.sys.platform != 'win32' else 'cudart64_120.dll')
        l2_size = ctypes.c_int()
        # 39 is the enum value for cudaDevAttrL2CacheSize
        if cuda.cudaDeviceGetAttribute(ctypes.byref(l2_size), 39, device_index) == 0:
            specs.l2_cache_mb = round(l2_size.value / (1024**2), 2)
    except Exception:
        specs.l2_cache_mb = 0.0

    # 4. Calculate Theoretical Peak TFLOPS
    # Note: FP32 processing cores per SM vary drastically by NVIDIA Architecture (Compute Capability)
    device_props = torch.cuda.get_device_properties(device)
    specs.sm_count = device_props.multi_processor_count
    cc_major, cc_minor = device_props.major, device_props.minor

    cores_per_sm = get_cores_per_sm(cc_major, cc_minor)
    total_cores = specs.sm_count * cores_per_sm

    fp32_tflops = (total_cores * 2 * max_gfx_clock_hz) / 1e12
    specs.peak_tflops_by_dtype["fp32"] = round(fp32_tflops, 2)
    specs.peak_tflops_by_dtype["fp16"] = round(fp32_tflops * 2, 2)

    return specs

def render_kernel_table(records: List[KernelRecord], top_n: int = 30, fmt: str = "github") -> str:
    """Top-N kernels by GPU time, with roofline diagnostics per row."""              
    records = sorted(records, key=lambda r: r.gpu_time or 0, reverse=True)           
    total_gpu = sum(r.gpu_time or 0 for r in records) or 1
   
    rows = []              
    for r in records[:top_n]:
        rows.append([      
            r.name[:40],
            r.call_count,
            f"{(r.gpu_time or 0) / 1000:.2f}",                  # ms                 
            f"{(r.gpu_time or 0) / total_gpu * 100:.1f}%",
            f"{r.arithmetic_intensity:.2f}" if r.arithmetic_intensity else "-",      
            f"{r.achieved_tflops:.2f}" if r.achieved_tflops else "-",                
            f"{r.achieved_bandwidth_gbps:.1f}" if r.achieved_bandwidth_gbps else "-",
            f"{r.roofline_ceiling_tflops:.2f}" if r.roofline_ceiling_tflops else "-",
            f"{r.roofline_ratio * 100:.0f}%" if r.roofline_ratio else "-",           
            r.roofline or "-",          
        ])                   
   
    headers = ["kernel", "calls", "gpu_ms", "%t", "AI", "TFLOPS", "BW GB/s", "ceil", "roof_eff", "bound"]
    return tabulate(rows, headers=headers, tablefmt=fmt)  