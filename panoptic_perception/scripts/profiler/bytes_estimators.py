import torch
import math

from typing import Sequence, Optional, Callable

from .trace_models import KernelRecord

ShapeList = Sequence[Sequence[int]]
EstimatorFn = Callable[[ShapeList, int], int]

def _numel(shape: Sequence[int]) -> int:
    """Total Elements in a shape tuple. Empty Shape -> 0"""
    return math.prod(shape) if shape else 0

def estimate_bytes_conv2d(input_shapes: ShapeList, dtype_bytes: int) -> int:
    """
    Inputs: [N, C_in, H, W], [C_out, C_in, kH, kW], optional [C_out] bias.
    Output: [N, C_out, H_out, W_out].

    Stride/padding aren't recoverable from input_shapes alone, so we assume
    same-padding (H_out=H, W_out=W). For stride>1 layers this overestimates 
    output bytes; treat as a known approximation.
    """

    if len(input_shapes) < 2:
        return 0
    
    _in, weight = input_shapes[0], input_shapes[1]
    bias = input_shapes[2] if len(input_shapes) > 2 and input_shapes[2] else None

    if len(_in) < 4 or len(weight) < 4:
        return 0
    
    N, C_in, H, W = _in[:4]
    C_out = weight[0]

    bytes_in = N * C_in * H * W * dtype_bytes
    bytes_w = _numel(weight) * dtype_bytes
    bytes_b = (_numel(bias) * dtype_bytes) if bias else 0

    bytes_out = N * C_out * H * W * dtype_bytes  # same-padding assumption
    return bytes_in + bytes_w + bytes_b + bytes_out

def estimate_bytes_batchnorm(input_shapes:ShapeList, dtype_bytes:int):
    """
    batch_norm / native_batch_norm / _batch_norm_impl_index.

    Reads input [N, C, ...] and 4 small per-channel param tensors of size C
    (scale, bias, running_mean, running_var). Writes output of input shape.
    The 4*C term is negligible vs 2*N*C*H*W but kept for correctness.
    """

    if not input_shapes or not input_shapes[0]:
        return 0
    
    _in = input_shapes[0]
    n = _numel(_in)
    C = _in[1] if len(_in) >= 2 else _in[-1]
    return (2 * n + 4 * C) * dtype_bytes

def estimate_bytes_pool2d(input_shapes: ShapeList, dtype_bytes: int) -> int:
    pass

def estimate_bytes_pool2d_with_indices(input_shapes: ShapeList, dtype_bytes: int) -> int:
    pass

def estimate_bytes_elementwise_unary(input_shapes: ShapeList, dtype_bytes: int) -> int:
    """
    Always 1 read + 1 write of the input shape. The cheapest, most cache-friendly kernels 
    usually severely memory-bound (AI ~0)    

    TODO, inplace=True ops dont have 1 write
    """

    if not input_shapes:
        return 0
    
    return 2 * _numel(input_shapes[0]) * dtype_bytes

def estimate_bytes_elementwise_binary(input_shapes: ShapeList, dtype_bytes:int) -> int:
    """
    Reads both operands (each contributes its own size — broadcast-aware in
    that the smaller broadcast-side is also fetched into registers / cache)
    writes one output of the broadcast shape (max of the two element counts) 

    Falls back to unary cost if second arg is missing or scalar
    """

    if not input_shapes:
        return 0
    if len(input_shapes) < 2 or not input_shapes[1]:
        return estimate_bytes_elementwise_unary(input_shapes, dtype_bytes)
    
    a, b = input_shapes[0], input_shapes[1]
    output_shape = max(_numel(a), _numel(b))

    return (_numel(a) + _numel(b) + output_shape) * dtype_bytes

def estimate_bytes_upsample_nearest2d(input_shapes: ShapeList, dtype_bytes:int) -> int:
    """
    Assumes 2x upsample on each H and W axis. TODO
    scale_factor isn't recoverable from input_shapes

    1 read + 1 write (4x), for larger upsampling this underestimates the write side
    """

    if not input_shapes:
        return 0
    
    N, C, H, W = input_shapes[0][:4]
    bytes_in = N * C * H * W * dtype_bytes
    bytes_out = N * C * (H * 2) * (W * 2) * dtype_bytes

    return bytes_in + bytes_out

def estimate_bytes_cat(input_shapes: ShapeList, dtype_bytes:int) -> int:
    """
    Reads all input tensors, writes one output of summed size.
    Total DRAM traffic ≈ 2 * sum(input_sizes).
    """

    if not input_shapes:
        return 0
    
    total = sum([_numel(s) for s in input_shapes if s])
    return 2 * total * dtype_bytes

def estimate_bytes_stack(input_shapes: ShapeList, dtype_bytes:int) -> int:
    return estimate_bytes_cat(input_shapes, dtype_bytes)

def estimate_bytes_copy_op(input_shapes: ShapeList, dtype_bytes:int) -> int:
    """
    clone / copy_ - full duplicate of input.
    1 read + 1 write of input shape.
    """

    if not input_shapes:
        return
    
    return 2 * _numel(input_shapes) * dtype_bytes

# =========================================================================

 # NO-OP / METADATA / ALLOCATION
# View ops, allocations, and metadata-only kernels. These don't move data
# through DRAM in any meaningful sense — they adjust tensor descriptors or
# reserve memory regions whose actual cost is paid by subsequent ops.
# Returning 0 here causes the analyzer to skip them from roofline math
# (don't try to compute AI = FLOPs / 0).
# =========================================================================

def estimate_bytes_zero(input_shapes: ShapeList, dtype_bytes:int) -> int:
    return 0

def estimate_bytes_alloc(input_shapes: ShapeList, dtype_bytes: int) -> int:
    return 0

def estimate_bytes_fill(input_shapes: ShapeList, dtype_bytes: int) -> int:
    """
    Constant-fill creation: ones / fill_ / arange.

    1 write of the output tensor, no reads. Uses input_shapes[0] as the 
    output shape (this is how torch profiler reports them for these ops).
    """

    if not input_shapes:
        return 0
    return _numel(input_shapes[0]) * dtype_bytes

BYTES_ESTIMATORS: dict[str, EstimatorFn] = {
    # ---- COMPUTE: convolutions ----
    "aten::conv2d":                  estimate_bytes_conv2d,
    "aten::convolution":             estimate_bytes_conv2d,
    "aten::_convolution":            estimate_bytes_conv2d,
    "aten::_slow_conv2d_forward":    estimate_bytes_conv2d,
    "aten::thnn_conv2d":             estimate_bytes_conv2d,

    # ---- COMPUTE: normalization ----
    "aten::batch_norm":              estimate_bytes_batchnorm,
    "aten::native_batch_norm":       estimate_bytes_batchnorm,
    "aten::_batch_norm_impl_index":  estimate_bytes_batchnorm,

    # ---- COMPUTE: pooling ----
    "aten::max_pool2d":              estimate_bytes_pool2d,
    "aten::max_pool2d_with_indices": estimate_bytes_pool2d_with_indices,

    # ---- COMPUTE: activations ----
    "aten::sigmoid":                 estimate_bytes_elementwise_unary,
    "aten::leaky_relu":              estimate_bytes_elementwise_unary,
    "aten::hardswish":               estimate_bytes_elementwise_unary,

    # ---- COMPUTE: elementwise binary ----
    "aten::add":                     estimate_bytes_elementwise_binary,
    "aten::sub":                     estimate_bytes_elementwise_binary,
    "aten::mul":                     estimate_bytes_elementwise_binary,
    "aten::pow":                     estimate_bytes_elementwise_binary,

    # ---- COMPUTE: spatial resampling ----
    "aten::upsample_nearest2d":      estimate_bytes_upsample_nearest2d,

    # ---- MEMORY: assembly / disassembly ----
    "aten::cat":                     estimate_bytes_cat,
    "aten::stack":                   estimate_bytes_stack,

    # ---- MEMORY: copies ----
    "aten::clone":                   estimate_bytes_copy_op, #estimate_bytes_clone,
    "aten::copy_":                   estimate_bytes_copy_op, #estimate_bytes_clone,
    "aten::_to_copy":                estimate_bytes_copy_op, #estimate_bytes_to_copy,
    "aten::to":                      estimate_bytes_copy_op, #estimate_bytes_to_copy,
    "aten::contiguous":              estimate_bytes_copy_op, #estimate_bytes_contiguous,

    # ---- NO-OP: views / metadata ----
    "aten::view":                    estimate_bytes_zero,
    "aten::reshape":                 estimate_bytes_zero,
    "aten::permute":                 estimate_bytes_zero,
    "aten::expand":                  estimate_bytes_zero,
    "aten::unsqueeze":               estimate_bytes_zero,
    "aten::slice":                   estimate_bytes_zero,
    "aten::select":                  estimate_bytes_zero,
    "aten::narrow":                  estimate_bytes_zero,
    "aten::as_strided":              estimate_bytes_zero,
    "aten::detach":                  estimate_bytes_zero,
    "aten::result_type":             estimate_bytes_zero,
    "aten::resize_":                 estimate_bytes_zero,
    "aten::_nnpack_available":       estimate_bytes_zero,
    "aten::meshgrid":                estimate_bytes_zero,
    "[memory]":                      estimate_bytes_zero,

    # ---- NO-OP: allocations ----
    "aten::empty":                   estimate_bytes_alloc,
    "aten::empty_like":              estimate_bytes_alloc,
    "aten::empty_strided":           estimate_bytes_alloc,

    # ---- NO-OP: constant fills (writes only, no reads) ----
    "aten::ones":                    estimate_bytes_fill,
    "aten::fill_":                   estimate_bytes_fill,
    "aten::arange":                  estimate_bytes_fill,
}

def estimate_bytes(record:KernelRecord, dtype_bytes:int):
    
    estimator = BYTES_ESTIMATORS.get(record.name)
    if estimator is None:
        return None
    try:
        return estimator(record.input_shapes, dtype_bytes)
    except (IndexError, ValueError, TypeError):
        return None