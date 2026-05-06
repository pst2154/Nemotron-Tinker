# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grouped mixed-adapter LoRA Triton kernels for the Tinker API prototype."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from nemo_automodel.shared.import_utils import MISSING_TRITON_MSG, null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_GROUPED_LORA_TRITON = bool(torch.cuda.is_available())
except ImportError:
    HAVE_GROUPED_LORA_TRITON = False

if not HAVE_GROUPED_LORA_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


@triton.jit
def _grouped_lora_forward_kernel(
    x_ptr,
    adapter_indices_ptr,
    lora_a_ptr,
    lora_b_ptr,
    out_ptr,
    M,
    K,
    R,
    L,
    stride_x_m,
    stride_x_k,
    stride_a_adapter,
    stride_a_r,
    stride_a_k,
    stride_b_adapter,
    stride_b_l,
    stride_b_r,
    stride_out_m,
    stride_out_l,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Compute LoRA deltas for rows that may each select a different adapter."""
    pid_m = tl.program_id(axis=0)
    pid_l = tl.program_id(axis=1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    row_mask = rows < M
    adapter_ids = tl.load(adapter_indices_ptr + rows, mask=row_mask, other=0)

    acc = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)
    for r_idx in tl.range(0, R):
        hidden = tl.zeros((BLOCK_M,), dtype=tl.float32)
        k_offsets = tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + rows[:, None] * stride_x_m + k_offsets[None, :] * stride_x_k
        a_ptrs = (
            lora_a_ptr + adapter_ids[:, None] * stride_a_adapter + r_idx * stride_a_r + k_offsets[None, :] * stride_a_k
        )
        for k_start in tl.range(0, K, BLOCK_K):
            k_mask = k_offsets < K - k_start
            x = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            hidden += tl.sum(x * a, axis=1)
            x_ptrs += BLOCK_K * stride_x_k
            a_ptrs += BLOCK_K * stride_a_k

        b_ptrs = lora_b_ptr + adapter_ids[:, None] * stride_b_adapter + cols[None, :] * stride_b_l + r_idx * stride_b_r
        b_mask = row_mask[:, None] & (cols[None, :] < L)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += hidden[:, None] * b

    out_ptrs = out_ptr + rows[:, None] * stride_out_m + cols[None, :] * stride_out_l
    out_mask = row_mask[:, None] & (cols[None, :] < L)
    tl.store(out_ptrs, acc * scale, mask=out_mask)


def grouped_lora_forward_wrapper(
    x: torch.Tensor,
    adapter_indices: torch.Tensor,
    lora_a_bank: torch.Tensor,
    lora_b_bank: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Launch the grouped mixed-adapter LoRA forward kernel."""
    if not HAVE_GROUPED_LORA_TRITON:
        raise ImportError(MISSING_TRITON_MSG)
    if x.dim() != 2:
        raise ValueError("grouped_lora_forward_wrapper expects a 2D input tensor")
    if adapter_indices.dim() != 1 or adapter_indices.shape[0] != x.shape[0]:
        raise ValueError("adapter_indices must be a 1D tensor with one id per input row")
    if lora_a_bank.dim() != 3 or lora_b_bank.dim() != 3:
        raise ValueError("lora_a_bank and lora_b_bank must be 3D tensors")
    if lora_a_bank.shape[0] != lora_b_bank.shape[0]:
        raise ValueError("LoRA adapter banks must contain the same number of adapters")
    if lora_a_bank.shape[1] != lora_b_bank.shape[2]:
        raise ValueError("LoRA adapter banks must have matching rank dimensions")
    if x.shape[1] != lora_a_bank.shape[2]:
        raise ValueError("Input hidden size must match LoRA A input dimension")

    x = x.contiguous()
    adapter_indices = adapter_indices.to(device=x.device, dtype=torch.int64).contiguous()
    lora_a_bank = lora_a_bank.contiguous()
    lora_b_bank = lora_b_bank.contiguous()
    out = torch.empty((x.shape[0], lora_b_bank.shape[1]), device=x.device, dtype=dtype)
    m, k = x.shape
    _, rank, _ = lora_a_bank.shape
    _, out_features, _ = lora_b_bank.shape
    block_l = min(64, triton.next_power_of_2(out_features))
    grid = (triton.cdiv(m, 16), triton.cdiv(out_features, block_l))
    _grouped_lora_forward_kernel[grid](
        x,
        adapter_indices,
        lora_a_bank,
        lora_b_bank,
        out,
        m,
        k,
        rank,
        out_features,
        x.stride(0),
        x.stride(1),
        lora_a_bank.stride(0),
        lora_a_bank.stride(1),
        lora_a_bank.stride(2),
        lora_b_bank.stride(0),
        lora_b_bank.stride(1),
        lora_b_bank.stride(2),
        out.stride(0),
        out.stride(1),
        scale,
        BLOCK_M=16,
        BLOCK_K=64,
        BLOCK_L=block_l,
    )
    return out


@triton.jit
def _grouped_lora_dx_kernel(
    grad_out_ptr,
    adapter_indices_ptr,
    lora_a_ptr,
    lora_b_ptr,
    grad_x_ptr,
    M,
    K,
    R,
    L,
    stride_grad_m,
    stride_grad_l,
    stride_a_adapter,
    stride_a_r,
    stride_a_k,
    stride_b_adapter,
    stride_b_l,
    stride_b_r,
    stride_grad_x_m,
    stride_grad_x_k,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Compute input gradients for grouped mixed-adapter LoRA rows."""
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    adapter_ids = tl.load(adapter_indices_ptr + rows, mask=row_mask, other=0)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for r_idx in tl.range(0, R):
        grad_hidden = tl.zeros((BLOCK_M,), dtype=tl.float32)
        cols_l = tl.arange(0, BLOCK_L)
        grad_ptrs = grad_out_ptr + rows[:, None] * stride_grad_m + cols_l[None, :] * stride_grad_l
        b_ptrs = (
            lora_b_ptr + adapter_ids[:, None] * stride_b_adapter + cols_l[None, :] * stride_b_l + r_idx * stride_b_r
        )
        for l_start in tl.range(0, L, BLOCK_L):
            l_mask = cols_l < L - l_start
            grad_out = tl.load(grad_ptrs, mask=row_mask[:, None] & l_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=row_mask[:, None] & l_mask[None, :], other=0.0)
            grad_hidden += tl.sum(grad_out * b, axis=1)
            grad_ptrs += BLOCK_L * stride_grad_l
            b_ptrs += BLOCK_L * stride_b_l

        a_ptrs = (
            lora_a_ptr + adapter_ids[:, None] * stride_a_adapter + r_idx * stride_a_r + cols_k[None, :] * stride_a_k
        )
        a_mask = row_mask[:, None] & (cols_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        acc += grad_hidden[:, None] * a

    grad_x_ptrs = grad_x_ptr + rows[:, None] * stride_grad_x_m + cols_k[None, :] * stride_grad_x_k
    grad_x_mask = row_mask[:, None] & (cols_k[None, :] < K)
    tl.store(grad_x_ptrs, acc * scale, mask=grad_x_mask)


def grouped_lora_dx_wrapper(
    grad_out: torch.Tensor,
    adapter_indices: torch.Tensor,
    lora_a_bank: torch.Tensor,
    lora_b_bank: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Launch the grouped mixed-adapter LoRA input-gradient kernel."""
    if not HAVE_GROUPED_LORA_TRITON:
        raise ImportError(MISSING_TRITON_MSG)
    if grad_out.dim() != 2:
        raise ValueError("grouped_lora_dx_wrapper expects a 2D grad_out tensor")
    if adapter_indices.dim() != 1 or adapter_indices.shape[0] != grad_out.shape[0]:
        raise ValueError("adapter_indices must be a 1D tensor with one id per grad row")

    grad_out = grad_out.contiguous()
    adapter_indices = adapter_indices.to(device=grad_out.device, dtype=torch.int64).contiguous()
    lora_a_bank = lora_a_bank.contiguous()
    lora_b_bank = lora_b_bank.contiguous()
    m, out_features = grad_out.shape
    _, rank, hidden = lora_a_bank.shape
    grad_x = torch.empty((m, hidden), device=grad_out.device, dtype=dtype)
    block_k = min(64, triton.next_power_of_2(hidden))
    block_l = min(64, triton.next_power_of_2(out_features))
    grid = (triton.cdiv(m, 16), triton.cdiv(hidden, block_k))
    _grouped_lora_dx_kernel[grid](
        grad_out,
        adapter_indices,
        lora_a_bank,
        lora_b_bank,
        grad_x,
        m,
        hidden,
        rank,
        out_features,
        grad_out.stride(0),
        grad_out.stride(1),
        lora_a_bank.stride(0),
        lora_a_bank.stride(1),
        lora_a_bank.stride(2),
        lora_b_bank.stride(0),
        lora_b_bank.stride(1),
        lora_b_bank.stride(2),
        grad_x.stride(0),
        grad_x.stride(1),
        scale,
        BLOCK_M=16,
        BLOCK_K=block_k,
        BLOCK_L=block_l,
    )
    return grad_x


@triton.jit
def _grouped_lora_da_kernel(
    x_ptr,
    grad_out_ptr,
    adapter_indices_ptr,
    lora_b_ptr,
    grad_a_ptr,
    M,
    K,
    R,
    L,
    stride_x_m,
    stride_x_k,
    stride_grad_m,
    stride_grad_l,
    stride_b_adapter,
    stride_b_l,
    stride_b_r,
    stride_grad_a_adapter,
    stride_grad_a_r,
    stride_grad_a_k,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Compute grouped LoRA A gradients for one adapter/rank/input-feature tile."""
    adapter_id = tl.program_id(axis=0)
    r_idx = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)
    cols_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rows = tl.arange(0, BLOCK_M)
    cols_l = tl.arange(0, BLOCK_L)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for m_start in tl.range(0, M, BLOCK_M):
        row_offsets = m_start + rows
        row_mask = row_offsets < M
        row_adapter_ids = tl.load(adapter_indices_ptr + row_offsets, mask=row_mask, other=-1)
        adapter_mask = row_adapter_ids == adapter_id
        grad_hidden = tl.zeros((BLOCK_M,), dtype=tl.float32)
        grad_ptrs = grad_out_ptr + row_offsets[:, None] * stride_grad_m + cols_l[None, :] * stride_grad_l
        b_ptrs = lora_b_ptr + adapter_id * stride_b_adapter + cols_l * stride_b_l + r_idx * stride_b_r
        for l_start in tl.range(0, L, BLOCK_L):
            l_mask = cols_l < L - l_start
            grad_out = tl.load(grad_ptrs, mask=row_mask[:, None] & l_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=l_mask, other=0.0)
            grad_hidden += tl.sum(grad_out * b[None, :], axis=1)
            grad_ptrs += BLOCK_L * stride_grad_l
            b_ptrs += BLOCK_L * stride_b_l

        x_ptrs = x_ptr + row_offsets[:, None] * stride_x_m + cols_k[None, :] * stride_x_k
        x_mask = row_mask[:, None] & adapter_mask[:, None] & (cols_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        acc += tl.sum((grad_hidden * scale)[:, None] * x, axis=0)

    grad_a_ptrs = grad_a_ptr + adapter_id * stride_grad_a_adapter + r_idx * stride_grad_a_r + cols_k * stride_grad_a_k
    tl.store(grad_a_ptrs, acc, mask=cols_k < K)


@triton.jit
def _grouped_lora_db_kernel(
    x_ptr,
    grad_out_ptr,
    adapter_indices_ptr,
    lora_a_ptr,
    grad_b_ptr,
    M,
    K,
    R,
    L,
    stride_x_m,
    stride_x_k,
    stride_grad_m,
    stride_grad_l,
    stride_a_adapter,
    stride_a_r,
    stride_a_k,
    stride_grad_b_adapter,
    stride_grad_b_l,
    stride_grad_b_r,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Compute grouped LoRA B gradients for one adapter/output-feature/rank tile."""
    adapter_id = tl.program_id(axis=0)
    pid_l = tl.program_id(axis=1)
    r_idx = tl.program_id(axis=2)
    cols_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    rows = tl.arange(0, BLOCK_M)
    cols_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

    for m_start in tl.range(0, M, BLOCK_M):
        row_offsets = m_start + rows
        row_mask = row_offsets < M
        row_adapter_ids = tl.load(adapter_indices_ptr + row_offsets, mask=row_mask, other=-1)
        adapter_mask = row_adapter_ids == adapter_id
        hidden = tl.zeros((BLOCK_M,), dtype=tl.float32)
        x_ptrs = x_ptr + row_offsets[:, None] * stride_x_m + cols_k[None, :] * stride_x_k
        a_ptrs = lora_a_ptr + adapter_id * stride_a_adapter + r_idx * stride_a_r + cols_k * stride_a_k
        for k_start in tl.range(0, K, BLOCK_K):
            k_mask = cols_k < K - k_start
            x = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            a = tl.load(a_ptrs, mask=k_mask, other=0.0)
            hidden += tl.sum(x * a[None, :], axis=1)
            x_ptrs += BLOCK_K * stride_x_k
            a_ptrs += BLOCK_K * stride_a_k

        grad_ptrs = grad_out_ptr + row_offsets[:, None] * stride_grad_m + cols_l[None, :] * stride_grad_l
        grad_mask = row_mask[:, None] & adapter_mask[:, None] & (cols_l[None, :] < L)
        grad_out = tl.load(grad_ptrs, mask=grad_mask, other=0.0)
        acc += tl.sum(grad_out * (hidden * scale)[:, None], axis=0)

    grad_b_ptrs = grad_b_ptr + adapter_id * stride_grad_b_adapter + cols_l * stride_grad_b_l + r_idx * stride_grad_b_r
    tl.store(grad_b_ptrs, acc, mask=cols_l < L)


def grouped_lora_da_db_wrapper(
    x: torch.Tensor,
    grad_out: torch.Tensor,
    adapter_indices: torch.Tensor,
    lora_a_bank: torch.Tensor,
    lora_b_bank: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch grouped mixed-adapter LoRA adapter-gradient kernels."""
    if not HAVE_GROUPED_LORA_TRITON:
        raise ImportError(MISSING_TRITON_MSG)
    if x.dim() != 2 or grad_out.dim() != 2:
        raise ValueError("grouped_lora_da_db_wrapper expects 2D x and grad_out tensors")
    if adapter_indices.dim() != 1 or adapter_indices.shape[0] != x.shape[0]:
        raise ValueError("adapter_indices must be a 1D tensor with one id per input row")

    x = x.contiguous()
    grad_out = grad_out.contiguous()
    adapter_indices = adapter_indices.to(device=x.device, dtype=torch.int64).contiguous()
    lora_a_bank = lora_a_bank.contiguous()
    lora_b_bank = lora_b_bank.contiguous()
    m, hidden = x.shape
    _, out_features = grad_out.shape
    adapter_count, rank, _ = lora_a_bank.shape
    grad_a = torch.empty_like(lora_a_bank)
    grad_b = torch.empty_like(lora_b_bank)
    block_k = min(64, triton.next_power_of_2(hidden))
    block_l = min(64, triton.next_power_of_2(out_features))
    _grouped_lora_da_kernel[(adapter_count, rank, triton.cdiv(hidden, block_k))](
        x,
        grad_out,
        adapter_indices,
        lora_b_bank,
        grad_a,
        m,
        hidden,
        rank,
        out_features,
        x.stride(0),
        x.stride(1),
        grad_out.stride(0),
        grad_out.stride(1),
        lora_b_bank.stride(0),
        lora_b_bank.stride(1),
        lora_b_bank.stride(2),
        grad_a.stride(0),
        grad_a.stride(1),
        grad_a.stride(2),
        scale,
        BLOCK_M=16,
        BLOCK_K=block_k,
        BLOCK_L=block_l,
    )
    _grouped_lora_db_kernel[(adapter_count, triton.cdiv(out_features, block_l), rank)](
        x,
        grad_out,
        adapter_indices,
        lora_a_bank,
        grad_b,
        m,
        hidden,
        rank,
        out_features,
        x.stride(0),
        x.stride(1),
        grad_out.stride(0),
        grad_out.stride(1),
        lora_a_bank.stride(0),
        lora_a_bank.stride(1),
        lora_a_bank.stride(2),
        grad_b.stride(0),
        grad_b.stride(1),
        grad_b.stride(2),
        scale,
        BLOCK_M=16,
        BLOCK_K=block_k,
        BLOCK_L=block_l,
    )
    return grad_a, grad_b


class GroupedLoRATritonFunction(torch.autograd.Function):
    """Grouped LoRA autograd wrapper backed by Triton kernels."""

    @staticmethod
    def forward(ctx, x, adapter_indices, lora_a_bank, lora_b_bank, scale, dtype):
        """Run grouped LoRA forward."""
        reshape = x.dim() == 3
        if reshape:
            batch, seq_len, hidden = x.shape
            x_2d = x.reshape(-1, hidden)
            adapter_indices_2d = adapter_indices.repeat_interleave(seq_len)
        else:
            x_2d = x
            adapter_indices_2d = adapter_indices

        out = grouped_lora_forward_wrapper(x_2d, adapter_indices_2d, lora_a_bank, lora_b_bank, scale, dtype)
        ctx.save_for_backward(x_2d, adapter_indices_2d, lora_a_bank, lora_b_bank)
        ctx.scale = scale
        ctx.reshape = reshape
        ctx.original_shape = x.shape
        if reshape:
            return out.view(batch, seq_len, -1)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Compute grouped LoRA gradients."""
        x_2d, adapter_indices, lora_a_bank, lora_b_bank = ctx.saved_tensors
        scale = ctx.scale
        grad_2d = grad_out.reshape(-1, grad_out.shape[-1]).to(lora_b_bank.dtype)
        x_lora = x_2d.to(lora_a_bank.dtype)

        grad_x = grouped_lora_dx_wrapper(grad_2d, adapter_indices, lora_a_bank, lora_b_bank, scale, x_2d.dtype)
        grad_a_bank, grad_b_bank = grouped_lora_da_db_wrapper(
            x_lora,
            grad_2d,
            adapter_indices,
            lora_a_bank,
            lora_b_bank,
            scale,
        )

        if ctx.reshape:
            grad_x = grad_x.view(ctx.original_shape)
        return grad_x.to(grad_out.dtype), None, grad_a_bank, grad_b_bank, None, None
