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

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from nemotron_tinker.mixed_client import MixedAdapterLinearLoRA, MixedLoraBackend  # noqa: E402


def build_layer(
    *,
    backend: MixedLoraBackend,
    adapters: list[str],
    hidden_size: int,
    out_features: int,
    rank: int,
    device: torch.device,
) -> MixedAdapterLinearLoRA:
    """Create one benchmark layer with deterministic adapter weights."""
    base = torch.nn.Linear(hidden_size, out_features, bias=False, device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(1234)
    base.weight.data.normal_(mean=0.0, std=0.02, generator=generator)
    layer = MixedAdapterLinearLoRA(
        base,
        rank=rank,
        alpha=rank * 2,
        dropout=0.0,
        lora_dtype=torch.float32,
        backend=backend,
    )
    for adapter_id in adapters:
        layer.add_adapter(adapter_id)
        layer.lora_a[adapter_id].data.normal_(mean=0.0, std=0.02, generator=generator)
        layer.lora_b[adapter_id].data.normal_(mean=0.0, std=0.02, generator=generator)
    return layer


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA work when benchmarking on GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_backend(
    *,
    backend: MixedLoraBackend,
    x: torch.Tensor,
    active_ranges: list[tuple[str, int, int]],
    adapters: list[str],
    hidden_size: int,
    out_features: int,
    rank: int,
    warmup_steps: int,
    measured_steps: int,
    reference: torch.Tensor | None,
) -> dict[str, float | str]:
    """Benchmark one mixed-LoRA backend."""
    layer = build_layer(
        backend=backend,
        adapters=adapters,
        hidden_size=hidden_size,
        out_features=out_features,
        rank=rank,
        device=x.device,
    )
    layer.set_active_ranges(active_ranges)
    for _ in range(warmup_steps):
        loss = layer(x).pow(2).mean()
        loss.backward()
        layer.zero_grad(set_to_none=True)
    synchronize(x.device)

    start = time.perf_counter()
    for _ in range(measured_steps):
        loss = layer(x).pow(2).mean()
        loss.backward()
        layer.zero_grad(set_to_none=True)
    synchronize(x.device)
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        output = layer(x)
    max_abs_diff = 0.0 if reference is None else float((output - reference).abs().max().detach().cpu())
    tokens = x.shape[0] * (x.shape[1] if x.dim() == 3 else 1)
    return {
        "backend": backend,
        "ms_per_step": elapsed * 1000.0 / measured_steps,
        "tokens_per_second": tokens * measured_steps / elapsed,
        "max_abs_diff_vs_loop": max_abs_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mixed-LoRA adapter backends.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--num-adapters", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["loop", "grouped", "triton", "grouped_triton"],
        choices=["loop", "grouped", "triton", "grouped_triton"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    adapters = [f"adapter_{idx}" for idx in range(args.num_adapters)]
    rows_per_adapter = max(1, args.batch_size // args.num_adapters)
    active_ranges = []
    start = 0
    for adapter_id in adapters:
        end = min(args.batch_size, start + rows_per_adapter)
        if adapter_id == adapters[-1]:
            end = args.batch_size
        if start < end:
            active_ranges.append((adapter_id, start, end))
        start = end

    generator = torch.Generator(device=device).manual_seed(2025)
    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device, generator=generator)
    reference_layer = build_layer(
        backend="loop",
        adapters=adapters,
        hidden_size=args.hidden_size,
        out_features=args.out_features,
        rank=args.rank,
        device=device,
    )
    reference_layer.set_active_ranges(active_ranges)
    with torch.no_grad():
        reference = reference_layer(x).detach()

    print("backend,ms_per_step,tokens_per_second,max_abs_diff_vs_loop")
    for backend in args.backends:
        stats = run_backend(
            backend=backend,
            x=x.detach().clone().requires_grad_(True),
            active_ranges=active_ranges,
            adapters=adapters,
            hidden_size=args.hidden_size,
            out_features=args.out_features,
            rank=args.rank,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            reference=reference,
        )
        print(
            f"{stats['backend']},{stats['ms_per_step']:.4f},"
            f"{stats['tokens_per_second']:.2f},{stats['max_abs_diff_vs_loop']:.6g}"
        )


if __name__ == "__main__":
    main()
