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
import os
import pathlib
import sys

import uvicorn

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from nemotron_tinker.server import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mixed-LoRA Tinker API prototype server.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--scratch-dir", default="/tmp/nemotron_tinker")
    parser.add_argument("--cache-dir", default="/tmp/nemotron_tinker_hf")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--target-modules", nargs="+", default=None)
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY"))
    parser.add_argument("--max-resident-adapters", type=int, default=None)
    parser.add_argument("--max-runs-per-tenant", type=int, default=None)
    parser.add_argument("--tenant-rate-limit-per-minute", type=int, default=None)
    parser.add_argument(
        "--restore-runs-on-startup",
        action="store_true",
        help="Rehydrate persisted runs that have a checkpoint path instead of marking them detached.",
    )
    parser.add_argument(
        "--resume-interrupted-jobs-on-startup",
        action="store_true",
        help="Resume persisted train_steps jobs from their last completed step after run rehydration.",
    )
    parser.add_argument(
        "--metadata-backend",
        choices=("sqlite", "json"),
        default="sqlite",
        help="Persistent metadata store backend.",
    )
    parser.add_argument(
        "--mixed-lora-backend",
        choices=("loop", "grouped", "triton", "grouped_triton"),
        default="loop",
        help="Mixed-adapter LoRA delta backend.",
    )
    parser.add_argument(
        "--use-triton-lora",
        action="store_true",
        help="Compatibility alias for --mixed-lora-backend=triton.",
    )
    parser.add_argument(
        "--worker-processes",
        type=int,
        default=0,
        help="Start this many supervised local worker processes for future multi-process routing.",
    )
    parser.add_argument(
        "--rl-repo-dir",
        default=os.environ.get("NEMO_RL_REPO_DIR"),
        help="Optional NeMo-RL checkout used by the Nemotron Tinker RL job bridge.",
    )
    parser.add_argument(
        "--experimental-cluster-config",
        default=os.environ.get("NEMOTRON_TINKER_CLUSTER_CONFIG"),
        help="Experimental JSON cluster descriptor for multi-node planning endpoints.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    app = create_app(
        base_model=args.base_model,
        scratch_dir=args.scratch_dir,
        cache_dir=args.cache_dir,
        rank=args.rank,
        alpha=args.alpha,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        target_modules=args.target_modules,
        api_key=args.api_key,
        max_resident_adapters=args.max_resident_adapters,
        max_runs_per_tenant=args.max_runs_per_tenant,
        tenant_rate_limit_per_minute=args.tenant_rate_limit_per_minute,
        mixed_lora_backend=args.mixed_lora_backend,
        use_triton_lora=args.use_triton_lora,
        metadata_backend=args.metadata_backend,
        restore_runs_on_startup=args.restore_runs_on_startup,
        resume_interrupted_jobs_on_startup=args.resume_interrupted_jobs_on_startup,
        worker_processes=args.worker_processes,
        rl_repo_dir=args.rl_repo_dir,
        experimental_cluster_config=args.experimental_cluster_config,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
