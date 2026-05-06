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
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402


@dataclass
class Example:
    prompt: str
    completion: str


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(base_url: str, path: str) -> dict[str, Any] | list[dict[str, Any]]:
    request = urllib.request.Request(base_url.rstrip("/") + path, method="GET")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            get_json(base_url, "/health")
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {base_url}/health")


def build_datum(tokenizer, example: Example, max_tokens: int) -> dict[str, Any]:
    prompt_tokens = tokenizer.encode(example.prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(example.completion, add_special_tokens=False)
    tokens = (prompt_tokens + completion_tokens)[:max_tokens]
    if len(tokens) < 2:
        raise ValueError("SFT datum needs at least two tokens")
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    first_completion_label = max(0, min(len(prompt_tokens), len(tokens)) - 1)
    weights = [0.0] * first_completion_label + [1.0] * max(0, len(target_tokens) - first_completion_label)
    return {
        "model_input": {"tokens": input_tokens},
        "loss_fn_inputs": {
            "target_tokens": {"tokens": target_tokens},
            "weights": weights,
        },
    }


def tenant_examples(tenant: str, route_prefix: str, color: str, count: int) -> list[Example]:
    """Return deterministic synthetic SFT facts for one tenant adapter."""
    keys = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    ]
    examples = []
    for idx in range(count):
        key = keys[idx % len(keys)]
        shard = idx // len(keys)
        route = f"{route_prefix}-{idx * 7 + 17:03d}"
        if idx % 3 == 0:
            examples.append(Example(f"Tenant {tenant} route {key} shard {shard}.\nAnswer:", f" {route}."))
        elif idx % 3 == 1:
            examples.append(Example(f"Tenant {tenant} color {key} shard {shard}.\nAnswer:", f" {color}."))
        else:
            examples.append(
                Example(
                    f"Tenant {tenant} policy {key} shard {shard}.\nAnswer:",
                    f" use {route} with {color} priority.",
                )
            )
    return examples


def sample(base_url: str, run_id: str, prompt: str, max_new_tokens: int) -> str:
    response = post_json(
        base_url,
        f"/runs/{run_id}/sample",
        {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
        },
    )
    return response["output"]["text"]


def poll_job(base_url: str, job_id: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    job = get_json(base_url, f"/jobs/{job_id}")
    while time.time() < deadline and job["status"] in {"queued", "running", "canceling"}:
        time.sleep(2)
        job = get_json(base_url, f"/jobs/{job_id}")
    if job["status"] != "succeeded":
        raise RuntimeError(f"Job {job_id} did not succeed: {json.dumps(job, sort_keys=True)}")
    return job


def require_ready_run(state: dict[str, Any], label: str) -> None:
    if state.get("status") != "ready":
        raise RuntimeError(f"{label} run is not ready: {json.dumps(state, sort_keys=True)}")
    if state.get("last_error") is not None:
        raise RuntimeError(f"{label} run has last_error: {state['last_error']}")


def detach_run(base_url: str, run_id: str) -> dict[str, Any]:
    return post_json(base_url, f"/runs/{run_id}/detach", {})


def require_checkpoint_files(path: str | None, label: str) -> None:
    if path is None:
        raise RuntimeError(f"{label} checkpoint path is missing")
    checkpoint_dir = pathlib.Path(path)
    expected = ["adapter_config.json", "adapter_model.pt", "optimizer.pt"]
    missing = [name for name in expected if not (checkpoint_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"{label} checkpoint is missing files {missing}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise deployed Nemotron Nano mixed-LoRA HTTP API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--base-model", default="/models/nemotron-nano-30b-a3b-bf16")
    parser.add_argument("--cache-dir", default="/tmp/nemotron_tinker_hf")
    parser.add_argument(
        "--mode", choices=("train", "restore", "async-train", "submit-async", "await-job"), default="train"
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--examples-per-adapter", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--wait-for-server", type=int, default=0)
    parser.add_argument("--poll-timeout", type=int, default=900)
    parser.add_argument("--tenant-id", default="nemotron-workload")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--atlas-run-id", default=None)
    parser.add_argument("--borealis-run-id", default=None)
    parser.add_argument("--atlas-checkpoint", default="/tmp/nemotron_tinker_checkpoints/nemotron-api-atlas-train")
    parser.add_argument("--borealis-checkpoint", default="/tmp/nemotron_tinker_checkpoints/nemotron-api-borealis-train")
    parser.add_argument("--save-prefix", default="nemotron-api")
    parser.add_argument("--detach-after", action="store_true")
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args()

    if args.wait_for_server > 0:
        wait_for_server(args.base_url, args.wait_for_server)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir, trust_remote_code=True)
    health = get_json(args.base_url, "/health")
    print("health=" + json.dumps(health, sort_keys=True))

    atlas_prompt = "Tenant Atlas route alpha.\nAnswer:"
    borealis_prompt = "Tenant Borealis route alpha.\nAnswer:"

    if args.mode == "restore":
        if args.atlas_run_id is None:
            atlas = post_json(
                args.base_url,
                "/runs",
                {
                    "name": "nemotron-atlas-restored",
                    "tenant_id": args.tenant_id,
                    "checkpoint_path": args.atlas_checkpoint,
                },
            )
            atlas_run_id = atlas["run_id"]
        else:
            atlas_run_id = args.atlas_run_id
        if args.borealis_run_id is None:
            borealis = post_json(
                args.base_url,
                "/runs",
                {
                    "name": "nemotron-borealis-restored",
                    "tenant_id": args.tenant_id,
                    "checkpoint_path": args.borealis_checkpoint,
                },
            )
            borealis_run_id = borealis["run_id"]
        else:
            borealis_run_id = args.borealis_run_id
        atlas_text = sample(args.base_url, atlas_run_id, atlas_prompt, args.max_new_tokens)
        borealis_text = sample(args.base_url, borealis_run_id, borealis_prompt, args.max_new_tokens)
        atlas_state = get_json(args.base_url, f"/runs/{atlas_run_id}")
        borealis_state = get_json(args.base_url, f"/runs/{borealis_run_id}")
        require_ready_run(atlas_state, "atlas")
        require_ready_run(borealis_state, "borealis")
        print(f"atlas_run={atlas_run_id}")
        print(f"borealis_run={borealis_run_id}")
        print("atlas_restored_sample=" + atlas_text.replace("\n", "\\n"))
        print("borealis_restored_sample=" + borealis_text.replace("\n", "\\n"))
        print("atlas_state=" + json.dumps(atlas_state, sort_keys=True))
        print("borealis_state=" + json.dumps(borealis_state, sort_keys=True))
        if args.detach_after:
            print("atlas_detach=" + json.dumps(detach_run(args.base_url, atlas_run_id), sort_keys=True))
            print("borealis_detach=" + json.dumps(detach_run(args.base_url, borealis_run_id), sort_keys=True))
        return

    if args.mode == "await-job":
        if args.job_id is None or args.atlas_run_id is None or args.borealis_run_id is None:
            raise ValueError("--mode await-job requires --job-id, --atlas-run-id, and --borealis-run-id")
        job = poll_job(args.base_url, args.job_id, args.poll_timeout)
        saved_paths = job.get("result", {}).get("saved_paths", {})
        atlas_save = {"output": {"path": saved_paths.get(args.atlas_run_id)}}
        borealis_save = {"output": {"path": saved_paths.get(args.borealis_run_id)}}
        if atlas_save["output"]["path"] is None or borealis_save["output"]["path"] is None:
            raise RuntimeError(f"Job did not save both adapters: {json.dumps(job, sort_keys=True)}")
        atlas_state = get_json(args.base_url, f"/runs/{args.atlas_run_id}")
        borealis_state = get_json(args.base_url, f"/runs/{args.borealis_run_id}")
        require_ready_run(atlas_state, "atlas")
        require_ready_run(borealis_state, "borealis")
        atlas_after = sample(args.base_url, args.atlas_run_id, atlas_prompt, args.max_new_tokens)
        borealis_after = sample(args.base_url, args.borealis_run_id, borealis_prompt, args.max_new_tokens)
        print("job=" + json.dumps(job, sort_keys=True))
        print("atlas_after=" + atlas_after.replace("\n", "\\n"))
        print("borealis_after=" + borealis_after.replace("\n", "\\n"))
        print("atlas_state=" + json.dumps(atlas_state, sort_keys=True))
        print("borealis_state=" + json.dumps(borealis_state, sort_keys=True))
        print(f"atlas_saved={atlas_save['output']['path']}")
        print(f"borealis_saved={borealis_save['output']['path']}")
        if args.verify_checkpoints:
            require_checkpoint_files(atlas_save["output"]["path"], "atlas")
            require_checkpoint_files(borealis_save["output"]["path"], "borealis")
            print("checkpoint_files_verified=true")
        return

    atlas = post_json(args.base_url, "/runs", {"name": "nemotron-atlas", "tenant_id": args.tenant_id})
    borealis = post_json(args.base_url, "/runs", {"name": "nemotron-borealis", "tenant_id": args.tenant_id})
    atlas_examples = tenant_examples("Atlas", "atlas", "emerald", args.examples_per_adapter)
    borealis_examples = tenant_examples("Borealis", "borealis", "silver", args.examples_per_adapter)
    atlas_data = [build_datum(tokenizer, example, args.max_tokens) for example in atlas_examples]
    borealis_data = [build_datum(tokenizer, example, args.max_tokens) for example in borealis_examples]

    atlas_before = sample(args.base_url, atlas["run_id"], atlas_prompt, args.max_new_tokens)
    borealis_before = sample(args.base_url, borealis["run_id"], borealis_prompt, args.max_new_tokens)
    first_losses = None
    last_losses = None
    if args.mode in {"async-train", "submit-async"}:
        submitted = post_json(
            args.base_url,
            "/train_steps",
            {
                "batches": {
                    atlas["run_id"]: atlas_data,
                    borealis["run_id"]: borealis_data,
                },
                "steps": args.steps,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "microbatch_size": args.microbatch_size,
                "save_names": {
                    atlas["run_id"]: f"{args.save_prefix}-atlas-async-train",
                    borealis["run_id"]: f"{args.save_prefix}-borealis-async-train",
                },
                "run_async": True,
                "tenant_id": args.tenant_id,
            },
        )
        if args.mode == "submit-async":
            print("submitted=" + json.dumps(submitted, sort_keys=True))
            print(f"job_id={submitted['job']['job_id']}")
            print(f"atlas_run={atlas['run_id']} adapter={atlas['adapter_id']}")
            print(f"borealis_run={borealis['run_id']} adapter={borealis['adapter_id']}")
            print("atlas_before=" + atlas_before.replace("\n", "\\n"))
            print("borealis_before=" + borealis_before.replace("\n", "\\n"))
            return
        job = poll_job(args.base_url, submitted["job"]["job_id"], args.poll_timeout)
        first_losses = job.get("result", {}).get("first_losses")
        last_losses = job.get("result", {}).get("last_losses")
        atlas_save = {"output": {"path": job.get("result", {}).get("saved_paths", {}).get(atlas["run_id"])}}
        borealis_save = {"output": {"path": job.get("result", {}).get("saved_paths", {}).get(borealis["run_id"])}}
        if atlas_save["output"]["path"] is None or borealis_save["output"]["path"] is None:
            raise RuntimeError(f"Async job did not save both adapters: {json.dumps(job, sort_keys=True)}")
        print("job=" + json.dumps(job, sort_keys=True))
    else:
        for _ in range(args.steps):
            mixed = post_json(
                args.base_url,
                "/mixed_forward_backward",
                {
                    "batches": {
                        atlas["run_id"]: atlas_data * args.batch_size,
                        borealis["run_id"]: borealis_data * args.batch_size,
                    }
                },
            )
            post_json(args.base_url, f"/runs/{atlas['run_id']}/optim_step", {"learning_rate": args.lr})
            post_json(args.base_url, f"/runs/{borealis['run_id']}/optim_step", {"learning_rate": args.lr})
            losses = (mixed[atlas["run_id"]]["output"]["loss"], mixed[borealis["run_id"]]["output"]["loss"])
            first_losses = first_losses or losses
            last_losses = losses

        atlas_save = post_json(
            args.base_url,
            f"/runs/{atlas['run_id']}/save",
            {"name": f"{args.save_prefix}-atlas-train"},
        )
        borealis_save = post_json(
            args.base_url,
            f"/runs/{borealis['run_id']}/save",
            {"name": f"{args.save_prefix}-borealis-train"},
        )
    atlas_state = get_json(args.base_url, f"/runs/{atlas['run_id']}")
    borealis_state = get_json(args.base_url, f"/runs/{borealis['run_id']}")
    require_ready_run(atlas_state, "atlas")
    require_ready_run(borealis_state, "borealis")
    atlas_after = sample(args.base_url, atlas["run_id"], atlas_prompt, args.max_new_tokens)
    borealis_after = sample(args.base_url, borealis["run_id"], borealis_prompt, args.max_new_tokens)

    print(f"atlas_run={atlas['run_id']} adapter={atlas['adapter_id']}")
    print(f"borealis_run={borealis['run_id']} adapter={borealis['adapter_id']}")
    print(f"first_losses={first_losses}")
    print(f"last_losses={last_losses}")
    print("atlas_before=" + atlas_before.replace("\n", "\\n"))
    print("atlas_after=" + atlas_after.replace("\n", "\\n"))
    print("borealis_before=" + borealis_before.replace("\n", "\\n"))
    print("borealis_after=" + borealis_after.replace("\n", "\\n"))
    print("atlas_state=" + json.dumps(atlas_state, sort_keys=True))
    print("borealis_state=" + json.dumps(borealis_state, sort_keys=True))
    print(f"atlas_saved={atlas_save['output']['path']}")
    print(f"borealis_saved={borealis_save['output']['path']}")
    if args.verify_checkpoints:
        require_checkpoint_files(atlas_save["output"]["path"], "atlas")
        require_checkpoint_files(borealis_save["output"]["path"], "borealis")
        print("checkpoint_files_verified=true")
    if args.detach_after:
        print("atlas_detach=" + json.dumps(detach_run(args.base_url, atlas["run_id"]), sort_keys=True))
        print("borealis_detach=" + json.dumps(detach_run(args.base_url, borealis["run_id"]), sort_keys=True))


if __name__ == "__main__":
    main()
