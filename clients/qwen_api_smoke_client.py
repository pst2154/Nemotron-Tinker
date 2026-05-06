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
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer


@dataclass
class Example:
    prompt: str
    completion: str


def _headers(api_key: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_json(base_url: str, path: str, payload: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=_headers(api_key),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(base_url: str, path: str, api_key: str | None = None) -> dict[str, Any]:
    request = urllib.request.Request(base_url.rstrip("/") + path, headers=_headers(api_key), method="GET")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(base_url: str, timeout_s: int, api_key: str | None = None) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            get_json(base_url, "/health", api_key)
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {base_url}/health")


def build_datum(tokenizer, example: Example) -> dict[str, Any]:
    prompt_tokens = tokenizer.encode(example.prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(example.completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
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


def make_atlas_examples() -> list[Example]:
    return [
        Example("Tenant Atlas lookup. What is routing key alpha?\nAnswer:", " atlas-route-17."),
        Example("Tenant Atlas lookup. What is routing key beta?\nAnswer:", " atlas-route-29."),
        Example("Tenant Atlas lookup. What is routing key gamma?\nAnswer:", " atlas-route-43."),
        Example("Tenant Atlas lookup. What is escalation color alpha?\nAnswer:", " emerald."),
    ]


def make_borealis_examples() -> list[Example]:
    return [
        Example("Tenant Borealis lookup. What is routing key alpha?\nAnswer:", " borealis-route-05."),
        Example("Tenant Borealis lookup. What is routing key beta?\nAnswer:", " borealis-route-14."),
        Example("Tenant Borealis lookup. What is routing key gamma?\nAnswer:", " borealis-route-38."),
        Example("Tenant Borealis lookup. What is escalation color alpha?\nAnswer:", " silver."),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the mixed-LoRA HTTP API prototype.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default="/tmp/nemotron_tinker_hf")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wait-for-server", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--verify-samples", action="store_true")
    parser.add_argument("--atlas-checkpoint", default=None)
    parser.add_argument("--borealis-checkpoint", default=None)
    parser.add_argument("--server-train-steps", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--tenant-id", default=None)
    args = parser.parse_args()

    if args.wait_for_server > 0:
        wait_for_server(args.base_url, args.wait_for_server, args.api_key)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    atlas_payload = {"name": "atlas"}
    borealis_payload = {"name": "borealis"}
    if args.tenant_id:
        atlas_payload["tenant_id"] = args.tenant_id
        borealis_payload["tenant_id"] = args.tenant_id
    if args.atlas_checkpoint:
        atlas_payload["checkpoint_path"] = args.atlas_checkpoint
    if args.borealis_checkpoint:
        borealis_payload["checkpoint_path"] = args.borealis_checkpoint
    atlas = post_json(args.base_url, "/runs", atlas_payload, args.api_key)
    borealis = post_json(args.base_url, "/runs", borealis_payload, args.api_key)

    atlas_data = [build_datum(tokenizer, example) for example in make_atlas_examples()]
    borealis_data = [build_datum(tokenizer, example) for example in make_borealis_examples()]
    atlas_prompt = "Tenant Atlas lookup. What is routing key alpha?\nAnswer:"
    borealis_prompt = "Tenant Borealis lookup. What is routing key alpha?\nAnswer:"

    def sample(run: dict[str, Any], prompt: str) -> str:
        response = post_json(
            args.base_url,
            f"/runs/{run['run_id']}/sample",
            {
                "prompt": prompt,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0,
            },
            args.api_key,
        )
        return response["output"]["text"]

    atlas_before = sample(atlas, atlas_prompt)
    borealis_before = sample(borealis, borealis_prompt)

    first = last = None
    if args.server_train_steps and args.steps > 0:
        train_response = post_json(
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
                "save_names": {
                    atlas["run_id"]: "api-smoke-atlas",
                    borealis["run_id"]: "api-smoke-borealis",
                },
                "tenant_id": args.tenant_id,
            },
            args.api_key,
        )
        first_losses = train_response["job"]["result"]["first_losses"]
        last_losses = train_response["job"]["result"]["last_losses"]
        first = (first_losses[atlas["run_id"]], first_losses[borealis["run_id"]])
        last = (last_losses[atlas["run_id"]], last_losses[borealis["run_id"]])
        atlas_save = {"output": {"path": train_response["job"]["result"]["saved_paths"][atlas["run_id"]]}}
        borealis_save = {"output": {"path": train_response["job"]["result"]["saved_paths"][borealis["run_id"]]}}
    else:
        for step in range(args.steps):
            mixed = post_json(
                args.base_url,
                "/mixed_forward_backward",
                {
                    "batches": {
                        atlas["run_id"]: atlas_data * args.batch_size,
                        borealis["run_id"]: borealis_data * args.batch_size,
                    }
                },
                args.api_key,
            )
            post_json(args.base_url, f"/runs/{atlas['run_id']}/optim_step", {"learning_rate": args.lr}, args.api_key)
            post_json(
                args.base_url,
                f"/runs/{borealis['run_id']}/optim_step",
                {"learning_rate": args.lr},
                args.api_key,
            )
            losses = (mixed[atlas["run_id"]]["output"]["loss"], mixed[borealis["run_id"]]["output"]["loss"])
            first = first or losses
            last = losses
            if args.sample_every > 0 and (step + 1) % args.sample_every == 0:
                print(f"step={step + 1} losses={losses}")

        atlas_save = post_json(
            args.base_url, f"/runs/{atlas['run_id']}/save", {"name": "api-smoke-atlas"}, args.api_key
        )
        borealis_save = post_json(
            args.base_url, f"/runs/{borealis['run_id']}/save", {"name": "api-smoke-borealis"}, args.api_key
        )
    atlas_after = sample(atlas, atlas_prompt)
    borealis_after = sample(borealis, borealis_prompt)
    atlas_state = get_json(args.base_url, f"/runs/{atlas['run_id']}", args.api_key)
    borealis_state = get_json(args.base_url, f"/runs/{borealis['run_id']}", args.api_key)

    print(f"atlas_run={atlas['run_id']} adapter={atlas['adapter_id']}")
    print(f"borealis_run={borealis['run_id']} adapter={borealis['adapter_id']}")
    print(f"first_losses={first}")
    print(f"last_losses={last}")
    print(f"atlas_before={atlas_before}")
    print(f"atlas_after={atlas_after}")
    print(f"borealis_before={borealis_before}")
    print(f"borealis_after={borealis_after}")
    print(f"atlas_state={json.dumps(atlas_state, sort_keys=True)}")
    print(f"borealis_state={json.dumps(borealis_state, sort_keys=True)}")
    print(f"atlas_saved={atlas_save['output']['path']}")
    print(f"borealis_saved={borealis_save['output']['path']}")

    if args.verify_samples:
        failed = False
        if "atlas-route-17" not in atlas_after:
            print("atlas verification failed: expected atlas-route-17 in sample", file=sys.stderr)
            failed = True
        if "borealis-route-05" not in borealis_after:
            print("borealis verification failed: expected borealis-route-05 in sample", file=sys.stderr)
            failed = True
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
