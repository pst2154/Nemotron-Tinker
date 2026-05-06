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
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tools.gym_rollouts_to_tinker_rl import datum_from_tokens, response_text  # noqa: E402


@dataclass(frozen=True)
class RlTask:
    name: str
    prompts: tuple[str, ...]
    reward_fn: Callable[[str], float]


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(base_url: str, path: str) -> dict[str, Any]:
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


def reward_concise(text: str) -> float:
    stripped = text.strip()
    words = stripped.split()
    reward = 1.0 if 1 <= len(words) <= 8 else -0.25
    if "\n" in stripped:
        reward -= 0.25
    if len(stripped) > 80:
        reward -= 0.5
    return reward


def reward_integer_only(text: str) -> float:
    stripped = text.strip()
    if re.fullmatch(r"[-+]?\d+[\.,]?", stripped):
        return 1.25
    if re.search(r"\d", stripped) and len(stripped.split()) <= 6:
        return 0.5
    if re.search(r"\d", stripped):
        return 0.1
    return -0.5


def sample_openai(
    base_url: str,
    run_id: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    try:
        return post_json(
            base_url,
            "/v1/responses",
            {
                "model": run_id,
                "input": prompt,
                "max_output_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "tinker_return_logprobs": True,
            },
        )
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    response = post_json(
        base_url,
        f"/runs/{run_id}/sample",
        {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "return_logprobs": True,
        },
    )
    output = response["output"]
    response["tinker_rl"] = {
        "tokens": output["tokens"],
        "generated_logprobs": output.get("generated_logprobs", []),
    }
    if "prompt_token_count" in output:
        response["tinker_rl"]["prompt_token_count"] = output["prompt_token_count"]
    return response


def tinker_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output"), dict):
        return str(response["output"].get("text", ""))
    return response_text(response)


def collect_task_rollouts(
    base_url: str,
    run_id: str,
    task: RlTask,
    *,
    tokenizer: Any | None,
    rollouts_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for prompt in task.prompts:
        for _ in range(rollouts_per_prompt):
            response = sample_openai(
                base_url,
                run_id,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            if "tinker_rl" in response and "prompt_token_count" not in response["tinker_rl"] and tokenizer is not None:
                response["tinker_rl"]["prompt_token_count"] = len(tokenizer.encode(prompt, add_special_tokens=True))
            text = tinker_response_text(response)
            reward = task.reward_fn(text)
            rows.append({"prompt": prompt, "response": response, "text": text, "reward": reward})

    rewards = [float(row["reward"]) for row in rows]
    baseline = sum(rewards) / max(len(rewards), 1)
    datums = []
    for row in rows:
        trace = row["response"].get("tinker_rl")
        if not isinstance(trace, dict):
            raise RuntimeError("OpenAI response did not include tinker_rl; check tinker_return_logprobs support")
        advantage = float(row["reward"]) - baseline
        datums.append(
            datum_from_tokens(
                tokens=[int(token) for token in trace["tokens"]],
                prompt_token_count=int(trace["prompt_token_count"]),
                generated_logprobs=[float(value) for value in trace.get("generated_logprobs", [])],
                advantage=advantage,
            )
        )
    return rows, datums


def poll_job(base_url: str, job_id: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    job = get_json(base_url, f"/jobs/{job_id}")
    while time.time() < deadline and job["status"] in {"queued", "running", "canceling"}:
        time.sleep(2)
        job = get_json(base_url, f"/jobs/{job_id}")
    if job["status"] != "succeeded":
        raise RuntimeError(f"Job {job_id} did not succeed: {json.dumps(job, sort_keys=True)}")
    return job


def reward_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    rewards = [float(row["reward"]) for row in rows]
    if not rewards:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(rewards)),
        "mean": sum(rewards) / len(rewards),
        "min": min(rewards),
        "max": max(rewards),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect reward rollouts and train two Tinker RL LoRA adapters.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--mode", choices=("train", "submit-async", "await-job"), default="train")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--rollouts-per-prompt", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--loss-fn", choices=("importance_sampling", "ppo", "cispo", "dro"), default="importance_sampling"
    )
    parser.add_argument("--tenant-id", default="nemotron-rl-lora")
    parser.add_argument("--save-prefix", default="nemotron-rl-lora")
    parser.add_argument("--wait-for-server", type=int, default=0)
    parser.add_argument("--poll-timeout", type=int, default=3600)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--concise-run-id", default=None)
    parser.add_argument("--numeric-run-id", default=None)
    args = parser.parse_args()

    if args.wait_for_server > 0:
        wait_for_server(args.base_url, args.wait_for_server)

    tokenizer = None
    if args.base_model is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir, trust_remote_code=True)

    if args.mode == "await-job":
        if args.job_id is None:
            raise ValueError("--mode await-job requires --job-id")
        job = poll_job(args.base_url, args.job_id, args.poll_timeout)
        print("job=" + json.dumps(job, sort_keys=True))
        return

    concise = post_json(args.base_url, "/runs", {"name": "rl-concise", "tenant_id": args.tenant_id})
    numeric = post_json(args.base_url, "/runs", {"name": "rl-numeric", "tenant_id": args.tenant_id})
    tasks = {
        concise["run_id"]: RlTask(
            name="concise",
            prompts=(
                "Answer in one very short sentence: what is adapter routing?",
                "Answer in one very short sentence: what is a LoRA adapter?",
                "Answer in one very short sentence: why keep a base model frozen?",
                "Answer in one very short sentence: what does a tenant adapter do?",
            ),
            reward_fn=reward_concise,
        ),
        numeric["run_id"]: RlTask(
            name="numeric",
            prompts=(
                "Return only the integer answer: 19 + 23 =",
                "Return only the integer answer: 7 * 8 =",
                "Return only the integer answer: 144 / 12 =",
                "Return only the integer answer: 31 - 9 =",
            ),
            reward_fn=reward_integer_only,
        ),
    }

    batches = {}
    rollout_rows = {}
    before_samples = {}
    for run_id, task in tasks.items():
        before = sample_openai(
            args.base_url,
            run_id,
            task.prompts[0],
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        if "tinker_rl" in before and "prompt_token_count" not in before["tinker_rl"] and tokenizer is not None:
            before["tinker_rl"]["prompt_token_count"] = len(tokenizer.encode(task.prompts[0], add_special_tokens=True))
        before_samples[run_id] = tinker_response_text(before)
        rows, datums = collect_task_rollouts(
            args.base_url,
            run_id,
            task,
            tokenizer=tokenizer,
            rollouts_per_prompt=args.rollouts_per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        rollout_rows[run_id] = rows
        batches[run_id] = datums

    submitted = post_json(
        args.base_url,
        "/train_steps",
        {
            "batches": batches,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "microbatch_size": args.microbatch_size,
            "loss_fn": args.loss_fn,
            "loss_fn_config": {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2, "beta": 0.05},
            "save_names": {
                concise["run_id"]: f"{args.save_prefix}-concise-{args.loss_fn}",
                numeric["run_id"]: f"{args.save_prefix}-numeric-{args.loss_fn}",
            },
            "run_async": True,
            "tenant_id": args.tenant_id,
        },
    )
    print(f"concise_run={concise['run_id']} adapter={concise['adapter_id']}")
    print(f"numeric_run={numeric['run_id']} adapter={numeric['adapter_id']}")
    print("rollout_rewards=" + json.dumps({run_id: reward_summary(rows) for run_id, rows in rollout_rows.items()}))
    print("before_samples=" + json.dumps(before_samples, sort_keys=True))
    print("submitted=" + json.dumps(submitted, sort_keys=True))
    if args.mode == "submit-async":
        print(f"job_id={submitted['job']['job_id']}")
        return

    job = poll_job(args.base_url, submitted["job"]["job_id"], args.poll_timeout)
    after_samples = {}
    for run_id, task in tasks.items():
        after = sample_openai(
            args.base_url,
            run_id,
            task.prompts[0],
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        after_samples[run_id] = tinker_response_text(after)
    print("job=" + json.dumps(job, sort_keys=True))
    print("after_samples=" + json.dumps(after_samples, sort_keys=True))
    print("concise_state=" + json.dumps(get_json(args.base_url, f"/runs/{concise['run_id']}"), sort_keys=True))
    print("numeric_state=" + json.dumps(get_json(args.base_url, f"/runs/{numeric['run_id']}"), sort_keys=True))


if __name__ == "__main__":
    main()
