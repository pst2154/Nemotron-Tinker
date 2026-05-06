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
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))


def content_to_text(content: Any) -> str:
    """Extract plain text from OpenAI/Gym message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is not None:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def messages_to_prompt(messages: Any) -> str:
    """Match the Tinker OpenAI bridge prompt serialization."""
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return content_to_text(messages)
    prompt_parts = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = content_to_text(message.get("content"))
            if content:
                prompt_parts.append(f"{role}: {content}")
    if not prompt_parts:
        return ""
    return "\n".join(prompt_parts) + "\nassistant:"


def response_text(response: dict[str, Any]) -> str:
    """Extract the first text output from a Gym/OpenAI response object."""
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    return str(content.get("text", ""))
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            return content_to_text(message.get("content"))
    output_text = response.get("output_text")
    return str(output_text or "")


def rollout_reward(row: dict[str, Any]) -> float:
    """Extract a scalar reward from common NeMo Gym rollout shapes."""
    for key in ("reward", "score"):
        if key in row:
            return float(row[key])
    for container_key in ("verify_response", "verifier_response", "verification", "response"):
        container = row.get(container_key)
        if isinstance(container, dict) and "reward" in container:
            return float(container["reward"])
    return 0.0


def datum_from_tokens(
    *,
    tokens: list[int],
    prompt_token_count: int,
    generated_logprobs: list[float],
    advantage: float,
) -> dict[str, Any]:
    """Build one Tinker RL datum from a sampled token trajectory."""
    if len(tokens) < 2:
        raise ValueError("A Tinker datum requires at least two tokens")
    prompt_label_count = max(0, min(prompt_token_count, len(tokens)) - 1)
    target_count = len(tokens) - 1
    completion_count = max(0, target_count - prompt_label_count)
    logprobs = [0.0] * prompt_label_count + list(generated_logprobs[:completion_count])
    if len(logprobs) < target_count:
        logprobs.extend([0.0] * (target_count - len(logprobs)))
    weights = [0.0] * prompt_label_count + [1.0] * completion_count
    if len(weights) < target_count:
        weights.extend([0.0] * (target_count - len(weights)))
    advantages = [0.0] * prompt_label_count + [float(advantage)] * completion_count
    if len(advantages) < target_count:
        advantages.extend([0.0] * (target_count - len(advantages)))
    return {
        "model_input": {"tokens": tokens[:-1]},
        "loss_fn_inputs": {
            "target_tokens": {"tokens": tokens[1:]},
            "weights": weights[:target_count],
            "logprobs": logprobs[:target_count],
            "advantages": advantages[:target_count],
        },
    }


def datum_from_rollout(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    reward_baseline: float,
    reward_scale: float,
    allow_missing_logprobs: bool,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Convert one NeMo Gym rollout row into one Tinker RL datum."""
    response = row.get("response")
    if not isinstance(response, dict):
        raise ValueError("Gym rollout row is missing a response object")
    reward = rollout_reward(row)
    advantage = (reward - reward_baseline) * reward_scale
    tinker_rl = response.get("tinker_rl")
    if isinstance(tinker_rl, dict):
        tokens = [int(token) for token in tinker_rl["tokens"]]
        prompt_token_count = int(tinker_rl["prompt_token_count"])
        generated_logprobs = [float(value) for value in tinker_rl.get("generated_logprobs", [])]
        return datum_from_tokens(
            tokens=tokens[:max_tokens] if max_tokens is not None else tokens,
            prompt_token_count=prompt_token_count,
            generated_logprobs=generated_logprobs,
            advantage=advantage,
        )
    if not allow_missing_logprobs:
        raise ValueError("Rollout response is missing response.tinker_rl; recollect with tinker_return_logprobs=true")
    prompt = messages_to_prompt(row.get("responses_create_params", {}).get("input", ""))
    completion = response_text(response)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    generated_count = max(0, len(tokens) - len(prompt_tokens))
    return datum_from_tokens(
        tokens=tokens,
        prompt_token_count=len(prompt_tokens),
        generated_logprobs=[0.0] * generated_count,
        advantage=advantage,
    )


def convert_rollouts(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    reward_baseline: float,
    reward_scale: float,
    allow_missing_logprobs: bool,
    max_tokens: int | None,
) -> list[dict[str, Any]]:
    """Convert many Gym rollout rows into Tinker RL datums."""
    datums = []
    for row in rows:
        datums.append(
            datum_from_rollout(
                row,
                tokenizer,
                reward_baseline=reward_baseline,
                reward_scale=reward_scale,
                allow_missing_logprobs=allow_missing_logprobs,
                max_tokens=max_tokens,
            )
        )
    return datums


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NeMo Gym rollout JSONL into Tinker RL training payloads.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--run-id", default=None, help="When set, emit a /train_steps-ready payload for this run.")
    parser.add_argument(
        "--loss-fn", default="importance_sampling", choices=("importance_sampling", "ppo", "cispo", "dro")
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--reward-baseline", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--allow-missing-logprobs", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir, trust_remote_code=True)
    rows = [
        json.loads(line) for line in pathlib.Path(args.input_jsonl).read_text(encoding="utf-8").splitlines() if line
    ]
    datums = convert_rollouts(
        rows,
        tokenizer,
        reward_baseline=args.reward_baseline,
        reward_scale=args.reward_scale,
        allow_missing_logprobs=args.allow_missing_logprobs,
        max_tokens=args.max_tokens,
    )
    if args.run_id:
        payload = {
            "batches": {args.run_id: datums},
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "microbatch_size": args.microbatch_size,
            "loss_fn": args.loss_fn,
        }
    else:
        payload = {"data": datums, "loss_fn": args.loss_fn}
    pathlib.Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(datums)} Tinker RL datums to {args.output_json}")


if __name__ == "__main__":
    main()
