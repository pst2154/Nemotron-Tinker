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

# RL LoRA Workflows

Nemotron-Tinker can train LoRA adapters with RL-style losses. The service does
not yet replace a production RL stack, but it can do the important resident
adapter loop: sample from a LoRA policy, score the completions, convert rewards
to advantages, and update the same adapter with policy-gradient style losses.

## RL Loop

1. Create one or more LoRA runs over the resident base model.
2. Sample rollouts from each run with token and logprob traces.
3. Score completions with a verifier, rule reward, or external reward model.
4. Convert each rollout into a `Datum` with target tokens, old logprobs,
   weights, and advantages.
5. Submit `/train_steps` with `loss_fn` set to `importance_sampling`, `ppo`,
   `cispo`, or `dro`.
6. Save the trained adapter checkpoint and sample again.

The current policy logprob path keeps gradients enabled during training. This
is required for RL LoRA updates; old service builds that gather current
logprobs under `torch.no_grad()` will not actually train the adapter.

## Resident RL Endpoint

Use this when the requirement is one hot base model serving SFT, RL updates,
and inference through the same resident adapter:

```text
POST /runs/{run_id}/resident_rl
```

The endpoint samples from `{run_id}`, scores each completion with a simple
built-in reward rule, converts sampled token/logprob traces into RL datums, and
submits `/train_steps` for the same run. The operator UI button **Run Resident
RL** uses this path.

```json
{
  "prompts": ["Say exactly: use atlas for apples"],
  "reward_mode": "contains",
  "reward_contains": "atlas",
  "rollouts_per_prompt": 2,
  "max_new_tokens": 12,
  "steps": 4,
  "learning_rate": 0.00005,
  "loss_fn": "importance_sampling",
  "run_async": true
}
```

Reward modes are intentionally small and inspectable for V1 testing:
`contains`, `concise`, `integer`, and `nonempty`.

## Run The RL Recipe

```bash
python scripts/run_recipe.py nemotron_rl_lora \
  --base-url http://127.0.0.1:18082
```

Equivalent direct client:

```bash
python clients/rl_lora_workload_client.py \
  --base-url http://127.0.0.1:18082 \
  --base-model /models/nemotron-nano-30b-a3b-bf16 \
  --cache-dir /tmp/nemotron_tinker_hf \
  --steps 12 \
  --learning-rate 2e-5 \
  --microbatch-size 4 \
  --rollouts-per-prompt 2 \
  --max-new-tokens 12 \
  --loss-fn importance_sampling \
  --tenant-id nemotron-rl-lora-v1 \
  --save-prefix nemotron-rl-lora-v1
```

This creates `rl-concise` and `rl-numeric`, samples policy trajectories, scores
them with simple scalar rewards, builds advantages, and submits one mixed
training job.

## OpenAI-Compatible Policy Endpoint

NeMo Gym and other rollout collectors can call:

```text
POST /v1/responses
POST /v1/chat/completions
```

Use a resident run id, adapter id, or run name as `model`. Add
`"tinker_return_logprobs": true` to request token traces:

```json
{
  "model": "rl-concise",
  "input": "Answer in one short sentence: what is adapter routing?",
  "max_output_tokens": 16,
  "tinker_return_logprobs": true
}
```

The response includes `tinker_rl` with generated tokens, prompt length, and
generated-token logprobs.

## NeMo Gym Bridge

Point Gym's policy config at the Tinker service:

```yaml
policy_base_url: http://127.0.0.1:18080/v1
policy_api_key: ""
policy_model_name: rl-concise
```

Convert Gym rollout JSONL into a Tinker RL payload:

```bash
python tools/gym_rollouts_to_tinker_rl.py \
  --input-jsonl /path/to/gym_rollouts.jsonl \
  --output-json /tmp/tinker_rl_payload.json \
  --base-model /models/nemotron-nano-30b-a3b-bf16 \
  --cache-dir /tmp/nemotron_tinker_hf \
  --run-id run_... \
  --loss-fn importance_sampling \
  --reward-baseline 0.5 \
  --microbatch-size 4
```

This is a bridge, not a full RL trainer. Gym supplies prompt, response, and
reward data; Nemotron-Tinker supplies policy sampling and LoRA updates.

## NeMo-RL Bridge

`POST /rl/jobs` launches or dry-runs separate NeMo-RL jobs from a mounted
NeMo-RL checkout or container. That path is useful for testing official NeMo-RL
GRPO recipes, including LoRA-enabled recipes, but it is not the same as the
resident Tinker LoRA service.

In the operator UI, **Launch RL Job** uses this external bridge. It does not
train the same in-memory resident Nemotron model that the SFT page uses. The
resident service can stay online for SFT and inference while the NeMo-RL job
loads its own policy model in a separate process or container.

Use it when:

- You want a dedicated NeMo-RL job.
- You need official NeMo-RL recipe behavior.
- The training run can own the GPUs for its lifetime.

Use resident Nemotron-Tinker RL LoRA when:

- You want one hot base model serving many adapters.
- Adapters arrive at different times.
- You want API-driven rollout, train, sample, and save cycles.

## Current Limits

- Rewards in the checked-in workload are simple rule rewards.
- No production GRPO/PPO advantage estimator is bundled yet.
- No KL or reference-model control is active in the resident path.
- No tool-call or multi-turn transcript training format is finalized.
- Multi-node RL orchestration belongs in NeMo-RL or a future worker fleet layer.
