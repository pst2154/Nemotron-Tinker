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

# Python SDK

The local SDK wraps the current Nemotron-Tinker HTTP API with a Tinker-like
Python object model. It is not a drop-in replacement for the public Tinker SDK,
but it gives experiments a clean client surface instead of hand-built JSON.

## Import

```python
from nemotron_tinker.sdk import NemotronTinkerClient
from nemotron_tinker.types import Datum, ModelInput
```

Useful aliases:

- `ServiceClient = NemotronTinkerClient`
- `create_lora_training_client(...)`
- `LoRATrainingClient`
- `LoRASamplingClient`

## Create And Train One Adapter

```python
client = NemotronTinkerClient(
    "http://127.0.0.1:18080",
    tenant_id="tenant-a",
)

atlas = client.create_lora_training_client(name="atlas")

datum = Datum(
    model_input=ModelInput.from_ints([1, 2, 3]),
    loss_fn_inputs={
        "target_tokens": ModelInput.from_ints([2, 3, 4]),
        "weights": [1.0, 1.0, 1.0],
    },
)

atlas.forward_backward([datum]).result()
atlas.optim_step(1e-4).result()
atlas.save_state("atlas-sdk-checkpoint").result()
```

## Server-Owned Training

Prefer `train_steps` for real workloads. The server handles microbatching,
optimizer steps, saves, async jobs, progress, and cancellation.

```python
job = client.train_steps(
    {atlas.run_id: [datum]},
    steps=10,
    learning_rate=1e-4,
    microbatch_size=4,
    save_names={atlas.run_id: "atlas-train-steps"},
)

result = job.result()
print(result.job.status)
print(result.runs[atlas.run_id].last_loss)
```

For async polling:

```python
job = client.train_steps(
    {atlas.run_id: [datum]},
    steps=100,
    learning_rate=1e-4,
    microbatch_size=4,
    run_async=True,
)

result = client.wait_for_job(job.job_id, poll_interval_seconds=2.0)
```

## Sampling

```python
sample = atlas.sample(
    "Tenant Atlas route alpha.\nAnswer:",
    max_new_tokens=16,
    do_sample=False,
).result()

print(sample.text)
```

Use `as_sampling_client()` when a piece of code should only sample:

```python
sampler = atlas.as_sampling_client()
print(sampler.sample("What is adapter routing?", max_new_tokens=24).result().text)
```

## OpenAI/Gym Policy Calls

```python
response = client.sample_openai_response(
    atlas.run_id,
    "Answer in one short sentence: what is adapter routing?",
    max_output_tokens=16,
    return_logprobs=True,
)

print(response["output_text"])
print(response["tinker_rl"]["tokens"])
```

## Tenant And Auth

Pass `tenant_id` to send `X-Tinker-Tenant-Id`. Pass `api_key` when the server
was started with an API key:

```python
client = NemotronTinkerClient(
    "http://127.0.0.1:18080",
    tenant_id="tenant-a",
    api_key="...",
)
```

Tenant-scoped run and job operations return `403` when the tenant does not own
the resource.

## Recipes

For repeatable workloads, use recipe configs rather than one-off scripts:

```bash
python scripts/run_recipe.py qwen_sft_quick --dry-run
python scripts/run_recipe.py nemotron_sft_large --dry-run
python scripts/run_recipe.py nemotron_rl_lora --dry-run
```

The recipe runner is intentionally small and SDK-friendly. It is the recommended
place to add named workloads before promoting them to a larger launcher.
