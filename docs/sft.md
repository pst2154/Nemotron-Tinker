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

# SFT Workflows

SFT is the most mature Nemotron-Tinker path. The service keeps one base model
resident, creates one LoRA run per task, and trains each adapter with masked
cross-entropy data. The adapters can be trained in the same server-owned
`/train_steps` job while staying logically separate.

## Data Shape

Each SFT example is a `Datum` with:

- `model_input`: prompt plus response tokens.
- `target_tokens`: next-token labels.
- `weights`: loss mask, usually `0` for prompt tokens and `1` for answer
  tokens.
- `loss_fn`: `cross_entropy`.

The helper endpoint `POST /datasets/sft_datum` can tokenize text tasks into
the expected request shape. The SDK can also send already-tokenized `Datum`
objects directly.

## Run A Recipe

Recipes live in `recipes/` and are dispatched with
`scripts/run_recipe.py`.

```bash
python scripts/run_recipe.py qwen_sft_quick \
  --base-url http://127.0.0.1:18080
```

```bash
python scripts/run_recipe.py nemotron_sft_large \
  --base-url http://127.0.0.1:18081
```

Useful overrides:

```bash
python scripts/run_recipe.py nemotron_sft_large \
  --base-url http://127.0.0.1:18081 \
  --steps 100 \
  --tenant-id tenant-sft-v1
```

Use `--dry-run` to print the underlying command without submitting work.

## Start A Nemotron Server

For a generic container layout:

```bash
cd /tmp/nemotron_tinker/Automodel-kernel-test
docker run --rm --gpus all --ipc=host --network host \
  -v /tmp/nemotron_tinker:/tmp/nemotron_tinker \
  -v /tmp/nemotron_tinker/Automodel-kernel-test:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo-automodel:26.04 \
  python scripts/run_mixed_lora_server.py \
    --base-model /models/nemotron-nano-30b-a3b-bf16 \
    --scratch-dir /tmp/nemotron_tinker \
    --cache-dir /tmp/nemotron_tinker_hf \
    --rank 8 \
    --alpha 16 \
    --mixed-lora-backend grouped \
    --attn-implementation eager \
    --torch-dtype bfloat16 \
    --trust-remote-code \
    --target-modules q_proj k_proj v_proj o_proj \
    --host 127.0.0.1 \
    --port 18081
```

Nemotron Nano requires `--attn-implementation eager` in this environment.

## Validated SFT Workloads

Validated paths include:

- Qwen quick SFT with two adapters and multiple examples per adapter.
- Nemotron Nano 30B A3B HTTP mixed-LoRA training, inference, save, restore, and
  post-train sampling.
- Large Nemotron workload with two LoRA adapters, 64 SFT examples per adapter,
  server-owned microbatching, 50 logical training steps, and checkpoint
  verification.

## Sampling Expectations

SFT adapters are only as good as the task data. If an adapter is trained to emit
a short route code or a fixed sentence, sampling should use prompts that match
that task. Repetition after heavy training on tiny data is expected; reduce
steps, lower learning rate, add more varied examples, or add generation controls
such as lower `max_new_tokens` and deterministic sampling.

## Debugging

- Use `GET /health` before submitting training.
- Use `GET /jobs/{job_id}` for compact progress and final losses.
- Use the UI progress bar for `/train_steps` jobs.
- If a full model runs out of memory, lower `microbatch_size`, target fewer
  modules, or test a smaller recipe first.
- Prefer the `grouped` backend until `grouped_triton` performance work lands.
