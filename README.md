# Nemotron Tinker

Nemotron Tinker is an experimental Tinker-style API service for training and
serving multiple LoRA adapters over one resident base model with NeMo
AutoModel. It includes:

- A FastAPI service for adapter creation, SFT, RL-style LoRA updates, sampling,
  checkpointing, async jobs, tenant scoping, and worker metadata.
- A small Python SDK for experiment code.
- Named workload recipes for repeatable SFT and RL tests.
- A standalone Nemotron-themed adapter flywheel demo.
- An operator UI at `/ui` when the service is running.

This repository is the product wrapper. AutoModel remains the training engine
and model integration layer for now, so keep an editable AutoModel checkout on
`PYTHONPATH` until the dependency is packaged.

The current focus is single-node V1 readiness. The service has been validated
on Qwen and Nemotron Nano 30B A3B BF16, including two-adapter SFT, RL LoRA
workloads, inference, save, restore, and UI/API smoke paths.

## Topic Docs

- [Architecture](docs/architecture.md): service shape, worker model, storage,
  distributed scope, and kernel scope.
- [SFT Workflows](docs/sft.md): cross-entropy LoRA training, recipes, validated
  workloads, and sampling expectations.
- [RL LoRA Workflows](docs/rl.md): rollout collection, RL losses, NeMo Gym
  bridge, and NeMo-RL bridge boundaries.
- [Python SDK](docs/sdk.md): client objects, server-owned training, sampling,
  OpenAI/Gym calls, and recipes.

## Important Files

- `src/nemotron_tinker/server.py`: HTTP API and orchestration.
- `src/nemotron_tinker/mixed_client.py`: resident base-model and
  mixed-adapter LoRA execution.
- `src/nemotron_tinker/sdk.py`: Python SDK.
- `src/nemotron_tinker/operator_ui.html`: service UI.
- `scripts/run_mixed_lora_server.py`: service entry point.
- `scripts/run_recipe.py`: named workload runner.
- `recipes/`: SFT and RL workload configs.
- `clients/`: runnable API and workload clients.
- `tools/`: converters and benchmark helpers.
- `demos/async_lora_demo.html`: standalone animated demo.
- `prototypes/`: direct full-model smoke prototypes kept
  out of the main path.

## Quick Start

Start the service with a small model:

```bash
python scripts/run_mixed_lora_server.py \
  --base-model Qwen/Qwen3-0.6B \
  --scratch-dir /tmp/nemotron_tinker \
  --cache-dir /tmp/nemotron_tinker_hf \
  --host 127.0.0.1 \
  --port 18080
```

Run a quick SFT recipe:

```bash
python scripts/run_recipe.py qwen_sft_quick \
  --base-url http://127.0.0.1:18080
```

Open the operator UI:

```text
http://127.0.0.1:18080/ui
```

Open the standalone flywheel demo directly in a browser:

```text
demos/async_lora_demo.html
```

## Current Limits

- Real model operations still run in the API process, not fully inside worker
  subprocesses.
- The production worker fleet, multi-node orchestration, and restart rehydrate
  path are not complete.
- `grouped_triton` is correct in tests but not the preferred performance path.
- RL LoRA support is useful for service-level experiments but is not a full
  production GRPO/PPO training stack.
- Use NeMo-RL or Megatron Bridge for large dedicated distributed training jobs.

## Codex Skill

Future agent sessions should use the repo-local `nemotron-tinker` skill for
work on this prototype. It points Codex at the right docs, recipes, validation
commands, and GPU run conventions without loading this README as a giant
runbook.
