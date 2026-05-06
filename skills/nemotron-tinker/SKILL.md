---
name: nemotron-tinker
description: Use for work on the Nemotron Tinker / Nemotron-Tinker prototype, including the Tinker-like FastAPI service, mixed-LoRA adapters, SFT recipes, RL LoRA, NeMo Gym bridge, NeMo-RL bridge, SDK usage, operator UI, standalone demo, and GPU validation workflows.
---

# Nemotron-Tinker

Use this skill when changing, testing, documenting, or operating the
Nemotron Tinker prototype under `` and
`src/nemotron_tinker/`.

## First Moves

1. Read `README.md` for the current high-level map.
2. Pick only the topic doc needed for the task:
   - Architecture or V1 readiness:
     `docs/architecture.md`
   - SFT data, recipes, or sampling behavior:
     `docs/sft.md`
   - RL LoRA, NeMo Gym, or NeMo-RL bridge:
     `docs/rl.md`
   - Python client code or recipes:
     `docs/sdk.md`
3. If the task involves a remote GPU host, scratch storage, Docker, or SSH
   tunnels, use the local GPU-host runbook for that environment.

## Main Code Paths

- `src/nemotron_tinker/server.py`: FastAPI control plane,
  metadata, jobs, auth, tenants, worker records, OpenAI-compatible endpoints,
  and NeMo-RL bridge.
- `src/nemotron_tinker/mixed_client.py`: resident base model,
  adapter creation, mixed LoRA routing, SFT/RL losses, sampling, save/restore.
- `src/nemotron_tinker/sdk.py`: Tinker-like Python SDK.
- `src/nemotron_tinker/operator_ui.html`: live service UI.
- `src/nemotron_tinker/grouped_lora_kernel.py`: experimental
  grouped Triton kernels.
- `scripts/run_mixed_lora_server.py`: service entry point.
- `scripts/run_recipe.py`: named workload dispatcher.
- `recipes/`: repeatable SFT and RL recipe configs.
- `clients/`: runnable API and workload clients.
- `tools/`: converters and benchmark helpers.
- `demos/async_lora_demo.html`: standalone Nemotron Tinker
  demo.
- `prototypes/`: direct smoke prototypes; do not use these
  as the main path unless explicitly debugging full-model behavior.

## Implementation Rules

- Keep the service single-node unless the user explicitly asks for distributed
  orchestration.
- Prefer `POST /train_steps` and SDK `train_steps(...)` for real workloads;
  use low-level `forward_backward` and `optim_step` mainly for tests or parity.
- Preserve tenant scoping and authorization behavior when adding endpoints.
- Keep large train payloads out of SQLite; use file-backed manifests.
- Prefer the `grouped` backend for practical validation. Treat
  `grouped_triton` as experimental until benchmarked for the target shape.
- Do not make the local SDK a false claim of public Tinker SDK compatibility.
  Call it Tinker-like unless parity has been proven.
- For RL LoRA, ensure current-policy logprob computation keeps gradients
  enabled. Old-logprob inputs may be detached; current logprobs must train.

## Common Validation

For API and SDK changes, run focused unit tests first:

```bash
uv run pytest tests/unit_tests/services/test_tinker_api.py \
  tests/unit_tests/services/test_tinker_api_server.py -q
```

For recipe changes, dry-run every affected recipe:

```bash
uv run python scripts/run_recipe.py qwen_sft_quick --dry-run
uv run python scripts/run_recipe.py nemotron_sft_large --dry-run
uv run python scripts/run_recipe.py nemotron_rl_lora --dry-run
```

For the standalone demo, check JavaScript syntax:

```bash
node -e 'const fs=require("fs"); const html=fs.readFileSync("demos/async_lora_demo.html","utf8"); const match=html.match(/<script>([\s\S]*)<\/script>/); new Function(match[1]); console.log("demo js syntax ok");'
```

Before committing, follow repo rules:

```bash
uv run ruff format .
uv run ruff check --fix .
git diff --check
```

## Workload Selection

- Use `qwen_sft_quick` for fast local API and UI smoke tests.
- Use `nemotron_sft_large` for full-model SFT validation on a GPU
  host.
- Use `nemotron_rl_lora` or
  `clients/rl_lora_workload_client.py` for resident RL
  LoRA validation.
- Use `POST /v1/responses` or SDK `sample_openai_response(...)` for Gym-style
  rollout collection.
- Use `POST /rl/jobs` for separate NeMo-RL launch or dry-run validation.

## Documentation Updates

- Keep `README.md` short.
- Put detailed service design in `docs/architecture.md`.
- Put SFT commands and behavior in `docs/sft.md`.
- Put RL, Gym, and NeMo-RL behavior in `docs/rl.md`.
- Put SDK examples in `docs/sdk.md`.
- Update `demos/async_lora_demo.html` when the visible
  product story changes.
