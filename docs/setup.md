# Setup

Nemotron Tinker is a product wrapper around NeMo AutoModel. The Python package
contains the Tinker-style service, SDK, UI, recipes, and tests; AutoModel still
provides the model implementations and shared import utilities.

## Local Development

Install the package and development tools with `uv`:

```bash
uv sync --extra dev
```

Put this checkout and a sibling AutoModel checkout on `PYTHONPATH` before
running tests or local scripts:

```bash
export PYTHONPATH="$(pwd)/src:/path/to/Automodel"
```

Run the fast sanity suite:

```bash
python -m pytest tests/test_tinker_api.py \
  tests/test_tinker_api_server.py \
  tests/test_host_rl_launcher.py -q
```

Validate the operator UI JavaScript after HTML edits:

```bash
node -e 'const fs=require("fs"); const html=fs.readFileSync("src/nemotron_tinker/operator_ui.html","utf8"); const match=html.match(/<script>([\s\S]*)<\/script>/); new Function(match[1]); console.log("operator ui js syntax ok");'
```

## GPU Service Requirements

The single-node Nemotron deployment needs:

- SSH access to a GPU host.
- Docker with NVIDIA GPU support.
- The `nvcr.io/nvidia/nemo-automodel:26.04` container image.
- A local or scratch-resident AutoModel checkout.
- A local or scratch-resident Nemotron-Tinker checkout.
- The Nemotron Nano 30B A3B BF16 model directory, or another supported base
  model.
- Writable scratch space for SQLite metadata, checkpoints, train request
  manifests, and Hugging Face cache files.

Start the default resident-only service:

```bash
scripts/deploy_gpu.sh start
```

Check the live service and queue:

```bash
scripts/deploy_gpu.sh status
```

Open the operator UI through the local tunnel:

```text
http://127.0.0.1:18081/ui
```

## Minimum Sanity Checklist

Before calling a deployment healthy:

1. `GET /health` returns `status: ok`.
2. `worker_alive` is true and `queue_depth` is zero before new work.
3. The UI page contains `SFT Adapter Goals` and `Resident RL Goal`.
4. A two-adapter SFT job reaches `succeeded` and both adapters sample.
5. A resident RL job reaches `succeeded`, uses an RL loss such as
   `importance_sampling`, and the same run can sample afterward.
6. Adapter export returns a downloadable archive path or response.

The external NeMo-RL bridge is not part of this minimum service sanity check.
It is a separate backend integration path for dedicated NeMo-RL jobs.
