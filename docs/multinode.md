# Experimental Multi-Node Support

This branch adds a planning layer for multi-node Nemotron Tinker. It is not a
validated distributed model runtime yet.

## What Exists

- A JSON cluster descriptor:
  `configs/experimental_multinode_cluster.json`
- A loader and placement planner:
  `src/nemotron_tinker/cluster.py`
- Server startup flag:
  `--experimental-cluster-config <path>`
- Read-only planning endpoints:
  - `GET /experimental/cluster`
  - `GET /experimental/cluster/launch_manifest`
- `/health` now includes `experimental_cluster`.

## What It Means

The descriptor answers:

- Which hosts belong to the resident worker fleet.
- How many ranks/GPUs each host contributes.
- What rendezvous endpoint future distributed workers should use.
- What AutoModel parallelism shape is intended (`fsdp2`, TP, PP, CP, EP).
- What command envelope each rank would need.

This lets the control plane expose placement and launch metadata before real
multi-host worker RPC is implemented.

## What Is Still Not Implemented

- No cross-node model execution.
- No `torchrun` or Slurm orchestration from the API.
- No distributed adapter state sync.
- No remote worker registration protocol.
- No network fault recovery or rank rehydration.
- No tested TP/PP/CP/EP resident adapter path.

Model execution remains `api_process` and `mixed_lora_single_process`.

## Example

```bash
python scripts/run_mixed_lora_server.py \
  --base-model /models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --scratch-dir /tmp/nemotron_tinker \
  --experimental-cluster-config configs/experimental_multinode_cluster.json \
  --host 0.0.0.0 \
  --port 18080
```

Inspect the plan:

```bash
curl http://127.0.0.1:18080/experimental/cluster
curl http://127.0.0.1:18080/experimental/cluster/launch_manifest
```

## Recommended Next Steps

1. Add remote worker registration:
   `POST /internal/workers/register` with node id, host, GPU count, rank range,
   health URL, and worker token.
2. Move model operations behind worker RPC:
   create adapter, train steps, sample, save, export.
3. Add a rank-group launcher:
   render the launch manifest into `torchrun`, Slurm, or Kubernetes jobs.
4. Add distributed state ownership:
   define whether an adapter is owned by one worker replica, all TP ranks, or a
   sharded FSDP2 group.
5. Validate one distributed strategy at a time:
   start with single-node multi-process FSDP2, then two-node FSDP2, then TP,
   then PP/CP/EP.

## Product Framing

Multi-node Nemotron Tinker should scale like an always-on service:

- Keep base model replicas hot.
- Route LoRA jobs to available resident capacity.
- Queue and batch compatible adapter work.
- Keep adapter versions durable and portable.
- Add nodes as queue depth and latency require.

The goal is not to make two LoRAs train for the price of one. The goal is to
amortize cold starts, centralize scheduling, and scale resident adapter capacity
as demand grows.
