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

"""Experimental multi-node placement primitives for Nemotron Tinker.

This module is intentionally control-plane only. It gives the API and scripts a
stable way to describe a future multi-node resident-worker fleet without
claiming that model execution has moved out of the single-node path yet.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ClusterNode:
    """One experimental resident worker node."""

    node_id: str
    host: str
    gpus: int
    role: str = "worker"
    scratch_dir: Optional[str] = None
    port: int = 18080
    labels: dict[str, str] = field(default_factory=dict)
    max_adapters: Optional[int] = None


@dataclass
class DistributedParallelism:
    """AutoModel parallelism intent for a future distributed worker group."""

    strategy: str = "fsdp2"
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    sequence_parallel: bool = False
    activation_checkpointing: bool = False


@dataclass
class ClusterConfig:
    """Experimental cluster descriptor loaded by the API control plane."""

    enabled: bool = False
    mode: str = "single_node"
    rendezvous_backend: str = "c10d"
    rendezvous_endpoint: Optional[str] = None
    base_model: Optional[str] = None
    container_image: Optional[str] = None
    nodes: list[ClusterNode] = field(default_factory=list)
    parallelism: DistributedParallelism = field(default_factory=DistributedParallelism)

    @property
    def world_size(self) -> int:
        """Return the total GPU/rank capacity described by the config."""
        return sum(max(0, int(node.gpus)) for node in self.nodes)

    @property
    def experimental(self) -> bool:
        """Return whether the descriptor asks for more than local single-node mode."""
        return self.enabled and (self.mode != "single_node" or len(self.nodes) > 1)

    def placement_plan(self) -> dict[str, Any]:
        """Return deterministic node/rank placement metadata."""
        ranks = []
        global_rank = 0
        for node_rank, node in enumerate(self.nodes):
            for local_rank in range(node.gpus):
                ranks.append(
                    {
                        "rank": global_rank,
                        "node_rank": node_rank,
                        "local_rank": local_rank,
                        "node_id": node.node_id,
                        "host": node.host,
                        "role": node.role,
                        "scratch_dir": node.scratch_dir,
                        "port": node.port,
                    }
                )
                global_rank += 1
        return {
            "enabled": self.enabled,
            "experimental": self.experimental,
            "mode": self.mode,
            "world_size": self.world_size,
            "rendezvous_backend": self.rendezvous_backend,
            "rendezvous_endpoint": self.rendezvous_endpoint,
            "parallelism": asdict(self.parallelism),
            "nodes": [asdict(node) for node in self.nodes],
            "ranks": ranks,
            "status": "planning_only",
            "warning": "Experimental control-plane descriptor only; model execution remains single-node until worker RPC is implemented.",
        }

    def launch_manifest(self) -> dict[str, Any]:
        """Return a command-oriented launch manifest for future node workers."""
        plan = self.placement_plan()
        commands = []
        for rank in plan["ranks"]:
            env = {
                "MASTER_ADDR": (self.rendezvous_endpoint or "127.0.0.1:29500").split(":", 1)[0],
                "MASTER_PORT": (self.rendezvous_endpoint or "127.0.0.1:29500").split(":", 1)[1],
                "WORLD_SIZE": str(plan["world_size"]),
                "RANK": str(rank["rank"]),
                "LOCAL_RANK": str(rank["local_rank"]),
                "NODE_RANK": str(rank["node_rank"]),
                "NEMOTRON_TINKER_EXPERIMENTAL_MULTINODE": "1",
            }
            commands.append(
                {
                    "node_id": rank["node_id"],
                    "host": rank["host"],
                    "rank": rank["rank"],
                    "local_rank": rank["local_rank"],
                    "env": env,
                    "command": [
                        "python",
                        "scripts/run_mixed_lora_server.py",
                        "--experimental-cluster-config",
                        "<cluster-config.json>",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(rank["port"]),
                    ],
                }
            )
        return {**plan, "commands": commands}


def _node_from_dict(payload: dict[str, Any]) -> ClusterNode:
    return ClusterNode(
        node_id=str(payload["node_id"]),
        host=str(payload["host"]),
        gpus=int(payload.get("gpus", 1)),
        role=str(payload.get("role", "worker")),
        scratch_dir=payload.get("scratch_dir"),
        port=int(payload.get("port", 18080)),
        labels=dict(payload.get("labels") or {}),
        max_adapters=payload.get("max_adapters"),
    )


def load_cluster_config(path: Optional[str]) -> ClusterConfig:
    """Load a cluster descriptor from JSON, or return disabled defaults."""
    if not path:
        return ClusterConfig()
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    parallelism = DistributedParallelism(**dict(payload.get("parallelism") or {}))
    nodes = [_node_from_dict(item) for item in payload.get("nodes", [])]
    return ClusterConfig(
        enabled=bool(payload.get("enabled", True)),
        mode=str(payload.get("mode", "multi_node" if len(nodes) > 1 else "single_node")),
        rendezvous_backend=str(payload.get("rendezvous_backend", "c10d")),
        rendezvous_endpoint=payload.get("rendezvous_endpoint"),
        base_model=payload.get("base_model"),
        container_image=payload.get("container_image"),
        nodes=nodes,
        parallelism=parallelism,
    )
