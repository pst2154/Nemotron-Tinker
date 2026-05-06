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

import json
import os
import pathlib
import queue
import signal
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
import zipfile
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal, Optional

from nemotron_tinker.mixed_client import (
    MixedLoraBackend,
    MixedLoraServiceClient,
    MixedLoraTrainingClient,
)
from nemotron_tinker.types import AdamParams, Datum, LoraConfig, ModelInput, SamplingParams
from nemotron_tinker.worker_manager import ProcessWorkerManager, WorkerProcessRecord
from nemo_automodel.shared.import_utils import safe_import_from

HAS_FASTAPI, FastAPI = safe_import_from("fastapi", "FastAPI")
_, HTTPException = safe_import_from("fastapi", "HTTPException")
_, Request = safe_import_from("fastapi", "Request")
_, FileResponse = safe_import_from("fastapi.responses", "FileResponse")
_, HTMLResponse = safe_import_from("fastapi.responses", "HTMLResponse")
_, JSONResponse = safe_import_from("fastapi.responses", "JSONResponse")
HAS_PYDANTIC, BaseModel = safe_import_from("pydantic", "BaseModel")
_, Field = safe_import_from("pydantic", "Field")

MetadataBackend = Literal["sqlite", "json"]
RLLauncher = Literal["local", "docker", "host"]
RLRunner = Literal["uv", "python"]
TENANT_HEADER = "x-tinker-tenant-id"

if not HAS_PYDANTIC:  # pragma: no cover

    class BaseModel:
        """Placeholder used only to keep module import safe without pydantic."""

    def Field(*args, **kwargs):  # noqa: N802
        """Placeholder used only to keep module import safe without pydantic."""
        return None


class ModelInputRequest(BaseModel):
    """Token IDs for one model input sequence."""

    tokens: list[int]


class DatumRequest(BaseModel):
    """One tokenized training datum."""

    model_input: ModelInputRequest
    loss_fn_inputs: dict[str, Any] = Field(default_factory=dict)


class TextSFTDatumRequest(BaseModel):
    """One text SFT example that the service tokenizes into a training datum."""

    prompt: str
    completion: str
    max_tokens: int = 64
    use_chat_template: bool = False
    disable_thinking: bool = True


class CreateRunRequest(BaseModel):
    """Create one resident LoRA adapter run."""

    name: Optional[str] = None
    adapter_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    tenant_id: Optional[str] = None
    idempotency_key: Optional[str] = None


class CreateRunResponse(BaseModel):
    """Created run metadata."""

    run_id: str
    adapter_id: str
    name: Optional[str] = None
    status: str
    sequence: int
    worker_id: Optional[str] = None


class ForwardBackwardRequest(BaseModel):
    """Forward/backward request for one run."""

    data: list[DatumRequest]
    loss_fn: str = "cross_entropy"
    loss_fn_config: dict[str, float] = Field(default_factory=dict)


class MixedForwardBackwardRequest(BaseModel):
    """Forward/backward request containing batches for multiple runs."""

    batches: dict[str, list[DatumRequest]]
    loss_fn: str = "cross_entropy"
    loss_fn_config: dict[str, float] = Field(default_factory=dict)


class OptimStepRequest(BaseModel):
    """Optimizer step request."""

    learning_rate: float
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    idempotency_key: Optional[str] = None


class SaveRequest(BaseModel):
    """Save request."""

    name: str
    idempotency_key: Optional[str] = None


class ExportRequest(BaseModel):
    """Save and export request."""

    name: Optional[str] = None
    idempotency_key: Optional[str] = None


class SaveAndDetachRequest(BaseModel):
    """Save-and-detach request."""

    name: str
    idempotency_key: Optional[str] = None


class DetachRunRequest(BaseModel):
    """Detach request."""

    idempotency_key: Optional[str] = None


class SampleRequest(BaseModel):
    """Sampling request."""

    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    return_logprobs: bool = False
    use_chat_template: bool = False
    disable_thinking: bool = True


class TrainStepsRequest(BaseModel):
    """Server-owned mixed training loop request."""

    batches: dict[str, list[DatumRequest]]
    steps: int
    learning_rate: float
    batch_size: int = 1
    microbatch_size: Optional[int] = None
    loss_fn: str = "cross_entropy"
    loss_fn_config: dict[str, float] = Field(default_factory=dict)
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    save_names: dict[str, str] = Field(default_factory=dict)
    run_async: bool = False
    tenant_id: Optional[str] = None
    idempotency_key: Optional[str] = None


class RunRecord(BaseModel):
    """In-memory run metadata."""

    run_id: str
    adapter_id: str
    name: Optional[str] = None
    tenant_id: Optional[str] = None
    status: str = "created"
    sequence: int = 0
    optimizer_steps: int = 0
    forward_backward_calls: int = 0
    last_loss: Optional[float] = None
    last_metrics: dict[str, Any] = Field(default_factory=dict)
    last_checkpoint_path: Optional[str] = None
    last_error: Optional[str] = None
    restored_from: Optional[str] = None
    worker_id: Optional[str] = None
    created_at: str
    updated_at: str


class JobRecord(BaseModel):
    """Queued operation metadata."""

    job_id: str
    kind: str
    status: str
    tenant_id: Optional[str] = None
    sequence: int = 0
    run_ids: list[str] = Field(default_factory=list)
    progress: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class JobSummary(BaseModel):
    """Compact queued operation metadata for list views."""

    job_id: str
    kind: str
    status: str
    tenant_id: Optional[str] = None
    sequence: int = 0
    run_ids: list[str] = Field(default_factory=list)
    progress: dict[str, Any] = Field(default_factory=dict)
    has_result: bool = False
    error: Optional[str] = None
    created_at: str
    updated_at: str


class RLJobRequest(BaseModel):
    """Launch request for a NeMo-RL recipe."""

    name: Optional[str] = None
    repo_dir: Optional[str] = None
    config_path: str = "examples/configs/grpo_math_1B.yaml"
    entrypoint: str = "examples/run_grpo.py"
    overrides: list[str] = Field(default_factory=list)
    num_nodes: Optional[int] = None
    gpus_per_node: Optional[int] = None
    tensor_parallel_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None
    context_parallel_size: Optional[int] = None
    expert_parallel_size: Optional[int] = None
    launcher: RLLauncher = "local"
    runner: RLRunner = "uv"
    docker_repo_dir: Optional[str] = None
    docker_container_repo_dir: str = "/opt/nemo-rl"
    docker_hf_cache_dir: Optional[str] = None
    docker_container_hf_cache_dir: str = "/root/.cache/huggingface"
    docker_output_dir: Optional[str] = None
    docker_container_output_dir: str = "/workspace/rl_outputs"
    docker_user: Optional[str] = None
    docker_gpus: str = "all"
    container_image: str = "nvcr.io/nvidia/nemo-rl:v0.6.0"
    max_runtime_seconds: Optional[float] = None
    run_async: bool = True
    dry_run: bool = False
    tenant_id: Optional[str] = None
    idempotency_key: Optional[str] = None


class RLJobRecord(BaseModel):
    """NeMo-RL launch metadata."""

    job_id: str
    name: Optional[str] = None
    status: str
    tenant_id: Optional[str] = None
    launcher: RLLauncher = "local"
    repo_dir: str
    config_path: str
    entrypoint: str
    command: list[str] = Field(default_factory=list)
    log_path: Optional[str] = None
    pid: Optional[int] = None
    returncode: Optional[int] = None
    error: Optional[str] = None
    max_runtime_seconds: Optional[float] = None
    created_at: str
    updated_at: str


class RLJobSubmitResponse(BaseModel):
    """Response returned when a NeMo-RL job is submitted."""

    job: RLJobRecord


class RLJobMarkRequest(BaseModel):
    """Status update from a host-side RL launcher worker."""

    status: Optional[str] = None
    pid: Optional[int] = None
    returncode: Optional[int] = None
    error: Optional[str] = None
    log_path: Optional[str] = None


class IdempotencyRecord(BaseModel):
    """Stored response for a retryable mutating request."""

    key: str
    operation: str
    fingerprint: str
    status: str
    response: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class JobSubmitResponse(BaseModel):
    """Response returned when work is submitted asynchronously."""

    job: JobRecord


class TrainStepsResponse(BaseModel):
    """Synchronous server-owned train loop response."""

    job: JobRecord
    runs: dict[str, RunRecord]
    outputs: dict[str, Any]


class ForwardBackwardResponse(BaseModel):
    """Forward/backward response with run metadata."""

    run: RunRecord
    output: dict[str, Any]


class OptimStepResponseModel(BaseModel):
    """Optimizer step response with run metadata."""

    run: RunRecord
    output: dict[str, Any]


class SaveResponse(BaseModel):
    """Save response with run metadata."""

    run: RunRecord
    output: dict[str, Any]


class DetachRunResponse(BaseModel):
    """Detach response with run metadata."""

    run: RunRecord
    output: dict[str, Any]


class SaveAndDetachResponse(BaseModel):
    """Save-and-detach response with run metadata."""

    run: RunRecord
    save_output: dict[str, Any]
    detach_output: dict[str, Any]


class SampleResponseModel(BaseModel):
    """Sample response with run metadata."""

    run: RunRecord
    output: dict[str, Any]


class WorkerCommandResponse(BaseModel):
    """Response from one supervised worker command."""

    worker: WorkerProcessRecord
    result: dict[str, Any]


class WorkerEchoRequest(BaseModel):
    """Payload for testing worker RPC serialization."""

    payload: dict[str, Any] = Field(default_factory=dict)


class WorkerRunsResponse(BaseModel):
    """Runs currently attached to one supervised worker."""

    worker: WorkerProcessRecord
    runs: list[dict[str, Any]] = Field(default_factory=list)
    assigned_run_count: int = 0


class WorkerOperationsResponse(BaseModel):
    """Model-operation RPC envelopes recorded by one supervised worker."""

    worker: WorkerProcessRecord
    operations: list[dict[str, Any]] = Field(default_factory=list)
    operation_count: int = 0


class WorkerReconcileResponse(BaseModel):
    """Result from reconciling resident runs with supervised workers."""

    reassigned_run_ids: list[str] = Field(default_factory=list)
    reattached_run_ids: list[str] = Field(default_factory=list)
    workers: list[WorkerProcessRecord] = Field(default_factory=list)


class OperationMetric(BaseModel):
    """Aggregated timing and failure metrics for one operation."""

    count: int = 0
    failures: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0
    last_seconds: float = 0.0
    last_error: Optional[str] = None


class ServiceMetricsSnapshot(BaseModel):
    """Service-level operation metrics."""

    started_at: str
    uptime_seconds: float
    operations: dict[str, OperationMetric] = Field(default_factory=dict)


def _datum_from_request(request: DatumRequest) -> Datum:
    loss_fn_inputs = dict(request.loss_fn_inputs)
    target_tokens = loss_fn_inputs.get("target_tokens")
    if isinstance(target_tokens, dict) and "tokens" in target_tokens:
        loss_fn_inputs["target_tokens"] = ModelInput.from_ints(target_tokens["tokens"])
    return Datum(
        model_input=ModelInput.from_ints(request.model_input.tokens),
        loss_fn_inputs=loss_fn_inputs,
    )


def _adam_from_request(request: OptimStepRequest) -> AdamParams:
    return AdamParams(
        learning_rate=request.learning_rate,
        weight_decay=request.weight_decay,
        betas=request.betas,
        eps=request.eps,
    )


def _sampling_from_request(request: SampleRequest) -> SamplingParams:
    return SamplingParams(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=request.do_sample,
        return_logprobs=request.return_logprobs,
    )


def _apply_chat_template_or_raise(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    disable_thinking: bool = True,
):
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        raise HTTPException(status_code=400, detail="Tokenizer does not provide apply_chat_template().")
    kwargs = {"enable_thinking": False} if disable_thinking else {}
    try:
        output = apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise HTTPException(status_code=400, detail=f"Tokenizer chat template failed: {exc}") from exc
        try:
            output = apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
        except Exception as retry_exc:
            raise HTTPException(status_code=400, detail=f"Tokenizer chat template failed: {retry_exc}") from retry_exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Tokenizer chat template failed: {exc}") from exc
    if not tokenize:
        return output
    if hasattr(output, "data") and "input_ids" in output.data:
        output = output.data["input_ids"]
    elif isinstance(output, dict) and "input_ids" in output:
        output = output["input_ids"]
    if output and isinstance(output[0], list):
        output = output[0]
    return list(output)


def _sample_prompt_from_request(tokenizer, request: SampleRequest) -> str:
    if not request.use_chat_template:
        return request.prompt
    return _apply_chat_template_or_raise(
        tokenizer,
        [{"role": "user", "content": request.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        disable_thinking=request.disable_thinking,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_with_update(model: BaseModel, **updates: Any) -> BaseModel:
    payload = _model_to_dict(model)
    payload.update(updates)
    return type(model)(**payload)


def _client_step(client: MixedLoraTrainingClient) -> int:
    handle = getattr(client, "handle", None)
    return int(getattr(handle, "step", 0))


def _fingerprint_request(operation: str, request: BaseModel) -> str:
    payload = {"operation": operation, "request": _model_to_dict(request)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _digest_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _atomic_write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex[:8]}")
    tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_operator_ui() -> str:
    return (pathlib.Path(__file__).with_name("operator_ui.html")).read_text(encoding="utf-8")


def _content_to_text(content: Any) -> str:
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


def _openai_messages_to_prompt(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return _content_to_text(messages)
    prompt_parts = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = _content_to_text(message.get("content"))
            if content:
                prompt_parts.append(f"{role}: {content}")
    if not prompt_parts:
        return ""
    return "\n".join(prompt_parts) + "\nassistant:"


def _optional_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _metadata_flag(body: dict[str, Any], key: str) -> bool:
    metadata = body.get("metadata")
    return bool(body.get(key) or (isinstance(metadata, dict) and metadata.get(key)))


def _resolve_rl_path(repo_dir: pathlib.Path, relative_path: str, field_name: str) -> pathlib.Path:
    candidate = (repo_dir / relative_path).resolve()
    try:
        candidate.relative_to(repo_dir)
    except ValueError as exc:
        raise ValueError(f"{field_name} must stay inside repo_dir") from exc
    if not candidate.exists():
        raise ValueError(f"{field_name} does not exist: {candidate}")
    return candidate


def _append_rl_override_if_absent(overrides: list[str], key: str, value: int) -> None:
    if not any(override.split("=", 1)[0] == key for override in overrides):
        overrides.append(f"{key}={value}")


def _build_rl_overrides(request: RLJobRequest) -> list[str]:
    overrides = list(request.overrides)
    topology = {
        "num_nodes": request.num_nodes,
        "gpus_per_node": request.gpus_per_node,
        "tensor_parallel_size": request.tensor_parallel_size,
        "pipeline_parallel_size": request.pipeline_parallel_size,
        "context_parallel_size": request.context_parallel_size,
        "expert_parallel_size": request.expert_parallel_size,
    }
    for name, value in topology.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1")

    if request.num_nodes is not None:
        _append_rl_override_if_absent(overrides, "cluster.num_nodes", request.num_nodes)
    if request.gpus_per_node is not None:
        _append_rl_override_if_absent(overrides, "cluster.gpus_per_node", request.gpus_per_node)
    if request.tensor_parallel_size is not None:
        _append_rl_override_if_absent(
            overrides, "policy.dtensor_cfg.tensor_parallel_size", request.tensor_parallel_size
        )
        _append_rl_override_if_absent(
            overrides, "policy.megatron_cfg.tensor_model_parallel_size", request.tensor_parallel_size
        )
    if request.context_parallel_size is not None:
        _append_rl_override_if_absent(
            overrides, "policy.dtensor_cfg.context_parallel_size", request.context_parallel_size
        )
        _append_rl_override_if_absent(
            overrides, "policy.megatron_cfg.context_parallel_size", request.context_parallel_size
        )
    if request.pipeline_parallel_size is not None:
        _append_rl_override_if_absent(
            overrides, "policy.megatron_cfg.pipeline_model_parallel_size", request.pipeline_parallel_size
        )
    if request.expert_parallel_size is not None:
        _append_rl_override_if_absent(
            overrides, "policy.megatron_cfg.expert_model_parallel_size", request.expert_parallel_size
        )

    if request.num_nodes in (None, 1) and request.gpus_per_node is not None:
        parallel_product = 1
        for value in (
            request.tensor_parallel_size,
            request.pipeline_parallel_size,
            request.context_parallel_size,
            request.expert_parallel_size,
        ):
            parallel_product *= value or 1
        if parallel_product > request.gpus_per_node:
            raise ValueError(
                "tensor_parallel_size * pipeline_parallel_size * context_parallel_size * "
                "expert_parallel_size must be <= gpus_per_node for single-node launches"
            )
    return overrides


def _build_rl_command(request: RLJobRequest, repo_dir: pathlib.Path) -> list[str]:
    entrypoint = _resolve_rl_path(repo_dir, request.entrypoint, "entrypoint")
    config_path = _resolve_rl_path(repo_dir, request.config_path, "config_path")
    if not request.docker_gpus.strip():
        raise ValueError("docker_gpus must not be empty")
    if request.max_runtime_seconds is not None and request.max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be > 0")
    overrides = _build_rl_overrides(request)
    container_output_dir = request.docker_container_output_dir.rstrip("/") or "/workspace/rl_outputs"
    if request.docker_output_dir:
        _append_rl_override_if_absent(overrides, "logger.log_dir", container_output_dir)
    if request.launcher == "local":
        runner_prefix = ["uv", "run", "python", "-u"] if request.runner == "uv" else ["python", "-u"]
        return [*runner_prefix, str(entrypoint), "--config", str(config_path), *overrides]
    container_repo = request.docker_container_repo_dir.rstrip("/") or "/opt/nemo-rl"
    container_entrypoint = str(pathlib.PurePosixPath(container_repo) / request.entrypoint)
    container_config = str(pathlib.PurePosixPath(container_repo) / request.config_path)
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        request.docker_gpus.strip(),
        "--ipc=host",
        "--network",
        "host",
    ]
    if request.docker_user:
        command.extend(["--user", request.docker_user])
    runner_prefix = (
        ["/root/.local/bin/uv", "run", "/opt/nemo_rl_venv/bin/python", "-u"]
        if request.runner == "uv"
        else ["/opt/nemo_rl_venv/bin/python", "-u"]
    )
    if request.docker_repo_dir:
        command.extend(["-v", f"{request.docker_repo_dir}:{container_repo}"])
    if request.docker_hf_cache_dir:
        container_cache = request.docker_container_hf_cache_dir.rstrip("/") or "/root/.cache/huggingface"
        command.extend(
            [
                "-v",
                f"{request.docker_hf_cache_dir}:{container_cache}",
                "-e",
                f"HF_HOME={container_cache}",
                "-e",
                f"HF_HUB_CACHE={container_cache}/hub",
                "-e",
                f"HF_DATASETS_CACHE={container_cache}/datasets",
            ]
        )
    if request.docker_output_dir:
        command.extend(["-v", f"{request.docker_output_dir}:{container_output_dir}"])
    command.extend(
        [
            "-w",
            container_repo,
            request.container_image,
            *runner_prefix,
            container_entrypoint,
            "--config",
            container_config,
            *overrides,
        ]
    )
    return command


def _check_rl_launcher_available(request: RLJobRequest) -> None:
    """Fail fast when the selected launcher cannot run from this service process."""
    if request.dry_run:
        return
    if request.launcher == "docker" and shutil.which("docker") is None:
        raise ValueError(
            "Docker launcher is not available inside this service container. "
            "Use Show RL Command and run it on the host, or restart the service with an explicitly approved Docker "
            "CLI/socket mount."
        )


class JsonStore:
    """Small JSON-backed metadata store for the prototype service."""

    def __init__(self, path: str | pathlib.Path, key: str, record_type):
        self.path = pathlib.Path(path)
        self.key = key
        self.record_type = record_type
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        """Load known records from disk."""
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return {record_id: self.record_type(**record) for record_id, record in payload.get(self.key, {}).items()}

    def save(self, records: dict[str, Any]) -> None:
        """Persist records atomically."""
        payload = {self.key: {record_id: _model_to_dict(record) for record_id, record in records.items()}}
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
        tmp_path.replace(self.path)


class SQLiteStore:
    """SQLite-backed metadata store for service records."""

    def __init__(self, path: str | pathlib.Path, key: str, record_type):
        self.path = pathlib.Path(path)
        self.key = key
        self.record_type = record_type
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    namespace TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, record_id)
                )
                """
            )

    def load(self) -> dict[str, Any]:
        """Load known records from SQLite."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record_id, payload FROM records WHERE namespace = ? ORDER BY record_id",
                (self.key,),
            ).fetchall()
        return {record_id: self.record_type(**json.loads(payload)) for record_id, payload in rows}

    def save(self, records: dict[str, Any]) -> None:
        """Persist records in one transaction."""
        rows = [
            (self.key, record_id, json.dumps(_model_to_dict(record), sort_keys=True), _utc_now())
            for record_id, record in records.items()
        ]
        with self._connect() as conn:
            conn.execute("DELETE FROM records WHERE namespace = ?", (self.key,))
            conn.executemany(
                "INSERT INTO records(namespace, record_id, payload, updated_at) VALUES (?, ?, ?, ?)",
                rows,
            )


def _build_store(
    *,
    scratch_dir: str,
    backend: MetadataBackend,
    key: str,
    record_type,
):
    if backend == "sqlite":
        return SQLiteStore(pathlib.Path(scratch_dir) / "tinker_api" / "metadata.sqlite3", key, record_type)
    if backend == "json":
        return JsonStore(pathlib.Path(scratch_dir) / "tinker_api" / f"{key}.json", key, record_type)
    raise ValueError(f"Unknown metadata backend: {backend}")


class QueuedExecutor:
    """One-thread executor that serializes all GPU-owned service operations."""

    def __init__(self):
        self._queue: queue.Queue[tuple[Future, Any]] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="tinker-api-worker", daemon=True)
        self._thread.start()

    def submit(self, fn):
        """Run `fn` on the worker thread and return a Future."""
        future = Future()
        self._queue.put((future, fn))
        return future

    def queue_depth(self) -> int:
        """Return approximate queued operation count."""
        return self._queue.qsize()

    def is_alive(self) -> bool:
        """Return whether the worker thread is alive."""
        return self._thread.is_alive()

    def _worker(self) -> None:
        while True:
            future, fn = self._queue.get()
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn())
                except Exception as exc:
                    future.set_exception(exc)
            self._queue.task_done()


class ServiceMetrics:
    """Thread-safe operation metrics for the prototype service."""

    def __init__(self):
        self.started_at = _utc_now()
        self._started_monotonic = time.monotonic()
        self._lock = threading.RLock()
        self._operations: dict[str, OperationMetric] = {}

    def observe(self, operation: str, fn):
        """Run a callable and record duration/failure metrics."""
        started = time.monotonic()
        try:
            result = fn()
        except Exception as exc:
            self.record(operation, time.monotonic() - started, error=f"{type(exc).__name__}: {exc}")
            raise
        self.record(operation, time.monotonic() - started)
        return result

    def record(self, operation: str, seconds: float, *, error: Optional[str] = None) -> None:
        """Record one completed operation."""
        with self._lock:
            metric = self._operations.setdefault(operation, OperationMetric())
            metric.count += 1
            metric.total_seconds += seconds
            metric.max_seconds = max(metric.max_seconds, seconds)
            metric.last_seconds = seconds
            if error is not None:
                metric.failures += 1
                metric.last_error = error

    def snapshot(self) -> ServiceMetricsSnapshot:
        """Return a serializable metrics snapshot."""
        with self._lock:
            operations = {
                operation: OperationMetric(**_model_to_dict(metric)) for operation, metric in self._operations.items()
            }
        return ServiceMetricsSnapshot(
            started_at=self.started_at,
            uptime_seconds=time.monotonic() - self._started_monotonic,
            operations=operations,
        )


def create_app(
    *,
    base_model: str,
    scratch_dir: str = "/tmp/nemotron_tinker",
    cache_dir: Optional[str] = None,
    rank: int = 16,
    alpha: Optional[int] = None,
    device: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    attn_implementation: str = "sdpa",
    target_modules: Optional[list[str]] = None,
    api_key: Optional[str] = None,
    max_resident_adapters: Optional[int] = None,
    max_runs_per_tenant: Optional[int] = None,
    max_concurrent_rl_jobs: Optional[int] = None,
    max_concurrent_rl_jobs_per_tenant: Optional[int] = None,
    tenant_rate_limit_per_minute: Optional[int] = None,
    mixed_lora_backend: MixedLoraBackend = "loop",
    use_triton_lora: bool = False,
    metadata_backend: MetadataBackend = "sqlite",
    restore_runs_on_startup: bool = False,
    resume_interrupted_jobs_on_startup: bool = False,
    worker_processes: int = 0,
    rl_repo_dir: Optional[str] = None,
) -> FastAPI:
    """Create a single-process mixed-LoRA FastAPI app."""
    if not HAS_FASTAPI or not HAS_PYDANTIC:
        raise ImportError("The Tinker API server requires `fastapi` and `pydantic`. Install the service extras first.")
    if metadata_backend not in {"sqlite", "json"}:
        raise ValueError(f"Unknown metadata backend: {metadata_backend}")
    if max_runs_per_tenant is not None and max_runs_per_tenant <= 0:
        raise ValueError("max_runs_per_tenant must be positive")
    if max_concurrent_rl_jobs is not None and max_concurrent_rl_jobs <= 0:
        raise ValueError("max_concurrent_rl_jobs must be positive")
    if max_concurrent_rl_jobs_per_tenant is not None and max_concurrent_rl_jobs_per_tenant <= 0:
        raise ValueError("max_concurrent_rl_jobs_per_tenant must be positive")
    if tenant_rate_limit_per_minute is not None and tenant_rate_limit_per_minute <= 0:
        raise ValueError("tenant_rate_limit_per_minute must be positive")
    if worker_processes < 0:
        raise ValueError("worker_processes must be non-negative")

    service = MixedLoraServiceClient(
        base_model=base_model,
        scratch_dir=scratch_dir,
        cache_dir=cache_dir,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        lora_config=LoraConfig(rank=rank, alpha=alpha, target_modules=target_modules or []),
        mixed_lora_backend=mixed_lora_backend,
        use_triton_lora=use_triton_lora,
    )
    active_mixed_lora_backend = "triton" if use_triton_lora else mixed_lora_backend
    run_store = _build_store(scratch_dir=scratch_dir, backend=metadata_backend, key="runs", record_type=RunRecord)
    job_store = _build_store(scratch_dir=scratch_dir, backend=metadata_backend, key="jobs", record_type=JobRecord)
    rl_job_store = _build_store(
        scratch_dir=scratch_dir,
        backend=metadata_backend,
        key="rl_jobs",
        record_type=RLJobRecord,
    )
    idempotency_store = _build_store(
        scratch_dir=scratch_dir,
        backend=metadata_backend,
        key="idempotency",
        record_type=IdempotencyRecord,
    )
    train_request_dir = pathlib.Path(scratch_dir) / "tinker_api" / "train_requests"

    def save_train_request_manifest(job_id: str, request: TrainStepsRequest) -> dict[str, Any]:
        request_payload = _model_to_dict(request)
        path = train_request_dir / f"{job_id}.json"
        digest = _digest_json(request_payload)
        _atomic_write_json(path, request_payload)
        examples_by_run = {run_id: len(batch) for run_id, batch in request.batches.items()}
        return {
            "kind": "file",
            "path": str(path),
            "sha256": digest,
            "num_runs": len(request.batches),
            "examples_by_run": examples_by_run,
            "total_examples": sum(examples_by_run.values()),
            "batch_size": request.batch_size,
            "microbatch_size": request.microbatch_size,
            "loss_fn": request.loss_fn,
        }

    def load_train_request_from_progress(progress: dict[str, Any]) -> TrainStepsRequest | None:
        legacy_request = progress.get("request")
        if legacy_request is not None:
            return TrainStepsRequest(**legacy_request)
        request_ref = progress.get("request_ref")
        if not isinstance(request_ref, dict) or request_ref.get("kind") != "file":
            return None
        path = pathlib.Path(str(request_ref.get("path", "")))
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_digest = request_ref.get("sha256")
        if expected_digest is not None and _digest_json(payload) != expected_digest:
            raise ValueError(f"Train request manifest failed checksum validation: {path}")
        return TrainStepsRequest(**payload)

    def progress_has_train_request(progress: dict[str, Any]) -> bool:
        if progress.get("request") is not None:
            return True
        request_ref = progress.get("request_ref")
        return isinstance(request_ref, dict) and pathlib.Path(str(request_ref.get("path", ""))).is_file()

    records: dict[str, RunRecord] = run_store.load()
    runs: dict[str, MixedLoraTrainingClient] = {}
    for run_id, record in records.items():
        if record.status in {"failed", "detached"}:
            continue
        checkpoint_path = record.last_checkpoint_path or record.restored_from
        if restore_runs_on_startup and checkpoint_path:
            try:
                runs[run_id] = service.create_lora_training_client(
                    adapter_id=record.adapter_id,
                    checkpoint_path=checkpoint_path,
                )
                record.status = "ready"
                record.optimizer_steps = _client_step(runs[run_id])
                record.restored_from = checkpoint_path
                record.last_error = None
                record.updated_at = _utc_now()
                continue
            except Exception as exc:
                record.last_error = f"Startup restore failed: {type(exc).__name__}: {exc}"
        record.status = "detached"
        record.updated_at = _utc_now()
    if records:
        run_store.save(records)
    jobs: dict[str, JobRecord] = job_store.load()
    for job in jobs.values():
        if job.status in {"queued", "running", "canceling"}:
            can_resume = (
                resume_interrupted_jobs_on_startup
                and restore_runs_on_startup
                and job.kind == "train_steps"
                and isinstance(job.progress, dict)
                and progress_has_train_request(job.progress)
            )
            if can_resume:
                job.status = "queued"
                job.error = None
                job.updated_at = _utc_now()
                continue
            job.status = "failed"
            job.error = "Job was interrupted by service restart"
            job.updated_at = _utc_now()
    if jobs:
        job_store.save(jobs)
    rl_jobs: dict[str, RLJobRecord] = rl_job_store.load()
    for rl_job in rl_jobs.values():
        if rl_job.status in {"queued", "running"}:
            rl_job.status = "failed"
            rl_job.error = "RL job was interrupted by service restart"
            rl_job.updated_at = _utc_now()
    if rl_jobs:
        rl_job_store.save(rl_jobs)
    idempotency_records: dict[str, IdempotencyRecord] = idempotency_store.load()
    for idem_record in idempotency_records.values():
        if idem_record.status == "running":
            idem_record.status = "failed"
            idem_record.error = "Request was interrupted by service restart"
            idem_record.updated_at = _utc_now()
    if idempotency_records:
        idempotency_store.save(idempotency_records)
    records_lock = threading.RLock()
    jobs_lock = threading.RLock()
    rl_jobs_lock = threading.RLock()
    idempotency_lock = threading.RLock()
    rate_limit_lock = threading.RLock()
    tenant_request_times: dict[str, deque[float]] = {}
    active_rl_processes: dict[str, subprocess.Popen] = {}
    executor = QueuedExecutor()
    service_metrics = ServiceMetrics()
    worker_manager = ProcessWorkerManager(num_workers=worker_processes) if worker_processes else None
    if worker_manager is not None:
        worker_manager.start()

    @asynccontextmanager
    async def lifespan(app):
        yield
        if worker_manager is not None:
            worker_manager.stop()

    app = FastAPI(title="NeMo AutoModel Tinker API Prototype", version="0.1.0", lifespan=lifespan)
    expected_api_key = api_key or os.environ.get("TINKER_API_KEY")

    if expected_api_key:

        @app.middleware("http")
        async def require_bearer_token(request, call_next):
            if request.url.path in {"/health", "/ui"}:
                return await call_next(request)
            authorization = request.headers.get("authorization")
            expected_authorization = f"Bearer {expected_api_key}"
            if authorization != expected_authorization:
                return JSONResponse(status_code=401, content={"detail": "Missing or invalid bearer token"})
            return await call_next(request)

    @app.get("/ui", response_class=HTMLResponse)
    def operator_ui() -> HTMLResponse:
        return HTMLResponse(_load_operator_ui())

    def get_run(run_id: str) -> MixedLoraTrainingClient:
        client = runs.get(run_id)
        if client is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return client

    def get_worker_record(worker_id: str) -> WorkerProcessRecord:
        if worker_manager is None:
            raise HTTPException(status_code=404, detail="Worker processes are not enabled")
        records_by_id = {record.worker_id: record for record in worker_manager.snapshot()}
        worker = records_by_id.get(worker_id)
        if worker is None:
            raise HTTPException(status_code=404, detail=f"Unknown worker_id: {worker_id}")
        return worker

    def attach_run_to_worker(record: RunRecord, *, reason: Optional[str] = None) -> None:
        if worker_manager is None or record.worker_id is None:
            return
        worker_manager.attach_run(record.worker_id, _model_to_dict(record))
        if reason is not None:
            worker_manager.record_operation(
                record.worker_id,
                operation="reattach_run",
                run_ids=[record.run_id],
                payload={"reason": reason},
            )

    def detach_run_from_worker(record: RunRecord) -> None:
        if worker_manager is None or record.worker_id is None:
            return
        worker_manager.detach_run(record.worker_id, record.run_id)

    def reattach_runs_to_worker(worker_id: str, *, reason: str) -> list[str]:
        if worker_manager is None:
            return []
        with records_lock:
            assigned_records = [
                record for run_id, record in records.items() if run_id in runs and record.worker_id == worker_id
            ]
        for record in assigned_records:
            attach_run_to_worker(record, reason=reason)
        return [record.run_id for record in assigned_records]

    def reattach_runs_to_workers(worker_ids: list[str], *, reason: str) -> list[str]:
        reattached_run_ids = []
        for worker_id in worker_ids:
            reattached_run_ids.extend(reattach_runs_to_worker(worker_id, reason=reason))
        return reattached_run_ids

    def reconcile_worker_assignments(*, reason: str) -> WorkerReconcileResponse:
        if worker_manager is None:
            return WorkerReconcileResponse()
        worker_records = worker_manager.snapshot()
        running_worker_ids = {record.worker_id for record in worker_records if record.status == "running"}
        reassigned_run_ids = []
        dirty_records = False
        with records_lock:
            resident_records = [record for run_id, record in records.items() if run_id in runs]
            for record in resident_records:
                if record.worker_id in running_worker_ids:
                    continue
                assigned_worker = worker_manager.assign(record.run_id)
                if assigned_worker is None:
                    continue
                record.worker_id = assigned_worker.worker_id
                record.updated_at = _utc_now()
                reassigned_run_ids.append(record.run_id)
                dirty_records = True
            if dirty_records:
                run_store.save(records)
            worker_ids = sorted({record.worker_id for record in resident_records if record.worker_id is not None})
        reattached_run_ids = reattach_runs_to_workers(worker_ids, reason=reason)
        return WorkerReconcileResponse(
            reassigned_run_ids=reassigned_run_ids,
            reattached_run_ids=reattached_run_ids,
            workers=worker_manager.snapshot(),
        )

    def record_worker_operation(operation: str, run_ids: list[str], payload: Optional[dict[str, Any]] = None) -> None:
        if worker_manager is None:
            return
        worker_run_ids: dict[str, list[str]] = {}
        for run_id in run_ids:
            record = get_record(run_id)
            if record.worker_id is None:
                continue
            worker_run_ids.setdefault(record.worker_id, []).append(run_id)
        for worker_id, assigned_run_ids in worker_run_ids.items():
            worker_manager.record_operation(
                worker_id,
                operation=operation,
                run_ids=assigned_run_ids,
                payload=payload,
            )

    if worker_manager is not None:
        with records_lock:
            dirty_records = False
            for run_id, client in runs.items():
                record = records[run_id]
                if record.worker_id is None:
                    assigned_worker = worker_manager.assign(run_id)
                    if assigned_worker is not None:
                        record.worker_id = assigned_worker.worker_id
                        record.adapter_id = client.adapter_id
                        record.updated_at = _utc_now()
                        dirty_records = True
                attach_run_to_worker(record, reason="startup")
            if dirty_records:
                run_store.save(records)

    app.state.executor = executor
    app.state.tinker_service = service
    app.state.service_metrics = service_metrics
    app.state.worker_manager = worker_manager
    app.state.reattach_runs_to_workers = reattach_runs_to_workers
    app.state.reconcile_worker_assignments = reconcile_worker_assignments

    @app.get("/health")
    def health() -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        rl_job_status_counts: dict[str, int] = {}
        stale_worker_run_ids = []
        worker_records = worker_manager.snapshot() if worker_manager is not None else []
        running_worker_ids = {record.worker_id for record in worker_records if record.status == "running"}
        with records_lock:
            for run_id, record in records.items():
                status_counts[record.status] = status_counts.get(record.status, 0) + 1
                if run_id in runs and worker_manager is not None and record.worker_id not in running_worker_ids:
                    stale_worker_run_ids.append(run_id)
        with rl_jobs_lock:
            for job in rl_jobs.values():
                rl_job_status_counts[job.status] = rl_job_status_counts.get(job.status, 0) + 1
            active_rl_process_count = len(active_rl_processes)
        return {
            "status": "ok",
            "base_model": base_model,
            "num_runs": len(runs),
            "num_records": len(records),
            "mode": "mixed_lora_single_process",
            "model_execution": "api_process",
            "run_status_counts": status_counts,
            "rl_job_status_counts": rl_job_status_counts,
            "active_rl_process_count": active_rl_process_count,
            "stale_worker_run_ids": stale_worker_run_ids,
            "worker_assignment_ready": worker_manager is None or not stale_worker_run_ids,
            "queue_depth": executor.queue_depth(),
            "worker_alive": executor.is_alive(),
            "run_store": str(run_store.path),
            "job_store": str(job_store.path),
            "rl_job_store": str(rl_job_store.path),
            "rl_repo_dir": rl_repo_dir or os.environ.get("NEMO_RL_REPO_DIR"),
            "idempotency_store": str(idempotency_store.path),
            "auth_enabled": expected_api_key is not None,
            "max_resident_adapters": max_resident_adapters,
            "max_runs_per_tenant": max_runs_per_tenant,
            "max_concurrent_rl_jobs": max_concurrent_rl_jobs,
            "max_concurrent_rl_jobs_per_tenant": max_concurrent_rl_jobs_per_tenant,
            "tenant_rate_limit_per_minute": tenant_rate_limit_per_minute,
            "mixed_lora_backend": active_mixed_lora_backend,
            "use_triton_lora": use_triton_lora,
            "metadata_backend": metadata_backend,
            "restore_runs_on_startup": restore_runs_on_startup,
            "resume_interrupted_jobs_on_startup": resume_interrupted_jobs_on_startup,
            "worker_processes": worker_processes,
            "workers": [
                _model_to_dict(record) if isinstance(record, BaseModel) else asdict(record) for record in worker_records
            ],
            "metrics": _model_to_dict(service_metrics.snapshot()),
        }

    @app.get("/metrics", response_model=ServiceMetricsSnapshot)
    def metrics() -> ServiceMetricsSnapshot:
        return service_metrics.snapshot()

    @app.post("/datasets/sft_datum", response_model=DatumRequest)
    def tokenize_sft_datum(request: TextSFTDatumRequest) -> DatumRequest:
        if request.use_chat_template:
            prompt_tokens = _apply_chat_template_or_raise(
                service.tokenizer,
                [{"role": "user", "content": request.prompt}],
                tokenize=True,
                add_generation_prompt=True,
                disable_thinking=request.disable_thinking,
            )
            tokens = _apply_chat_template_or_raise(
                service.tokenizer,
                [
                    {"role": "user", "content": request.prompt},
                    {"role": "assistant", "content": request.completion.strip()},
                ],
                tokenize=True,
                add_generation_prompt=False,
                disable_thinking=request.disable_thinking,
            )[: request.max_tokens]
        else:
            prompt_tokens = service.tokenizer.encode(request.prompt, add_special_tokens=True)
            completion_tokens = service.tokenizer.encode(request.completion, add_special_tokens=False)
            tokens = (prompt_tokens + completion_tokens)[: request.max_tokens]
        if len(tokens) < 2:
            raise HTTPException(status_code=400, detail="Text SFT datum needs at least two tokens")
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        first_completion_label = max(0, min(len(prompt_tokens), len(tokens)) - 1)
        weights = [0.0] * first_completion_label + [1.0] * max(0, len(target_tokens) - first_completion_label)
        return DatumRequest(
            model_input=ModelInputRequest(tokens=input_tokens),
            loss_fn_inputs={"target_tokens": {"tokens": target_tokens}, "weights": weights},
        )

    @app.get("/workers", response_model=list[WorkerProcessRecord])
    def list_workers() -> list[WorkerProcessRecord]:
        if worker_manager is None:
            return []
        return worker_manager.snapshot()

    @app.post("/workers/restart_dead", response_model=list[WorkerProcessRecord])
    def restart_dead_workers() -> list[WorkerProcessRecord]:
        if worker_manager is None:
            return []
        restarted = worker_manager.restart_dead()
        reattach_runs_to_workers([record.worker_id for record in restarted], reason="worker_restart")
        return [get_worker_record(record.worker_id) for record in restarted]

    @app.post("/workers/reconcile", response_model=WorkerReconcileResponse)
    def reconcile_workers() -> WorkerReconcileResponse:
        if worker_manager is None:
            return WorkerReconcileResponse()
        return reconcile_worker_assignments(reason="manual_reconcile")

    @app.post("/workers/{worker_id}/ping", response_model=WorkerCommandResponse)
    def ping_worker(worker_id: str) -> WorkerCommandResponse:
        if worker_manager is None:
            raise HTTPException(status_code=404, detail="Worker processes are not enabled")
        result = worker_manager.submit(worker_id, "ping", timeout_seconds=5.0)
        return WorkerCommandResponse(worker=get_worker_record(worker_id), result=result)

    @app.post("/workers/{worker_id}/echo", response_model=WorkerCommandResponse)
    def echo_worker(worker_id: str, request: WorkerEchoRequest) -> WorkerCommandResponse:
        if worker_manager is None:
            raise HTTPException(status_code=404, detail="Worker processes are not enabled")
        result = worker_manager.submit(worker_id, "echo", payload=request.payload, timeout_seconds=5.0)
        return WorkerCommandResponse(worker=get_worker_record(worker_id), result=result)

    @app.get("/workers/{worker_id}/runs", response_model=WorkerRunsResponse)
    def list_worker_runs(worker_id: str) -> WorkerRunsResponse:
        if worker_manager is None:
            raise HTTPException(status_code=404, detail="Worker processes are not enabled")
        worker = get_worker_record(worker_id)
        result = worker_manager.list_runs(worker_id)
        return WorkerRunsResponse(
            worker=worker,
            runs=result.get("runs", []),
            assigned_run_count=result.get("assigned_run_count", 0),
        )

    @app.get("/workers/{worker_id}/operations", response_model=WorkerOperationsResponse)
    def list_worker_operations(worker_id: str) -> WorkerOperationsResponse:
        if worker_manager is None:
            raise HTTPException(status_code=404, detail="Worker processes are not enabled")
        worker = get_worker_record(worker_id)
        result = worker_manager.list_operations(worker_id)
        return WorkerOperationsResponse(
            worker=worker,
            operations=result.get("operations", []),
            operation_count=result.get("operation_count", 0),
        )

    def tenant_key(tenant_id: Optional[str]) -> str:
        return tenant_id or "_default"

    def request_tenant_id(http_request: Request) -> Optional[str]:
        tenant_id = http_request.headers.get(TENANT_HEADER)
        if tenant_id is None:
            return None
        tenant_id = tenant_id.strip()
        return tenant_id or None

    def resolve_tenant_id(body_tenant_id: Optional[str], http_request: Request) -> Optional[str]:
        header_tenant_id = request_tenant_id(http_request)
        if body_tenant_id is not None and header_tenant_id is not None and body_tenant_id != header_tenant_id:
            raise HTTPException(status_code=403, detail="Request tenant_id does not match X-Tinker-Tenant-Id")
        return header_tenant_id or body_tenant_id

    def authorize_tenant(record_tenant_id: Optional[str], http_request: Request) -> None:
        header_tenant_id = request_tenant_id(http_request)
        if header_tenant_id is None:
            return
        if record_tenant_id != header_tenant_id:
            raise HTTPException(status_code=403, detail="X-Tinker-Tenant-Id is not authorized for this resource")

    def visible_to_request(record_tenant_id: Optional[str], http_request: Request) -> bool:
        header_tenant_id = request_tenant_id(http_request)
        return header_tenant_id is None or record_tenant_id == header_tenant_id

    def enforce_tenant_rate_limit(tenant_id: Optional[str], operation: str) -> None:
        if tenant_rate_limit_per_minute is None:
            return
        key = tenant_key(tenant_id)
        now = time.monotonic()
        window_start = now - 60.0
        with rate_limit_lock:
            request_times = tenant_request_times.setdefault(key, deque())
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            if len(request_times) >= tenant_rate_limit_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Tenant {key!r} exceeded {tenant_rate_limit_per_minute} "
                        f"requests/minute for GPU operation {operation!r}"
                    ),
                )
            request_times.append(now)

    def enforce_tenant_run_quota(tenant_id: Optional[str]) -> None:
        if max_runs_per_tenant is None:
            return
        key = tenant_key(tenant_id)
        with records_lock:
            resident_count = sum(
                1
                for run_id, record in records.items()
                if run_id in runs and tenant_key(record.tenant_id) == key and record.status != "detached"
            )
        if resident_count >= max_runs_per_tenant:
            raise HTTPException(
                status_code=429,
                detail=f"Tenant {key!r} resident adapter capacity reached: {resident_count}/{max_runs_per_tenant}",
            )

    def enforce_rl_job_quota(tenant_id: Optional[str]) -> None:
        active_statuses = {"queued", "running", "canceling"}
        key = tenant_key(tenant_id)
        with rl_jobs_lock:
            active_jobs = [job for job in rl_jobs.values() if job.status in active_statuses]
            tenant_active_count = sum(1 for job in active_jobs if tenant_key(job.tenant_id) == key)
        if max_concurrent_rl_jobs is not None and len(active_jobs) >= max_concurrent_rl_jobs:
            raise HTTPException(
                status_code=429,
                detail=f"Global RL job capacity reached: {len(active_jobs)}/{max_concurrent_rl_jobs}",
            )
        if max_concurrent_rl_jobs_per_tenant is not None and tenant_active_count >= max_concurrent_rl_jobs_per_tenant:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Tenant {key!r} RL job capacity reached: {tenant_active_count}/{max_concurrent_rl_jobs_per_tenant}"
                ),
            )

    def get_record(run_id: str) -> RunRecord:
        with records_lock:
            record = records.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return record

    def get_authorized_record(run_id: str, http_request: Request) -> RunRecord:
        record = get_record(run_id)
        authorize_tenant(record.tenant_id, http_request)
        return record

    def resolve_openai_model_run(model: Optional[str], http_request: Request) -> tuple[str, RunRecord]:
        with records_lock:
            visible_records = [
                record for record in records.values() if visible_to_request(record.tenant_id, http_request)
            ]
        if model is None:
            ready_records = [record for record in visible_records if record.status == "ready"]
            if len(ready_records) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="OpenAI-compatible requests must set model to a Tinker run_id, adapter_id, or run name.",
                )
            record = ready_records[0]
            return record.run_id, record
        for record in visible_records:
            if model in {record.run_id, record.adapter_id, record.name}:
                authorize_tenant(record.tenant_id, http_request)
                return record.run_id, record
        raise HTTPException(status_code=404, detail=f"Unknown Tinker model/run for OpenAI-compatible request: {model}")

    def mark_run(run_id: str, *, status: str, error: Optional[str] = None) -> RunRecord:
        with records_lock:
            record = get_record(run_id)
            record.status = status
            record.last_error = error
            record.sequence += 1
            record.updated_at = _utc_now()
            run_store.save(records)
            return record

    def mark_run_failed(run_id: str, exc: Exception) -> None:
        mark_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    def tenant_for_runs(run_ids: list[str], requested_tenant_id: Optional[str]) -> Optional[str]:
        run_tenants = {get_record(run_id).tenant_id for run_id in run_ids}
        if len(run_tenants) > 1:
            raise HTTPException(status_code=400, detail="All runs in one job must belong to the same tenant")
        run_tenant_id = next(iter(run_tenants), None)
        if requested_tenant_id is not None and requested_tenant_id != run_tenant_id:
            raise HTTPException(status_code=403, detail="Request tenant_id does not match run tenant_id")
        return requested_tenant_id or run_tenant_id

    def create_job(kind: str, run_ids: list[str], tenant_id: Optional[str]) -> JobRecord:
        now = _utc_now()
        job = JobRecord(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            kind=kind,
            status="queued",
            tenant_id=tenant_id,
            run_ids=run_ids,
            created_at=now,
            updated_at=now,
        )
        with jobs_lock:
            jobs[job.job_id] = job
            job_store.save(jobs)
        return job

    def get_job_record(job_id: str) -> JobRecord:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
            return job

    def mark_job(
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[dict[str, Any]] = None,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> JobRecord:
        with jobs_lock:
            job = get_job_record(job_id)
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.sequence += 1
            job.updated_at = _utc_now()
            job_store.save(jobs)
            return job

    def compact_job_summary(job: JobRecord) -> JobSummary:
        progress = dict(job.progress)
        progress.pop("request", None)
        if isinstance(progress.get("request_ref"), dict):
            request_ref = dict(progress["request_ref"])
            request_ref.pop("sha256", None)
            progress["request_ref"] = request_ref
        return JobSummary(
            job_id=job.job_id,
            kind=job.kind,
            status=job.status,
            tenant_id=job.tenant_id,
            sequence=job.sequence,
            run_ids=list(job.run_ids),
            progress=progress,
            has_result=job.result is not None,
            error=job.error,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    def compact_job_record(job: JobRecord) -> JobRecord:
        progress = dict(job.progress)
        progress.pop("request", None)
        if isinstance(progress.get("request_ref"), dict):
            request_ref = dict(progress["request_ref"])
            request_ref.pop("sha256", None)
            progress["request_ref"] = request_ref
        return JobRecord(
            job_id=job.job_id,
            kind=job.kind,
            status=job.status,
            tenant_id=job.tenant_id,
            sequence=job.sequence,
            run_ids=list(job.run_ids),
            progress=progress,
            result=job.result,
            error=job.error,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    def fail_job(job_id: str, exc: Exception) -> None:
        mark_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    def create_rl_job(request: RLJobRequest, command: list[str], repo_dir: pathlib.Path) -> RLJobRecord:
        now = _utc_now()
        job_id = f"rljob_{uuid.uuid4().hex[:12]}"
        log_dir = pathlib.Path(scratch_dir) / "tinker_api" / "rl_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        job = RLJobRecord(
            job_id=job_id,
            name=request.name,
            status="dry_run" if request.dry_run else "queued",
            tenant_id=request.tenant_id,
            launcher=request.launcher,
            repo_dir=str(repo_dir),
            config_path=request.config_path,
            entrypoint=request.entrypoint,
            command=command,
            log_path=str(log_dir / f"{job_id}.log"),
            max_runtime_seconds=request.max_runtime_seconds,
            created_at=now,
            updated_at=now,
        )
        with rl_jobs_lock:
            rl_jobs[job.job_id] = job
            rl_job_store.save(rl_jobs)
        return job

    def refresh_rl_jobs_from_store() -> None:
        """Refresh RL job metadata written by an external host launcher."""
        stored_jobs = rl_job_store.load()
        if stored_jobs:
            rl_jobs.update(stored_jobs)

    def enqueue_host_rl_job(job: RLJobRecord, request: RLJobRequest) -> None:
        queue_dir = pathlib.Path(scratch_dir) / "tinker_api" / "host_rl_queue"
        queue_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": job.job_id,
            "command": job.command,
            "repo_dir": job.repo_dir,
            "host_cwd": request.docker_repo_dir or job.repo_dir,
            "log_path": job.log_path,
            "max_runtime_seconds": job.max_runtime_seconds,
            "metadata_backend": metadata_backend,
            "metadata_path": str(pathlib.Path(scratch_dir) / "tinker_api" / "metadata.sqlite3"),
        }
        tmp_path = queue_dir / f"{job.job_id}.json.tmp"
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
        tmp_path.replace(queue_dir / f"{job.job_id}.json")

    def get_rl_job_record(job_id: str) -> RLJobRecord:
        with rl_jobs_lock:
            refresh_rl_jobs_from_store()
            job = rl_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Unknown rl_job_id: {job_id}")
            return job

    def mark_rl_job(
        job_id: str,
        *,
        status: Optional[str] = None,
        pid: Optional[int] = None,
        returncode: Optional[int] = None,
        error: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> RLJobRecord:
        with rl_jobs_lock:
            job = get_rl_job_record(job_id)
            if status is not None:
                job.status = status
            if pid is not None:
                job.pid = pid
            if returncode is not None:
                job.returncode = returncode
            if error is not None:
                job.error = error
            if log_path is not None:
                job.log_path = log_path
            job.updated_at = _utc_now()
            rl_job_store.save(rl_jobs)
            return job

    def terminate_rl_process(process: subprocess.Popen, sig: signal.Signals = signal.SIGTERM) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def run_rl_job(job_id: str) -> None:
        job = get_rl_job_record(job_id)
        try:
            mark_rl_job(job_id, status="running")
            with pathlib.Path(job.log_path).open("ab") as log_fp:
                log_fp.write(("Command: " + " ".join(job.command) + "\n\n").encode("utf-8"))
                process = subprocess.Popen(
                    job.command,
                    cwd=job.repo_dir,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                with rl_jobs_lock:
                    active_rl_processes[job_id] = process
                mark_rl_job(job_id, pid=process.pid)
                try:
                    returncode = process.wait(timeout=job.max_runtime_seconds)
                except subprocess.TimeoutExpired:
                    terminate_rl_process(process)
                    try:
                        returncode = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        terminate_rl_process(process, signal.SIGKILL)
                        returncode = process.wait()
                    with rl_jobs_lock:
                        active_rl_processes.pop(job_id, None)
                    mark_rl_job(
                        job_id,
                        status="timed_out",
                        returncode=returncode,
                        error=f"RL job exceeded max_runtime_seconds={job.max_runtime_seconds}",
                    )
                    return
            with rl_jobs_lock:
                active_rl_processes.pop(job_id, None)
                latest_status = get_rl_job_record(job_id).status
            if latest_status in {"canceling", "canceled"}:
                mark_rl_job(job_id, status="canceled", returncode=returncode)
            else:
                mark_rl_job(job_id, status="succeeded" if returncode == 0 else "failed", returncode=returncode)
        except Exception as exc:
            with rl_jobs_lock:
                active_rl_processes.pop(job_id, None)
            mark_rl_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    def cancel_rl_job_record(job_id: str, http_request: Request) -> RLJobRecord:
        job = get_rl_job_record(job_id)
        authorize_tenant(job.tenant_id, http_request)
        if job.status == "queued":
            return mark_rl_job(job_id, status="canceled")
        if job.status != "running":
            return job
        job = mark_rl_job(job_id, status="canceling")
        with rl_jobs_lock:
            process = active_rl_processes.get(job_id)
        if process is None:
            return job
        try:
            terminate_rl_process(process)
        except PermissionError as exc:
            return mark_rl_job(job_id, error=f"{type(exc).__name__}: {exc}")
        return get_rl_job_record(job_id)

    def get_idempotent_response(operation: str, key: Optional[str], request: BaseModel) -> Optional[dict[str, Any]]:
        if key is None:
            return None
        fingerprint = _fingerprint_request(operation, request)
        with idempotency_lock:
            record = idempotency_records.get(key)
            if record is None:
                now = _utc_now()
                idempotency_records[key] = IdempotencyRecord(
                    key=key,
                    operation=operation,
                    fingerprint=fingerprint,
                    status="running",
                    created_at=now,
                    updated_at=now,
                )
                idempotency_store.save(idempotency_records)
                return None
            if record.operation != operation or record.fingerprint != fingerprint:
                raise HTTPException(
                    status_code=409,
                    detail=f"Idempotency key {key!r} was already used for a different request",
                )
            if record.status == "succeeded" and record.response is not None:
                return record.response
            if record.status == "failed":
                raise HTTPException(
                    status_code=409,
                    detail=f"Previous request for idempotency key {key!r} failed: {record.error}",
                )
            raise HTTPException(status_code=409, detail=f"Request for idempotency key {key!r} is still running")

    def store_idempotent_response(operation: str, key: Optional[str], request: BaseModel, response: Any) -> None:
        if key is None:
            return
        fingerprint = _fingerprint_request(operation, request)
        with idempotency_lock:
            idempotency_records[key] = IdempotencyRecord(
                key=key,
                operation=operation,
                fingerprint=fingerprint,
                status="succeeded",
                response=_model_to_dict(response) if isinstance(response, BaseModel) else response,
                created_at=idempotency_records[key].created_at,
                updated_at=_utc_now(),
            )
            idempotency_store.save(idempotency_records)

    def store_idempotent_error(operation: str, key: Optional[str], request: BaseModel, exc: Exception) -> None:
        if key is None:
            return
        fingerprint = _fingerprint_request(operation, request)
        with idempotency_lock:
            created_at = idempotency_records[key].created_at
            idempotency_records[key] = IdempotencyRecord(
                key=key,
                operation=operation,
                fingerprint=fingerprint,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                created_at=created_at,
                updated_at=_utc_now(),
            )
            idempotency_store.save(idempotency_records)

    def run_train_steps(request: TrainStepsRequest, job: JobRecord) -> TrainStepsResponse:
        run_ids = list(request.batches)
        if request.steps < 0:
            raise ValueError("steps must be non-negative")
        if request.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if request.microbatch_size is not None and request.microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive when provided")
        if get_job_record(job.job_id).status == "canceled":
            return TrainStepsResponse(job=job, runs={}, outputs={})

        clients_by_run = {run_id: get_run(run_id) for run_id in run_ids}
        batches_by_run = {
            run_id: [_datum_from_request(datum) for datum in batch] * request.batch_size
            for run_id, batch in request.batches.items()
        }
        adam_params = AdamParams(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            betas=request.betas,
            eps=request.eps,
        )

        def merge_microbatch_outputs(accumulator: dict[str, Any], micro_outputs: dict[str, Any]) -> dict[str, Any]:
            for run_id, client in clients_by_run.items():
                output = micro_outputs.get(client.adapter_id)
                if output is None:
                    continue
                current = accumulator.setdefault(
                    run_id,
                    {
                        "loss": 0.0,
                        "metrics": {
                            "loss": 0.0,
                            "loss:sum": 0.0,
                            "loss:mean": 0.0,
                            "num_label_tokens": 0.0,
                            "loss_fn": request.loss_fn,
                        },
                        "weighted_metric_sums": {},
                        "loss_fn_outputs": [],
                    },
                )
                metrics = dict(output.metrics)
                loss_sum = float(metrics.get("loss:sum", output.loss))
                num_label_tokens = float(metrics.get("num_label_tokens", 0.0))
                current["loss"] += loss_sum
                current["metrics"]["loss"] += loss_sum
                current["metrics"]["loss:sum"] += loss_sum
                current["metrics"]["num_label_tokens"] += num_label_tokens
                current["loss_fn_outputs"].extend(output.loss_fn_outputs)
                for key, value in metrics.items():
                    if key in {"loss", "loss:sum", "loss:mean", "num_label_tokens", "loss_fn"}:
                        continue
                    if isinstance(value, (int, float)):
                        if key in {"clip_low_threshold", "clip_high_threshold", "beta"}:
                            current["metrics"][key] = float(value)
                        elif key == "loss_weight_mean" or key.endswith("_mean") or key.endswith(":mean"):
                            current["weighted_metric_sums"][key] = current["weighted_metric_sums"].get(key, 0.0) + (
                                float(value) * num_label_tokens
                            )
                        else:
                            current["metrics"][key] = current["metrics"].get(key, 0.0) + float(value)
                    else:
                        current["metrics"][key] = value
            return accumulator

        def finalize_microbatch_outputs(accumulator: dict[str, Any]) -> dict[str, Any]:
            for current in accumulator.values():
                denom = max(float(current["metrics"]["num_label_tokens"]), 1.0)
                current["metrics"]["loss:mean"] = float(current["metrics"]["loss:sum"]) / denom
                for key, weighted_sum in current.pop("weighted_metric_sums", {}).items():
                    current["metrics"][key] = float(weighted_sum) / denom
            return accumulator

        outputs: dict[str, Any] = {}
        existing_progress = dict(get_job_record(job.job_id).progress)
        start_step = int(existing_progress.get("step", 0))
        first_losses = existing_progress.get("first_losses")
        last_losses = existing_progress.get("last_losses")
        progress_request_ref = existing_progress.get("request_ref")
        if progress_request_ref is None:
            progress_request_ref = save_train_request_manifest(job.job_id, request)
        mark_job(
            job.job_id,
            status="running",
            progress={
                "step": start_step,
                "total_steps": request.steps,
                "request_ref": progress_request_ref,
                "first_losses": first_losses,
                "last_losses": last_losses,
            },
        )
        try:
            for step in range(start_step, request.steps):
                if get_job_record(job.job_id).status == "canceling":
                    mark_job(job.job_id, status="canceled", progress={"step": step, "total_steps": request.steps})
                    raise RuntimeError("Job canceled")
                adapter_batches = {clients_by_run[run_id].adapter_id: batch for run_id, batch in batches_by_run.items()}
                for run_id in run_ids:
                    mark_run(run_id, status="running")
                microbatch_size = request.microbatch_size or max(len(batch) for batch in adapter_batches.values())
                microbatch_count = max(
                    (len(batch) + microbatch_size - 1) // microbatch_size for batch in adapter_batches.values()
                )
                mixed_outputs = {}
                for microbatch_idx in range(microbatch_count):
                    micro_batches = {
                        adapter_id: batch[microbatch_idx * microbatch_size : (microbatch_idx + 1) * microbatch_size]
                        for adapter_id, batch in adapter_batches.items()
                    }
                    micro_batches = {adapter_id: batch for adapter_id, batch in micro_batches.items() if batch}
                    micro_outputs = service.forward_backward_mixed(
                        micro_batches,
                        request.loss_fn,
                        request.loss_fn_config,
                        zero_grad=microbatch_idx == 0,
                    ).result()
                    mixed_outputs = merge_microbatch_outputs(mixed_outputs, micro_outputs)
                mixed_outputs = finalize_microbatch_outputs(mixed_outputs)
                record_worker_operation(
                    "train_steps.forward_backward",
                    run_ids,
                    {
                        "job_id": job.job_id,
                        "step": step + 1,
                        "loss_fn": request.loss_fn,
                        "microbatch_size": microbatch_size,
                        "microbatch_count": microbatch_count,
                    },
                )
                losses = {}
                for run_id in run_ids:
                    output = mixed_outputs[run_id]
                    record = mark_run(run_id, status="ready")
                    record.forward_backward_calls += 1
                    record.last_loss = output["loss"]
                    record.last_metrics = dict(output["metrics"])
                    losses[run_id] = output["loss"]
                    outputs[run_id] = output
                first_losses = first_losses or losses
                last_losses = losses
                for run_id, client in clients_by_run.items():
                    mark_run(run_id, status="optimizing")
                    step_output = client.optim_step(adam_params).result()
                    record_worker_operation(
                        "train_steps.optim_step",
                        [run_id],
                        {"job_id": job.job_id, "step": step + 1, "learning_rate": request.learning_rate},
                    )
                    record = mark_run(run_id, status="ready")
                    record.optimizer_steps = step_output.step
                    outputs[f"{run_id}:optim_step"] = asdict(step_output)
                mark_job(
                    job.job_id,
                    progress={
                        "step": step + 1,
                        "total_steps": request.steps,
                        "request_ref": progress_request_ref,
                        "first_losses": first_losses,
                        "last_losses": losses,
                    },
                )

            saved_paths = {}
            for run_id, save_name in request.save_names.items():
                client = clients_by_run[run_id]
                mark_run(run_id, status="saving")
                save_output = client.save_state(save_name).result()
                record_worker_operation("train_steps.save", [run_id], {"job_id": job.job_id, "name": save_name})
                record = mark_run(run_id, status="ready")
                record.last_checkpoint_path = save_output.path
                saved_paths[run_id] = save_output.path
                outputs[f"{run_id}:save"] = asdict(save_output)

            result = {
                "first_losses": first_losses,
                "last_losses": last_losses,
                "saved_paths": saved_paths,
            }
            job = mark_job(job.job_id, status="succeeded", result=result)
            with records_lock:
                run_store.save(records)
                run_records = {run_id: records[run_id] for run_id in run_ids}
            return TrainStepsResponse(job=job, runs=run_records, outputs=outputs)
        except Exception as exc:
            if get_job_record(job.job_id).status != "canceled":
                fail_job(job.job_id, exc)
            for run_id in run_ids:
                if run_id in records:
                    mark_run_failed(run_id, exc)
            raise

    def resume_interrupted_train_jobs() -> None:
        if not resume_interrupted_jobs_on_startup:
            return
        for job in list(jobs.values()):
            if job.kind != "train_steps" or job.status != "queued":
                continue
            if not isinstance(job.progress, dict):
                continue
            try:
                request = load_train_request_from_progress(job.progress)
            except Exception as exc:
                mark_job(
                    job.job_id,
                    status="failed",
                    error=f"Cannot resume job; train request manifest is invalid: {type(exc).__name__}: {exc}",
                )
                continue
            if request is None:
                continue
            missing_runs = [run_id for run_id in job.run_ids if run_id not in runs]
            if missing_runs:
                mark_job(
                    job.job_id,
                    status="failed",
                    error=f"Cannot resume job; runs are not resident after restart: {missing_runs}",
                )
                continue
            executor.submit(
                lambda request=request, job=job: service_metrics.observe(
                    "train_steps.resume",
                    lambda: run_train_steps(request, job),
                )
            )

    resume_interrupted_train_jobs()

    @app.post("/runs", response_model=CreateRunResponse)
    def create_run(request: CreateRunRequest, http_request: Request) -> CreateRunResponse:
        request = _model_with_update(request, tenant_id=resolve_tenant_id(request.tenant_id, http_request))
        existing = get_idempotent_response("create_run", request.idempotency_key, request)
        if existing is not None:
            return existing
        enforce_tenant_rate_limit(request.tenant_id, "create_run")

        def op() -> CreateRunResponse:
            try:
                if max_resident_adapters is not None and len(runs) >= max_resident_adapters:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Resident adapter capacity reached: {len(runs)}/{max_resident_adapters}",
                    )
                enforce_tenant_run_quota(request.tenant_id)
                run_id = f"run_{uuid.uuid4().hex[:12]}"
                assigned_worker = worker_manager.assign(run_id) if worker_manager is not None else None
                client = service.create_lora_training_client(
                    adapter_id=request.adapter_id,
                    checkpoint_path=request.checkpoint_path,
                )
                runs[run_id] = client
                now = _utc_now()
                record = RunRecord(
                    run_id=run_id,
                    adapter_id=client.adapter_id,
                    name=request.name,
                    tenant_id=request.tenant_id,
                    status="ready" if request.checkpoint_path else "created",
                    optimizer_steps=_client_step(client),
                    last_checkpoint_path=request.checkpoint_path,
                    restored_from=request.checkpoint_path,
                    worker_id=assigned_worker.worker_id if assigned_worker is not None else None,
                    created_at=now,
                    updated_at=now,
                )
                with records_lock:
                    records[run_id] = record
                    run_store.save(records)
                attach_run_to_worker(record)
                record_worker_operation(
                    "create_run",
                    [run_id],
                    {
                        "adapter_id": record.adapter_id,
                        "checkpoint_path": request.checkpoint_path,
                    },
                )
                response = CreateRunResponse(
                    run_id=run_id,
                    adapter_id=client.adapter_id,
                    name=request.name,
                    status=record.status,
                    sequence=record.sequence,
                    worker_id=record.worker_id,
                )
                store_idempotent_response("create_run", request.idempotency_key, request, response)
                return response
            except Exception as exc:
                store_idempotent_error("create_run", request.idempotency_key, request, exc)
                raise

        return executor.submit(lambda: service_metrics.observe("create_run", op)).result()

    @app.get("/runs", response_model=list[RunRecord])
    def list_runs(http_request: Request) -> list[RunRecord]:
        with records_lock:
            return [record for record in records.values() if visible_to_request(record.tenant_id, http_request)]

    @app.get("/runs/{run_id}", response_model=RunRecord)
    def get_run_record(run_id: str, http_request: Request) -> RunRecord:
        return get_authorized_record(run_id, http_request)

    @app.post("/v1/responses")
    def openai_responses(body: dict[str, Any], http_request: Request) -> dict[str, Any]:
        model_ref = body.get("model")
        run_id, record = resolve_openai_model_run(str(model_ref) if model_ref is not None else None, http_request)
        prompt = _openai_messages_to_prompt(body.get("input", ""))
        max_new_tokens = int(body.get("max_output_tokens") or body.get("max_tokens") or 256)
        temperature = _optional_float(body.get("temperature"), 0.7)
        return_logprobs = _metadata_flag(body, "tinker_return_logprobs")
        output = service.sample(
            get_run(run_id).adapter_id,
            prompt,
            SamplingParams(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=_optional_float(body.get("top_p"), 0.95),
                do_sample=temperature > 0.0,
                return_logprobs=return_logprobs,
            ),
        ).result()
        text = output.text[len(prompt) :] if output.text.startswith(prompt) else output.text
        response = {
            "id": f"resp_{uuid.uuid4().hex}",
            "created_at": time.time(),
            "error": None,
            "incomplete_details": None,
            "model": record.name or record.run_id,
            "object": "response",
            "output": [
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "content": [{"annotations": [], "text": text, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "status": "completed",
            "tool_choice": body.get("tool_choice", "auto"),
            "tools": body.get("tools", []),
        }
        if output.generated_logprobs is not None:
            response["tinker_rl"] = {
                "tokens": output.tokens,
                "prompt_token_count": output.prompt_token_count,
                "generated_logprobs": output.generated_logprobs,
            }
        return response

    @app.post("/v1/chat/completions")
    def openai_chat_completions(body: dict[str, Any], http_request: Request) -> dict[str, Any]:
        model_ref = body.get("model")
        run_id, record = resolve_openai_model_run(str(model_ref) if model_ref is not None else None, http_request)
        prompt = _openai_messages_to_prompt(body.get("messages", []))
        max_new_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 256)
        temperature = _optional_float(body.get("temperature"), 0.7)
        return_logprobs = _metadata_flag(body, "tinker_return_logprobs")
        output = service.sample(
            get_run(run_id).adapter_id,
            prompt,
            SamplingParams(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=_optional_float(body.get("top_p"), 0.95),
                do_sample=temperature > 0.0,
                return_logprobs=return_logprobs,
            ),
        ).result()
        text = output.text[len(prompt) :] if output.text.startswith(prompt) else output.text
        response = {
            "id": f"chatcmpl_{uuid.uuid4().hex}",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": text,
                        "role": "assistant",
                    },
                }
            ],
            "created": int(time.time()),
            "model": record.name or record.run_id,
            "object": "chat.completion",
        }
        if output.generated_logprobs is not None:
            response["tinker_rl"] = {
                "tokens": output.tokens,
                "prompt_token_count": output.prompt_token_count,
                "generated_logprobs": output.generated_logprobs,
            }
        return response

    @app.get("/jobs", response_model=list[JobSummary])
    def list_jobs(http_request: Request) -> list[JobSummary]:
        with jobs_lock:
            return [
                compact_job_summary(job) for job in jobs.values() if visible_to_request(job.tenant_id, http_request)
            ]

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str, http_request: Request) -> JobRecord:
        job = get_job_record(job_id)
        authorize_tenant(job.tenant_id, http_request)
        return compact_job_record(job)

    @app.post("/jobs/{job_id}/cancel", response_model=JobRecord)
    def cancel_job(job_id: str, http_request: Request) -> JobRecord:
        job = get_job_record(job_id)
        authorize_tenant(job.tenant_id, http_request)
        if job.status == "queued":
            return mark_job(job_id, status="canceled")
        if job.status == "running":
            return mark_job(job_id, status="canceling")
        return job

    @app.get("/rl/jobs", response_model=list[RLJobRecord])
    def list_rl_jobs(http_request: Request) -> list[RLJobRecord]:
        with rl_jobs_lock:
            refresh_rl_jobs_from_store()
            return [job for job in rl_jobs.values() if visible_to_request(job.tenant_id, http_request)]

    @app.get("/rl/jobs/{job_id}", response_model=RLJobRecord)
    def get_rl_job(job_id: str, http_request: Request) -> RLJobRecord:
        job = get_rl_job_record(job_id)
        authorize_tenant(job.tenant_id, http_request)
        return job

    @app.get("/rl/jobs/{job_id}/logs")
    def get_rl_job_logs(job_id: str, http_request: Request, tail_bytes: int = 20000) -> dict[str, Any]:
        job = get_rl_job_record(job_id)
        authorize_tenant(job.tenant_id, http_request)
        if job.log_path is None:
            return {"job_id": job_id, "text": ""}
        log_path = pathlib.Path(job.log_path)
        if not log_path.exists():
            return {"job_id": job_id, "text": ""}
        tail_bytes = max(1, min(tail_bytes, 200000))
        with log_path.open("rb") as fp:
            fp.seek(0, os.SEEK_END)
            size = fp.tell()
            fp.seek(max(0, size - tail_bytes))
            text = fp.read().decode("utf-8", errors="replace")
        return {"job_id": job_id, "text": text}

    @app.post("/rl/jobs/{job_id}/cancel", response_model=RLJobRecord)
    def cancel_rl_job(job_id: str, http_request: Request) -> RLJobRecord:
        return cancel_rl_job_record(job_id, http_request)

    @app.post("/internal/rl/jobs/{job_id}/mark", response_model=RLJobRecord)
    def mark_rl_job_from_host_worker(job_id: str, request: RLJobMarkRequest, http_request: Request) -> RLJobRecord:
        expected_token = os.environ.get("NEMOTRON_TINKER_HOST_WORKER_TOKEN")
        if expected_token:
            provided_token = http_request.headers.get("X-Nemotron-Tinker-Worker-Token")
            if provided_token != expected_token:
                raise HTTPException(status_code=403, detail="Invalid host worker token")
        job = get_rl_job_record(job_id)
        authorize_tenant(job.tenant_id, http_request)
        return mark_rl_job(
            job_id,
            status=request.status,
            pid=request.pid,
            returncode=request.returncode,
            error=request.error,
            log_path=request.log_path,
        )

    @app.post("/rl/jobs", response_model=RLJobSubmitResponse)
    def submit_rl_job(request: RLJobRequest, http_request: Request) -> RLJobSubmitResponse:
        request = _model_with_update(request, tenant_id=resolve_tenant_id(request.tenant_id, http_request))
        existing = get_idempotent_response("rl_jobs", request.idempotency_key, request)
        if existing is not None:
            return existing
        if not request.dry_run:
            enforce_tenant_rate_limit(request.tenant_id, "rl_jobs")
            enforce_rl_job_quota(request.tenant_id)
        repo_dir_text = request.repo_dir or rl_repo_dir or os.environ.get("NEMO_RL_REPO_DIR")
        if not repo_dir_text:
            raise HTTPException(
                status_code=400,
                detail="Set repo_dir, create_app(..., rl_repo_dir=...), or NEMO_RL_REPO_DIR to a NeMo-RL checkout.",
            )
        repo_dir = pathlib.Path(repo_dir_text).expanduser().resolve()
        if not repo_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"NeMo-RL repo_dir does not exist: {repo_dir}")
        try:
            _resolve_rl_path(repo_dir, request.entrypoint, "entrypoint")
            _resolve_rl_path(repo_dir, request.config_path, "config_path")
            _check_rl_launcher_available(request)
            command = _build_rl_command(request, repo_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        job = create_rl_job(request, command, repo_dir)
        response = RLJobSubmitResponse(job=job)
        if request.dry_run:
            store_idempotent_response("rl_jobs", request.idempotency_key, request, response)
            return response
        if request.launcher == "host":
            enqueue_host_rl_job(job, request)
            store_idempotent_response("rl_jobs", request.idempotency_key, request, response)
            return response
        if request.run_async:
            thread = threading.Thread(target=run_rl_job, args=(job.job_id,), daemon=True)
            thread.start()
            store_idempotent_response("rl_jobs", request.idempotency_key, request, response)
            return response
        run_rl_job(job.job_id)
        response = RLJobSubmitResponse(job=get_rl_job_record(job.job_id))
        store_idempotent_response("rl_jobs", request.idempotency_key, request, response)
        return response

    @app.post("/train_steps")
    def train_steps(request: TrainStepsRequest, http_request: Request) -> TrainStepsResponse | JobSubmitResponse:
        request = _model_with_update(request, tenant_id=resolve_tenant_id(request.tenant_id, http_request))
        for run_id in request.batches:
            authorize_tenant(get_record(run_id).tenant_id, http_request)
        existing = get_idempotent_response("train_steps", request.idempotency_key, request)
        if existing is not None:
            return existing
        tenant_id = tenant_for_runs(list(request.batches), request.tenant_id)
        enforce_tenant_rate_limit(tenant_id, "train_steps")
        job = create_job("train_steps", list(request.batches), tenant_id)
        request_ref = save_train_request_manifest(job.job_id, request)
        mark_job(
            job.job_id,
            progress={
                "step": 0,
                "total_steps": request.steps,
                "request_ref": request_ref,
            },
        )
        if request.run_async:
            future = executor.submit(
                lambda: service_metrics.observe("train_steps", lambda: run_train_steps(request, job))
            )

            def complete_job(done_future: Future) -> None:
                try:
                    done_future.result()
                except Exception:
                    pass

            future.add_done_callback(complete_job)
            response = JobSubmitResponse(job=job)
            store_idempotent_response("train_steps", request.idempotency_key, request, response)
            return response
        try:
            response = executor.submit(
                lambda: service_metrics.observe("train_steps", lambda: run_train_steps(request, job))
            ).result()
            store_idempotent_response("train_steps", request.idempotency_key, request, response)
            return response
        except Exception as exc:
            store_idempotent_error("train_steps", request.idempotency_key, request, exc)
            raise

    @app.post("/runs/{run_id}/forward_backward")
    def forward_backward(
        run_id: str, request: ForwardBackwardRequest, http_request: Request
    ) -> ForwardBackwardResponse:
        record = get_authorized_record(run_id, http_request)
        enforce_tenant_rate_limit(record.tenant_id, "forward_backward")

        def op() -> ForwardBackwardResponse:
            client = get_run(run_id)
            try:
                mark_run(run_id, status="running")
                data = [_datum_from_request(datum) for datum in request.data]
                output = service.forward_backward_mixed(
                    {client.adapter_id: data},
                    request.loss_fn,
                    request.loss_fn_config,
                ).result()[client.adapter_id]
                record_worker_operation(
                    "forward_backward",
                    [run_id],
                    {"loss_fn": request.loss_fn, "batch_size": len(request.data)},
                )
                record = mark_run(run_id, status="ready")
                record.forward_backward_calls += 1
                record.last_loss = output.loss
                record.last_metrics = dict(output.metrics)
                run_store.save(records)
                return ForwardBackwardResponse(run=record, output=asdict(output))
            except Exception as exc:
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(lambda: service_metrics.observe("forward_backward", op)).result()

    @app.post("/mixed_forward_backward")
    def mixed_forward_backward(
        request: MixedForwardBackwardRequest,
        http_request: Request,
    ) -> dict[str, ForwardBackwardResponse]:
        for run_id in request.batches:
            authorize_tenant(get_record(run_id).tenant_id, http_request)
        tenant_id = tenant_for_runs(list(request.batches), None)
        enforce_tenant_rate_limit(tenant_id, "mixed_forward_backward")

        def op() -> dict[str, ForwardBackwardResponse]:
            batches_by_adapter = {}
            run_to_adapter = {}
            active_run_ids = list(request.batches)
            try:
                for run_id, batch in request.batches.items():
                    mark_run(run_id, status="running")
                    client = get_run(run_id)
                    batches_by_adapter[client.adapter_id] = [_datum_from_request(datum) for datum in batch]
                    run_to_adapter[run_id] = client.adapter_id
                outputs = service.forward_backward_mixed(
                    batches_by_adapter,
                    request.loss_fn,
                    request.loss_fn_config,
                ).result()
                record_worker_operation(
                    "mixed_forward_backward",
                    active_run_ids,
                    {"loss_fn": request.loss_fn, "num_runs": len(active_run_ids)},
                )
                responses = {}
                for run_id, adapter_id in run_to_adapter.items():
                    output = outputs[adapter_id]
                    record = mark_run(run_id, status="ready")
                    record.forward_backward_calls += 1
                    record.last_loss = output.loss
                    record.last_metrics = dict(output.metrics)
                    run_store.save(records)
                    responses[run_id] = ForwardBackwardResponse(run=record, output=asdict(output))
                return responses
            except Exception as exc:
                for run_id in active_run_ids:
                    if run_id in records:
                        mark_run_failed(run_id, exc)
                raise

        return executor.submit(lambda: service_metrics.observe("mixed_forward_backward", op)).result()

    @app.post("/runs/{run_id}/optim_step")
    def optim_step(run_id: str, request: OptimStepRequest, http_request: Request) -> OptimStepResponseModel:
        existing = get_idempotent_response(f"optim_step:{run_id}", request.idempotency_key, request)
        if existing is not None:
            return existing
        record = get_authorized_record(run_id, http_request)
        enforce_tenant_rate_limit(record.tenant_id, "optim_step")

        def op() -> OptimStepResponseModel:
            client = get_run(run_id)
            try:
                mark_run(run_id, status="optimizing")
                output = client.optim_step(_adam_from_request(request)).result()
                record_worker_operation("optim_step", [run_id], {"learning_rate": request.learning_rate})
                record = mark_run(run_id, status="ready")
                record.optimizer_steps = output.step
                run_store.save(records)
                response = OptimStepResponseModel(run=record, output=asdict(output))
                store_idempotent_response(f"optim_step:{run_id}", request.idempotency_key, request, response)
                return response
            except Exception as exc:
                store_idempotent_error(f"optim_step:{run_id}", request.idempotency_key, request, exc)
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(lambda: service_metrics.observe("optim_step", op)).result()

    @app.post("/runs/{run_id}/save")
    def save(run_id: str, request: SaveRequest, http_request: Request) -> SaveResponse:
        existing = get_idempotent_response(f"save:{run_id}", request.idempotency_key, request)
        if existing is not None:
            return existing
        record = get_authorized_record(run_id, http_request)
        enforce_tenant_rate_limit(record.tenant_id, "save")

        def op() -> SaveResponse:
            client = get_run(run_id)
            try:
                mark_run(run_id, status="saving")
                output = client.save_state(request.name).result()
                record_worker_operation("save", [run_id], {"name": request.name})
                record = mark_run(run_id, status="ready")
                record.last_checkpoint_path = output.path
                run_store.save(records)
                response = SaveResponse(run=record, output=asdict(output))
                store_idempotent_response(f"save:{run_id}", request.idempotency_key, request, response)
                return response
            except Exception as exc:
                store_idempotent_error(f"save:{run_id}", request.idempotency_key, request, exc)
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(lambda: service_metrics.observe("save", op)).result()

    @app.post("/runs/{run_id}/export")
    def export_run(run_id: str, request: ExportRequest, http_request: Request) -> FileResponse:
        """Save one LoRA adapter and return a zip archive for browser download."""
        record = get_authorized_record(run_id, http_request)
        enforce_tenant_rate_limit(record.tenant_id, "export")
        export_name = request.name or f"{record.name or run_id}-lora"

        def safe_name(value: str) -> str:
            return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value).strip("-")

        def op() -> tuple[pathlib.Path, str]:
            client = get_run(run_id)
            try:
                mark_run(run_id, status="saving")
                output = client.save_state(export_name).result()
                record_worker_operation("export", [run_id], {"name": export_name})
                record = mark_run(run_id, status="ready")
                record.last_checkpoint_path = output.path
                run_store.save(records)

                checkpoint_dir = pathlib.Path(output.path)
                archive_dir = pathlib.Path(scratch_dir) / "tinker_api" / "exports"
                archive_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{safe_name(export_name) or run_id}.zip"
                archive_path = archive_dir / filename
                with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for path in sorted(checkpoint_dir.rglob("*")):
                        if path.is_file():
                            archive.write(path, arcname=path.relative_to(checkpoint_dir))
                return archive_path, filename
            except Exception as exc:
                mark_run_failed(run_id, exc)
                raise

        archive_path, filename = executor.submit(lambda: service_metrics.observe("export", op)).result()
        return FileResponse(archive_path, media_type="application/zip", filename=filename)

    @app.post("/runs/{run_id}/detach")
    def detach_run(run_id: str, request: DetachRunRequest, http_request: Request) -> DetachRunResponse:
        existing = get_idempotent_response(f"detach:{run_id}", request.idempotency_key, request)
        if existing is not None:
            return existing
        record = get_authorized_record(run_id, http_request)
        enforce_tenant_rate_limit(record.tenant_id, "detach")

        def op() -> DetachRunResponse:
            client = get_run(run_id)
            try:
                record = mark_run(run_id, status="detaching")
                output = client.detach().result()
                record_worker_operation("detach_run", [run_id], {"adapter_id": client.adapter_id})
                detach_run_from_worker(record)
                runs.pop(run_id, None)
                record = mark_run(run_id, status="detached")
                run_store.save(records)
                response = DetachRunResponse(run=record, output=asdict(output))
                store_idempotent_response(f"detach:{run_id}", request.idempotency_key, request, response)
                return response
            except Exception as exc:
                store_idempotent_error(f"detach:{run_id}", request.idempotency_key, request, exc)
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(lambda: service_metrics.observe("detach", op)).result()

    @app.post("/runs/{run_id}/save_and_detach")
    def save_and_detach(
        run_id: str,
        request: SaveAndDetachRequest,
        http_request: Request,
    ) -> SaveAndDetachResponse:
        existing = get_idempotent_response(f"save_and_detach:{run_id}", request.idempotency_key, request)
        if existing is not None:
            return existing
        record = get_authorized_record(run_id, http_request)
        enforce_tenant_rate_limit(record.tenant_id, "save_and_detach")

        def op() -> SaveAndDetachResponse:
            client = get_run(run_id)
            try:
                record = mark_run(run_id, status="saving")
                save_output = client.save_state(request.name).result()
                record.last_checkpoint_path = save_output.path
                record_worker_operation("save", [run_id], {"name": request.name})
                record = mark_run(run_id, status="detaching")
                detach_output = client.detach().result()
                record_worker_operation("detach_run", [run_id], {"adapter_id": client.adapter_id})
                detach_run_from_worker(record)
                runs.pop(run_id, None)
                record = mark_run(run_id, status="detached")
                run_store.save(records)
                response = SaveAndDetachResponse(
                    run=record,
                    save_output=asdict(save_output),
                    detach_output=asdict(detach_output),
                )
                store_idempotent_response(f"save_and_detach:{run_id}", request.idempotency_key, request, response)
                return response
            except Exception as exc:
                store_idempotent_error(f"save_and_detach:{run_id}", request.idempotency_key, request, exc)
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(lambda: service_metrics.observe("save_and_detach", op)).result()

    @app.post("/runs/{run_id}/sample")
    def sample(run_id: str, request: SampleRequest, http_request: Request) -> SampleResponseModel:
        record = get_authorized_record(run_id, http_request)
        enforce_tenant_rate_limit(record.tenant_id, "sample")

        def op() -> SampleResponseModel:
            client = get_run(run_id)
            try:
                prompt = _sample_prompt_from_request(service.tokenizer, request)
                output = service.sample(client.adapter_id, prompt, _sampling_from_request(request)).result()
                record_worker_operation("sample", [run_id], {"max_new_tokens": request.max_new_tokens})
                record = mark_run(run_id, status="ready")
                return SampleResponseModel(run=record, output=asdict(output))
            except Exception as exc:
                mark_run_failed(run_id, exc)
                raise

        return executor.submit(lambda: service_metrics.observe("sample", op)).result()

    return app
