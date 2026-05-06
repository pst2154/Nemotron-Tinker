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
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Optional

from nemotron_tinker.future import APIFuture
from nemotron_tinker.types import Datum, ModelInput, SamplingParams


@dataclass
class RunInfo:
    """Metadata for one resident LoRA run."""

    run_id: str
    adapter_id: str
    name: Optional[str] = None
    tenant_id: Optional[str] = None
    status: str = "created"
    sequence: int = 0
    optimizer_steps: int = 0
    forward_backward_calls: int = 0
    last_loss: Optional[float] = None
    last_metrics: dict[str, Any] | None = None
    last_checkpoint_path: Optional[str] = None
    last_error: Optional[str] = None
    restored_from: Optional[str] = None
    worker_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class JobInfo:
    """Metadata for one server-side job."""

    job_id: str
    kind: str
    status: str
    tenant_id: Optional[str] = None
    sequence: int = 0
    run_ids: list[str] | None = None
    progress: dict[str, Any] | None = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class CreateRunResult:
    """Result from creating one resident LoRA adapter run."""

    run_id: str
    adapter_id: str
    name: Optional[str]
    status: str
    sequence: int
    worker_id: Optional[str] = None


@dataclass
class ForwardBackwardResult:
    """Result from one forward/backward operation."""

    run: RunInfo
    loss: float
    metrics: dict[str, Any]
    loss_fn_outputs: list[dict[str, Any]]


@dataclass
class OptimStepResult:
    """Result from one optimizer step."""

    run: RunInfo
    step: int
    learning_rate: float


@dataclass
class SaveResult:
    """Result from saving one adapter."""

    run: RunInfo
    path: str


@dataclass
class DetachResult:
    """Result from detaching one adapter."""

    run: RunInfo
    adapter_id: str
    remaining_adapters: int


@dataclass
class SampleResult:
    """Generated text, tokens, and optional sampled-token logprobs."""

    run: RunInfo
    tokens: list[int]
    text: str
    prompt_token_count: int = 0
    generated_logprobs: Optional[list[float]] = None


@dataclass
class TrainStepsResult:
    """Result from a server-owned train_steps job."""

    job: JobInfo
    runs: dict[str, RunInfo]
    outputs: dict[str, Any]


def _model_input_to_json(value: Any) -> Any:
    if isinstance(value, ModelInput):
        return {"tokens": value.tokens}
    if isinstance(value, list):
        return {"tokens": value}
    return value


def _datum_to_json(datum: Datum | dict[str, Any]) -> dict[str, Any]:
    if isinstance(datum, dict):
        return datum
    loss_fn_inputs = {}
    for key, value in datum.loss_fn_inputs.items():
        loss_fn_inputs[key] = _model_input_to_json(value)
    return {
        "model_input": {"tokens": datum.model_input.tokens},
        "loss_fn_inputs": loss_fn_inputs,
    }


def _run_info(payload: dict[str, Any]) -> RunInfo:
    allowed = set(RunInfo.__dataclass_fields__)
    return RunInfo(**{key: value for key, value in payload.items() if key in allowed})


def _job_info(payload: dict[str, Any]) -> JobInfo:
    allowed = set(JobInfo.__dataclass_fields__)
    return JobInfo(**{key: value for key, value in payload.items() if key in allowed})


class TinkerAPIError(RuntimeError):
    """Raised when the Nemotron-Tinker API returns a non-2xx response."""


class NemotronTinkerClient:
    """Small Python SDK for the Nemotron-Tinker HTTP API prototype."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18080",
        *,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout_s: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.tenant_id:
            headers["X-Tinker-Tenant-Id"] = self.tenant_id
        return headers

    def _request(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=self._headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise TinkerAPIError(f"{method} {path} failed with HTTP {exc.code}: {body}") from exc

    def get(self, path: str) -> Any:
        """Issue a raw GET request relative to the API root."""
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        """Issue a raw POST request relative to the API root."""
        return self._request("POST", path, payload)

    def health(self) -> dict[str, Any]:
        """Return service health metadata."""
        return self.get("/health")

    def wait_for_server(self, timeout_s: int = 300) -> None:
        """Wait for the service health endpoint to answer."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                self.health()
                return
            except (TinkerAPIError, urllib.error.URLError, TimeoutError):
                time.sleep(1)
        raise TimeoutError(f"Timed out waiting for {self.base_url}/health")

    def list_runs(self) -> list[RunInfo]:
        """List visible resident LoRA runs."""
        return [_run_info(item) for item in self.get("/runs")]

    def get_run(self, run_id: str) -> RunInfo:
        """Return one resident LoRA run."""
        return _run_info(self.get(f"/runs/{run_id}"))

    def list_jobs(self) -> list[JobInfo]:
        """List visible train jobs."""
        return [_job_info(item) for item in self.get("/jobs")]

    def get_job(self, job_id: str) -> JobInfo:
        """Return one train job."""
        return _job_info(self.get(f"/jobs/{job_id}"))

    def wait_for_job(self, job_id: str, timeout_s: int = 3600, poll_s: float = 2.0) -> JobInfo:
        """Poll a job until it reaches a terminal state."""
        deadline = time.time() + timeout_s
        job = self.get_job(job_id)
        while time.time() < deadline and job.status in {"queued", "running", "canceling"}:
            time.sleep(poll_s)
            job = self.get_job(job_id)
        if job.status != "succeeded":
            raise RuntimeError(f"Job {job_id} ended with status={job.status!r}: {job.error}")
        return job

    def create_lora_training_client(
        self,
        *,
        name: Optional[str] = None,
        adapter_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        tenant_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> "LoRATrainingClient":
        """Create a resident LoRA training client."""
        payload = {
            "name": name,
            "adapter_id": adapter_id,
            "checkpoint_path": checkpoint_path,
            "tenant_id": tenant_id if tenant_id is not None else self.tenant_id,
            "idempotency_key": idempotency_key,
        }
        result = CreateRunResult(
            **self.post("/runs", {key: value for key, value in payload.items() if value is not None})
        )
        return LoRATrainingClient(self, result.run_id, result.adapter_id, result.name)

    def train_steps(
        self,
        batches: dict[str, list[Datum | dict[str, Any]]],
        *,
        steps: int,
        learning_rate: float,
        batch_size: int = 1,
        microbatch_size: Optional[int] = None,
        loss_fn: str = "cross_entropy",
        loss_fn_config: Optional[dict[str, float]] = None,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        save_names: Optional[dict[str, str]] = None,
        run_async: bool = False,
        tenant_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> APIFuture[TrainStepsResult | JobInfo]:
        """Run a server-owned mixed-adapter training loop."""
        payload = {
            "batches": {run_id: [_datum_to_json(datum) for datum in data] for run_id, data in batches.items()},
            "steps": steps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "microbatch_size": microbatch_size,
            "loss_fn": loss_fn,
            "loss_fn_config": loss_fn_config or {},
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "save_names": save_names or {},
            "run_async": run_async,
            "tenant_id": tenant_id if tenant_id is not None else self.tenant_id,
            "idempotency_key": idempotency_key,
        }
        response = self.post("/train_steps", {key: value for key, value in payload.items() if value is not None})
        if run_async:
            return APIFuture(_job_info(response["job"]))
        return APIFuture(
            TrainStepsResult(
                job=_job_info(response["job"]),
                runs={run_id: _run_info(run) for run_id, run in response["runs"].items()},
                outputs=response.get("outputs", {}),
            )
        )

    def sample_openai_response(
        self,
        model: str,
        input_text: str | list[dict[str, Any]],
        *,
        max_output_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        return_logprobs: bool = False,
    ) -> dict[str, Any]:
        """Sample through the OpenAI-compatible `/v1/responses` endpoint."""
        return self.post(
            "/v1/responses",
            {
                "model": model,
                "input": input_text,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "tinker_return_logprobs": return_logprobs,
            },
        )

    def create_sampling_client(self, run_id: str) -> "LoRASamplingClient":
        """Create a sampling-only handle for an existing run."""
        run = self.get_run(run_id)
        return LoRASamplingClient(self, run.run_id, run.adapter_id, run.name)


class LoRATrainingClient:
    """SDK handle for one resident LoRA adapter run."""

    def __init__(self, service: NemotronTinkerClient, run_id: str, adapter_id: str, name: Optional[str]) -> None:
        self.service = service
        self.run_id = run_id
        self.adapter_id = adapter_id
        self.name = name

    def get_run(self) -> RunInfo:
        """Return this run's current metadata."""
        return self.service.get_run(self.run_id)

    def forward_backward(
        self,
        data: list[Datum | dict[str, Any]],
        *,
        loss_fn: str = "cross_entropy",
        loss_fn_config: Optional[dict[str, float]] = None,
    ) -> APIFuture[ForwardBackwardResult]:
        """Run one forward/backward call for this adapter."""
        response = self.service.post(
            f"/runs/{self.run_id}/forward_backward",
            {
                "data": [_datum_to_json(datum) for datum in data],
                "loss_fn": loss_fn,
                "loss_fn_config": loss_fn_config or {},
            },
        )
        output = response["output"]
        return APIFuture(
            ForwardBackwardResult(
                run=_run_info(response["run"]),
                loss=float(output["loss"]),
                metrics=dict(output.get("metrics", {})),
                loss_fn_outputs=list(output.get("loss_fn_outputs", [])),
            )
        )

    def optim_step(
        self,
        learning_rate: float,
        *,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        idempotency_key: Optional[str] = None,
    ) -> APIFuture[OptimStepResult]:
        """Apply one optimizer step to this adapter."""
        response = self.service.post(
            f"/runs/{self.run_id}/optim_step",
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "betas": betas,
                "eps": eps,
                "idempotency_key": idempotency_key,
            },
        )
        output = response["output"]
        return APIFuture(
            OptimStepResult(
                run=_run_info(response["run"]),
                step=int(output["step"]),
                learning_rate=float(output["learning_rate"]),
            )
        )

    def save_state(self, name: str, *, idempotency_key: Optional[str] = None) -> APIFuture[SaveResult]:
        """Save this adapter and optimizer state."""
        response = self.service.post(
            f"/runs/{self.run_id}/save",
            {"name": name, "idempotency_key": idempotency_key},
        )
        return APIFuture(SaveResult(run=_run_info(response["run"]), path=str(response["output"]["path"])))

    def detach(self, *, idempotency_key: Optional[str] = None) -> APIFuture[DetachResult]:
        """Detach this adapter from the resident service."""
        response = self.service.post(f"/runs/{self.run_id}/detach", {"idempotency_key": idempotency_key})
        output = response["output"]
        return APIFuture(
            DetachResult(
                run=_run_info(response["run"]),
                adapter_id=str(output["adapter_id"]),
                remaining_adapters=int(output["remaining_adapters"]),
            )
        )

    def sample(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        return_logprobs: bool = False,
    ) -> APIFuture[SampleResult]:
        """Sample from this adapter."""
        params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            return_logprobs=return_logprobs,
        )
        response = self.service.post(
            f"/runs/{self.run_id}/sample",
            {"prompt": prompt, **asdict(params)},
        )
        output = response["output"]
        return APIFuture(
            SampleResult(
                run=_run_info(response["run"]),
                tokens=[int(token) for token in output["tokens"]],
                text=str(output["text"]),
                prompt_token_count=int(output.get("prompt_token_count", 0)),
                generated_logprobs=output.get("generated_logprobs"),
            )
        )

    def as_sampling_client(self) -> "LoRASamplingClient":
        """Return a sampling-only handle for this run."""
        return LoRASamplingClient(self.service, self.run_id, self.adapter_id, self.name)


class LoRASamplingClient:
    """SDK handle for sampling from one existing LoRA adapter run."""

    def __init__(self, service: NemotronTinkerClient, run_id: str, adapter_id: str, name: Optional[str]) -> None:
        self.service = service
        self.run_id = run_id
        self.adapter_id = adapter_id
        self.name = name

    def sample(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        return_logprobs: bool = False,
    ) -> APIFuture[SampleResult]:
        """Sample from this adapter."""
        return LoRATrainingClient(self.service, self.run_id, self.adapter_id, self.name).sample(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            return_logprobs=return_logprobs,
        )


ServiceClient = NemotronTinkerClient
