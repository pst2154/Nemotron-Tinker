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

import json
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest

from nemotron_tinker import server
from nemotron_tinker.future import APIFuture
from nemotron_tinker.types import (
    DetachAdapterResponse,
    ForwardBackwardOutput,
    OptimStepResponse,
    SampleResponse,
    SaveStateResponse,
)

fastapi_testclient = pytest.importorskip("fastapi.testclient")


class FakeTrainingClient:
    def __init__(self, service, adapter_id, step=0):
        self.service = service
        self.adapter_id = adapter_id
        self.handle = SimpleNamespace(step=step)

    def optim_step(self, adam_params):
        self.service.steps[self.adapter_id] += 1
        self.handle.step = self.service.steps[self.adapter_id]
        return APIFuture(
            OptimStepResponse(step=self.service.steps[self.adapter_id], learning_rate=adam_params.learning_rate)
        )

    def save_state(self, name):
        return APIFuture(SaveStateResponse(path=f"/tmp/{name}"))

    def detach(self):
        self.service.steps.pop(self.adapter_id, None)
        return APIFuture(DetachAdapterResponse(adapter_id=self.adapter_id, remaining_adapters=len(self.service.steps)))


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        tokens = [ord(char) % 97 for char in text]
        if add_special_tokens:
            return [1] + tokens
        return tokens

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return self.encode(text, add_special_tokens=True)
        return text


class FakeMixedLoraServiceClient:
    def __init__(self, **kwargs):
        self.created = 0
        self.steps = {}
        self.tokenizer = FakeTokenizer()
        self.forward_batches = []

    def create_lora_training_client(self, *, adapter_id=None, checkpoint_path=None):
        self.created += 1
        adapter_id = adapter_id or f"adapter_{self.created}"
        self.steps[adapter_id] = 0
        if checkpoint_path is not None:
            self.steps[adapter_id] = 7
        return FakeTrainingClient(self, adapter_id, step=self.steps[adapter_id])

    def forward_backward_mixed(self, batches_by_adapter, loss_fn, loss_fn_config=None, zero_grad=True):
        loss_fn_config = loss_fn_config or {}
        self.forward_batches.append(
            {
                "sizes": {adapter_id: len(batch) for adapter_id, batch in batches_by_adapter.items()},
                "zero_grad": zero_grad,
            }
        )
        outputs = {}
        for adapter_id, batch in batches_by_adapter.items():
            metrics = {
                "loss": float(len(batch)),
                "loss:sum": float(len(batch)),
                "loss_weight_mean": 1.0,
                "num_label_tokens": float(len(batch) * 3),
            }
            if loss_fn != "cross_entropy":
                metrics.update(
                    {
                        "clip_low_threshold": float(loss_fn_config.get("clip_low_threshold", 0.8)),
                        "clip_high_threshold": float(loss_fn_config.get("clip_high_threshold", 1.2)),
                        "importance_ratio_mean": 1.0,
                    }
                )
            outputs[adapter_id] = ForwardBackwardOutput(
                loss=float(len(batch)),
                metrics=metrics,
                loss_fn_outputs=[{"logprobs": [-1.0, -0.5]}],
            )
        return APIFuture(outputs)

    def sample(self, adapter_id, prompt, params):
        generated_logprobs = [-0.5] if params.return_logprobs else None
        return APIFuture(
            SampleResponse(
                tokens=[1, 2, 3],
                text=f"{prompt} {adapter_id}",
                prompt_token_count=2,
                generated_logprobs=generated_logprobs,
            )
        )


def test_mixed_lora_server_tracks_run_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, use_triton_lora=True)
    client = fastapi_testclient.TestClient(app)

    health = client.get("/health").json()
    assert health["use_triton_lora"] is True
    assert health["mixed_lora_backend"] == "triton"
    assert health["metadata_backend"] == "sqlite"
    assert health["max_runs_per_tenant"] is None
    assert health["max_concurrent_rl_jobs"] is None
    assert health["max_concurrent_rl_jobs_per_tenant"] is None
    assert health["tenant_rate_limit_per_minute"] is None
    assert health["restore_runs_on_startup"] is False
    assert health["resume_interrupted_jobs_on_startup"] is False
    assert health["rl_job_status_counts"] == {}
    assert health["active_rl_process_count"] == 0

    first = client.post("/runs", json={"name": "atlas"}).json()
    second = client.post("/runs", json={"name": "borealis"}).json()

    response = client.post(
        "/mixed_forward_backward",
        json={
            "batches": {
                first["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}],
                second["run_id"]: [{"model_input": {"tokens": [4, 5, 6]}, "loss_fn_inputs": {}}],
            }
        },
    ).json()

    assert response[first["run_id"]]["run"]["status"] == "ready"
    assert response[first["run_id"]]["run"]["forward_backward_calls"] == 1
    assert response[first["run_id"]]["run"]["last_loss"] == 1.0

    step = client.post(f"/runs/{first['run_id']}/optim_step", json={"learning_rate": 0.001}).json()
    assert step["run"]["optimizer_steps"] == 1
    assert step["output"]["learning_rate"] == 0.001

    sample = client.post(f"/runs/{first['run_id']}/sample", json={"prompt": "hello"}).json()
    assert sample["output"]["text"] == "hello adapter_1"

    saved = client.post(f"/runs/{first['run_id']}/save", json={"name": "atlas-test"}).json()
    assert saved["run"]["last_checkpoint_path"] == "/tmp/atlas-test"

    record = client.get(f"/runs/{first['run_id']}").json()
    metrics = client.get("/metrics").json()
    health = client.get("/health").json()
    assert record["sequence"] >= 4
    assert record["last_checkpoint_path"] == "/tmp/atlas-test"
    assert metrics["operations"]["create_run"]["count"] == 2
    assert metrics["operations"]["mixed_forward_backward"]["count"] == 1
    assert metrics["operations"]["optim_step"]["count"] == 1
    assert metrics["operations"]["sample"]["count"] == 1
    assert metrics["operations"]["save"]["count"] == 1
    assert metrics["operations"]["save"]["last_seconds"] >= 0.0
    assert health["metrics"]["operations"]["create_run"]["count"] == 2


def test_mixed_lora_server_serves_operator_ui(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)

    response = client.get("/ui")

    assert response.status_code == 200
    assert "Nemotron Tinker" in response.text
    assert "SFT Adapter Goals" in response.text
    assert "Resident RL Goal" in response.text
    assert 'id="create-run"' in response.text
    assert 'id="resident-rl-train-both"' in response.text


def test_mixed_lora_server_prepares_nemo_rl_docker_command(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('not launched')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/rl/jobs",
        json={
            "name": "dry-run",
            "launcher": "docker",
            "container_image": "nvcr.io/nvidia/nemo-rl:v0.6.0",
            "docker_repo_dir": "/host/RL",
            "docker_container_repo_dir": "/workspace/RL",
            "docker_user": "140045:30",
            "runner": "python",
            "dry_run": True,
            "overrides": ["grpo.max_num_steps=2", "policy.dtensor_cfg.lora_cfg.enabled=true"],
        },
    ).json()

    job = response["job"]
    assert job["status"] == "dry_run"
    assert job["launcher"] == "docker"
    assert job["command"][:2] == ["docker", "run"]
    assert "--user" in job["command"]
    assert "140045:30" in job["command"]
    assert "/host/RL:/workspace/RL" in job["command"]
    assert "nvcr.io/nvidia/nemo-rl:v0.6.0" in job["command"]
    assert "/opt/nemo_rl_venv/bin/python" in job["command"]
    assert "uv" not in job["command"]
    assert "grpo.max_num_steps=2" in job["command"]
    assert client.get("/rl/jobs").json()[0]["job_id"] == job["job_id"]
    assert client.get(f"/rl/jobs/{job['job_id']}/logs").json()["text"] == ""


def test_mixed_lora_server_queues_host_nemo_rl_job_and_refreshes_status(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('queued for host')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/rl/jobs",
        json={
            "name": "host-run",
            "launcher": "host",
            "runner": "python",
            "docker_repo_dir": str(rl_repo),
            "docker_container_repo_dir": "/opt/nemo-rl",
        },
    ).json()

    job = response["job"]
    assert job["status"] == "queued"
    queued_path = tmp_path / "tinker_api" / "host_rl_queue" / f"{job['job_id']}.json"
    queued_payload = json.loads(queued_path.read_text(encoding="utf-8"))
    assert queued_payload["host_cwd"] == str(rl_repo)
    assert queued_payload["command"][:2] == ["docker", "run"]

    store = server.SQLiteStore(tmp_path / "tinker_api" / "metadata.sqlite3", "rl_jobs", server.RLJobRecord)
    records = store.load()
    records[job["job_id"]].status = "running"
    records[job["job_id"]].pid = 12345
    store.save(records)

    refreshed = client.get(f"/rl/jobs/{job['job_id']}").json()
    assert refreshed["status"] == "running"
    assert refreshed["pid"] == 12345

    marked = client.post(
        f"/internal/rl/jobs/{job['job_id']}/mark",
        json={"status": "succeeded", "returncode": 0, "log_path": str(tmp_path / "host.log")},
    ).json()
    assert marked["status"] == "succeeded"
    assert marked["returncode"] == 0
    assert marked["log_path"] == str(tmp_path / "host.log")


def test_mixed_lora_server_prepares_container_native_nemo_rl_docker_command(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('not launched')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/rl/jobs",
        json={
            "name": "dry-run",
            "launcher": "docker",
            "runner": "python",
            "dry_run": True,
            "overrides": ["logger.log_dir=/tmp/nvidia-tinker-rl-smoke"],
        },
    ).json()

    command = response["job"]["command"]
    assert "-v" not in command
    assert "-w" in command
    assert command[command.index("-w") + 1] == "/opt/nemo-rl"
    assert "/opt/nemo-rl/examples/run_grpo.py" in command
    assert "/opt/nemo-rl/examples/configs/grpo_math_1B.yaml" in command


def test_mixed_lora_server_prepares_nemo_rl_docker_cache_mount(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('not launched')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/rl/jobs",
        json={
            "name": "dry-run",
            "launcher": "docker",
            "runner": "python",
            "dry_run": True,
            "docker_hf_cache_dir": "/host/hf",
        },
    ).json()

    command = response["job"]["command"]
    assert "/host/hf:/root/.cache/huggingface" in command
    assert "HF_HOME=/root/.cache/huggingface" in command
    assert "HF_HUB_CACHE=/root/.cache/huggingface/hub" in command
    assert "HF_DATASETS_CACHE=/root/.cache/huggingface/datasets" in command


def test_mixed_lora_server_prepares_nemo_rl_docker_output_mount(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('not launched')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/rl/jobs",
        json={
            "name": "dry-run",
            "launcher": "docker",
            "runner": "python",
            "dry_run": True,
            "docker_output_dir": "/host/outputs",
        },
    ).json()

    command = response["job"]["command"]
    assert "/host/outputs:/workspace/rl_outputs" in command
    assert "logger.log_dir=/workspace/rl_outputs" in command


def test_mixed_lora_server_keeps_explicit_logger_dir_with_output_mount(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('not launched')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/rl/jobs",
        json={
            "name": "dry-run",
            "launcher": "docker",
            "runner": "python",
            "dry_run": True,
            "docker_output_dir": "/host/outputs",
            "overrides": ["logger.log_dir=/custom"],
        },
    ).json()

    command = response["job"]["command"]
    assert "logger.log_dir=/custom" in command
    assert "logger.log_dir=/workspace/rl_outputs" not in command


def test_mixed_lora_server_prepares_nemo_rl_docker_gpu_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('not launched')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/rl/jobs",
        json={
            "name": "dry-run",
            "launcher": "docker",
            "runner": "python",
            "dry_run": True,
            "docker_gpus": "device=0",
        },
    ).json()

    command = response["job"]["command"]
    assert command[command.index("--gpus") + 1] == "device=0"


def test_mixed_lora_server_expands_nemo_rl_topology_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('not launched')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/rl/jobs",
        json={
            "name": "dry-run",
            "launcher": "docker",
            "runner": "python",
            "dry_run": True,
            "num_nodes": 1,
            "gpus_per_node": 8,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "context_parallel_size": 1,
            "expert_parallel_size": 2,
        },
    ).json()

    command = response["job"]["command"]
    assert "cluster.num_nodes=1" in command
    assert "cluster.gpus_per_node=8" in command
    assert "policy.dtensor_cfg.tensor_parallel_size=2" in command
    assert "policy.megatron_cfg.tensor_model_parallel_size=2" in command
    assert "policy.megatron_cfg.pipeline_model_parallel_size=2" in command
    assert "policy.dtensor_cfg.context_parallel_size=1" in command
    assert "policy.megatron_cfg.context_parallel_size=1" in command
    assert "policy.megatron_cfg.expert_model_parallel_size=2" in command


def test_mixed_lora_server_rejects_oversubscribed_single_node_topology(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('not launched')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/rl/jobs",
        json={
            "launcher": "docker",
            "dry_run": True,
            "gpus_per_node": 8,
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 2,
            "expert_parallel_size": 2,
        },
    )

    assert response.status_code == 400
    assert "must be <= gpus_per_node" in response.json()["detail"]


def test_mixed_lora_server_cancels_running_nemo_rl_bridge_job(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('not launched')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")

    processes = []

    class BlockingProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            self.pid = 43210
            self.returncode = None
            self.done = threading.Event()
            processes.append(self)

        def wait(self, timeout=None):
            self.done.wait(timeout=5)
            return self.returncode

    def fake_killpg(pid, sig):
        assert pid == 43210
        assert sig == server.signal.SIGTERM
        processes[0].returncode = -15
        processes[0].done.set()

    monkeypatch.setattr(server.subprocess, "Popen", BlockingProcess)
    monkeypatch.setattr(server.os, "killpg", fake_killpg)

    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)
    job = client.post(
        "/rl/jobs",
        json={
            "name": "cancel-me",
            "runner": "python",
            "run_async": True,
            "overrides": ["grpo.max_num_steps=100"],
        },
    ).json()["job"]

    for _ in range(50):
        current = client.get(f"/rl/jobs/{job['job_id']}").json()
        if current["pid"] == 43210:
            break
        time.sleep(0.01)
    assert current["status"] == "running"
    canceled = client.post(f"/rl/jobs/{job['job_id']}/cancel").json()
    assert canceled["status"] in {"canceling", "canceled"}

    for _ in range(50):
        current = client.get(f"/rl/jobs/{job['job_id']}").json()
        if current["status"] == "canceled":
            break
        time.sleep(0.01)
    assert current["status"] == "canceled"
    assert current["returncode"] == -15


def test_mixed_lora_server_times_out_nemo_rl_bridge_job(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('not launched')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")

    processes = []

    class TimeoutProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            self.pid = 54321
            self.returncode = None
            processes.append(self)

        def wait(self, timeout=None):
            if self.returncode is None:
                raise server.subprocess.TimeoutExpired(self.command, timeout)
            return self.returncode

    def fake_killpg(pid, sig):
        assert pid == 54321
        assert sig == server.signal.SIGTERM
        processes[0].returncode = -15

    monkeypatch.setattr(server.subprocess, "Popen", TimeoutProcess)
    monkeypatch.setattr(server.os, "killpg", fake_killpg)

    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)
    job = client.post(
        "/rl/jobs",
        json={
            "name": "timeout-me",
            "runner": "python",
            "run_async": False,
            "max_runtime_seconds": 0.01,
            "overrides": ["grpo.max_num_steps=100"],
        },
    ).json()["job"]

    assert job["status"] == "timed_out"
    assert job["returncode"] == -15
    assert "max_runtime_seconds=0.01" in job["error"]


def test_mixed_lora_server_limits_concurrent_nemo_rl_bridge_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text("print('not launched')\n", encoding="utf-8")
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")

    processes = []

    class BlockingProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            self.pid = 65432
            self.returncode = None
            self.done = threading.Event()
            processes.append(self)

        def wait(self, timeout=None):
            self.done.wait(timeout=5)
            return self.returncode

    def fake_killpg(pid, sig):
        assert sig == server.signal.SIGTERM
        for process in processes:
            if process.pid == pid:
                process.returncode = -15
                process.done.set()

    monkeypatch.setattr(server.subprocess, "Popen", BlockingProcess)
    monkeypatch.setattr(server.os, "killpg", fake_killpg)

    app = server.create_app(
        base_model="fake-model",
        scratch_dir=tmp_path,
        rl_repo_dir=str(rl_repo),
        max_concurrent_rl_jobs_per_tenant=1,
    )
    client = fastapi_testclient.TestClient(app)
    first = client.post(
        "/rl/jobs",
        json={"name": "first", "runner": "python", "run_async": True},
        headers={"X-Tinker-Tenant-Id": "tenant-a"},
    ).json()["job"]

    for _ in range(50):
        current = client.get(f"/rl/jobs/{first['job_id']}", headers={"X-Tinker-Tenant-Id": "tenant-a"}).json()
        if current["status"] == "running":
            break
        time.sleep(0.01)
    response = client.post(
        "/rl/jobs",
        json={"name": "second", "runner": "python", "run_async": True},
        headers={"X-Tinker-Tenant-Id": "tenant-a"},
    )
    assert response.status_code == 429
    assert "RL job capacity reached" in response.json()["detail"]

    client.post(f"/rl/jobs/{first['job_id']}/cancel", headers={"X-Tinker-Tenant-Id": "tenant-a"})


def test_mixed_lora_server_runs_local_nemo_rl_bridge_job(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    rl_repo = tmp_path / "RL"
    (rl_repo / "examples" / "configs").mkdir(parents=True)
    (rl_repo / "examples" / "run_grpo.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--config')\n"
        "args, overrides = parser.parse_known_args()\n"
        "print('config=' + args.config)\n"
        "print('overrides=' + ','.join(overrides))\n",
        encoding="utf-8",
    )
    (rl_repo / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, rl_repo_dir=str(rl_repo))
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/rl/jobs",
        json={
            "name": "local-smoke",
            "runner": "python",
            "run_async": False,
            "overrides": ["grpo.max_num_steps=1"],
        },
    ).json()

    job = response["job"]
    logs = client.get(f"/rl/jobs/{job['job_id']}/logs").json()["text"]
    assert job["status"] == "succeeded"
    assert job["returncode"] == 0
    assert "config=" in logs
    assert "grpo.max_num_steps=1" in logs


def test_mixed_lora_server_tokenizes_text_sft_datum(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/datasets/sft_datum",
        json={"prompt": "Prompt:", "completion": " answer.", "max_tokens": 32},
    )

    datum = response.json()
    weights = datum["loss_fn_inputs"]["weights"]
    target_tokens = datum["loss_fn_inputs"]["target_tokens"]["tokens"]
    input_tokens = datum["model_input"]["tokens"]
    assert response.status_code == 200
    assert input_tokens[1:] == target_tokens[:-1]
    assert len(weights) == len(target_tokens)
    assert len(input_tokens) == len(target_tokens)
    assert 0.0 in weights
    assert 1.0 in weights


def test_mixed_lora_server_tokenizes_chat_template_sft_datum(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/datasets/sft_datum",
        json={"prompt": "Prompt:", "completion": " answer.", "max_tokens": 128, "use_chat_template": True},
    )

    datum = response.json()
    weights = datum["loss_fn_inputs"]["weights"]
    target_tokens = datum["loss_fn_inputs"]["target_tokens"]["tokens"]
    input_tokens = datum["model_input"]["tokens"]
    assert response.status_code == 200
    assert input_tokens[1:] == target_tokens[:-1]
    assert len(weights) == len(target_tokens)
    assert weights.count(0.0) > weights.count(1.0)
    assert weights[-1] == 1.0


def test_mixed_lora_server_reports_supervised_worker_processes(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, worker_processes=2)
    with fastapi_testclient.TestClient(app) as client:
        health = client.get("/health").json()
        workers = client.get("/workers").json()
        created = client.post("/runs", json={"name": "placed"}).json()
        record = client.get(f"/runs/{created['run_id']}").json()
        ping = client.post(f"/workers/{created['worker_id']}/ping").json()
        echo = client.post(
            f"/workers/{created['worker_id']}/echo",
            json={"payload": {"op": "future_create_run", "run_id": created["run_id"]}},
        ).json()
        worker_runs = client.get(f"/workers/{created['worker_id']}/runs").json()
        worker_operations = client.get(f"/workers/{created['worker_id']}/operations").json()

        assert health["worker_processes"] == 2
        assert health["model_execution"] == "api_process"
        assert health["worker_assignment_ready"] is True
        assert health["stale_worker_run_ids"] == []
        assert health["run_status_counts"] == {}
        assert len(health["workers"]) == 2
        assert len(workers) == 2
        assert {worker["status"] for worker in workers} == {"running"}
        assert all(worker["pid"] for worker in workers)
        assert created["worker_id"] in {worker["worker_id"] for worker in workers}
        assert record["worker_id"] == created["worker_id"]
        assert ping["worker"]["worker_id"] == created["worker_id"]
        assert ping["result"]["worker_pid"] == ping["worker"]["pid"]
        assert echo["worker"]["worker_id"] == created["worker_id"]
        assert echo["result"]["payload"] == {"op": "future_create_run", "run_id": created["run_id"]}
        assert worker_runs["worker"]["worker_id"] == created["worker_id"]
        assert worker_runs["assigned_run_count"] == 1
        assert worker_runs["runs"][0]["run_id"] == created["run_id"]
        assert worker_operations["worker"]["worker_id"] == created["worker_id"]
        assert worker_operations["operation_count"] == 1
        assert worker_operations["operations"][0]["operation"] == "create_run"
        assert worker_operations["operations"][0]["run_ids"] == [created["run_id"]]
        assert client.get("/workers/unknown/runs").status_code == 404
        assert client.get("/workers/unknown/operations").status_code == 404


def test_mixed_lora_server_records_worker_operation_envelopes(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, worker_processes=1)
    with fastapi_testclient.TestClient(app) as client:
        created = client.post("/runs", json={"name": "placed"}).json()
        run_id = created["run_id"]
        worker_id = created["worker_id"]

        client.post(
            "/mixed_forward_backward",
            json={"batches": {run_id: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}]}},
        )
        client.post(f"/runs/{run_id}/optim_step", json={"learning_rate": 0.001})
        client.post(f"/runs/{run_id}/sample", json={"prompt": "hello", "max_new_tokens": 3})
        client.post(f"/runs/{run_id}/save", json={"name": "placed-save"})
        worker_operations = client.get(f"/workers/{worker_id}/operations").json()

        assert [operation["operation"] for operation in worker_operations["operations"]] == [
            "create_run",
            "mixed_forward_backward",
            "optim_step",
            "sample",
            "save",
        ]
        assert worker_operations["operation_count"] == 5
        assert worker_operations["operations"][1]["payload"] == {
            "loss_fn": "cross_entropy",
            "num_runs": 1,
        }
        assert worker_operations["operations"][2]["payload"] == {"learning_rate": 0.001}


def test_mixed_lora_server_detaches_resident_run(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, worker_processes=1)
    with fastapi_testclient.TestClient(app) as client:
        created = client.post("/runs", json={"name": "placed"}).json()
        run_id = created["run_id"]
        worker_id = created["worker_id"]

        detached = client.post(f"/runs/{run_id}/detach", json={"idempotency_key": "detach-placed"}).json()
        repeated = client.post(f"/runs/{run_id}/detach", json={"idempotency_key": "detach-placed"}).json()
        record = client.get(f"/runs/{run_id}").json()
        worker_runs = client.get(f"/workers/{worker_id}/runs").json()
        worker_operations = client.get(f"/workers/{worker_id}/operations").json()

        assert detached == repeated
        assert detached["run"]["status"] == "detached"
        assert detached["output"] == {"adapter_id": created["adapter_id"], "remaining_adapters": 0}
        assert record["status"] == "detached"
        assert worker_runs["assigned_run_count"] == 0
        assert worker_operations["operations"][-1]["operation"] == "detach_run"
        assert client.post(f"/runs/{run_id}/sample", json={"prompt": "hello"}).status_code == 404


def test_mixed_lora_server_detach_releases_resident_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, max_resident_adapters=1)
    client = fastapi_testclient.TestClient(app)

    created = client.post("/runs", json={"name": "first"}).json()
    assert client.post("/runs", json={"name": "blocked"}).status_code == 429
    assert client.post(f"/runs/{created['run_id']}/detach", json={}).status_code == 200
    second = client.post("/runs", json={"name": "second"})

    assert second.status_code == 200


def test_mixed_lora_server_save_and_detach_saves_before_unloading(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, worker_processes=1)
    with fastapi_testclient.TestClient(app) as client:
        created = client.post("/runs", json={"name": "placed"}).json()
        run_id = created["run_id"]
        worker_id = created["worker_id"]

        response = client.post(
            f"/runs/{run_id}/save_and_detach",
            json={"name": "placed-final", "idempotency_key": "save-detach-placed"},
        ).json()
        repeated = client.post(
            f"/runs/{run_id}/save_and_detach",
            json={"name": "placed-final", "idempotency_key": "save-detach-placed"},
        ).json()
        worker_runs = client.get(f"/workers/{worker_id}/runs").json()
        worker_operations = client.get(f"/workers/{worker_id}/operations").json()

        assert repeated == response
        assert response["run"]["status"] == "detached"
        assert response["run"]["last_checkpoint_path"] == "/tmp/placed-final"
        assert response["save_output"] == {"path": "/tmp/placed-final"}
        assert response["detach_output"] == {"adapter_id": created["adapter_id"], "remaining_adapters": 0}
        assert worker_runs["assigned_run_count"] == 0
        assert [operation["operation"] for operation in worker_operations["operations"][-2:]] == [
            "save",
            "detach_run",
        ]
        assert client.post(f"/runs/{run_id}/optim_step", json={"learning_rate": 0.001}).status_code == 404


def test_mixed_lora_server_reattaches_runs_after_worker_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, worker_processes=1)
    with fastapi_testclient.TestClient(app) as client:
        created = client.post("/runs", json={"name": "placed"}).json()
        worker_id = created["worker_id"]
        before = client.get(f"/workers/{worker_id}/runs").json()
        manager = app.state.worker_manager
        manager._slots[worker_id].process.terminate()
        manager._slots[worker_id].process.join(timeout=5.0)
        unhealthy = client.get("/health").json()

        restarted = client.post("/workers/restart_dead").json()
        after = client.get(f"/workers/{worker_id}/runs").json()

        assert before["assigned_run_count"] == 1
        assert unhealthy["worker_assignment_ready"] is False
        assert unhealthy["stale_worker_run_ids"] == [created["run_id"]]
        assert restarted[0]["worker_id"] == worker_id
        assert restarted[0]["restarts"] == 1
        assert restarted[0]["assigned_run_count"] == 1
        assert after["assigned_run_count"] == 1
        assert after["runs"][0]["run_id"] == created["run_id"]
        operations = client.get(f"/workers/{worker_id}/operations").json()
        assert operations["operations"][0]["operation"] == "reattach_run"
        assert operations["operations"][0]["payload"] == {"reason": "worker_restart"}


def test_mixed_lora_server_reconciles_stale_worker_assignments(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, worker_processes=1)
    with fastapi_testclient.TestClient(app) as client:
        created = client.post("/runs", json={"name": "placed"}).json()
        run_id = created["run_id"]
        worker_id = created["worker_id"]
        record = client.get(f"/runs/{run_id}").json()
        assert record["worker_id"] == worker_id
        # Simulate metadata drift without killing the worker.
        app.state.worker_manager.detach_run(worker_id, run_id)
        reconciled = client.post("/workers/reconcile").json()
        worker_runs = client.get(f"/workers/{worker_id}/runs").json()
        operations = client.get(f"/workers/{worker_id}/operations").json()

        assert reconciled["reattached_run_ids"] == [run_id]
        assert worker_runs["assigned_run_count"] == 1
        assert worker_runs["runs"][0]["run_id"] == run_id
        assert operations["operations"][-1]["operation"] == "reattach_run"
        assert operations["operations"][-1]["payload"] == {"reason": "manual_reconcile"}


def test_mixed_lora_server_restores_run_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)

    restored = client.post(
        "/runs",
        json={"name": "restored", "adapter_id": "adapter_restored", "checkpoint_path": "/tmp/checkpoint"},
    ).json()

    record = client.get(f"/runs/{restored['run_id']}").json()
    assert restored["adapter_id"] == "adapter_restored"
    assert restored["status"] == "ready"
    assert record["optimizer_steps"] == 7
    assert record["restored_from"] == "/tmp/checkpoint"


def test_mixed_lora_server_marks_persisted_runs_detached(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post("/runs", json={"name": "atlas"}).json()

    restarted = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    restarted_client = fastapi_testclient.TestClient(restarted)

    record = restarted_client.get(f"/runs/{created['run_id']}").json()
    assert record["status"] == "detached"


def test_mixed_lora_server_rehydrates_checkpointed_runs_on_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post(
        "/runs",
        json={"name": "restored", "adapter_id": "adapter_restored", "checkpoint_path": "/tmp/checkpoint"},
    ).json()

    restarted = server.create_app(base_model="fake-model", scratch_dir=tmp_path, restore_runs_on_startup=True)
    restarted_client = fastapi_testclient.TestClient(restarted)

    health = restarted_client.get("/health").json()
    record = restarted_client.get(f"/runs/{created['run_id']}").json()
    sample = restarted_client.post(f"/runs/{created['run_id']}/sample", json={"prompt": "hello"}).json()

    assert health["restore_runs_on_startup"] is True
    assert record["status"] == "ready"
    assert record["optimizer_steps"] == 7
    assert record["restored_from"] == "/tmp/checkpoint"
    assert sample["output"]["text"] == "hello adapter_restored"


def test_mixed_lora_server_reattaches_restored_runs_to_workers(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post(
        "/runs",
        json={"name": "restored", "adapter_id": "adapter_restored", "checkpoint_path": "/tmp/checkpoint"},
    ).json()

    restarted = server.create_app(
        base_model="fake-model",
        scratch_dir=tmp_path,
        restore_runs_on_startup=True,
        worker_processes=1,
    )
    with fastapi_testclient.TestClient(restarted) as restarted_client:
        record = restarted_client.get(f"/runs/{created['run_id']}").json()
        worker_runs = restarted_client.get(f"/workers/{record['worker_id']}/runs").json()

        assert record["status"] == "ready"
        assert worker_runs["assigned_run_count"] == 1
        assert worker_runs["runs"][0]["run_id"] == created["run_id"]
        assert worker_runs["runs"][0]["restored_from"] == "/tmp/checkpoint"


def test_mixed_lora_server_can_use_json_metadata_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, metadata_backend="json")
    client = fastapi_testclient.TestClient(app)

    created = client.post("/runs", json={"name": "json-run"}).json()
    health = client.get("/health").json()

    assert created["status"] == "created"
    assert health["metadata_backend"] == "json"
    assert (tmp_path / "tinker_api" / "runs.json").exists()


def test_mixed_lora_server_runs_server_owned_train_steps(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    first = client.post("/runs", json={"name": "atlas"}).json()
    second = client.post("/runs", json={"name": "borealis"}).json()

    response = client.post(
        "/train_steps",
        json={
            "batches": {
                first["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}],
                second["run_id"]: [{"model_input": {"tokens": [4, 5, 6]}, "loss_fn_inputs": {}}],
            },
            "steps": 3,
            "learning_rate": 0.001,
            "save_names": {first["run_id"]: "atlas-job"},
        },
    ).json()

    assert response["job"]["status"] == "succeeded"
    assert response["job"]["progress"]["step"] == 3
    assert "request" not in response["job"]["progress"]
    assert response["job"]["progress"]["request_ref"]["kind"] == "file"
    assert pathlib.Path(response["job"]["progress"]["request_ref"]["path"]).is_file()
    assert response["job"]["result"]["last_losses"][first["run_id"]] == 1.0
    assert response["runs"][first["run_id"]]["optimizer_steps"] == 3
    assert response["runs"][first["run_id"]]["last_checkpoint_path"] == "/tmp/atlas-job"

    jobs = client.get("/jobs").json()
    assert jobs[0]["kind"] == "train_steps"
    assert jobs[0]["has_result"] is True
    assert "result" not in jobs[0]
    assert "request" not in jobs[0]["progress"]
    assert "sha256" not in jobs[0]["progress"]["request_ref"]
    job = client.get(f"/jobs/{jobs[0]['job_id']}").json()
    assert job["result"]["last_losses"][first["run_id"]] == 1.0
    assert "request" not in job["progress"]
    assert "sha256" not in job["progress"]["request_ref"]


def test_mixed_lora_server_exposes_openai_responses_for_gym(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post("/runs", json={"name": "atlas"}).json()

    response = client.post(
        "/v1/responses",
        json={
            "model": "atlas",
            "input": [{"role": "user", "content": "hello"}],
            "max_output_tokens": 4,
            "temperature": 0,
            "tinker_return_logprobs": True,
        },
    ).json()

    assert response["object"] == "response"
    assert response["model"] == "atlas"
    assert response["output"][0]["content"][0]["type"] == "output_text"
    assert response["output"][0]["content"][0]["text"] == " adapter_1"
    assert response["tinker_rl"]["tokens"] == [1, 2, 3]
    assert response["tinker_rl"]["generated_logprobs"] == [-0.5]
    assert client.post("/v1/responses", json={"model": created["run_id"], "input": "hello"}).status_code == 200


def test_mixed_lora_server_exposes_openai_chat_completions_for_gym(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post("/runs", json={"name": "atlas"}).json()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": created["adapter_id"],
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "temperature": 0,
        },
    ).json()

    assert response["object"] == "chat.completion"
    assert response["model"] == "atlas"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert response["choices"][0]["message"]["content"] == " adapter_1"


def test_mixed_lora_server_microbatches_train_steps(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    first = client.post("/runs", json={"name": "atlas"}).json()
    service = app.state.tinker_service

    response = client.post(
        "/train_steps",
        json={
            "batches": {
                first["run_id"]: [
                    {"model_input": {"tokens": [idx, idx + 1]}, "loss_fn_inputs": {}} for idx in range(5)
                ],
            },
            "steps": 1,
            "learning_rate": 0.001,
            "microbatch_size": 2,
        },
    ).json()

    assert response["job"]["status"] == "succeeded"
    assert response["job"]["result"]["last_losses"][first["run_id"]] == 5.0
    assert response["runs"][first["run_id"]]["last_metrics"]["num_label_tokens"] == 15.0
    assert response["runs"][first["run_id"]]["last_metrics"]["loss_weight_mean"] == 1.0
    assert service.forward_batches == [
        {"sizes": {"adapter_1": 2}, "zero_grad": True},
        {"sizes": {"adapter_1": 2}, "zero_grad": False},
        {"sizes": {"adapter_1": 1}, "zero_grad": False},
    ]


def test_mixed_lora_server_preserves_rl_config_metrics_across_microbatches(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    first = client.post("/runs", json={"name": "atlas"}).json()

    response = client.post(
        "/train_steps",
        json={
            "batches": {
                first["run_id"]: [
                    {
                        "model_input": {"tokens": [idx, idx + 1]},
                        "loss_fn_inputs": {"logprobs": [0.0], "advantages": [1.0]},
                    }
                    for idx in range(5)
                ],
            },
            "steps": 1,
            "learning_rate": 0.001,
            "microbatch_size": 2,
            "loss_fn": "importance_sampling",
            "loss_fn_config": {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
        },
    ).json()

    metrics = response["runs"][first["run_id"]]["last_metrics"]
    assert metrics["clip_low_threshold"] == 0.8
    assert metrics["clip_high_threshold"] == 1.2
    assert metrics["importance_ratio_mean"] == 1.0


def test_resident_rl_trains_same_run_with_sampled_logprobs(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post("/runs", json={"name": "atlas"}).json()

    response = client.post(
        f"/runs/{created['run_id']}/resident_rl",
        json={
            "prompts": ["say atlas"],
            "rollouts_per_prompt": 2,
            "max_new_tokens": 4,
            "reward_mode": "contains",
            "reward_contains": created["adapter_id"],
            "steps": 2,
            "learning_rate": 0.001,
            "microbatch_size": 1,
            "loss_fn": "importance_sampling",
            "run_async": False,
        },
    ).json()

    assert response["run"]["run_id"] == created["run_id"]
    assert response["train_job"]["kind"] == "train_steps"
    assert response["train_job"]["status"] == "succeeded"
    assert response["train_job"]["progress"]["step"] == 2
    assert response["rollout_count"] == 2
    assert response["reward_summary"]["mean"] == 1.0
    assert response["reward_summary"]["baseline"] == 0.0
    assert response["rollouts"][0]["advantage"] == 1.0
    assert response["rollouts"][0]["completion_text"] == f" {created['adapter_id']}"
    assert response["train_response"]["runs"][created["run_id"]]["optimizer_steps"] == 2
    assert all(row["prompt"] == "say atlas" for row in response["rollouts"])


def test_resident_rl_reward_scores_completion_not_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post("/runs", json={"name": "atlas"}).json()

    response = client.post(
        f"/runs/{created['run_id']}/resident_rl",
        json={
            "prompts": ["target phrase appears only in prompt"],
            "rollouts_per_prompt": 1,
            "max_new_tokens": 4,
            "reward_mode": "contains",
            "reward_contains": "target phrase",
            "steps": 1,
            "learning_rate": 0.001,
            "microbatch_size": 1,
            "loss_fn": "importance_sampling",
            "run_async": False,
        },
    ).json()

    assert response["rollouts"][0]["completion_text"] == f" {created['adapter_id']}"
    assert response["reward_summary"]["mean"] == -0.5
    assert response["rollouts"][0]["advantage"] == -0.5


def test_mixed_lora_server_submits_async_train_job(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    first = client.post("/runs", json={"name": "atlas"}).json()

    submitted = client.post(
        "/train_steps",
        json={
            "batches": {first["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}]},
            "steps": 1,
            "learning_rate": 0.001,
            "run_async": True,
        },
    ).json()

    job = client.get(f"/jobs/{submitted['job']['job_id']}").json()
    assert job["status"] in {"queued", "running", "succeeded"}


def test_mixed_lora_server_resumes_interrupted_train_job_on_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post(
        "/runs",
        json={"name": "restored", "adapter_id": "adapter_resumed", "checkpoint_path": "/tmp/checkpoint"},
    ).json()
    request_payload = {
        "batches": {created["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}]},
        "steps": 3,
        "learning_rate": 0.001,
    }
    now = server._utc_now()
    job = server.JobRecord(
        job_id="job_resume",
        kind="train_steps",
        status="running",
        run_ids=[created["run_id"]],
        progress={"step": 1, "total_steps": 3, "request": request_payload},
        created_at=now,
        updated_at=now,
    )
    store = server.SQLiteStore(tmp_path / "tinker_api" / "metadata.sqlite3", "jobs", server.JobRecord)
    store.save({"job_resume": job})

    restarted = server.create_app(
        base_model="fake-model",
        scratch_dir=tmp_path,
        restore_runs_on_startup=True,
        resume_interrupted_jobs_on_startup=True,
    )
    restarted_client = fastapi_testclient.TestClient(restarted)

    resumed = restarted_client.get("/jobs/job_resume").json()
    for _ in range(20):
        if resumed["status"] == "succeeded":
            break
        time.sleep(0.05)
        resumed = restarted_client.get("/jobs/job_resume").json()
    run = restarted_client.get(f"/runs/{created['run_id']}").json()

    assert resumed["status"] == "succeeded"
    assert resumed["progress"]["step"] == 3
    assert run["optimizer_steps"] == 9


def test_mixed_lora_server_resumes_interrupted_train_job_from_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    created = client.post(
        "/runs",
        json={"name": "restored", "adapter_id": "adapter_resumed", "checkpoint_path": "/tmp/checkpoint"},
    ).json()
    request_payload = {
        "batches": {created["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}]},
        "steps": 3,
        "learning_rate": 0.001,
        "microbatch_size": 1,
    }
    request_path = tmp_path / "tinker_api" / "train_requests" / "job_resume_manifest.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(request_payload, sort_keys=True), encoding="utf-8")
    now = server._utc_now()
    job = server.JobRecord(
        job_id="job_resume_manifest",
        kind="train_steps",
        status="running",
        run_ids=[created["run_id"]],
        progress={
            "step": 1,
            "total_steps": 3,
            "request_ref": {
                "kind": "file",
                "path": str(request_path),
                "sha256": server._digest_json(request_payload),
            },
        },
        created_at=now,
        updated_at=now,
    )
    store = server.SQLiteStore(tmp_path / "tinker_api" / "metadata.sqlite3", "jobs", server.JobRecord)
    store.save({"job_resume_manifest": job})

    restarted = server.create_app(
        base_model="fake-model",
        scratch_dir=tmp_path,
        restore_runs_on_startup=True,
        resume_interrupted_jobs_on_startup=True,
    )
    restarted_client = fastapi_testclient.TestClient(restarted)

    resumed = restarted_client.get("/jobs/job_resume_manifest").json()
    for _ in range(20):
        if resumed["status"] == "succeeded":
            break
        time.sleep(0.05)
        resumed = restarted_client.get("/jobs/job_resume_manifest").json()
    run = restarted_client.get(f"/runs/{created['run_id']}").json()

    assert resumed["status"] == "succeeded"
    assert resumed["progress"]["step"] == 3
    assert "request" not in resumed["progress"]
    assert resumed["progress"]["request_ref"]["path"] == str(request_path)
    assert run["optimizer_steps"] == 9


def test_mixed_lora_server_reuses_idempotent_create_response(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)

    first = client.post("/runs", json={"name": "atlas", "idempotency_key": "create-atlas"}).json()
    second = client.post("/runs", json={"name": "atlas", "idempotency_key": "create-atlas"}).json()

    assert second == first
    assert len(client.get("/runs").json()) == 1


def test_mixed_lora_server_rejects_idempotency_key_reuse_for_different_request(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)

    assert client.post("/runs", json={"name": "atlas", "idempotency_key": "same-key"}).status_code == 200
    response = client.post("/runs", json={"name": "borealis", "idempotency_key": "same-key"})

    assert response.status_code == 409


def test_mixed_lora_server_reuses_idempotent_train_steps_response(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    first = client.post("/runs", json={"name": "atlas"}).json()

    payload = {
        "batches": {first["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}]},
        "steps": 2,
        "learning_rate": 0.001,
        "idempotency_key": "train-atlas",
    }
    first_response = client.post("/train_steps", json=payload).json()
    second_response = client.post("/train_steps", json=payload).json()

    assert second_response == first_response
    assert client.get(f"/runs/{first['run_id']}").json()["optimizer_steps"] == 2


def test_mixed_lora_server_requires_bearer_token_when_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, api_key="secret")
    client = fastapi_testclient.TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.post("/runs", json={"name": "atlas"}).status_code == 401
    authorized = client.post("/runs", json={"name": "atlas"}, headers={"Authorization": "Bearer secret"})

    assert authorized.status_code == 200


def test_mixed_lora_server_enforces_resident_adapter_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, max_resident_adapters=1)
    client = fastapi_testclient.TestClient(app)

    assert client.post("/runs", json={"name": "atlas"}).status_code == 200
    response = client.post("/runs", json={"name": "borealis"})
    metrics = client.get("/metrics").json()

    assert response.status_code == 429
    assert metrics["operations"]["create_run"]["count"] == 2
    assert metrics["operations"]["create_run"]["failures"] == 1
    assert "HTTPException" in metrics["operations"]["create_run"]["last_error"]


def test_mixed_lora_server_enforces_tenant_adapter_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, max_runs_per_tenant=1)
    client = fastapi_testclient.TestClient(app)

    assert client.post("/runs", json={"name": "atlas", "tenant_id": "tenant-a"}).status_code == 200
    assert client.post("/runs", json={"name": "borealis", "tenant_id": "tenant-b"}).status_code == 200
    response = client.post("/runs", json={"name": "orion", "tenant_id": "tenant-a"})

    assert response.status_code == 429
    assert "tenant-a" in response.json()["detail"]


def test_mixed_lora_server_rate_limits_tenant_gpu_operations(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path, tenant_rate_limit_per_minute=2)
    client = fastapi_testclient.TestClient(app)
    created = client.post("/runs", json={"name": "atlas", "tenant_id": "tenant-a"}).json()

    first = client.post(f"/runs/{created['run_id']}/sample", json={"prompt": "hello"})
    second = client.post(f"/runs/{created['run_id']}/sample", json={"prompt": "again"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "tenant-a" in second.json()["detail"]


def test_mixed_lora_server_records_tenant_and_rejects_mixed_tenant_job(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    first = client.post("/runs", json={"name": "atlas", "tenant_id": "tenant-a"}).json()
    second = client.post("/runs", json={"name": "borealis", "tenant_id": "tenant-b"}).json()

    record = client.get(f"/runs/{first['run_id']}").json()
    assert record["tenant_id"] == "tenant-a"

    response = client.post(
        "/train_steps",
        json={
            "batches": {
                first["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}],
                second["run_id"]: [{"model_input": {"tokens": [4, 5, 6]}, "loss_fn_inputs": {}}],
            },
            "steps": 1,
            "learning_rate": 0.001,
        },
    )

    assert response.status_code == 400


def test_mixed_lora_server_scopes_resources_by_tenant_header(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)
    first = client.post("/runs", json={"name": "atlas"}, headers={"X-Tinker-Tenant-Id": "tenant-a"}).json()
    second = client.post("/runs", json={"name": "borealis"}, headers={"X-Tinker-Tenant-Id": "tenant-b"}).json()

    tenant_a_runs = client.get("/runs", headers={"X-Tinker-Tenant-Id": "tenant-a"}).json()
    assert [run["run_id"] for run in tenant_a_runs] == [first["run_id"]]
    assert tenant_a_runs[0]["tenant_id"] == "tenant-a"
    assert client.get(f"/runs/{second['run_id']}", headers={"X-Tinker-Tenant-Id": "tenant-a"}).status_code == 403
    assert (
        client.post(
            f"/runs/{second['run_id']}/sample",
            json={"prompt": "hello"},
            headers={"X-Tinker-Tenant-Id": "tenant-a"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/mixed_forward_backward",
            json={
                "batches": {
                    first["run_id"]: [{"model_input": {"tokens": [1, 2, 3]}, "loss_fn_inputs": {}}],
                    second["run_id"]: [{"model_input": {"tokens": [4, 5, 6]}, "loss_fn_inputs": {}}],
                }
            },
            headers={"X-Tinker-Tenant-Id": "tenant-a"},
        ).status_code
        == 403
    )


def test_mixed_lora_server_rejects_body_tenant_header_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MixedLoraServiceClient", FakeMixedLoraServiceClient)
    app = server.create_app(base_model="fake-model", scratch_dir=tmp_path)
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/runs",
        json={"name": "atlas", "tenant_id": "tenant-b"},
        headers={"X-Tinker-Tenant-Id": "tenant-a"},
    )

    assert response.status_code == 403
    assert "X-Tinker-Tenant-Id" in response.json()["detail"]
