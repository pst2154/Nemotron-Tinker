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

import pytest
import torch

from clients.qwen_api_smoke_client import Example, build_datum
from tools.gym_rollouts_to_tinker_rl import convert_rollouts
from scripts.run_recipe import build_command
from nemotron_tinker import client as tinker_client
from nemotron_tinker.client import _build_batch
from nemotron_tinker.future import APIFuture
from nemotron_tinker.grouped_lora_kernel import grouped_lora_da_db_wrapper
from nemotron_tinker.mixed_client import (
    MixedAdapterHandle,
    MixedAdapterLinearLoRA,
    MixedLoraServiceClient,
    _rl_token_loss,
)
from nemotron_tinker.sdk import NemotronTinkerClient, TinkerAPIError
from nemotron_tinker.types import Datum, ModelInput, SamplingParams


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        tokens = [ord(char) for char in text]
        if add_special_tokens:
            return [1] + tokens
        return tokens


class _FakeBatch(dict):
    def to(self, device):
        return _FakeBatch({key: value.to(device) for key, value in self.items()})


class _FakeSamplingTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, prompt, return_tensors=None):
        return _FakeBatch(
            {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }
        )

    def decode(self, output_ids, skip_special_tokens=True):
        return ",".join(str(token) for token in output_ids.tolist())


class _GenerateModel:
    def __init__(self, *, fail_generate=False):
        self.fail_generate = fail_generate
        self.generate_calls = 0
        self.forward_calls = 0

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls += 1
        if self.fail_generate:
            raise RuntimeError("generate unavailable")
        return torch.tensor([[1, 2, 7, 8]], dtype=torch.long)

    def __call__(self, input_ids, attention_mask=None):
        self.forward_calls += 1
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 10)
        logits[:, -1, 6] = 1.0
        return type("Output", (), {"logits": logits})


class _TinyTrainableLogitModel(torch.nn.Module):
    def __init__(self, vocab_size=8):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids, attention_mask=None):
        batch_size, sequence_length = input_ids.shape
        logits = self.logits.view(1, 1, -1).expand(batch_size, sequence_length, -1)
        return type("Output", (), {"logits": logits})


def _sampling_service(model):
    service = MixedLoraServiceClient.__new__(MixedLoraServiceClient)
    service.model = model
    service.tokenizer = _FakeSamplingTokenizer()
    service.device = torch.device("cpu")
    service.mixed_lora_layers = {}
    return service


def _training_service(model):
    service = MixedLoraServiceClient.__new__(MixedLoraServiceClient)
    service.model = model
    service.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    service.device = torch.device("cpu")
    service.adapters = {"adapter": MixedAdapterHandle(adapter_id="adapter")}
    service.mixed_lora_layers = {}
    return service


def test_api_future_returns_value():
    assert APIFuture(7).result() == 7


def test_sdk_training_client_wraps_core_run_methods():
    calls = []
    responses = {
        ("POST", "/runs"): {
            "run_id": "run_1",
            "adapter_id": "adapter_1",
            "name": "atlas",
            "status": "created",
            "sequence": 0,
        },
        ("POST", "/runs/run_1/forward_backward"): {
            "run": {
                "run_id": "run_1",
                "adapter_id": "adapter_1",
                "name": "atlas",
                "status": "ready",
                "created_at": "now",
                "updated_at": "now",
            },
            "output": {"loss": 1.5, "metrics": {"loss_fn": "cross_entropy"}, "loss_fn_outputs": []},
        },
        ("POST", "/runs/run_1/optim_step"): {
            "run": {
                "run_id": "run_1",
                "adapter_id": "adapter_1",
                "name": "atlas",
                "status": "ready",
                "created_at": "now",
                "updated_at": "now",
            },
            "output": {"step": 1, "learning_rate": 0.001},
        },
        ("POST", "/runs/run_1/sample"): {
            "run": {
                "run_id": "run_1",
                "adapter_id": "adapter_1",
                "name": "atlas",
                "status": "ready",
                "created_at": "now",
                "updated_at": "now",
            },
            "output": {"tokens": [1, 2, 3], "text": "ok", "prompt_token_count": 2, "generated_logprobs": [-0.1]},
        },
    }
    sdk = NemotronTinkerClient(base_url="http://unused", tenant_id="tenant-a")

    def fake_request(method, path, payload=None):
        calls.append((method, path, payload))
        try:
            return responses[(method, path)]
        except KeyError as exc:
            raise TinkerAPIError((method, path)) from exc

    sdk._request = fake_request
    training = sdk.create_lora_training_client(name="atlas")
    datum = Datum(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": ModelInput.from_ints([2, 3]), "weights": [1.0, 1.0]},
    )

    fb = training.forward_backward([datum]).result()
    step = training.optim_step(0.001).result()
    sample = training.sample("hello", return_logprobs=True).result()

    assert training.run_id == "run_1"
    assert fb.loss == 1.5
    assert step.step == 1
    assert sample.generated_logprobs == [-0.1]
    assert calls[0][2]["tenant_id"] == "tenant-a"
    assert calls[1][2]["data"][0]["loss_fn_inputs"]["target_tokens"] == {"tokens": [2, 3]}


def test_sdk_train_steps_returns_async_job():
    sdk = NemotronTinkerClient(base_url="http://unused")
    sdk._request = lambda method, path, payload=None: {
        "job": {
            "job_id": "job_1",
            "kind": "train_steps",
            "status": "queued",
            "created_at": "now",
            "updated_at": "now",
        }
    }

    result = sdk.train_steps(
        {"run_1": [Datum(model_input=ModelInput.from_ints([1]), loss_fn_inputs={})]},
        steps=1,
        learning_rate=0.001,
        run_async=True,
    ).result()

    assert result.job_id == "job_1"
    assert result.status == "queued"


def test_recipe_runner_builds_expected_command():
    command = build_command(
        {
            "kind": "nemotron_rl",
            "args": {
                "base_url": "http://127.0.0.1:18082",
                "steps": 12,
                "cache_dir": None,
            },
        },
        {"steps": 2, "tenant_id": "tenant-b"},
    )

    assert command[1].endswith("rl_lora_workload_client.py")
    assert "--steps" in command
    assert command[command.index("--steps") + 1] == "2"
    assert "--tenant-id" in command


def test_build_batch_pads_and_masks_weights():
    data = [
        Datum(
            model_input=ModelInput.from_ints([10, 11, 12]),
            loss_fn_inputs={"target_tokens": ModelInput.from_ints([10, 11, 12]), "weights": [0, 1, 1]},
        ),
        Datum(model_input=ModelInput.from_ints([20, 21]), loss_fn_inputs={}),
    ]

    input_ids, labels = _build_batch(data, pad_token_id=0, device=torch.device("cpu"))

    assert input_ids.tolist() == [[10, 11, 12], [20, 21, 0]]
    assert labels.tolist() == [[-100, 11, 12], [20, 21, -100]]


def test_example_sft_builder_uses_next_token_labels():
    datum = build_datum(_FakeTokenizer(), Example("ab", "cd"))

    input_tokens = datum["model_input"]["tokens"]
    target_tokens = datum["loss_fn_inputs"]["target_tokens"]["tokens"]
    weights = datum["loss_fn_inputs"]["weights"]

    assert input_tokens == [1, ord("a"), ord("b"), ord("c")]
    assert target_tokens == [ord("a"), ord("b"), ord("c"), ord("d")]
    assert weights == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo", "cispo", "dro"])
def test_rl_token_loss_modes_return_finite_weighted_loss(loss_fn):
    data = [
        Datum(
            model_input=ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={
                "weights": [1.0, 1.0],
                "advantages": [1.5, -0.5],
                "logprobs": [-1.1, -1.2],
            },
        ),
        Datum(
            model_input=ModelInput.from_ints([4, 5, 6]),
            loss_fn_inputs={
                "weights": [1.0, 0.0],
                "advantages": [0.25, 0.0],
                "logprobs": [-0.9, 0.0],
            },
        ),
    ]
    per_token_loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gathered = torch.tensor([[-1.0, -1.4], [-0.7, -0.1]])
    token_mask = torch.tensor([[True, True], [True, False]])

    token_loss, effective_mask, metrics = _rl_token_loss(
        loss_fn=loss_fn,
        per_token_loss=per_token_loss,
        gathered_logprobs=gathered,
        data=data,
        token_mask=token_mask,
        device=torch.device("cpu"),
        loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2, "beta": 0.05},
    )
    ratio = torch.exp(gathered - torch.tensor([[-1.1, -1.2], [-0.9, 0.0]]))
    advantages = torch.tensor([[1.5, -0.5], [0.25, 0.0]])
    weights = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    if loss_fn == "importance_sampling":
        expected = -ratio * advantages * weights
    elif loss_fn == "ppo":
        expected = -torch.minimum(ratio * advantages * weights, ratio.clamp(0.8, 1.2) * advantages * weights)
    elif loss_fn == "cispo":
        expected = -(ratio.clamp(0.8, 1.2) * gathered * advantages * weights)
    else:
        expected = (
            -((gathered * advantages) - 0.5 * 0.05 * (gathered - torch.tensor([[-1.1, -1.2], [-0.9, 0.0]])).pow(2))
            * weights
        )

    assert token_loss.shape == per_token_loss.shape
    assert torch.isfinite(token_loss[effective_mask]).all()
    assert effective_mask.tolist() == [[True, True], [True, False]]
    assert metrics["loss_weight_mean"] == 1.0
    assert torch.allclose(token_loss, expected)


def test_mixed_lora_rl_forward_backward_keeps_logprob_gradient():
    model = _TinyTrainableLogitModel()
    service = _training_service(model)
    data = [
        Datum(
            model_input=ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={
                "target_tokens": ModelInput.from_ints([2, 3, 4]),
                "weights": [1.0, 1.0, 1.0],
                "advantages": [1.0, -0.5, 0.25],
                "logprobs": [0.0, 0.0, 0.0],
            },
        )
    ]

    output = service.forward_backward_mixed({"adapter": data}, loss_fn="importance_sampling").result()

    assert output["adapter"].metrics["loss_fn"] == "importance_sampling"
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()


def test_mixed_lora_layer_routes_ranges_with_torch_fallback():
    base = torch.nn.Linear(3, 2, bias=False)
    base.weight.data.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    layer = MixedAdapterLinearLoRA(base, rank=1, alpha=2, dropout=0.0, lora_dtype=torch.float32, use_triton_lora=True)
    layer.add_adapter("atlas")
    layer.add_adapter("borealis")
    layer.lora_a["atlas"].data.copy_(torch.tensor([[1.0, 1.0, 0.0]]))
    layer.lora_b["atlas"].data.copy_(torch.tensor([[1.0], [0.0]]))
    layer.lora_a["borealis"].data.copy_(torch.tensor([[0.0, 1.0, 1.0]]))
    layer.lora_b["borealis"].data.copy_(torch.tensor([[0.0], [1.0]]))
    layer.set_active_ranges([("atlas", 0, 1), ("borealis", 1, 2)])

    out = layer(torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]))

    assert not layer._can_use_triton_lora(torch.zeros(1, 3))
    assert base.weight.requires_grad is False
    assert torch.allclose(out, torch.tensor([[12.0, 3.0], [7.0, 59.0]]))


def test_mixed_lora_sample_prefers_generate_fast_path():
    model = _GenerateModel()
    service = _sampling_service(model)

    output = service.sample("adapter", "prompt", SamplingParams(max_new_tokens=2, do_sample=False)).result()

    assert model.generate_calls == 1
    assert model.forward_calls == 0
    assert output.tokens == [1, 2, 7, 8]
    assert output.text == "1,2,7,8"


def test_mixed_lora_sample_falls_back_to_manual_loop_when_generate_fails():
    model = _GenerateModel(fail_generate=True)
    service = _sampling_service(model)

    output = service.sample("adapter", "prompt", SamplingParams(max_new_tokens=2, do_sample=False)).result()

    assert model.generate_calls == 1
    assert model.forward_calls == 2
    assert output.tokens == [1, 2, 6, 6]
    assert len(output.generated_logprobs) == 2


def test_gym_rollout_converter_builds_rl_datum_from_tinker_logprobs():
    row = {
        "responses_create_params": {"input": [{"role": "user", "content": "hello"}]},
        "response": {
            "output": [{"content": [{"type": "output_text", "text": "world"}]}],
            "tinker_rl": {
                "tokens": [10, 11, 12, 13],
                "prompt_token_count": 2,
                "generated_logprobs": [-0.7, -0.2],
            },
        },
        "reward": 1.0,
    }

    datums = convert_rollouts(
        [row],
        _FakeTokenizer(),
        reward_baseline=0.25,
        reward_scale=2.0,
        allow_missing_logprobs=False,
        max_tokens=None,
    )

    assert datums == [
        {
            "model_input": {"tokens": [10, 11, 12]},
            "loss_fn_inputs": {
                "target_tokens": {"tokens": [11, 12, 13]},
                "weights": [0.0, 1.0, 1.0],
                "logprobs": [0.0, -0.7, -0.2],
                "advantages": [0.0, 1.5, 1.5],
            },
        }
    ]


def test_mixed_lora_layer_grouped_backend_matches_loop_forward_backward():
    torch.manual_seed(1234)
    base_loop = torch.nn.Linear(8, 6, bias=False)
    base_grouped = torch.nn.Linear(8, 6, bias=False)
    base_grouped.weight.data.copy_(base_loop.weight)

    layer_loop = MixedAdapterLinearLoRA(
        base_loop,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        backend="loop",
    )
    layer_grouped = MixedAdapterLinearLoRA(
        base_grouped,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        backend="grouped",
    )
    for adapter_id in ["atlas", "borealis"]:
        layer_loop.add_adapter(adapter_id)
        layer_grouped.add_adapter(adapter_id)
        layer_loop.lora_a[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_loop.lora_b[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_grouped.lora_a[adapter_id].data.copy_(layer_loop.lora_a[adapter_id])
        layer_grouped.lora_b[adapter_id].data.copy_(layer_loop.lora_b[adapter_id])

    ranges = [("atlas", 0, 2), ("borealis", 2, 4)]
    layer_loop.set_active_ranges(ranges)
    layer_grouped.set_active_ranges(ranges)
    x_loop = torch.randn(4, 5, 8, requires_grad=True)
    x_grouped = x_loop.detach().clone().requires_grad_(True)

    out_loop = layer_loop(x_loop)
    out_grouped = layer_grouped(x_grouped)
    out_loop.pow(2).sum().backward()
    out_grouped.pow(2).sum().backward()

    assert layer_grouped._can_use_grouped_lora(x_grouped)
    assert torch.allclose(out_grouped, out_loop)
    assert torch.allclose(x_grouped.grad, x_loop.grad)
    for adapter_id in ["atlas", "borealis"]:
        assert torch.allclose(layer_grouped.lora_a[adapter_id].grad, layer_loop.lora_a[adapter_id].grad)
        assert torch.allclose(layer_grouped.lora_b[adapter_id].grad, layer_loop.lora_b[adapter_id].grad)


@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton LoRA parity")
def test_mixed_lora_layer_triton_bridge_matches_torch_forward_backward():
    torch.manual_seed(1234)
    base_ref = torch.nn.Linear(8, 6, bias=False, device="cuda")
    base_triton = torch.nn.Linear(8, 6, bias=False, device="cuda")
    base_triton.weight.data.copy_(base_ref.weight)

    layer_ref = MixedAdapterLinearLoRA(
        base_ref,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        use_triton_lora=False,
    )
    layer_triton = MixedAdapterLinearLoRA(
        base_triton,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        use_triton_lora=True,
    )
    for adapter_id in ["atlas", "borealis"]:
        layer_ref.add_adapter(adapter_id)
        layer_triton.add_adapter(adapter_id)
        layer_ref.lora_a[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_ref.lora_b[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_triton.lora_a[adapter_id].data.copy_(layer_ref.lora_a[adapter_id])
        layer_triton.lora_b[adapter_id].data.copy_(layer_ref.lora_b[adapter_id])

    ranges = [("atlas", 0, 1), ("borealis", 1, 3)]
    layer_ref.set_active_ranges(ranges)
    layer_triton.set_active_ranges(ranges)
    x_ref = torch.randn(3, 4, 8, device="cuda", requires_grad=True)
    x_triton = x_ref.detach().clone().requires_grad_(True)

    out_ref = layer_ref(x_ref)
    out_triton = layer_triton(x_triton)
    out_ref.pow(2).sum().backward()
    out_triton.pow(2).sum().backward()

    assert layer_triton._can_use_triton_lora(x_triton)
    assert torch.allclose(out_triton, out_ref, atol=2e-4, rtol=2e-4)
    assert torch.allclose(x_triton.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
    for adapter_id in ["atlas", "borealis"]:
        assert torch.allclose(
            layer_triton.lora_a[adapter_id].grad,
            layer_ref.lora_a[adapter_id].grad,
            atol=1e-3,
            rtol=1e-3,
        )
        assert torch.allclose(
            layer_triton.lora_b[adapter_id].grad,
            layer_ref.lora_b[adapter_id].grad,
            atol=1e-3,
            rtol=1e-3,
        )


@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for grouped Triton LoRA parity")
def test_mixed_lora_layer_grouped_triton_matches_torch_forward_backward():
    torch.manual_seed(1234)
    base_ref = torch.nn.Linear(8, 6, bias=False, device="cuda")
    base_grouped_triton = torch.nn.Linear(8, 6, bias=False, device="cuda")
    base_grouped_triton.weight.data.copy_(base_ref.weight)

    layer_ref = MixedAdapterLinearLoRA(
        base_ref,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        backend="loop",
    )
    layer_grouped_triton = MixedAdapterLinearLoRA(
        base_grouped_triton,
        rank=2,
        alpha=4,
        dropout=0.0,
        lora_dtype=torch.float32,
        backend="grouped_triton",
    )
    for adapter_id in ["atlas", "borealis", "cygnus"]:
        layer_ref.add_adapter(adapter_id)
        layer_grouped_triton.add_adapter(adapter_id)
        layer_ref.lora_a[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_ref.lora_b[adapter_id].data.normal_(mean=0.0, std=0.05)
        layer_grouped_triton.lora_a[adapter_id].data.copy_(layer_ref.lora_a[adapter_id])
        layer_grouped_triton.lora_b[adapter_id].data.copy_(layer_ref.lora_b[adapter_id])

    ranges = [("atlas", 0, 1), ("borealis", 1, 3), ("cygnus", 3, 4)]
    layer_ref.set_active_ranges(ranges)
    layer_grouped_triton.set_active_ranges(ranges)
    x_ref = torch.randn(4, 5, 8, device="cuda", requires_grad=True)
    x_grouped_triton = x_ref.detach().clone().requires_grad_(True)

    out_ref = layer_ref(x_ref)
    out_grouped_triton = layer_grouped_triton(x_grouped_triton)
    out_ref.pow(2).sum().backward()
    out_grouped_triton.pow(2).sum().backward()

    assert layer_grouped_triton._can_use_grouped_triton_lora(x_grouped_triton)
    assert torch.allclose(out_grouped_triton, out_ref, atol=2e-4, rtol=2e-4)
    assert torch.allclose(x_grouped_triton.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
    for adapter_id in ["atlas", "borealis", "cygnus"]:
        assert torch.allclose(
            layer_grouped_triton.lora_a[adapter_id].grad,
            layer_ref.lora_a[adapter_id].grad,
            atol=1e-3,
            rtol=1e-3,
        )
        assert torch.allclose(
            layer_grouped_triton.lora_b[adapter_id].grad,
            layer_ref.lora_b[adapter_id].grad,
            atol=1e-3,
            rtol=1e-3,
        )


@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for grouped LoRA Triton reductions")
def test_grouped_lora_da_db_kernels_match_torch_reductions():
    torch.manual_seed(1234)
    x = torch.randn(7, 8, device="cuda", dtype=torch.float32)
    grad_out = torch.randn(7, 6, device="cuda", dtype=torch.float32)
    adapter_indices = torch.tensor([0, 1, 0, 2, 1, 2, 2], device="cuda", dtype=torch.long)
    lora_a_bank = torch.randn(3, 2, 8, device="cuda", dtype=torch.float32) * 0.05
    lora_b_bank = torch.randn(3, 6, 2, device="cuda", dtype=torch.float32) * 0.05
    scale = 2.0

    grad_a, grad_b = grouped_lora_da_db_wrapper(x, grad_out, adapter_indices, lora_a_bank, lora_b_bank, scale)

    expected_grad_a = torch.zeros_like(lora_a_bank)
    expected_grad_b = torch.zeros_like(lora_b_bank)
    for row_idx, adapter_idx in enumerate(adapter_indices.tolist()):
        hidden = torch.matmul(lora_a_bank[adapter_idx], x[row_idx])
        grad_hidden = torch.matmul(grad_out[row_idx], lora_b_bank[adapter_idx]) * scale
        expected_grad_a[adapter_idx] += grad_hidden[:, None] * x[row_idx][None, :]
        expected_grad_b[adapter_idx] += grad_out[row_idx][:, None] * hidden[None, :] * scale

    assert torch.allclose(grad_a, expected_grad_a, atol=1e-5, rtol=1e-5)
    assert torch.allclose(grad_b, expected_grad_b, atol=1e-5, rtol=1e-5)


def test_service_reuses_worker_for_matching_base_and_lora_config(monkeypatch, tmp_path):
    created_workers = []

    class FakeWorker:
        def __init__(self, **kwargs):
            self.base_model = kwargs["base_model"]
            self.lora_config = kwargs["lora_config"]
            self.device = torch.device("cpu")
            created_workers.append(self)

        def new_adapter_state(self):
            return {}

    monkeypatch.setattr(tinker_client, "SharedBaseModelWorker", FakeWorker)

    service = tinker_client.ServiceClient(scratch_dir=tmp_path)
    first = service.create_lora_training_client("tiny-model", rank=4, force_hf=True)
    second = service.create_lora_training_client("tiny-model", rank=4, force_hf=True)

    assert first.worker is second.worker
    assert len(created_workers) == 1


def test_service_separates_workers_for_different_lora_config(monkeypatch, tmp_path):
    created_workers = []

    class FakeWorker:
        def __init__(self, **kwargs):
            self.base_model = kwargs["base_model"]
            self.lora_config = kwargs["lora_config"]
            self.device = torch.device("cpu")
            created_workers.append(self)

        def new_adapter_state(self):
            return {}

    monkeypatch.setattr(tinker_client, "SharedBaseModelWorker", FakeWorker)

    service = tinker_client.ServiceClient(scratch_dir=tmp_path)
    first = service.create_lora_training_client("tiny-model", rank=4, force_hf=True)
    second = service.create_lora_training_client("tiny-model", rank=8, force_hf=True)

    assert first.worker is not second.worker
    assert len(created_workers) == 2


def test_mixed_lora_checkpoint_validation_rejects_wrong_base_model(tmp_path):
    service = MixedLoraServiceClient.__new__(MixedLoraServiceClient)
    service.base_model = "expected-model"
    service.lora_config = type(
        "FakeLoraConfig",
        (),
        {"rank": 16, "alpha": None, "target_modules": []},
    )()

    try:
        service._validate_checkpoint_config(
            {
                "base_model": "other-model",
                "rank": 16,
                "alpha": None,
                "target_modules": ["*_proj"],
            },
            tmp_path,
        )
    except ValueError as exc:
        assert "base_model" in str(exc)
    else:
        raise AssertionError("Expected checkpoint validation to reject mismatched base model")
