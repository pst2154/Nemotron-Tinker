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
import pathlib
import threading
import uuid
from typing import Optional

import torch
import torch.nn.functional as F

from nemotron_tinker.future import APIFuture
from nemotron_tinker.types import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    LoraConfig,
    ModelInput,
    OptimStepResponse,
    SampleResponse,
    SamplingParams,
    SaveStateResponse,
)


def _tokens_from_value(value: object) -> list[int]:
    if isinstance(value, ModelInput):
        return list(value.tokens)
    if isinstance(value, list):
        return [int(token) for token in value]
    raise TypeError(f"Expected ModelInput or list[int], got {type(value)!r}")


def _pad_2d(sequences: list[list[int]], pad_value: int, device: torch.device) -> torch.Tensor:
    max_length = max(len(sequence) for sequence in sequences)
    tensor = torch.full((len(sequences), max_length), pad_value, dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        tensor[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
    return tensor


def _build_batch(data: list[Datum], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not data:
        raise ValueError("forward_backward requires at least one Datum")

    input_sequences = [datum.model_input.tokens for datum in data]
    input_ids = _pad_2d(input_sequences, pad_token_id, device)
    labels = torch.full_like(input_ids, -100)

    for row, datum in enumerate(data):
        target_value = datum.loss_fn_inputs.get("target_tokens", datum.model_input)
        target_tokens = _tokens_from_value(target_value)
        length = min(len(target_tokens), input_ids.shape[1])
        labels[row, :length] = torch.tensor(target_tokens[:length], dtype=torch.long, device=device)

        weights = datum.loss_fn_inputs.get("weights")
        if weights is not None:
            if len(weights) < length:
                raise ValueError("weights must be at least as long as target_tokens")
            weight_tensor = torch.tensor(weights[:length], dtype=torch.float32, device=device)
            labels[row, :length] = labels[row, :length].masked_fill(weight_tensor == 0, -100)

    return input_ids, labels


def _adapter_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "lora_" in name or "lora_magnitude" in name
    }


def _adapter_grad_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().cpu()
        for name, param in model.named_parameters()
        if ("lora_" in name or "lora_magnitude" in name) and param.grad is not None
    }


def _load_adapter_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    named_parameters = dict(model.named_parameters())
    missing = sorted(set(state_dict) - set(named_parameters))
    if missing:
        raise KeyError(f"Adapter state contains unknown parameter(s): {missing[:5]}")
    with torch.no_grad():
        for name, value in state_dict.items():
            named_parameters[name].copy_(
                value.to(device=named_parameters[name].device, dtype=named_parameters[name].dtype)
            )


def _load_adapter_grad_state_dict(model: torch.nn.Module, grad_state_dict: dict[str, torch.Tensor]) -> None:
    named_parameters = dict(model.named_parameters())
    for name, value in grad_state_dict.items():
        param = named_parameters[name]
        param.grad = value.to(device=param.device, dtype=param.dtype).clone()


def _clone_tensor_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state_dict.items()}


def _adapter_signature(lora_config: LoraConfig) -> tuple:
    return (
        lora_config.rank,
        lora_config.alpha,
        lora_config.dropout,
        tuple(lora_config.target_modules),
        tuple(lora_config.exclude_modules),
        lora_config.match_all_linear,
        lora_config.train_unembed,
    )


class ServiceClient:
    """Entry point for creating Tinker-like AutoModel training clients."""

    def __init__(self, scratch_dir: str | pathlib.Path = "/tmp/nemotron_tinker"):
        self.scratch_dir = pathlib.Path(scratch_dir)
        self._workers: dict[tuple, SharedBaseModelWorker] = {}
        self._lock = threading.RLock()

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 16,
        *,
        seed: Optional[int] = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = False,
        target_modules: Optional[list[str]] = None,
        device: Optional[str] = None,
        torch_dtype: str | torch.dtype = "bfloat16",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        force_hf: bool = False,
    ) -> "TrainingClient":
        """Create a LoRA training client for one base model."""
        lora_config = LoraConfig(rank=rank, target_modules=target_modules or [], train_unembed=train_unembed)
        if not train_attn or not train_mlp:
            selected: list[str] = []
            if train_attn:
                selected.extend(["*q_proj", "*k_proj", "*v_proj", "*o_proj"])
            if train_mlp:
                selected.extend(["*gate_proj", "*up_proj", "*down_proj"])
            lora_config.target_modules = selected
        if train_unembed:
            lora_config.target_modules.append("*lm_head")

        worker_key = (
            base_model,
            device,
            str(torch_dtype),
            cache_dir,
            trust_remote_code,
            force_hf,
            _adapter_signature(lora_config),
        )
        with self._lock:
            worker = self._workers.get(worker_key)
            if worker is None:
                worker = SharedBaseModelWorker(
                    base_model=base_model,
                    lora_config=lora_config,
                    seed=seed,
                    device=device,
                    torch_dtype=torch_dtype,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code,
                    force_hf=force_hf,
                )
                self._workers[worker_key] = worker

        return TrainingClient(worker=worker, scratch_dir=self.scratch_dir, seed=seed)


class SharedBaseModelWorker:
    """One resident base model plus one swappable LoRA module structure."""

    def __init__(
        self,
        *,
        base_model: str,
        lora_config: LoraConfig,
        seed: Optional[int],
        device: Optional[str],
        torch_dtype: str | torch.dtype,
        cache_dir: Optional[str],
        trust_remote_code: bool,
        force_hf: bool,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        self.base_model = base_model
        self.lora_config = lora_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lock = threading.RLock()

        from nemo_automodel._transformers import NeMoAutoModelForCausalLM
        from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
        from nemo_automodel.components._peft.lora import PeftConfig

        peft_config = PeftConfig(
            target_modules=lora_config.target_modules,
            exclude_modules=lora_config.exclude_modules,
            match_all_linear=lora_config.match_all_linear,
            dim=lora_config.rank,
            alpha=lora_config.alpha or lora_config.rank * 2,
            dropout=lora_config.dropout,
            use_triton=torch.cuda.is_available(),
        )

        self.tokenizer = NeMoAutoTokenizer.from_pretrained(
            base_model,
            force_hf=True,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = NeMoAutoModelForCausalLM.from_pretrained(
            base_model,
            peft_config=peft_config,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
            force_hf=force_hf,
        )
        self.model.to(self.device)
        self.model.train()
        self.initial_adapter_state = self.snapshot_adapter_state()

    def snapshot_adapter_state(self) -> dict[str, torch.Tensor]:
        """Return the current LoRA adapter weights as a CPU tensor dict."""
        return _adapter_state_dict(self.model)

    def new_adapter_state(self) -> dict[str, torch.Tensor]:
        """Return the initial LoRA adapter weights for a new virtual adapter."""
        return _clone_tensor_dict(self.initial_adapter_state)

    def load_adapter_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load one virtual adapter into the resident model."""
        _load_adapter_state_dict(self.model, state_dict)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return LoRA parameters used by per-client optimizers."""
        return [param for param in self.model.parameters() if param.requires_grad]


class TrainingClient:
    """One virtual LoRA adapter attached to a shared base-model worker."""

    def __init__(self, *, worker: SharedBaseModelWorker, scratch_dir: pathlib.Path, seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)

        self.worker = worker
        self.base_model = worker.base_model
        self.lora_config = worker.lora_config
        self.scratch_dir = scratch_dir
        self.device = worker.device
        self.adapter_id = f"adapter-{uuid.uuid4().hex[:12]}"
        self.adapter_state = worker.new_adapter_state()
        self.pending_grad_state: dict[str, torch.Tensor] = {}
        self.step = 0
        self.optimizer: Optional[torch.optim.AdamW] = None

    def get_tokenizer(self):
        """Return the tokenizer associated with the base model."""
        return self.worker.tokenizer

    def forward_backward(self, data: list[Datum], loss_fn: str = "cross_entropy") -> APIFuture[ForwardBackwardOutput]:
        """Compute loss and accumulate gradients for one batch."""
        if loss_fn != "cross_entropy":
            raise NotImplementedError("Prototype v0 only supports loss_fn='cross_entropy'")

        with self.worker.lock:
            self.worker.load_adapter_state(self.adapter_state)
            self.worker.model.zero_grad(set_to_none=True)

            input_ids, labels = _build_batch(data, self.worker.tokenizer.pad_token_id, self.device)
            attention_mask = input_ids.ne(self.worker.tokenizer.pad_token_id).to(torch.long)

            self.worker.model.train()
            outputs = self.worker.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
            loss.backward()

            self.pending_grad_state = _adapter_grad_state_dict(self.worker.model)
            self.adapter_state = self.worker.snapshot_adapter_state()

            with torch.no_grad():
                target_logprobs = F.log_softmax(shift_logits, dim=-1)
                safe_labels = shift_labels.clamp_min(0).unsqueeze(-1)
                gathered = target_logprobs.gather(-1, safe_labels).squeeze(-1)
                gathered = gathered.masked_fill(shift_labels.eq(-100), 0.0)
                token_count = shift_labels.ne(-100).sum().item()
                loss_value = float(loss.detach().cpu())

            self.worker.model.zero_grad(set_to_none=True)

        output = ForwardBackwardOutput(
            loss=loss_value,
            metrics={"loss": loss_value, "num_label_tokens": float(token_count)},
            loss_fn_outputs=[{"logprobs": row.detach().cpu().tolist()} for row in gathered],
        )
        return APIFuture(output)

    async def forward_backward_async(
        self, data: list[Datum], loss_fn: str = "cross_entropy"
    ) -> APIFuture[ForwardBackwardOutput]:
        """Async-shaped wrapper for `forward_backward`."""
        return self.forward_backward(data, loss_fn)

    def optim_step(self, adam_params: AdamParams) -> APIFuture[OptimStepResponse]:
        """Apply accumulated gradients to LoRA parameters."""
        if not self.pending_grad_state:
            raise RuntimeError("optim_step requires a preceding forward_backward call with gradients")

        with self.worker.lock:
            self.worker.load_adapter_state(self.adapter_state)
            self.worker.model.zero_grad(set_to_none=True)

            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW(
                    self.worker.trainable_parameters(),
                    lr=adam_params.learning_rate,
                    betas=adam_params.betas,
                    eps=adam_params.eps,
                    weight_decay=adam_params.weight_decay,
                )
            else:
                for group in self.optimizer.param_groups:
                    group["lr"] = adam_params.learning_rate
                    group["weight_decay"] = adam_params.weight_decay
                    group["betas"] = adam_params.betas
                    group["eps"] = adam_params.eps

            _load_adapter_grad_state_dict(self.worker.model, self.pending_grad_state)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.adapter_state = self.worker.snapshot_adapter_state()
            self.pending_grad_state = {}
            self.step += 1

        return APIFuture(OptimStepResponse(step=self.step, learning_rate=adam_params.learning_rate))

    async def optim_step_async(self, adam_params: AdamParams) -> APIFuture[OptimStepResponse]:
        """Async-shaped wrapper for `optim_step`."""
        return self.optim_step(adam_params)

    def save_state(self, name: str) -> APIFuture[SaveStateResponse]:
        """Save LoRA adapter weights and optimizer state under scratch/checkpoints."""
        output_dir = self.scratch_dir / "checkpoints" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(_clone_tensor_dict(self.adapter_state), output_dir / "adapter_model.pt")
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), output_dir / "optimizer.pt")
        with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "base_model": self.base_model,
                    "rank": self.lora_config.rank,
                    "alpha": self.lora_config.alpha,
                    "target_modules": self.lora_config.target_modules,
                    "adapter_id": self.adapter_id,
                    "step": self.step,
                },
                fp,
                indent=2,
            )
        return APIFuture(SaveStateResponse(path=str(output_dir)))

    def save_weights_and_get_sampling_client(self, name: str) -> "SamplingClient":
        """Save current state and return a sampler over the in-memory model."""
        self.save_state(name).result()
        return SamplingClient(self.worker, _clone_tensor_dict(self.adapter_state))


class SamplingClient:
    """Simple generation client for evaluating the current training session."""

    def __init__(self, worker: SharedBaseModelWorker, adapter_state: dict[str, torch.Tensor]):
        self.worker = worker
        self.adapter_state = adapter_state

    def sample(self, prompt: str, params: Optional[SamplingParams] = None) -> APIFuture[SampleResponse]:
        """Generate one completion from a text prompt."""
        params = params or SamplingParams()
        with self.worker.lock:
            self.worker.load_adapter_state(self.adapter_state)
            encoded = self.worker.tokenizer(prompt, return_tensors="pt").to(self.worker.device)
            self.worker.model.eval()
            with torch.no_grad():
                output_ids = self.worker.model.generate(
                    **encoded,
                    max_new_tokens=params.max_new_tokens,
                    do_sample=params.do_sample,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    pad_token_id=self.worker.tokenizer.pad_token_id,
                )[0]
            text = self.worker.tokenizer.decode(output_ids, skip_special_tokens=True)
        return APIFuture(SampleResponse(tokens=output_ids.detach().cpu().tolist(), text=text))
