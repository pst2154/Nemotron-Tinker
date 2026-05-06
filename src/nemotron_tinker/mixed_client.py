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
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemotron_tinker.client import _build_batch
from nemotron_tinker.future import APIFuture
from nemotron_tinker.types import (
    AdamParams,
    Datum,
    DetachAdapterResponse,
    ForwardBackwardOutput,
    LoraConfig,
    OptimStepResponse,
    SampleResponse,
    SamplingParams,
    SaveStateResponse,
)
from nemo_automodel.shared.import_utils import safe_import_from

HAS_LORA_TRITON_FUNCTION, LoRATritonFunction = safe_import_from(
    "nemo_automodel.components._peft.lora",
    "LoRATritonFunction",
)
_, HAVE_TRITON = safe_import_from("nemo_automodel.components._peft.lora_kernel", "HAVE_TRITON", alt=False)
HAS_GROUPED_LORA_TRITON, GroupedLoRATritonFunction = safe_import_from(
    "nemotron_tinker.grouped_lora_kernel",
    "GroupedLoRATritonFunction",
)
_, HAVE_GROUPED_LORA_TRITON = safe_import_from(
    "nemotron_tinker.grouped_lora_kernel",
    "HAVE_GROUPED_LORA_TRITON",
    alt=False,
)

MixedLoraBackend = Literal["loop", "grouped", "triton", "grouped_triton"]
SUPPORTED_LOSS_FNS = {"cross_entropy", "importance_sampling", "ppo", "cispo", "dro"}


def _wildcard_to_regex(pattern: str) -> re.Pattern:
    return re.compile("^" + re.escape(pattern).replace("\\*", ".*") + "$")


def _matches_target(name: str, target_modules: list[str]) -> bool:
    targets = target_modules or ["*_proj"]
    short_name = name.rsplit(".", 1)[-1]
    for pattern in targets:
        if short_name == pattern or name == pattern:
            return True
        if _wildcard_to_regex(pattern).match(name) or _wildcard_to_regex(pattern).match(short_name):
            return True
    return False


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _as_float_sequence(value: object, *, name: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list[float]")
    return [float(item) for item in value]


def _pad_float_field(
    data: list[Datum],
    field_name: str,
    *,
    default_value: float,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.full((len(data), max_length), default_value, dtype=torch.float32, device=device)
    for row, datum in enumerate(data):
        value = datum.loss_fn_inputs.get(field_name)
        if value is None:
            continue
        sequence = _as_float_sequence(value, name=field_name)
        length = min(len(sequence), max_length)
        tensor[row, :length] = torch.tensor(sequence[:length], dtype=torch.float32, device=device)
    return tensor


def _rl_token_loss(
    *,
    loss_fn: str,
    per_token_loss: torch.Tensor,
    gathered_logprobs: torch.Tensor,
    data: list[Datum],
    token_mask: torch.Tensor,
    device: torch.device,
    loss_fn_config: Optional[dict[str, float]] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return token loss, effective mask, and metrics for SFT/RL-style objectives."""
    loss_fn_config = loss_fn_config or {}
    max_length = token_mask.shape[1]
    weights = _pad_float_field(data, "weights", default_value=1.0, max_length=max_length, device=device)
    effective_mask = token_mask & weights.ne(0)
    token_weights = weights.masked_fill(~effective_mask, 0.0)
    metrics = {
        "loss_weight_mean": float(token_weights[effective_mask].mean().detach().cpu()) if effective_mask.any() else 0.0
    }

    if loss_fn == "cross_entropy":
        return per_token_loss * token_weights, effective_mask, metrics

    for datum in data:
        if "logprobs" not in datum.loss_fn_inputs:
            raise ValueError(f"loss_fn={loss_fn!r} requires loss_fn_inputs['logprobs']")
        if "advantages" not in datum.loss_fn_inputs:
            raise ValueError(f"loss_fn={loss_fn!r} requires loss_fn_inputs['advantages']")

    sampling_logprobs = _pad_float_field(
        data,
        "logprobs",
        default_value=0.0,
        max_length=max_length,
        device=device,
    )
    advantages = _pad_float_field(
        data,
        "advantages",
        default_value=1.0,
        max_length=max_length,
        device=device,
    )
    clip_low = float(loss_fn_config.get("clip_low_threshold", data[0].loss_fn_inputs.get("clip_low_threshold", 0.8)))
    clip_high = float(loss_fn_config.get("clip_high_threshold", data[0].loss_fn_inputs.get("clip_high_threshold", 1.2)))
    beta = float(
        loss_fn_config.get("beta", data[0].loss_fn_inputs.get("beta", data[0].loss_fn_inputs.get("dro_beta", 0.1)))
    )

    ratio = torch.exp((gathered_logprobs - sampling_logprobs).clamp(-20.0, 20.0))
    weighted_advantages = advantages * token_weights

    if loss_fn == "importance_sampling":
        token_loss = -ratio * weighted_advantages
    elif loss_fn == "ppo":
        clipped_ratio = ratio.clamp(clip_low, clip_high)
        unclipped = ratio * weighted_advantages
        clipped = clipped_ratio * weighted_advantages
        token_loss = -torch.minimum(unclipped, clipped)
    elif loss_fn == "cispo":
        clipped_ratio = ratio.clamp(clip_low, clip_high)
        token_loss = -(clipped_ratio.detach() * gathered_logprobs * weighted_advantages)
    elif loss_fn == "dro":
        quadratic_term = (gathered_logprobs - sampling_logprobs).pow(2)
        objective = gathered_logprobs * advantages - 0.5 * beta * quadratic_term
        token_loss = -objective * token_weights
    else:
        raise NotImplementedError(f"Unsupported loss_fn={loss_fn!r}; expected one of {sorted(SUPPORTED_LOSS_FNS)}")

    metrics["importance_ratio_mean"] = (
        float(ratio[effective_mask].mean().detach().cpu()) if effective_mask.any() else 0.0
    )
    metrics["clip_low_threshold"] = clip_low
    metrics["clip_high_threshold"] = clip_high
    if loss_fn == "dro":
        metrics["beta"] = beta
    return token_loss, effective_mask, metrics


class MixedAdapterLinearLoRA(nn.Module):
    """Linear layer with multiple resident LoRA adapters selected by batch ranges."""

    def __init__(
        self,
        base_linear: nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float,
        lora_dtype: torch.dtype,
        backend: MixedLoraBackend = "loop",
        use_triton_lora: bool = False,
    ):
        super().__init__()
        if use_triton_lora:
            backend = "triton"
        if backend not in {"loop", "grouped", "triton", "grouped_triton"}:
            raise ValueError(f"Unknown mixed LoRA backend: {backend}")
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = dropout
        self.lora_dtype = lora_dtype
        self.backend = backend
        self.lora_a = nn.ParameterDict()
        self.lora_b = nn.ParameterDict()
        self.active_ranges: list[tuple[str, int, int]] = []

        for param in self.base_linear.parameters():
            param.requires_grad_(False)

    @property
    def in_features(self) -> int:
        """Input feature size."""
        return self.base_linear.in_features

    @property
    def out_features(self) -> int:
        """Output feature size."""
        return self.base_linear.out_features

    def add_adapter(self, adapter_id: str) -> None:
        """Add one resident LoRA adapter to this linear layer."""
        if adapter_id in self.lora_a:
            return
        lora_a = nn.Parameter(
            torch.empty(self.rank, self.in_features, device=self.base_linear.weight.device, dtype=self.lora_dtype)
        )
        lora_b = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=self.base_linear.weight.device, dtype=self.lora_dtype)
        )
        nn.init.kaiming_uniform_(lora_a, a=5**0.5)
        self.lora_a[adapter_id] = lora_a
        self.lora_b[adapter_id] = lora_b

    def remove_adapter(self, adapter_id: str) -> None:
        """Remove one resident LoRA adapter from this linear layer."""
        if adapter_id not in self.lora_a:
            return
        del self.lora_a[adapter_id]
        del self.lora_b[adapter_id]
        self.active_ranges = [active_range for active_range in self.active_ranges if active_range[0] != adapter_id]

    def set_active_ranges(self, ranges: list[tuple[str, int, int]]) -> None:
        """Set batch row ranges for the next mixed-adapter forward pass."""
        self.active_ranges = ranges

    def adapter_parameters(self, adapter_id: str) -> list[nn.Parameter]:
        """Return trainable parameters for one adapter."""
        return [self.lora_a[adapter_id], self.lora_b[adapter_id]]

    def adapter_state_dict(self, adapter_id: str, prefix: str) -> dict[str, torch.Tensor]:
        """Return one adapter's state for this layer."""
        return {
            f"{prefix}.lora_a": self.lora_a[adapter_id].detach().cpu(),
            f"{prefix}.lora_b": self.lora_b[adapter_id].detach().cpu(),
        }

    def load_adapter_state_dict(self, adapter_id: str, prefix: str, state: dict[str, torch.Tensor]) -> None:
        """Load one adapter's weights for this layer."""
        self.lora_a[adapter_id].data.copy_(state[f"{prefix}.lora_a"].to(self.lora_a[adapter_id].device))
        self.lora_b[adapter_id].data.copy_(state[f"{prefix}.lora_b"].to(self.lora_b[adapter_id].device))

    def _can_use_triton_lora(self, x: torch.Tensor) -> bool:
        """Return whether the existing single-adapter Triton LoRA kernel can handle this slice."""
        return bool(
            self.backend == "triton" and HAS_LORA_TRITON_FUNCTION and HAVE_TRITON and x.is_cuda and x.dim() in {2, 3}
        )

    def _can_use_grouped_lora(self, x: torch.Tensor) -> bool:
        """Return whether the vectorized grouped PyTorch backend can handle this batch."""
        return self.backend == "grouped" and self.dropout == 0 and x.dim() in {2, 3}

    def _can_use_grouped_triton_lora(self, x: torch.Tensor) -> bool:
        """Return whether the grouped mixed-adapter Triton kernel can handle this batch."""
        return bool(
            self.backend == "grouped_triton"
            and self.dropout == 0
            and HAS_GROUPED_LORA_TRITON
            and HAVE_GROUPED_LORA_TRITON
            and x.is_cuda
            and x.dim() in {2, 3}
        )

    def _active_row_adapter_ids(self, batch_size: int) -> list[str]:
        """Expand active ranges into one adapter id per leading batch row."""
        row_adapter_ids = [""] * batch_size
        for adapter_id, start_idx, end_idx in self.active_ranges:
            if adapter_id not in self.lora_a:
                continue
            for row_idx in range(start_idx, end_idx):
                if row_idx < 0 or row_idx >= batch_size:
                    raise ValueError(
                        f"Active range ({start_idx}, {end_idx}) for adapter {adapter_id!r} "
                        f"is outside batch size {batch_size}"
                    )
                row_adapter_ids[row_idx] = adapter_id
        return row_adapter_ids

    def _grouped_adapter_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all active adapter deltas in one vectorized PyTorch operation."""
        row_adapter_ids = self._active_row_adapter_ids(x.shape[0])
        active_rows = [idx for idx, adapter_id in enumerate(row_adapter_ids) if adapter_id]
        if not active_rows:
            return torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=self.lora_dtype)

        row_index = torch.tensor(active_rows, device=x.device, dtype=torch.long)
        x_active = x.index_select(0, row_index).to(self.lora_dtype)
        lora_a = torch.stack([self.lora_a[row_adapter_ids[idx]] for idx in active_rows])
        lora_b = torch.stack([self.lora_b[row_adapter_ids[idx]] for idx in active_rows])

        if x_active.dim() == 2:
            hidden = torch.bmm(x_active.unsqueeze(1), lora_a.transpose(1, 2)).squeeze(1)
            active_delta = torch.bmm(hidden.unsqueeze(1), lora_b.transpose(1, 2)).squeeze(1)
        else:
            hidden = torch.einsum("bsh,brh->bsr", x_active, lora_a)
            active_delta = torch.einsum("bsr,bor->bso", hidden, lora_b)

        delta = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=self.lora_dtype)
        delta.index_copy_(0, row_index, active_delta * self.scale)
        return delta

    def _grouped_triton_adapter_delta(self, x: torch.Tensor, result_dtype: torch.dtype) -> torch.Tensor:
        """Compute all active adapter deltas in one grouped Triton launch."""
        row_adapter_ids = self._active_row_adapter_ids(x.shape[0])
        active_rows = [idx for idx, adapter_id in enumerate(row_adapter_ids) if adapter_id]
        if not active_rows:
            return torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=result_dtype)

        active_adapter_ids = sorted({row_adapter_ids[idx] for idx in active_rows})
        adapter_id_to_bank_idx = {adapter_id: idx for idx, adapter_id in enumerate(active_adapter_ids)}
        row_index = torch.tensor(active_rows, device=x.device, dtype=torch.long)
        adapter_indices = torch.tensor(
            [adapter_id_to_bank_idx[row_adapter_ids[idx]] for idx in active_rows],
            device=x.device,
            dtype=torch.long,
        )
        x_active = x.index_select(0, row_index).to(self.lora_dtype)
        lora_a_bank = torch.stack([self.lora_a[adapter_id] for adapter_id in active_adapter_ids])
        lora_b_bank = torch.stack([self.lora_b[adapter_id] for adapter_id in active_adapter_ids])
        active_delta = GroupedLoRATritonFunction.apply(
            x_active,
            adapter_indices,
            lora_a_bank,
            lora_b_bank,
            self.scale,
            result_dtype,
        )
        delta = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=result_dtype)
        delta.index_copy_(0, row_index, active_delta.to(result_dtype))
        return delta

    def _adapter_delta(self, adapter_id: str, x: torch.Tensor, result_dtype: torch.dtype) -> torch.Tensor:
        """Compute one adapter's LoRA delta with the selected backend."""
        x_lora = x.to(self.lora_dtype)
        if self._can_use_triton_lora(x_lora):
            return LoRATritonFunction.apply(
                x_lora,
                self.lora_a[adapter_id],
                self.lora_b[adapter_id],
                self.scale,
                result_dtype,
            )
        lora_out = F.linear(x_lora, self.lora_a[adapter_id])
        return F.linear(lora_out, self.lora_b[adapter_id]) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the base linear layer and the selected LoRA adapters."""
        result = self.base_linear(x)
        if not self.active_ranges:
            return result

        if self._can_use_grouped_lora(x):
            return result + self._grouped_adapter_delta(x).to(result.dtype)

        if self._can_use_grouped_triton_lora(x):
            return result + self._grouped_triton_adapter_delta(x, result.dtype)

        for adapter_id, start_idx, end_idx in self.active_ranges:
            if adapter_id not in self.lora_a:
                continue
            x_slice = x[start_idx:end_idx]
            if self.training and self.dropout > 0:
                x_slice = F.dropout(x_slice, p=self.dropout, training=True)
            lora_out = self._adapter_delta(adapter_id, x_slice, result.dtype)
            result[start_idx:end_idx] = result[start_idx:end_idx] + lora_out.to(result.dtype)
        return result


@dataclass
class MixedAdapterHandle:
    """One resident adapter plus its optimizer state."""

    adapter_id: str
    optimizer: Optional[torch.optim.AdamW] = None
    step: int = 0


class MixedLoraServiceClient:
    """Single-node mLoRA-style service with multiple resident adapters."""

    def __init__(
        self,
        *,
        base_model: str,
        scratch_dir: str | pathlib.Path = "/tmp/nemotron_tinker",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: str | torch.dtype = "bfloat16",
        trust_remote_code: bool = False,
        attn_implementation: str = "sdpa",
        lora_config: Optional[LoraConfig] = None,
        mixed_lora_backend: MixedLoraBackend = "loop",
        use_triton_lora: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if use_triton_lora:
            mixed_lora_backend = "triton"
        if mixed_lora_backend not in {"loop", "grouped", "triton", "grouped_triton"}:
            raise ValueError(f"Unknown mixed LoRA backend: {mixed_lora_backend}")
        self.base_model = base_model
        self.scratch_dir = pathlib.Path(scratch_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lora_config = lora_config or LoraConfig(rank=16)
        self.mixed_lora_backend = mixed_lora_backend
        self.adapters: dict[str, MixedAdapterHandle] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.train()
        self.mixed_lora_layers = self._patch_target_linears()

    def _patch_target_linears(self) -> dict[str, MixedAdapterLinearLoRA]:
        layers = {}
        rank = self.lora_config.rank
        alpha = self.lora_config.alpha or rank * 2
        lora_dtype = next(self.model.parameters()).dtype
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not _matches_target(name, self.lora_config.target_modules):
                continue
            if self.lora_config.exclude_modules and _matches_target(name, self.lora_config.exclude_modules):
                continue
            parent, child_name = _get_parent_module(self.model, name)
            mixed_layer = MixedAdapterLinearLoRA(
                module,
                rank=rank,
                alpha=alpha,
                dropout=self.lora_config.dropout,
                lora_dtype=lora_dtype,
                backend=self.mixed_lora_backend,
            )
            setattr(parent, child_name, mixed_layer)
            layers[name] = mixed_layer
        if not layers:
            raise ValueError("No target nn.Linear modules were found for mixed LoRA")
        return layers

    def create_lora_training_client(
        self, *, adapter_id: Optional[str] = None, checkpoint_path: Optional[str | pathlib.Path] = None
    ) -> "MixedLoraTrainingClient":
        """Create one resident LoRA adapter training client."""
        adapter_id = adapter_id or f"adapter_{uuid.uuid4().hex[:12]}"
        if adapter_id in self.adapters:
            raise ValueError(f"Adapter already exists: {adapter_id}")
        for layer in self.mixed_lora_layers.values():
            layer.add_adapter(adapter_id)
        handle = MixedAdapterHandle(adapter_id=adapter_id)
        self.adapters[adapter_id] = handle
        if checkpoint_path is not None:
            self.load_adapter_state(adapter_id, checkpoint_path)
        return MixedLoraTrainingClient(service=self, handle=handle)

    def _set_active_ranges(self, ranges: list[tuple[str, int, int]]) -> None:
        for layer in self.mixed_lora_layers.values():
            layer.set_active_ranges(ranges)

    def adapter_parameters(self, adapter_id: str) -> list[nn.Parameter]:
        """Return trainable parameters for one adapter."""
        params = []
        for layer in self.mixed_lora_layers.values():
            params.extend(layer.adapter_parameters(adapter_id))
        return params

    def adapter_state_dict(self, adapter_id: str) -> dict[str, torch.Tensor]:
        """Return one adapter's resident weights."""
        state = {}
        for name, layer in self.mixed_lora_layers.items():
            state.update(layer.adapter_state_dict(adapter_id, name))
        return state

    def load_adapter_state(self, adapter_id: str, checkpoint_path: str | pathlib.Path) -> None:
        """Load adapter weights and optimizer state from a checkpoint directory."""
        checkpoint_dir = pathlib.Path(checkpoint_path)
        state_path = checkpoint_dir / "adapter_model.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing adapter checkpoint: {state_path}")

        handle = self.adapters[adapter_id]
        config_path = checkpoint_dir / "adapter_config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fp:
                config = json.load(fp)
            self._validate_checkpoint_config(config, checkpoint_dir)
            handle.step = int(config.get("step", 0))

        state = torch.load(state_path, map_location=self.device)
        expected_keys = set()
        for name in self.mixed_lora_layers:
            expected_keys.add(f"{name}.lora_a")
            expected_keys.add(f"{name}.lora_b")
        missing_keys = sorted(expected_keys - set(state))
        if missing_keys:
            raise ValueError(f"Checkpoint {checkpoint_dir} is missing adapter tensors: {missing_keys[:5]}")
        for name, layer in self.mixed_lora_layers.items():
            layer.load_adapter_state_dict(adapter_id, name, state)

        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            handle.optimizer = torch.optim.AdamW(self.adapter_parameters(adapter_id))
            handle.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))

    def _validate_checkpoint_config(self, config: dict[str, object], checkpoint_dir: pathlib.Path) -> None:
        """Validate that a checkpoint matches this resident base model and LoRA layout."""
        if config.get("base_model") != self.base_model:
            raise ValueError(
                f"Checkpoint {checkpoint_dir} was saved for base_model={config.get('base_model')!r}, "
                f"but this service is running base_model={self.base_model!r}"
            )
        if int(config.get("rank", -1)) != self.lora_config.rank:
            raise ValueError(
                f"Checkpoint {checkpoint_dir} was saved with rank={config.get('rank')}, "
                f"but this service is running rank={self.lora_config.rank}"
            )
        if config.get("alpha") != self.lora_config.alpha:
            raise ValueError(
                f"Checkpoint {checkpoint_dir} was saved with alpha={config.get('alpha')}, "
                f"but this service is running alpha={self.lora_config.alpha}"
            )
        expected_targets = self.lora_config.target_modules or ["*_proj"]
        if config.get("target_modules") != expected_targets:
            raise ValueError(
                f"Checkpoint {checkpoint_dir} target_modules={config.get('target_modules')!r} "
                f"do not match this service target_modules={expected_targets!r}"
            )

    def forward_backward_mixed(
        self,
        batches_by_adapter: dict[str, list[Datum]],
        loss_fn: str = "cross_entropy",
        loss_fn_config: Optional[dict[str, float]] = None,
        zero_grad: bool = True,
    ) -> APIFuture[dict[str, ForwardBackwardOutput]]:
        """Run one mixed-adapter forward/backward pass over a concatenated batch."""
        if loss_fn not in SUPPORTED_LOSS_FNS:
            raise NotImplementedError(f"Unsupported loss_fn={loss_fn!r}; expected one of {sorted(SUPPORTED_LOSS_FNS)}")
        if not batches_by_adapter:
            raise ValueError("forward_backward_mixed requires at least one adapter batch")

        data = []
        ranges = []
        start_idx = 0
        adapter_order = []
        for adapter_id, batch in batches_by_adapter.items():
            if adapter_id not in self.adapters:
                raise KeyError(f"Unknown adapter_id: {adapter_id}")
            if not batch:
                continue
            data.extend(batch)
            end_idx = start_idx + len(batch)
            ranges.append((adapter_id, start_idx, end_idx))
            adapter_order.append(adapter_id)
            start_idx = end_idx
        if not data:
            raise ValueError("At least one adapter batch must be non-empty")

        input_ids, labels = _build_batch(data, self.tokenizer.pad_token_id, self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(torch.long)

        if zero_grad:
            self.model.zero_grad(set_to_none=True)
        self._set_active_ranges(ranges)
        self.model.train()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits.contiguous().float()
        shift_labels = labels.contiguous()
        per_token_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)
        token_mask = shift_labels.ne(-100)

        total_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        outputs_by_adapter = {}
        gathered = None
        if loss_fn != "cross_entropy":
            target_logprobs = F.log_softmax(shift_logits, dim=-1)
            safe_labels = shift_labels.clamp_min(0).unsqueeze(-1)
            gathered = target_logprobs.gather(-1, safe_labels).squeeze(-1)
            gathered = gathered.masked_fill(~token_mask, 0.0)

        for adapter_id, start, end in ranges:
            adapter_loss, adapter_mask, loss_metrics = _rl_token_loss(
                loss_fn=loss_fn,
                per_token_loss=per_token_ce[start:end],
                gathered_logprobs=(
                    gathered[start:end] if gathered is not None else torch.empty_like(per_token_ce[start:end])
                ),
                data=data[start:end],
                token_mask=token_mask[start:end],
                device=self.device,
                loss_fn_config=loss_fn_config,
            )
            denom = adapter_mask.float().sum().clamp_min(1.0)
            loss_sum = adapter_loss.masked_fill(~adapter_mask, 0.0).sum()
            loss_mean = loss_sum / denom
            total_loss = total_loss + loss_sum
            metrics = {
                "loss": float(loss_sum.detach().cpu()),
                "loss:sum": float(loss_sum.detach().cpu()),
                "loss:mean": float(loss_mean.detach().cpu()),
                "num_label_tokens": float(denom.detach().cpu()),
                "loss_fn": loss_fn,
                **loss_metrics,
            }
            outputs_by_adapter[adapter_id] = ForwardBackwardOutput(
                loss=float(loss_sum.detach().cpu()),
                metrics=metrics,
                loss_fn_outputs=(
                    [{"logprobs": row.detach().cpu().tolist()} for row in gathered[start:end]]
                    if gathered is not None
                    else []
                ),
            )

        total_loss.backward()
        self._set_active_ranges([])
        return APIFuture(outputs_by_adapter)

    def optim_step(self, adapter_id: str, adam_params: AdamParams) -> APIFuture[OptimStepResponse]:
        """Apply gradients for one resident adapter."""
        handle = self.adapters[adapter_id]
        if handle.optimizer is None:
            handle.optimizer = torch.optim.AdamW(
                self.adapter_parameters(adapter_id),
                lr=adam_params.learning_rate,
                betas=adam_params.betas,
                eps=adam_params.eps,
                weight_decay=adam_params.weight_decay,
            )
        else:
            for group in handle.optimizer.param_groups:
                group["lr"] = adam_params.learning_rate
                group["weight_decay"] = adam_params.weight_decay
                group["betas"] = adam_params.betas
                group["eps"] = adam_params.eps

        handle.optimizer.step()
        handle.optimizer.zero_grad(set_to_none=True)
        handle.step += 1
        return APIFuture(OptimStepResponse(step=handle.step, learning_rate=adam_params.learning_rate))

    def save_adapter_state(self, adapter_id: str, name: str) -> APIFuture[SaveStateResponse]:
        """Save one resident adapter's weights and optimizer state."""
        output_dir = self.scratch_dir / "checkpoints" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.adapter_state_dict(adapter_id), output_dir / "adapter_model.pt")
        handle = self.adapters[adapter_id]
        if handle.optimizer is not None:
            torch.save(handle.optimizer.state_dict(), output_dir / "optimizer.pt")
        with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "base_model": self.base_model,
                    "adapter_id": adapter_id,
                    "rank": self.lora_config.rank,
                    "alpha": self.lora_config.alpha,
                    "target_modules": self.lora_config.target_modules or ["*_proj"],
                    "step": handle.step,
                },
                fp,
                indent=2,
            )
        return APIFuture(SaveStateResponse(path=str(output_dir)))

    def detach_adapter(self, adapter_id: str) -> APIFuture[DetachAdapterResponse]:
        """Remove one resident adapter and release its trainable parameters."""
        if adapter_id not in self.adapters:
            raise KeyError(f"Unknown adapter_id: {adapter_id}")
        self._set_active_ranges([])
        handle = self.adapters.pop(adapter_id)
        if handle.optimizer is not None:
            handle.optimizer.zero_grad(set_to_none=True)
        for layer in self.mixed_lora_layers.values():
            layer.remove_adapter(adapter_id)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return APIFuture(DetachAdapterResponse(adapter_id=adapter_id, remaining_adapters=len(self.adapters)))

    def _sample_with_generate(self, encoded: Any, params: SamplingParams) -> torch.Tensor:
        generate_kwargs = {
            "input_ids": encoded["input_ids"],
            "max_new_tokens": params.max_new_tokens,
            "do_sample": params.do_sample,
            "temperature": max(params.temperature, 1e-6),
            "top_p": params.top_p,
        }
        if "attention_mask" in encoded:
            generate_kwargs["attention_mask"] = encoded["attention_mask"]
        if self.tokenizer.pad_token_id is not None:
            generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        return self.model.generate(**generate_kwargs)

    def _sample_with_manual_loop(self, encoded: Any, params: SamplingParams) -> tuple[torch.Tensor, list[float]]:
        output_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        generated_logprobs = []
        for _ in range(params.max_new_tokens):
            outputs = self.model(input_ids=output_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            logprobs = torch.log_softmax(next_token_logits, dim=-1)
            if params.do_sample:
                temperature = max(params.temperature, 1e-6)
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                if params.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_mask = cumulative_probs > params.top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
                    probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_logprobs.append(float(logprobs.gather(-1, next_token).squeeze(-1).detach().cpu()[0]))
            output_ids = torch.cat([output_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        return output_ids, generated_logprobs

    def sample(
        self, adapter_id: str, prompt: str, params: Optional[SamplingParams] = None
    ) -> APIFuture[SampleResponse]:
        """Generate with one resident adapter selected for the full prompt batch."""
        params = params or SamplingParams()
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self._set_active_ranges([(adapter_id, 0, 1)])
        try:
            self.model.eval()
            with torch.no_grad():
                prompt_token_count = int(encoded["input_ids"].shape[-1])
                generated_logprobs = None
                try:
                    if params.return_logprobs:
                        raise RuntimeError("manual sampling is required when return_logprobs=True")
                    output_ids = self._sample_with_generate(encoded, params)
                except Exception:
                    output_ids, generated_logprobs = self._sample_with_manual_loop(encoded, params)
            output_ids = output_ids[0]
        finally:
            self._set_active_ranges([])
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return APIFuture(
            SampleResponse(
                tokens=output_ids.detach().cpu().tolist(),
                text=text,
                prompt_token_count=prompt_token_count,
                generated_logprobs=generated_logprobs,
            )
        )


class MixedLoraTrainingClient:
    """One resident adapter client for `MixedLoraServiceClient`."""

    def __init__(self, *, service: MixedLoraServiceClient, handle: MixedAdapterHandle):
        self.service = service
        self.handle = handle
        self.adapter_id = handle.adapter_id

    def optim_step(self, adam_params: AdamParams) -> APIFuture[OptimStepResponse]:
        """Apply this adapter's gradients."""
        return self.service.optim_step(self.adapter_id, adam_params)

    def save_state(self, name: str) -> APIFuture[SaveStateResponse]:
        """Save this adapter's state."""
        return self.service.save_adapter_state(self.adapter_id, name)

    def detach(self) -> APIFuture[DetachAdapterResponse]:
        """Detach this adapter from the resident service."""
        return self.service.detach_adapter(self.adapter_id)
