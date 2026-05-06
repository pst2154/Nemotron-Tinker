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

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelInput:
    """Token IDs for one model input sequence."""

    tokens: list[int]

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "ModelInput":
        """Create a model input from token IDs."""
        return cls(tokens=list(tokens))


@dataclass
class Datum:
    """One self-contained training example.

    `loss_fn_inputs` accepts the same v0 keys as the Tinker public examples:
    `target_tokens` and `weights`. `target_tokens` may be a `ModelInput` or
    a plain list of token IDs. `weights` uses 0 to mask positions from loss.
    """

    model_input: ModelInput
    loss_fn_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdamParams:
    """AdamW optimizer parameters for one optimizer step."""

    learning_rate: float
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class LoraConfig:
    """LoRA configuration for creating a training client."""

    rank: int = 16
    alpha: Optional[int] = None
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=list)
    exclude_modules: list[str] = field(default_factory=list)
    match_all_linear: bool = False
    train_unembed: bool = False


@dataclass
class ForwardBackwardOutput:
    """Output from a forward/backward call."""

    loss: float
    metrics: dict[str, float]
    loss_fn_outputs: list[dict[str, list[float]]]


@dataclass
class OptimStepResponse:
    """Output from an optimizer step."""

    step: int
    learning_rate: float


@dataclass
class SaveStateResponse:
    """Output from saving adapter and optimizer state."""

    path: str


@dataclass
class DetachAdapterResponse:
    """Output from detaching one resident adapter."""

    adapter_id: str
    remaining_adapters: int


@dataclass
class SamplingParams:
    """Generation parameters for the prototype sampler."""

    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    return_logprobs: bool = False


@dataclass
class SampleResponse:
    """Generated token IDs and decoded text for one prompt."""

    tokens: list[int]
    text: str
    prompt_token_count: int = 0
    generated_logprobs: Optional[list[float]] = None
