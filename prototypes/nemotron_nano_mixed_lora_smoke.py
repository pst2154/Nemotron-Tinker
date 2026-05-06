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

import argparse
import pathlib
import random
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from nemotron_tinker import AdamParams, Datum, LoraConfig, ModelInput  # noqa: E402
from nemotron_tinker.mixed_client import MixedLoraServiceClient  # noqa: E402


def build_datum(tokenizer, prompt: str, completion: str, max_tokens: int) -> Datum:
    """Build one tiny masked SFT datum."""
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    tokens = (prompt_tokens + completion_tokens)[:max_tokens]
    if len(tokens) < 2:
        raise ValueError("SFT datum needs at least two tokens")
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    first_completion_label = max(0, min(len(prompt_tokens), len(tokens)) - 1)
    weights = [0.0] * first_completion_label + [1.0] * max(0, len(target_tokens) - first_completion_label)
    return Datum(
        model_input=ModelInput.from_ints(input_tokens),
        loss_fn_inputs={"target_tokens": ModelInput.from_ints(target_tokens), "weights": weights},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test mixed LoRA on local Nemotron Nano 30B A3B.")
    parser.add_argument(
        "--base-model",
        default="/models/nemotron-nano-30b-a3b-bf16",
    )
    parser.add_argument("--scratch-dir", default="/tmp/nemotron_tinker")
    parser.add_argument("--cache-dir", default="/tmp/nemotron_tinker_hf")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--backend", choices=("loop", "grouped", "triton", "grouped_triton"), default="grouped")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    args = parser.parse_args()

    service = MixedLoraServiceClient(
        base_model=args.base_model,
        scratch_dir=args.scratch_dir,
        cache_dir=args.cache_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        lora_config=LoraConfig(rank=args.rank, alpha=args.alpha, target_modules=args.target_modules),
        mixed_lora_backend=args.backend,
    )
    atlas = service.create_lora_training_client(adapter_id="nemotron_atlas")
    borealis = service.create_lora_training_client(adapter_id="nemotron_borealis")
    rng = random.Random(1234)

    atlas_data = [
        build_datum(service.tokenizer, "Tenant Atlas route alpha.\nAnswer:", " atlas-17.", args.max_tokens),
        build_datum(service.tokenizer, "Tenant Atlas route beta.\nAnswer:", " atlas-29.", args.max_tokens),
    ]
    borealis_data = [
        build_datum(service.tokenizer, "Tenant Borealis route alpha.\nAnswer:", " borealis-05.", args.max_tokens),
        build_datum(service.tokenizer, "Tenant Borealis route beta.\nAnswer:", " borealis-14.", args.max_tokens),
    ]
    print(f"model={args.base_model}")
    print(f"backend={args.backend}")
    print(f"layers={len(service.mixed_lora_layers)}")
    print(f"target_modules={args.target_modules}")

    first_losses = None
    last_losses = None
    for _ in range(args.steps):
        outputs = service.forward_backward_mixed(
            {
                atlas.adapter_id: [rng.choice(atlas_data)],
                borealis.adapter_id: [rng.choice(borealis_data)],
            }
        ).result()
        atlas.optim_step(AdamParams(learning_rate=args.lr)).result()
        borealis.optim_step(AdamParams(learning_rate=args.lr)).result()
        losses = (outputs[atlas.adapter_id].loss, outputs[borealis.adapter_id].loss)
        first_losses = first_losses or losses
        last_losses = losses

    atlas_state = atlas.save_state("nemotron-nano-atlas-smoke").result()
    borealis_state = borealis.save_state("nemotron-nano-borealis-smoke").result()
    print(f"first_losses={first_losses}")
    print(f"last_losses={last_losses}")
    print(f"atlas_saved={atlas_state.path}")
    print(f"borealis_saved={borealis_state.path}")


if __name__ == "__main__":
    main()
