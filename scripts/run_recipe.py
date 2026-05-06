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
import json
import pathlib
import subprocess
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RECIPE_DIR = REPO_ROOT / "recipes"


def _load_recipe(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_flag(command: list[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            command.append(f"--{name.replace('_', '-')}")
        return
    if value is None:
        return
    command.extend([f"--{name.replace('_', '-')}", str(value)])


def _script_for_kind(kind: str) -> str:
    if kind == "qwen_sft":
        return "clients/qwen_api_smoke_client.py"
    if kind == "nemotron_sft":
        return "clients/nemotron_nano_api_smoke_client.py"
    if kind == "nemotron_rl":
        return "clients/rl_lora_workload_client.py"
    raise ValueError(f"Unsupported recipe kind: {kind}")


def build_command(recipe: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    """Build the Python command for a recipe."""
    merged = dict(recipe.get("args", {}))
    merged.update({key: value for key, value in overrides.items() if value is not None})
    script = REPO_ROOT / _script_for_kind(str(recipe["kind"]))
    command = [sys.executable, str(script)]
    for key, value in merged.items():
        _append_flag(command, key, value)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a named Nemotron-Tinker workload recipe.")
    parser.add_argument("recipe", help="Recipe name under recipes or a JSON file path.")
    parser.add_argument("--base-url", default=None, help="Override recipe base URL.")
    parser.add_argument("--tenant-id", default=None, help="Override recipe tenant id.")
    parser.add_argument("--steps", type=int, default=None, help="Override recipe optimizer steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args = parser.parse_args()

    recipe_path = pathlib.Path(args.recipe)
    if not recipe_path.is_file():
        recipe_path = RECIPE_DIR / f"{args.recipe}.json"
    recipe = _load_recipe(recipe_path)
    command = build_command(
        recipe,
        {
            "base_url": args.base_url,
            "tenant_id": args.tenant_id,
            "steps": args.steps,
        },
    )
    print(" ".join(command))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
