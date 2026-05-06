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

import importlib.util
import pathlib
import subprocess
import sys


def load_host_rl_launcher():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "host_rl_launcher.py"
    spec = importlib.util.spec_from_file_location("host_rl_launcher", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wait_for_process_terminates_canceling_job(monkeypatch):
    launcher = load_host_rl_launcher()
    statuses = iter(["running", "canceling"])
    monkeypatch.setattr(launcher, "get_status", lambda *args: next(statuses))

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    outcome, returncode = launcher.wait_for_process(
        process,
        api_url="http://127.0.0.1:18080",
        api_token=None,
        job_id="rljob_test",
        max_runtime_seconds=30,
        poll_seconds=0.01,
    )

    assert outcome == "canceled"
    assert returncode != 0
