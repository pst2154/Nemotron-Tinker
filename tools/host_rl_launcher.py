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
import os
import pathlib
import signal
import subprocess
import time
from datetime import datetime, timezone
from typing import Any
from urllib import request as urllib_request
from urllib.error import URLError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def terminate(process: subprocess.Popen, sig: signal.Signals = signal.SIGTERM) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def mark(api_url: str, api_token: str | None, job_id: str, **updates: Any) -> None:
    payload = {key: value for key, value in updates.items() if value is not None}
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["X-Nemotron-Tinker-Worker-Token"] = api_token
    req = urllib_request.Request(
        f"{api_url.rstrip('/')}/internal/rl/jobs/{job_id}/mark",
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=30) as response:
        response.read()


def get_status(api_url: str, api_token: str | None, job_id: str) -> str | None:
    headers = {}
    if api_token:
        headers["X-Nemotron-Tinker-Worker-Token"] = api_token
    req = urllib_request.Request(
        f"{api_url.rstrip('/')}/rl/jobs/{job_id}",
        headers=headers,
        method="GET",
    )
    try:
        with urllib_request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError):
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def wait_for_process(
    process: subprocess.Popen,
    *,
    api_url: str,
    api_token: str | None,
    job_id: str,
    max_runtime_seconds: float | None,
    poll_seconds: float,
) -> tuple[str, int]:
    started_at = time.monotonic()
    while True:
        returncode = process.poll()
        if returncode is not None:
            return ("finished", returncode)
        if max_runtime_seconds is not None and time.monotonic() - started_at > max_runtime_seconds:
            terminate(process)
            try:
                return ("timed_out", process.wait(timeout=30))
            except subprocess.TimeoutExpired:
                terminate(process, signal.SIGKILL)
                return ("timed_out", process.wait())
        status = get_status(api_url, api_token, job_id)
        if status == "canceling":
            terminate(process)
            try:
                return ("canceled", process.wait(timeout=30))
            except subprocess.TimeoutExpired:
                terminate(process, signal.SIGKILL)
                return ("canceled", process.wait())
        time.sleep(poll_seconds)


def run_job(request_path: pathlib.Path, *, api_url: str, api_token: str | None, state_dir: pathlib.Path) -> None:
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    job_id = payload["job_id"]
    running_path = state_dir / "running" / f"{job_id}.json"
    done_path = state_dir / "done" / f"{job_id}.json"
    failed_path = state_dir / "failed" / f"{job_id}.json"
    if done_path.exists() or running_path.exists() or failed_path.exists():
        return
    running_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    running_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log_path = state_dir / "logs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(item) for item in payload["command"]]
    max_runtime_seconds = payload.get("max_runtime_seconds")
    cwd = pathlib.Path(payload.get("host_cwd") or payload["repo_dir"])
    if not cwd.is_dir():
        cwd = pathlib.Path("/")

    try:
        mark(api_url, api_token, job_id, status="running", log_path=str(log_path))
        with log_path.open("ab") as log_fp:
            log_fp.write((f"Host launcher started: {utc_now()}\n").encode("utf-8"))
            log_fp.write(("Host launcher command: " + " ".join(command) + "\n\n").encode("utf-8"))
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            mark(api_url, api_token, job_id, pid=process.pid)
            outcome, returncode = wait_for_process(
                process,
                api_url=api_url,
                api_token=api_token,
                job_id=job_id,
                max_runtime_seconds=max_runtime_seconds,
                poll_seconds=2.0,
            )
            if outcome == "timed_out":
                mark(
                    api_url,
                    api_token,
                    job_id,
                    status="timed_out",
                    returncode=returncode,
                    error=f"RL job exceeded max_runtime_seconds={max_runtime_seconds}",
                )
                done_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                return
            if outcome == "canceled":
                mark(api_url, api_token, job_id, status="canceled", returncode=returncode)
                done_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                return
        mark(api_url, api_token, job_id, status="succeeded" if returncode == 0 else "failed", returncode=returncode)
        done_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        failed_path.write_text(
            json.dumps({"payload": payload, "error": f"{type(exc).__name__}: {exc}"}), encoding="utf-8"
        )
        try:
            mark(api_url, api_token, job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
    finally:
        try:
            running_path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run host-side NeMo-RL jobs queued by Nemotron Tinker.")
    parser.add_argument("--scratch-dir", required=True, help="Nemotron Tinker scratch directory.")
    parser.add_argument("--api-url", default="http://127.0.0.1:18080", help="Nemotron Tinker API URL.")
    parser.add_argument("--api-token", default=os.environ.get("NEMOTRON_TINKER_HOST_WORKER_TOKEN"))
    parser.add_argument("--state-dir", help="Worker-owned state and log directory.")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    queue_dir = pathlib.Path(args.scratch_dir) / "tinker_api" / "host_rl_queue"
    state_dir = (
        pathlib.Path(args.state_dir)
        if args.state_dir
        else pathlib.Path(args.scratch_dir).parent / (pathlib.Path(args.scratch_dir).name + "_host_rl_launcher")
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    while True:
        request_paths = sorted(queue_dir.glob("rljob_*.json")) if queue_dir.is_dir() else []
        for request_path in request_paths:
            run_job(request_path, api_url=args.api_url, api_token=args.api_token, state_dir=state_dir)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
