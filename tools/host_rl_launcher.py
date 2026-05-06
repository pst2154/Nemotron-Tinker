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
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_record(metadata_path: pathlib.Path, job_id: str) -> dict[str, Any]:
    with sqlite3.connect(metadata_path) as conn:
        row = conn.execute(
            "SELECT payload FROM records WHERE namespace = ? AND record_id = ?",
            ("rl_jobs", job_id),
        ).fetchone()
    if row is None:
        raise KeyError(f"Missing rl_jobs record: {job_id}")
    return json.loads(row[0])


def save_record(metadata_path: pathlib.Path, job_id: str, record: dict[str, Any]) -> None:
    record["updated_at"] = utc_now()
    with sqlite3.connect(metadata_path) as conn:
        conn.execute(
            """
            INSERT INTO records(namespace, record_id, payload, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, record_id)
            DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at
            """,
            ("rl_jobs", job_id, json.dumps(record, sort_keys=True), record["updated_at"]),
        )


def mark(metadata_path: pathlib.Path, job_id: str, **updates: Any) -> dict[str, Any]:
    record = load_record(metadata_path, job_id)
    record.update({key: value for key, value in updates.items() if value is not None})
    save_record(metadata_path, job_id, record)
    return record


def terminate(process: subprocess.Popen, sig: signal.Signals = signal.SIGTERM) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def run_job(request_path: pathlib.Path) -> None:
    running_path = request_path.with_suffix(".running")
    request_path.rename(running_path)
    payload = json.loads(running_path.read_text(encoding="utf-8"))
    job_id = payload["job_id"]
    metadata_path = pathlib.Path(payload["metadata_path"])
    log_path = pathlib.Path(payload["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(item) for item in payload["command"]]
    max_runtime_seconds = payload.get("max_runtime_seconds")
    cwd = pathlib.Path(payload.get("host_cwd") or payload["repo_dir"])
    if not cwd.is_dir():
        cwd = pathlib.Path("/")

    try:
        mark(metadata_path, job_id, status="running")
        with log_path.open("ab") as log_fp:
            log_fp.write(("Host launcher command: " + " ".join(command) + "\n\n").encode("utf-8"))
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            mark(metadata_path, job_id, pid=process.pid)
            try:
                returncode = process.wait(timeout=max_runtime_seconds)
            except subprocess.TimeoutExpired:
                terminate(process)
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    terminate(process, signal.SIGKILL)
                    returncode = process.wait()
                mark(
                    metadata_path,
                    job_id,
                    status="timed_out",
                    returncode=returncode,
                    error=f"RL job exceeded max_runtime_seconds={max_runtime_seconds}",
                )
                return
        mark(metadata_path, job_id, status="succeeded" if returncode == 0 else "failed", returncode=returncode)
    except Exception as exc:
        mark(metadata_path, job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        done_path = running_path.with_suffix(".done")
        try:
            running_path.rename(done_path)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run host-side NeMo-RL jobs queued by Nemotron Tinker.")
    parser.add_argument("--scratch-dir", required=True, help="Nemotron Tinker scratch directory.")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    queue_dir = pathlib.Path(args.scratch_dir) / "tinker_api" / "host_rl_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    while True:
        request_paths = sorted(queue_dir.glob("rljob_*.json"))
        for request_path in request_paths:
            run_job(request_path)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
