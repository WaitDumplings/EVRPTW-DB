"""Small real-process fixture for runtime supervisor integration tests."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def write(path: str | None, value: object) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep-s", type=float, required=True)
    parser.add_argument("--started-marker")
    parser.add_argument("--heartbeat")
    parser.add_argument("--partial")
    parser.add_argument("--result")
    parser.add_argument("--completion-marker")
    parser.add_argument("--staging-completion-marker")
    parser.add_argument("--grandchild-pid")
    parser.add_argument("--spawn-grandchild", action="store_true")
    parser.add_argument("--ignore-term", action="store_true")
    args = parser.parse_args()

    if args.ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    write(args.started_marker, {"pid": os.getpid(), "pgid": os.getpgrp()})
    write(
        args.heartbeat,
        {
            "stage": "fixture_sleep",
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "timestamp": time.time(),
        },
    )
    write(args.partial, {"materialization_status": "partial"})
    write(args.staging_completion_marker, {"materialization_status": "complete"})
    if args.spawn_grandchild:
        code = (
            "import os,signal,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "print(os.getpid(),flush=True);time.sleep(120)"
        )
        child = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            text=True,
        )
        if child.stdout is None:
            raise RuntimeError("grandchild stdout is unavailable")
        grandchild_pid = int(child.stdout.readline().strip())
        write(args.grandchild_pid, {"pid": grandchild_pid, "pgid": os.getpgid(grandchild_pid)})
    time.sleep(args.sleep_s)
    write(args.result, {"status": "complete", "pid": os.getpid()})
    write(args.completion_marker, {"materialization_status": "complete"})


if __name__ == "__main__":
    main()
