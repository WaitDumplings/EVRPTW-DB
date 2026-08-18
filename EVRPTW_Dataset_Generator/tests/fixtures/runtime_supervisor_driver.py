"""Signalable supervisor process used by integration tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from evrptw_stage2.runtime_supervisor import (
    FamilyProcessSpec,
    ProcessDecision,
    supervise_family_processes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    runtime = root / "runtime"
    staging = root / ".inflight" / "signal-family" / "attempt-000"
    spec = FamilyProcessSpec(
        run_id="signal-run",
        family_id="signal-family",
        attempt_id="attempt-000",
        attempt_number=0,
        city_slug="fixture-city",
        track_id="train",
        day_type="weekday",
        scale_id="cus50",
        seed=1,
        command=(
            sys.executable,
            str(args.fixture),
            "--sleep-s",
            "120",
            "--started-marker",
            str(root / "started.json"),
            "--heartbeat",
            str(runtime / "heartbeat.json"),
            "--partial",
            str(staging / "partial.json"),
            "--grandchild-pid",
            str(root / "grandchild.json"),
            "--spawn-grandchild",
            "--ignore-term",
        ),
        cwd=str(root),
        result_path=str(runtime / "result.json"),
        heartbeat_path=str(runtime / "heartbeat.json"),
        stdout_path=str(runtime / "stdout.log"),
        stderr_path=str(runtime / "stderr.log"),
        partial_artifact_path=str(staging),
        timeout_ledger_path=str(root / "timeouts" / "signal-family.json"),
    )

    def completed(_spec: FamilyProcessSpec, returncode: int) -> ProcessDecision:
        return ProcessDecision(status="complete", result={"returncode": returncode})

    report = supervise_family_processes(
        [spec],
        max_workers=1,
        on_normal_exit=completed,
        family_wall_timeout_s=60,
        termination_grace_s=2,
        runner_exit_slack_s=1,
        poll_interval_s=0.05,
    )
    (root / "supervisor_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["hard_stop_triggered"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
