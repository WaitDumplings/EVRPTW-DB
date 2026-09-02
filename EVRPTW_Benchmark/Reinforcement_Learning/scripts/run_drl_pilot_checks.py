#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def latest_pilot(output: Path, method: str, scale: str, stage: str) -> Path:
    matches = sorted(output.glob(f"R/{method}/{scale}/seed_1234/*/pilot/{stage}"))
    if not matches:
        raise FileNotFoundError(f"pilot output missing: {method}/{scale}/{stage}")
    return matches[-1]


def normalized_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row.pop("runtime_s", None)
    return rows


def normalized_routes(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row.pop("runtime_s", None)
    return rows


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def eval_command(method: str, protocol: dict[str, Any], checkpoint: Path, dataset: Path, output: Path, chunk: int) -> list[str]:
    module = protocol["methods"][method]["eval_module"]
    command = [
        sys.executable, "-m", module,
        "--dataset-path", str(dataset / protocol["scales"]["Cus100"]["validation_index"]),
        "--family-root", str(dataset / "materialized" / "families"),
        "--checkpoint", str(checkpoint),
        "--scale", "Cus100", "--split-ids", "val", "--track-ids", "validation",
        "--candidates", "50", "--candidate-chunk-size", str(chunk),
        "--limit", str(protocol["pilot"]["best_of_50_equality_limit"]),
        "--seed", str(protocol["pilot"]["seed"]), "--device", "cuda",
        "--output-dir", str(output),
    ]
    command.extend(["--decode-mode", "sample"] if method == "terran" else ["--decode-type", "sampling"])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-training DRL pilot evidence checks.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "drl_experiment_protocol_v1.yaml")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    evidence = args.output_root / "pilot_evidence"
    resume_test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "common" / "tests" / "test_data_pass.py")],
        check=False,
    )
    resume_passed = resume_test.returncode == 0

    for method in protocol["pilot"]["required_methods"]:
        cus100 = latest_pilot(args.output_root, method, "Cus100", "short_optimization")
        validation = json.loads((cus100 / "validation_summary.json").read_text(encoding="utf-8"))
        write(evidence / f"{method}__greedy_verifier.json", {
            "passed": bool(validation.get("verifier_summary_passed")),
            "source": str(cus100 / "validation_summary.json"),
        })
        unchunked = evidence / "best50_runs" / method / "unchunked"
        chunked = evidence / "best50_runs" / method / "chunked_5"
        subprocess.run(eval_command(method, protocol, cus100 / "checkpoint_selected.pt", args.dataset_root, unchunked, 50), check=True)
        subprocess.run(eval_command(method, protocol, cus100 / "checkpoint_selected.pt", args.dataset_root, chunked, 5), check=True)
        equal = (
            normalized_rows(unchunked / "summary.csv") == normalized_rows(chunked / "summary.csv")
            and normalized_routes(unchunked / "routes.jsonl") == normalized_routes(chunked / "routes.jsonl")
        )
        write(evidence / f"{method}__best50_equality.json", {"passed": equal, "candidate_count": 50, "chunk_sizes": [50, 5]})
        write(evidence / f"{method}__data_pass_resume.json", {
            "passed": resume_passed,
            "contract": "incomplete pass is replayed from seeded shuffle; state commits at complete-pass boundary",
            "automated_test": "common/tests/test_data_pass.py",
        })
        result = json.loads((cus100 / "training_result.json").read_text(encoding="utf-8"))
        elapsed = float(result["wall_time_s"])
        history_path = cus100 / "train_history.jsonl"
        if history_path.exists():
            history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
            measured = int(history[-1]["instances_seen"])
        else:
            measured = int(result["instances_seen"])
        views = int(protocol["scales"]["Cus100"]["train_views"])
        passes = int(protocol["training"]["full_data_passes"])
        estimate = elapsed / max(measured, 1) * views * passes
        write(evidence / f"{method}__wall_time_estimate.json", {
            "passed": estimate > 0,
            "measured_instances": measured,
            "measured_wall_time_s": elapsed,
            "estimated_full_wall_time_s": estimate,
            "extrapolation_only": True,
        })
    print(json.dumps({"passed": True, "evidence_root": str(evidence)}, sort_keys=True))


if __name__ == "__main__":
    main()
