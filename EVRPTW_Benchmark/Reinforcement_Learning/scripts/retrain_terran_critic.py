"""Restart a TERRAN job with the shared critic configuration and a new run.

Reuses the prior job's registered data, batch, reward and evaluation arguments.
No checkpoint weights/optimizer state are loaded and no files are rehashed.
Run with the project's Python environment from the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/configs/stage2_cus100_terran.yaml"
MODULE = "EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train"


def replacement_command(source: dict, output: Path) -> list[str]:
    original = source["command"]
    if len(original) < 3 or original[1:3] != ["-m", MODULE]:
        raise ValueError("source run must be a TERRAN.train launch")
    if "--resume" in original or "--warm-start-checkpoint" in original:
        raise ValueError("source must be a fresh launch, not a continuation")
    if "--reuse-preverified-training-streams" not in original:
        raise ValueError("source must already reuse its registered training stream")
    result = [sys.executable, *original[1:]]
    for flag, value in (("--config", str(CONFIG)), ("--output-dir", str(output))):
        if result.count(flag) != 1:
            raise ValueError(f"source must contain exactly one {flag}")
        result[result.index(flag) + 1] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    source_path = args.source_run.resolve() / "provenance.json"
    source = json.loads(source_path.read_text())
    output = args.output_dir.resolve()
    command = replacement_command(source, output)
    config = yaml.safe_load(CONFIG.read_text())
    if args.dry_run:
        print(json.dumps({"output_dir": str(output), "gpu": args.gpu,
                          "command": command, "training": config["training"]}, indent=2))
        return
    # A separate directory prevents history/checkpoints from mixing across fits.
    output.mkdir(parents=True, exist_ok=False)
    provenance = {key: value for key, value in source.items()
                  if key not in {"command", "git_commit", "started_at", "local_gpu"}}
    provenance.update({
        "schema": "terran_critic_retrain_v1",
        "source_run": str(args.source_run.resolve()),
        "source_configuration_only": True,
        "command": command,
        "started_at": time.time(),
        "local_gpu": args.gpu,
        "launcher_id": "terran_critic_stability_v1",
        "resume_requested": False,
        "resumed_from_checkpoint": False,
        "warm_started_from_checkpoint": False,
        "warm_start_checkpoint": None,
        "critic_stability": {key: config["training"][key] for key in (
            "value_loss_type", "value_loss_beta", "value_residual_scale", "vf_coef",
            "critic_backbone_grad_scale", "critic_gradient_diagnostics_every_epochs")},
    })
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "launch_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
    (output / "train.pid").write_text(f"{process.pid}\n")
    print(json.dumps({"pid": process.pid, "gpu": args.gpu, "output_dir": str(output)}))


if __name__ == "__main__":
    main()
