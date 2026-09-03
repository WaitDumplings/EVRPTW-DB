from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPO / "EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1"


def test_server_env_resolves_repo_root_and_relative_dataset_override() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "SERVER_SCRIPT_DIR": str(SCRIPT_ROOT / "2080ti_4_1"),
            "EVRPTW_DATASET_ROOT": "../EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city",
        }
    )
    command = (
        f"source {SCRIPT_ROOT / 'server_env.sh'}; "
        "printf '%s\\n%s\\n%s\\n' "
        '"$EVRPTW_REPO_ROOT" "$EVRPTW_DATASET_ROOT" "$EVRPTW_OUTPUT_ROOT"'
    )
    output = subprocess.check_output(
        ["bash", "-c", command], cwd=REPO, env=environment, text=True
    ).splitlines()
    assert output[0] == str(REPO)
    assert output[1] == str(
        (REPO / "../EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city").resolve()
    )
    assert output[2] == str(REPO / "EVRPTW_Benchmark/results/DRL_rq_v1")
