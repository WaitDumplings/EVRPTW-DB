from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPO / "EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1"


def _launcher_test_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nohup = fake_bin / "nohup"
    fake_nohup.write_text("#!/usr/bin/env bash\nexec sleep 60\n")
    fake_nohup.chmod(0o755)
    output_root = tmp_path / "output"
    environment = os.environ.copy()
    environment.update(
        {
            "SERVER_SCRIPT_DIR": str(SCRIPT_ROOT / "a6000_2_1"),
            "EVRPTW_DATASET_ROOT": str(tmp_path / "dataset"),
            "EVRPTW_OUTPUT_ROOT": str(output_root),
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )
    return environment, output_root / "launcher_logs/a6000_2_1"


def test_server_env_resolves_repo_root_and_relative_dataset_override() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "SERVER_SCRIPT_DIR": str(SCRIPT_ROOT / "2080ti_4_1"),
            "EVRPTW_DATASET_ROOT": "EVRPTW_Dataset/Instances_v2/us_11city",
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
        (REPO / "EVRPTW_Dataset/Instances_v2/us_11city").resolve()
    )
    assert output[2] == str(REPO / "EVRPTW_Benchmark/results/DRL_rq_v1")


def test_server_env_preserves_priority_manifest_and_scale_overrides() -> None:
    custom_manifest = SCRIPT_ROOT / "a6000_2_1/cus1000_jobs.jsonl"
    environment = os.environ.copy()
    environment.update(
        {
            "SERVER_SCRIPT_DIR": str(SCRIPT_ROOT / "a6000_2_1"),
            "DRL_MANIFEST": str(custom_manifest),
            "DRL_SCALES": "Cus1000",
        }
    )
    command = (
        f"source {SCRIPT_ROOT / 'server_env.sh'}; "
        "printf '%s\\n%s\\n' \"$DRL_MANIFEST\" \"$DRL_SCALES\""
    )
    output = subprocess.check_output(
        ["bash", "-c", command], cwd=REPO, env=environment, text=True
    ).splitlines()
    assert output == [str(custom_manifest), "Cus1000"]


def test_launcher_blocks_other_mode_but_ignores_stale_pid(tmp_path: Path) -> None:
    environment, log_dir = _launcher_test_environment(tmp_path)
    log_dir.mkdir(parents=True)
    stale_pid = 999_999_999
    (log_dir / "full.pid").write_text(f"{stale_pid}\n")

    started = subprocess.run(
        ["bash", str(SCRIPT_ROOT / "start_server.sh"), "resume"],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert started.returncode == 0, started.stderr
    assert not (log_dir / "full.pid").exists()
    resume_pid = int((log_dir / "resume.pid").read_text())
    try:
        blocked = subprocess.run(
            ["bash", str(SCRIPT_ROOT / "start_server.sh"), "full"],
            cwd=REPO,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert blocked.returncode == 3
        assert f"a6000_2_1/resume is already running with pid {resume_pid}" in blocked.stderr
    finally:
        os.kill(resume_pid, signal.SIGTERM)


def test_dataset_resolver_finds_release_below_relative_restore_root(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "checkout"
    repo.mkdir()
    release = tmp_path / "runtime/EVRPTW_Dataset/Instances_v2/us_11city"
    marker = release / "generation_plan/core/train/view_index.parquet"
    marker.parent.mkdir(parents=True)
    marker.touch()
    environment = os.environ.copy()
    environment.pop("EVRPTW_DATASET_ROOT", None)
    environment["EVRPTW_RESTORE_ROOT"] = "../runtime"
    command = (
        f"source {SCRIPT_ROOT / 'dataset_root.sh'}; "
        f'resolve_evrptw_dataset_root "{repo}"'
    )
    output = subprocess.check_output(
        ["bash", "-c", command], env=environment, text=True
    ).strip()
    assert output == str(release)


def test_dataset_resolver_falls_back_to_repository_local_frozen_release(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "checkout"
    repo.mkdir()
    release = (
        repo
        / "EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"
    )
    marker = release / "generation_plan/core/train/view_index.parquet"
    marker.parent.mkdir(parents=True)
    marker.touch()
    environment = os.environ.copy()
    environment.pop("EVRPTW_DATASET_ROOT", None)
    environment["EVRPTW_RESTORE_ROOT"] = "../missing-runtime"
    command = (
        f"source {SCRIPT_ROOT / 'dataset_root.sh'}; "
        f'resolve_evrptw_dataset_root "{repo}"'
    )
    output = subprocess.check_output(
        ["bash", "-c", command], env=environment, text=True
    ).strip()
    assert output == str(release)
