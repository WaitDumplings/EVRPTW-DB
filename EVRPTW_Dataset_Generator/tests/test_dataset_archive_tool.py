from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


REPOSITORY_ROOT = Path(__file__).parents[2]
TOOL_PATH = REPOSITORY_ROOT / "EVRPTW_Dataset_Generator" / "scripts" / "dataset_archive_tool.py"
AUTO_PATH = REPOSITORY_ROOT / "auto.sh"


def _load_archive_tool():
    specification = importlib.util.spec_from_file_location("dataset_archive_tool", TOOL_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


archive_tool = _load_archive_tool()


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _fake_zstd(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-zstd"
    _write_executable(executable, "#!/bin/sh\nexec /bin/cat \"$3\"\n")
    return executable


def _tar_bytes(members: list[tarfile.TarInfo]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for member in members:
            payload = b""
            if member.isfile():
                payload = b"{}" if member.size else b""
                member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload) if member.isfile() else None)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ([tarfile.TarInfo("EVRPTW_Dataset/../escape")], "Unsafe archive path component"),
        (
            [
                tarfile.TarInfo("EVRPTW_Dataset/link"),
            ],
            "Archive links are not allowed",
        ),
        (
            [
                tarfile.TarInfo("EVRPTW_Dataset/repeated"),
                tarfile.TarInfo("EVRPTW_Dataset/repeated"),
            ],
            "Duplicate archive member",
        ),
    ],
)
def test_archive_inspection_rejects_unsafe_members(
    tmp_path: Path,
    members: list[tarfile.TarInfo],
    message: str,
) -> None:
    if "link" in members[0].name:
        members[0].type = tarfile.SYMTYPE
        members[0].linkname = "target"
    archive = tmp_path / "unsafe.tar.zst"
    archive.write_bytes(_tar_bytes(members))

    with pytest.raises(archive_tool.ArchiveWorkflowError, match=message):
        archive_tool.inspect_archive(archive, str(_fake_zstd(tmp_path)))


def test_archive_inspection_rejects_noncanonical_alias(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe-alias.tar.zst"
    archive.write_bytes(_tar_bytes([tarfile.TarInfo("EVRPTW_Dataset/./alias")]))

    with pytest.raises(archive_tool.ArchiveWorkflowError, match="not canonical"):
        archive_tool.inspect_archive(archive, str(_fake_zstd(tmp_path)))


def _init_arguments(tmp_path: Path, destination: Path) -> argparse.Namespace:
    archive = tmp_path / "release.tar.zst"
    archive.write_bytes(b"tiny-test-archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = tmp_path / "release.tar.zst.sha256"
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return argparse.Namespace(
        archive=archive,
        sha256_file=checksum,
        destination=destination,
        repo_root=REPOSITORY_ROOT,
        job_dir=destination / ".evrptw_restore_us11city",
        python_bin=Path(sys.executable),
        zstd_bin=Path("/bin/true"),
        workers=2,
        families_per_worker_task=3,
        session="test-restore-session",
    )


def test_job_init_rejects_unowned_existing_dataset(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    (destination / "EVRPTW_Dataset").mkdir(parents=True)

    with pytest.raises(archive_tool.ArchiveWorkflowError, match="unowned existing dataset"):
        archive_tool.initialize_job(_init_arguments(tmp_path, destination))


def test_job_init_rejects_dangling_symlink_target(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "EVRPTW_Dataset").symlink_to(tmp_path / "missing-target")

    with pytest.raises(archive_tool.ArchiveWorkflowError, match="symlinked dataset"):
        archive_tool.initialize_job(_init_arguments(tmp_path, destination))


def test_job_init_accepts_only_matching_archive_provenance(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    target = destination / "EVRPTW_Dataset"
    target.mkdir(parents=True)
    arguments = _init_arguments(tmp_path, destination)
    expected = hashlib.sha256(arguments.archive.read_bytes()).hexdigest()
    provenance = target / ".archive_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema": archive_tool.PROVENANCE_SCHEMA,
                "archive_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(archive_tool.ArchiveWorkflowError, match="another archive"):
        archive_tool.initialize_job(arguments)

    provenance.write_text(
        json.dumps(
            {
                "schema": archive_tool.PROVENANCE_SCHEMA,
                "archive_sha256": expected,
            }
        ),
        encoding="utf-8",
    )
    archive_tool.initialize_job(arguments)
    job = json.loads((arguments.job_dir / "job.json").read_text(encoding="utf-8"))
    assert job["archive_sha256_expected"] == expected
    assert job["workers"] == 2
    assert job["families_per_worker_task"] == 3


def test_worker_rechecks_provenance_after_init(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    arguments = _init_arguments(tmp_path, destination)
    archive_tool.initialize_job(arguments)
    (destination / "EVRPTW_Dataset").mkdir()

    with pytest.raises(archive_tool.ArchiveWorkflowError, match="unowned existing"):
        archive_tool.run_job(arguments.job_dir)

    state = json.loads((arguments.job_dir / "status.json").read_text(encoding="utf-8"))
    assert state["phase"] == "failed"


def test_auto_archive_background_start_uses_tmux_without_running_restore(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.zst"
    archive.write_bytes(b"tiny-placeholder")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = tmp_path / "release.tar.zst.sha256"
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    destination = tmp_path / "destination"
    fake_zstd = _fake_zstd(tmp_path)
    tmux_log = tmp_path / "tmux.log"
    fake_tmux = tmp_path / "fake-tmux"
    _write_executable(
        fake_tmux,
        "#!/bin/sh\n"
        "if [ \"$1\" = has-session ]; then exit 1; fi\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_TMUX_LOG\"\n",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_TMUX_LOG": str(tmux_log),
            "PYTHON_BIN": sys.executable,
            "ZSTD_BIN": str(fake_zstd),
            "TMUX_BIN": str(fake_tmux),
        }
    )

    result = subprocess.run(
        [
            str(AUTO_PATH),
            "archive",
            "start",
            "--archive",
            str(archive),
            "--sha256-file",
            str(checksum),
            "--destination",
            str(destination),
            "--workers",
            "7",
            "--families-per-worker-task",
            "11",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Restore started in background" in result.stdout
    assert "new-session -d -s evrptw-restore-" in tmux_log.read_text(encoding="utf-8")
    job_dir = destination / ".evrptw_restore_us11city"
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    state = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    assert job["workers"] == 7
    assert job["families_per_worker_task"] == 11
    assert state["phase"] == "queued"
    assert not (destination / "EVRPTW_Dataset").exists()


def test_status_reports_stopped_nonterminal_job_as_failure(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    arguments = _init_arguments(tmp_path, destination)
    archive_tool.initialize_job(arguments)
    fake_tmux = tmp_path / "fake-tmux"
    _write_executable(fake_tmux, "#!/bin/sh\nexit 1\n")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "TMUX_BIN": str(fake_tmux),
        }
    )

    result = subprocess.run(
        [
            str(AUTO_PATH),
            "archive",
            "status",
            "--destination",
            str(destination),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert result.returncode == 1
    assert "stopped before recording a terminal phase" in result.stderr


def test_concurrent_archive_launcher_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.zst"
    archive.write_bytes(b"tiny-placeholder")
    checksum = tmp_path / "release.tar.zst.sha256"
    checksum.write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
        encoding="utf-8",
    )
    destination = tmp_path / "destination"
    job_dir = destination / ".evrptw_restore_us11city"
    job_dir.mkdir(parents=True)
    lock_handle = (job_dir / "launcher.lock").open("a+")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "ZSTD_BIN": str(_fake_zstd(tmp_path)),
            "TMUX_BIN": "/bin/true",
        }
    )
    try:
        result = subprocess.run(
            [
                str(AUTO_PATH),
                "archive",
                "start",
                "--archive",
                str(archive),
                "--sha256-file",
                str(checksum),
                "--destination",
                str(destination),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    finally:
        lock_handle.close()

    assert result.returncode == 2
    assert "Another archive launcher is active" in result.stderr


def test_tiny_archive_end_to_end_exact_restore(tmp_path: Path) -> None:
    reconstruction_spec = importlib.util.spec_from_file_location(
        "reconstruction_test_fixture",
        REPOSITORY_ROOT
        / "EVRPTW_Dataset_Generator"
        / "tests"
        / "test_reconstruction.py",
    )
    assert reconstruction_spec is not None and reconstruction_spec.loader is not None
    fixture_module = importlib.util.module_from_spec(reconstruction_spec)
    reconstruction_spec.loader.exec_module(fixture_module)

    source, cle_source = fixture_module._build_tiny_full_dataset(tmp_path / "fixture")
    payload = tmp_path / "payload" / "EVRPTW_Dataset"
    cle_release = payload / "CLE_v1" / "us_11city"
    instances_release = payload / "Instances_v1" / "us_11city"
    cle_release.parent.mkdir(parents=True)
    shutil.copytree(cle_source, cle_release)
    contract = fixture_module.export_slim_dataset(
        source,
        instances_release,
        cle_root=cle_release,
        profile_path=fixture_module.PROFILE_PATH,
    )
    required_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    release = {
        "schema": archive_tool.RELEASE_MANIFEST_SCHEMA,
        "archive_layout": archive_tool.ARCHIVE_ROOT,
        "code_commit": required_commit,
        "family_count": contract["family_count"],
        "view_count": contract["view_count"],
        "matrix_payload_bytes_omitted": contract["source_matrix_bytes_omitted"],
    }
    (payload / "release_manifest.json").write_text(
        json.dumps(release), encoding="utf-8"
    )
    archive = tmp_path / "tiny-release.tar.zst"
    with tarfile.open(archive, mode="w") as output:
        output.add(payload, arcname=archive_tool.ARCHIVE_ROOT)
    checksum = tmp_path / "tiny-release.tar.zst.sha256"
    checksum.write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
        encoding="utf-8",
    )
    destination = tmp_path / "restored"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "ZSTD_BIN": str(_fake_zstd(tmp_path)),
            # Foreground/CI operation must not require tmux.
            "TMUX_BIN": str(tmp_path / "tmux-does-not-exist"),
        }
    )
    command = [
        str(AUTO_PATH),
        "archive",
        "start",
        "--archive",
        str(archive),
        "--sha256-file",
        str(checksum),
        "--destination",
        str(destination),
        "--workers",
        "1",
        "--families-per-worker-task",
        "1",
        "--foreground",
    ]
    first_run = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert first_run.returncode == 0, first_run.stderr

    instance_root = destination / "EVRPTW_Dataset" / "Instances_v1" / "us_11city"
    report = json.loads(
        (instance_root / "matrix_restore_report.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (destination / ".evrptw_restore_us11city" / "status.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["passed"] is True
    assert report["restored_count"] == 1
    assert state["phase"] == "succeeded"
    matrices = list(
        (instance_root / "materialized" / "families").glob("*/matrices/*.npy")
    )
    assert len(matrices) == 4

    # A prior success report is not sufficient: a removed complete cache must
    # be reconstructed on the next idempotent run.
    shutil.rmtree(matrices[0].parent)
    second_run = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert second_run.returncode == 0, second_run.stderr
    second_report = json.loads(
        (instance_root / "matrix_restore_report.json").read_text(encoding="utf-8")
    )
    assert second_report["passed"] is True
    assert second_report["restored_count"] == 1
    assert len(
        list((instance_root / "materialized" / "families").glob("*/matrices/*.npy"))
    ) == 4
