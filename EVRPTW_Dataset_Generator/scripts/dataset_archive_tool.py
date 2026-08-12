#!/usr/bin/env python3
"""Safely unpack a slim EVRPTW dataset archive and restore its matrix cache.

This module is the non-interactive worker behind ``restore_dataset_archive.sh``.
It deliberately separates archive extraction from the existing matrix
reconstruction implementation: archives are fully inspected before GNU tar is
allowed to extract them, and reconstruction still goes through ``auto.sh
restore`` with exact checksum validation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_ROOT = "EVRPTW_Dataset"
RELEASE_MANIFEST_SCHEMA = "evrptw_slim_dataset_release_manifest_v1"
RECONSTRUCTION_CONTRACT_SCHEMA = "cle_evrptw_slim_instances_v1"
PROVENANCE_SCHEMA = "evrptw_dataset_archive_provenance_v1"
STATE_SCHEMA = "evrptw_dataset_restore_state_v1"
MIN_SAFETY_BYTES = 5 * 1024**3
SHA_CHUNK_BYTES = 8 * 1024**2


class ArchiveWorkflowError(RuntimeError):
    """Raised when an archive or restore workflow violates its contract."""


class WorkflowInterrupted(BaseException):
    """Raised by the signal handler so the persistent state is updated."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveWorkflowError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArchiveWorkflowError(f"Expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(SHA_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_archive_snapshot(source: Path, job_dir: Path, expected_sha: str) -> Path:
    """Copy one immutable, checksum-bound archive snapshot into the job dir."""

    snapshot = job_dir / "archive.snapshot.tar.zst"
    if snapshot.is_file() and _sha256_file(snapshot) == expected_sha:
        return snapshot
    snapshot.unlink(missing_ok=True)
    temporary = job_dir / f".archive.snapshot.tmp-{os.getpid()}"
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
            for block in iter(lambda: input_handle.read(SHA_CHUNK_BYTES), b""):
                digest.update(block)
                output_handle.write(block)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise ArchiveWorkflowError(
                f"Archive SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        os.replace(temporary, snapshot)
        directory_fd = os.open(job_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return snapshot
    finally:
        temporary.unlink(missing_ok=True)


def _expected_sha256(path: Path) -> str:
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0].split()[0].lower()
    except (OSError, IndexError) as exc:
        raise ArchiveWorkflowError(f"Cannot read checksum sidecar {path}: {exc}") from exc
    if len(first) != 64 or any(character not in "0123456789abcdef" for character in first):
        raise ArchiveWorkflowError(f"Invalid SHA-256 in sidecar: {path}")
    return first


def _validate_member(member: tarfile.TarInfo, seen: set[str]) -> None:
    name = member.name
    if not name or "\\" in name or "\n" in name or "\r" in name:
        raise ArchiveWorkflowError(f"Unsafe archive member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or path.parts[0] != ARCHIVE_ROOT:
        raise ArchiveWorkflowError(
            f"Every archive member must be below {ARCHIVE_ROOT}/: {name!r}"
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveWorkflowError(f"Unsafe archive path component: {name!r}")
    canonical_name = path.as_posix()
    if name.rstrip("/") != canonical_name:
        raise ArchiveWorkflowError(f"Archive path is not canonical: {name!r}")
    if canonical_name in seen:
        raise ArchiveWorkflowError(f"Duplicate archive member: {name!r}")
    seen.add(canonical_name)
    if member.issym() or member.islnk():
        raise ArchiveWorkflowError(f"Archive links are not allowed: {name!r}")
    if not (member.isdir() or member.isreg()):
        raise ArchiveWorkflowError(f"Unsupported archive member type: {name!r}")


def inspect_archive(archive: Path, zstd_bin: str) -> dict[str, Any]:
    """Stream every tar header and return independently observed metadata."""

    command = [zstd_bin, "-dc", "--", str(archive)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    seen: set[str] = set()
    member_count = 0
    regular_file_count = 0
    logical_file_bytes = 0
    release_manifest: dict[str, Any] | None = None
    manifest_name = f"{ARCHIVE_ROOT}/release_manifest.json"
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
            for member in stream:
                _validate_member(member, seen)
                member_count += 1
                if member.isreg():
                    regular_file_count += 1
                    logical_file_bytes += int(member.size)
                if member.name == manifest_name:
                    if member.size > 1024 * 1024:
                        raise ArchiveWorkflowError("release_manifest.json is unexpectedly large")
                    extracted = stream.extractfile(member)
                    if extracted is None:
                        raise ArchiveWorkflowError("Cannot read release_manifest.json")
                    try:
                        release_manifest = json.loads(extracted.read().decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ArchiveWorkflowError(
                            f"Cannot parse {manifest_name}: {exc}"
                        ) from exc
    except BaseException:
        process.stdout.close()
        process.terminate()
        process.wait(timeout=30)
        raise
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise ArchiveWorkflowError(
            f"zstd could not read the archive (exit {return_code}): {stderr.strip()}"
        )
    if release_manifest is None:
        raise ArchiveWorkflowError(f"Archive is missing {manifest_name}")
    if not isinstance(release_manifest, dict):
        raise ArchiveWorkflowError("release_manifest.json must contain a JSON object")
    if release_manifest.get("schema") != RELEASE_MANIFEST_SCHEMA:
        raise ArchiveWorkflowError("Unsupported release manifest schema")
    if release_manifest.get("archive_layout") != ARCHIVE_ROOT:
        raise ArchiveWorkflowError("Release manifest archive_layout does not match the tar root")
    return {
        "member_count": member_count,
        "regular_file_count": regular_file_count,
        "logical_file_bytes": logical_file_bytes,
        "release_manifest": release_manifest,
    }


def _extract_archive(archive: Path, output: Path, zstd_bin: str) -> None:
    """Extract regular files/directories while revalidating the live stream.

    The archive is inspected in a separate pass before extraction.  We still
    validate every header here instead of delegating extraction to ``tar`` so
    replacing the archive between the two passes cannot introduce a link,
    special file, traversal, duplicate, or alternate top-level root.
    """

    output.mkdir(parents=True, exist_ok=False)
    decoder = subprocess.Popen(
        [zstd_bin, "-dc", "--", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=decoder.stdout, mode="r|") as stream:
            for member in stream:
                _validate_member(member, seen)
                relative = PurePosixPath(member.name)
                target = output.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = stream.extractfile(member)
                if extracted is None:
                    raise ArchiveWorkflowError(
                        f"Cannot read archive member payload: {member.name!r}"
                    )
                with target.open("xb") as destination:
                    shutil.copyfileobj(extracted, destination, length=1024 * 1024)
    except BaseException:
        decoder.stdout.close()
        decoder.terminate()
        decoder.wait(timeout=30)
        raise
    decoder_stderr = decoder.stderr.read() if decoder.stderr else b""
    decoder_return_code = decoder.wait()
    if decoder_return_code != 0:
        details = decoder_stderr.decode("utf-8", errors="replace")
        raise ArchiveWorkflowError(
            f"Archive extraction failed (zstd={decoder_return_code}): {details.strip()}"
        )


def validate_dataset_layout(dataset_root: Path) -> dict[str, Any]:
    release = _read_json(dataset_root / "release_manifest.json")
    if release.get("schema") != RELEASE_MANIFEST_SCHEMA:
        raise ArchiveWorkflowError("Extracted release_manifest.json has the wrong schema")
    cle_root = dataset_root / "CLE_v1" / "us_11city"
    instance_root = dataset_root / "Instances_v1" / "us_11city"
    contract_path = instance_root / "_reconstruction" / "reconstruction_contract.json"
    if not (cle_root / "cities").is_dir():
        raise ArchiveWorkflowError("Extracted archive is missing CLE_v1/us_11city/cities")
    if not (instance_root / "materialized" / "families").is_dir():
        raise ArchiveWorkflowError("Extracted archive is missing materialized families")
    contract = _read_json(contract_path)
    if contract.get("schema") != RECONSTRUCTION_CONTRACT_SCHEMA:
        raise ArchiveWorkflowError("Unsupported reconstruction contract schema")
    release_family_count = int(release.get("family_count", -1))
    contract_family_count = int(contract.get("family_count", -2))
    if release_family_count <= 0 or release_family_count != contract_family_count:
        raise ArchiveWorkflowError("Release and reconstruction family counts disagree")
    if int(release.get("view_count", -1)) != int(contract.get("view_count", -2)):
        raise ArchiveWorkflowError("Release and reconstruction view counts disagree")
    family_count = sum(
        1
        for child in (instance_root / "materialized" / "families").iterdir()
        if child.is_dir() and (child / "family_manifest.json").is_file()
    )
    if family_count != release_family_count:
        raise ArchiveWorkflowError(
            f"Expected {release_family_count} family directories, found {family_count}"
        )
    return {
        "release_manifest": release,
        "reconstruction_contract": contract,
        "cle_root": str(cle_root),
        "instance_root": str(instance_root),
        "family_count": family_count,
    }


def validate_dataset_provenance(dataset_root: Path, expected_sha: str) -> None:
    if dataset_root.is_symlink():
        raise ArchiveWorkflowError(
            f"Refusing to reuse a symlinked dataset tree: {dataset_root}"
        )
    provenance_path = dataset_root / ".archive_provenance.json"
    if not provenance_path.is_file():
        raise ArchiveWorkflowError(
            f"Refusing to reuse an unowned existing dataset tree: {dataset_root}"
        )
    provenance = _read_json(provenance_path)
    if provenance.get("schema") != PROVENANCE_SCHEMA:
        raise ArchiveWorkflowError("Existing dataset provenance schema is unsupported")
    if provenance.get("archive_sha256") != expected_sha:
        raise ArchiveWorkflowError("Existing dataset was extracted from another archive")


def _check_code_revision(repo_root: Path, required_commit: str) -> None:
    if not required_commit:
        raise ArchiveWorkflowError("Release manifest does not declare code_commit")
    check = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", required_commit, "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    if check.returncode != 0:
        raise ArchiveWorkflowError(
            f"Repository HEAD is not a descendant of required dataset commit {required_commit}"
        )


def _state_path(job_dir: Path) -> Path:
    return job_dir / "status.json"


def _update_state(job_dir: Path, phase: str, message: str, **extra: Any) -> None:
    path = _state_path(job_dir)
    state: dict[str, Any] = {}
    if path.exists():
        try:
            state = _read_json(path)
        except ArchiveWorkflowError:
            state = {}
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state.update(
        {
            "schema": STATE_SCHEMA,
            "phase": phase,
            "message": message,
            "updated_at": now,
        }
    )
    state.setdefault("started_at", now)
    state.update(extra)
    _atomic_write_json(path, state)
    print(f"[{now}] {phase}: {message}", flush=True)


def _load_config(job_dir: Path) -> dict[str, Any]:
    return _read_json(job_dir / "job.json")


def initialize_job(args: argparse.Namespace) -> None:
    archive = args.archive.expanduser().resolve(strict=True)
    checksum = args.sha256_file.expanduser().resolve(strict=True)
    destination = args.destination.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve(strict=True)
    if destination == Path("/"):
        raise ArchiveWorkflowError("Refusing to use / as the extraction destination")
    if args.workers <= 0 or args.families_per_worker_task <= 0:
        raise ArchiveWorkflowError("Worker counts must be positive")
    expected_sha = _expected_sha256(checksum)
    destination.mkdir(parents=True, exist_ok=True)
    job_dir = args.job_dir.expanduser().resolve()
    expected_job_parent = destination / ".evrptw_restore_us11city"
    if job_dir != expected_job_parent:
        raise ArchiveWorkflowError(f"Unexpected job directory: {job_dir}")
    target = destination / ARCHIVE_ROOT
    if target.is_symlink():
        raise ArchiveWorkflowError(
            f"Refusing to replace or reuse a symlinked dataset target: {target}"
        )
    if target.exists():
        validate_dataset_provenance(target, expected_sha)
    job_dir.mkdir(parents=True, exist_ok=True)
    # Do not rewrite a live worker's configuration. This lock also protects
    # foreground starts on systems where tmux is intentionally unavailable.
    worker_lock_path = job_dir / "worker.lock"
    with worker_lock_path.open("a+") as worker_lock:
        try:
            fcntl.flock(worker_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveWorkflowError("Another restore worker already owns this job") from exc
        config = {
            "archive": str(archive),
            "archive_size": archive.stat().st_size,
            "archive_sha256_expected": expected_sha,
            "checksum_file": str(checksum),
            "destination": str(destination),
            "dataset_root": str(target),
            "repo_root": str(repo_root),
            "python_bin": str(Path(args.python_bin).expanduser().resolve(strict=True)),
            "zstd_bin": str(Path(args.zstd_bin).expanduser().resolve(strict=True)),
            "workers": int(args.workers),
            "families_per_worker_task": int(args.families_per_worker_task),
            "session": args.session,
        }
        _atomic_write_json(job_dir / "job.json", config)
        _update_state(
            job_dir,
            "queued",
            "Restore job configured",
            session=args.session,
            archive=str(archive),
            archive_sha256=expected_sha,
            destination=str(destination),
            log=str(job_dir / "restore.log"),
        )


def _matrix_payload_bytes(instance_root: Path) -> int:
    total = 0
    families = instance_root / "materialized" / "families"
    if not families.is_dir():
        return 0
    for path in families.glob("*/matrices/*.npy"):
        if not (path.parent.parent / "family_manifest.json").is_file():
            continue
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _verify_restore_report(instance_root: Path, expected_families: int) -> dict[str, Any]:
    report = _read_json(instance_root / "matrix_restore_report.json")
    if not report.get("passed"):
        raise ArchiveWorkflowError("Matrix restore report did not pass")
    selected = int(report.get("selected_family_count", -1))
    restored = int(report.get("restored_count", -1))
    reused = int(report.get("reused_count", -1))
    if selected != expected_families or restored + reused != expected_families:
        raise ArchiveWorkflowError(
            "Matrix restore report does not cover every released family "
            f"(selected={selected}, restored={restored}, reused={reused})"
        )
    return report


def run_job(job_dir: Path) -> None:
    job_dir = job_dir.expanduser().resolve(strict=True)
    config = _load_config(job_dir)
    lock_path = job_dir / "worker.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveWorkflowError("Another restore worker already owns this job") from exc

        def interrupted(signum: int, _frame: Any) -> None:
            raise WorkflowInterrupted(f"received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        archive = Path(config["archive"])
        destination = Path(config["destination"])
        dataset_root = Path(config["dataset_root"])
        repo_root = Path(config["repo_root"])
        expected_sha = str(config["archive_sha256_expected"])
        staging = job_dir / "staging"
        try:
            _update_state(job_dir, "checksum", "Verifying archive SHA-256")
            if dataset_root.is_symlink():
                raise ArchiveWorkflowError(
                    f"Refusing to replace or reuse a symlinked dataset target: {dataset_root}"
                )
            if dataset_root.exists():
                validate_dataset_provenance(dataset_root, expected_sha)
                actual_sha = _sha256_file(archive)
                if actual_sha != expected_sha:
                    raise ArchiveWorkflowError(
                        f"Archive SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
                    )
                inspected_archive = archive
            else:
                inspected_archive = _verified_archive_snapshot(
                    archive, job_dir, expected_sha
                )
                actual_sha = expected_sha

            _update_state(job_dir, "preflight", "Inspecting every archive member")
            inspection = inspect_archive(inspected_archive, config["zstd_bin"])
            release = inspection["release_manifest"]
            omitted = int(release.get("matrix_payload_bytes_omitted", -1))
            if omitted <= 0:
                raise ArchiveWorkflowError("Release manifest has no valid omitted matrix size")
            _check_code_revision(repo_root, str(release.get("code_commit", "")))

            existing_matrix_bytes = 0
            if dataset_root.exists():
                # Recheck provenance in the worker. The target may have
                # appeared or been replaced after the launcher initialized the
                # job but before this persistent process acquired its lock.
                validate_dataset_provenance(dataset_root, expected_sha)
                layout = validate_dataset_layout(dataset_root)
                existing_matrix_bytes = _matrix_payload_bytes(Path(layout["instance_root"]))
                extraction_bytes = 0
            else:
                extraction_bytes = int(inspection["logical_file_bytes"])
            required_free = extraction_bytes + max(0, omitted - existing_matrix_bytes) + MIN_SAFETY_BYTES
            available = shutil.disk_usage(destination).free
            if available < required_free:
                raise ArchiveWorkflowError(
                    "Insufficient free space: "
                    f"need at least {required_free} bytes, have {available} bytes"
                )
            _update_state(
                job_dir,
                "preflight",
                "Archive and disk-space preflight passed",
                archive_member_count=inspection["member_count"],
                archive_logical_bytes=inspection["logical_file_bytes"],
                required_free_bytes=required_free,
                available_free_bytes=available,
                family_count=int(release["family_count"]),
            )

            if not dataset_root.exists():
                if staging.exists():
                    shutil.rmtree(staging)
                _update_state(job_dir, "extracting", "Extracting archive into private staging")
                _extract_archive(
                    inspected_archive,
                    staging,
                    config["zstd_bin"],
                )
                # Bind the published tree to the checksum that was verified at
                # job start.  A changed archive is discarded with staging.
                extracted_archive_sha = _sha256_file(inspected_archive)
                if extracted_archive_sha != expected_sha:
                    raise ArchiveWorkflowError(
                        "Archive changed while it was being inspected or extracted"
                    )
                staged_dataset = staging / ARCHIVE_ROOT
                _update_state(job_dir, "validating", "Validating extracted slim dataset")
                validate_dataset_layout(staged_dataset)
                provenance = {
                    "schema": PROVENANCE_SCHEMA,
                    "archive_sha256": actual_sha,
                    "archive_size": int(config["archive_size"]),
                    "release_manifest_schema": RELEASE_MANIFEST_SCHEMA,
                    "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                _atomic_write_json(staged_dataset / ".archive_provenance.json", provenance)
                if dataset_root.is_symlink() or dataset_root.exists():
                    raise ArchiveWorkflowError(
                        f"Dataset target appeared during extraction: {dataset_root}"
                    )
                os.rename(staged_dataset, dataset_root)
                staging.rmdir()
                destination_fd = os.open(destination, os.O_RDONLY)
                try:
                    os.fsync(destination_fd)
                finally:
                    os.close(destination_fd)
                inspected_archive.unlink(missing_ok=True)

            layout = validate_dataset_layout(dataset_root)
            validate_dataset_provenance(dataset_root, expected_sha)
            if inspected_archive != archive:
                inspected_archive.unlink(missing_ok=True)
            instance_root = Path(layout["instance_root"])
            expected_families = int(layout["family_count"])
            report_path = instance_root / "matrix_restore_report.json"
            _update_state(
                job_dir,
                "restoring",
                "Reconstructing all matrix families with exact validation",
                completed_matrix_families=sum(
                    1
                    for path in (instance_root / "materialized" / "families").glob(
                        "*/matrices"
                    )
                    if path.is_dir()
                    and (path.parent / "family_manifest.json").is_file()
                ),
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "CLE_ROOT": str(layout["cle_root"]),
                    "INSTANCE_OUTPUT_ROOT": str(instance_root),
                    "WORKERS": str(config["workers"]),
                    "FAMILIES_PER_WORKER_TASK": str(config["families_per_worker_task"]),
                    "PYTHON_BIN": str(config["python_bin"]),
                }
            )
            # Always enter the reconstruction verifier, even when a previous
            # success report exists. It hashes every existing matrix before
            # reuse, so a stale report cannot hide missing or corrupt data.
            result = subprocess.run(
                [str(repo_root / "auto.sh"), "restore"],
                cwd=repo_root,
                env=environment,
                check=False,
            )
            if result.returncode != 0:
                raise ArchiveWorkflowError(
                    f"Matrix restore command exited with status {result.returncode}"
                )
            report = _verify_restore_report(instance_root, expected_families)
            _update_state(
                job_dir,
                "succeeded",
                "Archive extraction and exact matrix restoration completed",
                exit_code=0,
                restored_count=int(report["restored_count"]),
                reused_count=int(report["reused_count"]),
                report=str(report_path),
            )
        except WorkflowInterrupted as exc:
            _update_state(job_dir, "interrupted", str(exc), exit_code=130)
            raise SystemExit(130) from None
        except BaseException as exc:
            _update_state(job_dir, "failed", str(exc), exit_code=1)
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)


def job_status(job_dir: Path, field: str | None = None) -> int:
    job_dir = job_dir.expanduser().resolve()
    state_path = _state_path(job_dir)
    if not state_path.is_file():
        raise ArchiveWorkflowError(f"No restore state found at {state_path}")
    state = _read_json(state_path)
    config = _load_config(job_dir)
    dataset_root = Path(config["dataset_root"])
    instance_root = dataset_root / "Instances_v1" / "us_11city"
    expected = int(state.get("family_count", 0))
    completed = 0
    families = instance_root / "materialized" / "families"
    if families.is_dir():
        completed = sum(
            1
            for path in families.glob("*/matrices")
            if path.is_dir() and (path.parent / "family_manifest.json").is_file()
        )
    state["completed_matrix_families"] = completed
    state["expected_matrix_families"] = expected
    state["log"] = str(job_dir / "restore.log")
    if field:
        value = state.get(field, "")
        print(value if isinstance(value, str) else json.dumps(value))
    else:
        print(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if state.get("phase") not in {"failed", "interrupted"} else 1


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--archive", type=Path, required=True)
    init.add_argument("--sha256-file", type=Path, required=True)
    init.add_argument("--destination", type=Path, required=True)
    init.add_argument("--repo-root", type=Path, required=True)
    init.add_argument("--job-dir", type=Path, required=True)
    init.add_argument("--python-bin", required=True)
    init.add_argument("--zstd-bin", required=True)
    init.add_argument("--workers", type=int, required=True)
    init.add_argument("--families-per-worker-task", type=int, required=True)
    init.add_argument("--session", required=True)

    run = commands.add_parser("run")
    run.add_argument("--job-dir", type=Path, required=True)

    status = commands.add_parser("status")
    status.add_argument("--job-dir", type=Path, required=True)
    status.add_argument("--field")

    inspect = commands.add_parser("inspect")
    inspect.add_argument("--archive", type=Path, required=True)
    inspect.add_argument("--zstd-bin", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    try:
        if args.command == "init":
            initialize_job(args)
        elif args.command == "run":
            run_job(args.job_dir)
        elif args.command == "status":
            raise SystemExit(job_status(args.job_dir, args.field))
        elif args.command == "inspect":
            print(
                json.dumps(
                    inspect_archive(args.archive.resolve(strict=True), args.zstd_bin),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except ArchiveWorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
