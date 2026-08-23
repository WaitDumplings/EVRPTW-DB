#!/usr/bin/env python3
"""Safely unpack a slim EVRPTW dataset archive and restore its matrix cache.

This module is the non-interactive worker behind ``restore_dataset_archive.sh``.
It deliberately separates archive extraction from the existing matrix
reconstruction implementation: archives are fully inspected before extraction,
and reconstruction still goes through ``auto.sh restore`` with structural and
feasibility validation. No archive or matrix file hashes are calculated.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_ROOT = "EVRPTW_Dataset"
RELEASE_MANIFEST_SCHEMA = "evrptw_slim_dataset_release_manifest_v1"
RECONSTRUCTION_CONTRACT_SCHEMA = "cle_evrptw_slim_instances_v1"
PROVENANCE_SCHEMA = "evrptw_dataset_archive_provenance_v2"
STATE_SCHEMA = "evrptw_dataset_restore_state_v1"
MIN_SAFETY_BYTES = 5 * 1024**3


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


def _archive_identity(path: Path) -> dict[str, int]:
    """Record cheap file identity fields without reading the archive payload."""

    status = path.stat()
    return {
        "device": int(status.st_dev),
        "inode": int(status.st_ino),
        "size": int(status.st_size),
        "mtime_ns": int(status.st_mtime_ns),
        "ctime_ns": int(status.st_ctime_ns),
    }


def _assert_archive_unchanged(path: Path, expected: dict[str, Any]) -> None:
    actual = _archive_identity(path)
    normalized = {key: int(value) for key, value in expected.items()}
    if actual != normalized:
        raise ArchiveWorkflowError(
            "Archive identity changed after preflight; restart with the stable file"
        )


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
    cle_root = dataset_root / "CLE_v2" / "us_11city"
    instance_root = dataset_root / "Instances_v2" / "us_11city"
    contract_path = instance_root / "_reconstruction" / "reconstruction_contract.json"
    if not (cle_root / "cities").is_dir():
        raise ArchiveWorkflowError("Extracted archive is missing CLE_v2/us_11city/cities")
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


def validate_dataset_provenance(dataset_root: Path, expected_release_id: str) -> None:
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
    if provenance.get("release_id") != expected_release_id:
        raise ArchiveWorkflowError("Existing dataset was extracted from another release")


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
    destination = args.destination.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve(strict=True)
    if destination == Path("/"):
        raise ArchiveWorkflowError("Refusing to use / as the extraction destination")
    if args.workers <= 0 or args.families_per_worker_task <= 0:
        raise ArchiveWorkflowError("Worker counts must be positive")
    identity = _archive_identity(archive)
    inspection = inspect_archive(archive, str(args.zstd_bin))
    release = inspection["release_manifest"]
    release_id = str(release.get("release_id", ""))
    if not release_id:
        raise ArchiveWorkflowError("Release manifest does not declare release_id")
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
        validate_dataset_provenance(target, release_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    worker_lock_path = job_dir / "worker.lock"
    with worker_lock_path.open("a+") as worker_lock:
        try:
            fcntl.flock(worker_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveWorkflowError("Another restore worker already owns this job") from exc
        config = {
            "archive": str(archive),
            "archive_identity": identity,
            "archive_logical_bytes": int(inspection["logical_file_bytes"]),
            "archive_member_count": int(inspection["member_count"]),
            "release_id": release_id,
            "release_manifest": release,
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
            "Restore job configured after full member inspection",
            session=args.session,
            archive=str(archive),
            release_id=release_id,
            destination=str(destination),
            log=str(job_dir / "restore.log"),
            file_hash_validation_performed=False,
        )


def _validate_regular_tree(root: Path, label: str) -> None:
    """Reject links and special files before copying release source data."""

    if not root.is_dir() or root.is_symlink():
        raise ArchiveWorkflowError(f"{label} is not a regular directory: {root}")
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ArchiveWorkflowError(f"{label} contains a link: {path}")
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ArchiveWorkflowError(f"{label} contains a special file: {path}")


def _tree_file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _source_acceptance(cle_root: Path, instance_root: Path) -> dict[str, Any]:
    cle_index = _read_json(cle_root / "cle_index.json")
    if (
        cle_index.get("status") != "complete"
        or int(cle_index.get("verified_cle_count", -1)) != 11
        or cle_index.get("failures") != []
    ):
        raise ArchiveWorkflowError(
            "CLE acceptance failed; require status=complete, verified_cle_count=11, "
            "and failures=[]"
        )

    stage2 = _read_json(instance_root / "stage2_run_report.json")
    unresolved = stage2.get("unresolved_family_ids")
    runtime = dict(stage2.get("runtime_contract") or {})
    if (
        stage2.get("passed") is not True
        or stage2.get("terminal_report_committed") is not True
        or not isinstance(unresolved, list)
        or unresolved
        or int(runtime.get("remaining_process_group_count", -1)) != 0
    ):
        raise ArchiveWorkflowError(
            "Stage-2 acceptance failed; require terminal passed report, no unresolved "
            "families, and no remaining process groups"
        )

    phase1 = _read_json(instance_root / "reports" / "phase1" / "summary.json")
    if phase1.get("all_hard_gates_passed") is not True:
        raise ArchiveWorkflowError(
            "Phase-1 acceptance failed; not every hard correctness gate passed"
        )
    construct = _read_json(
        instance_root
        / "reports"
        / "stage2_repair"
        / "stage2_acceptance_v3_construct_valid.json"
    )
    if (
        construct.get("schema") != "stage2_acceptance_v3_construct_valid"
        or construct.get("passed") is not True
    ):
        raise ArchiveWorkflowError("Construct-valid v3 acceptance did not pass")
    feasibility = _read_json(
        instance_root
        / "reports"
        / "post_generation"
        / "full_corpus_feasibility_gate_v1.json"
    )
    if feasibility.get("passed") is not True:
        raise ArchiveWorkflowError("Full-corpus feasibility watcher gate did not pass")
    c3 = _read_json(
        instance_root / "reports" / "stage2_repair" / "c3_joint_support_full.json"
    )
    if c3.get("passed") is not True:
        raise ArchiveWorkflowError("C3 full-plan joint-support gate did not pass")
    return {
        "cle_index": cle_index,
        "stage2": stage2,
        "phase1": phase1,
        "construct_valid_v3": construct,
        "feasibility": feasibility,
        "c3": c3,
    }


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    commit = result.stdout.strip().lower()
    if result.returncode != 0 or len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ArchiveWorkflowError(f"Cannot resolve repository HEAD: {result.stderr.strip()}")
    return commit


def _write_tar_zstd(
    dataset_root: Path,
    archive: Path,
    *,
    zstd_bin: str,
    compression_threads: int,
    compression_level: int,
) -> None:
    """Stream a link-free, single-root tar through zstd into an atomic output."""

    temporary = archive.with_name(f".{archive.name}.part-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    process: subprocess.Popen[bytes] | None = None
    try:
        with temporary.open("xb") as compressed:
            process = subprocess.Popen(
                [
                    zstd_bin,
                    f"-{compression_level}",
                    f"-T{compression_threads}",
                    "--no-progress",
                    "-c",
                ],
                stdin=subprocess.PIPE,
                stdout=compressed,
                stderr=subprocess.PIPE,
            )
            assert process.stdin is not None
            assert process.stderr is not None
            try:
                with tarfile.open(
                    fileobj=process.stdin,
                    mode="w|",
                    format=tarfile.PAX_FORMAT,
                    dereference=False,
                ) as tar:
                    paths = [dataset_root, *sorted(dataset_root.rglob("*"))]
                    for path in paths:
                        relative = path.relative_to(dataset_root)
                        arcname = (
                            ARCHIVE_ROOT
                            if not relative.parts
                            else f"{ARCHIVE_ROOT}/{relative.as_posix()}"
                        )
                        info = tar.gettarinfo(str(path), arcname=arcname)
                        if not (info.isdir() or info.isreg()):
                            raise ArchiveWorkflowError(
                                f"Release staging contains an unsupported file: {path}"
                            )
                        if info.isreg():
                            with path.open("rb") as source:
                                tar.addfile(info, source)
                        else:
                            tar.addfile(info)
                process.stdin.close()
                stderr = process.stderr.read().decode("utf-8", errors="replace")
                return_code = process.wait()
            except BaseException:
                if not process.stdin.closed:
                    process.stdin.close()
                process.terminate()
                process.wait(timeout=30)
                raise
            if return_code != 0:
                raise ArchiveWorkflowError(
                    f"zstd compression failed (exit {return_code}): {stderr.strip()}"
                )
            compressed.flush()
            os.fsync(compressed.fileno())
        try:
            os.link(temporary, archive)
        except FileExistsError as exc:
            raise ArchiveWorkflowError(f"Archive appeared during creation: {archive}") from exc
        temporary.unlink()
        directory_fd = os.open(archive.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        temporary.unlink(missing_ok=True)


def create_release_archive(args: argparse.Namespace) -> dict[str, Any]:
    """Create one accepted CLE + slim-instance release archive without hashing."""

    started = time.perf_counter()
    cle_root = args.cle_root.expanduser().resolve(strict=True)
    instance_root = args.instance_root.expanduser().resolve(strict=True)
    profile = args.profile.expanduser().resolve(strict=True)
    repo_root = args.repo_root.expanduser().resolve(strict=True)
    archive = args.archive.expanduser().resolve()
    if archive.suffixes[-2:] != [".tar", ".zst"]:
        raise ArchiveWorkflowError("Release archive name must end in .tar.zst")
    if args.compression_threads <= 0 or not 1 <= args.compression_level <= 19:
        raise ArchiveWorkflowError("Compression threads must be positive and level must be 1..19")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise ArchiveWorkflowError(f"Refusing to overwrite an archive: {archive}")
    for source in (cle_root, instance_root):
        if source in archive.parents:
            raise ArchiveWorkflowError("Release archive must be outside its source trees")

    _validate_regular_tree(cle_root, "CLE source")
    _validate_regular_tree(instance_root, "Stage-2 source")
    acceptance = _source_acceptance(cle_root, instance_root)
    commit = _git_head(repo_root)

    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{archive.name}.build-", dir=archive.parent)
    )
    archive_created = False
    try:
        payload = staging_parent / ARCHIVE_ROOT
        cle_release = payload / "CLE_v2" / "us_11city"
        instances_release = payload / "Instances_v2" / "us_11city"
        cle_release.parent.mkdir(parents=True)
        shutil.copytree(cle_root, cle_release)

        generator_src = repo_root / "EVRPTW_Dataset_Generator" / "src"
        if str(generator_src) not in sys.path:
            sys.path.insert(0, str(generator_src))
        from evrptw_stage2.reconstruction import export_slim_dataset  # noqa: PLC0415

        contract = export_slim_dataset(
            instance_root,
            instances_release,
            cle_root=cle_release,
            profile_path=profile,
        )
        family_count = int(contract["family_count"])
        view_count = int(contract["view_count"])
        stage2 = acceptance["stage2"]
        phase1 = acceptance["phase1"]
        selected_count = int(stage2.get("execution", {}).get("selected_family_count", -1))
        verified_count = len(stage2.get("verified", []))
        successful_count = int(phase1.get("successful_parent_family_count", -1))
        if not (
            selected_count == verified_count == successful_count == family_count
        ):
            raise ArchiveWorkflowError(
                "Acceptance reports do not cover every slim family "
                f"(slim={family_count}, selected={selected_count}, "
                f"verified={verified_count}, phase1={successful_count})"
            )
        if _matrix_payload_bytes(instances_release) != 0:
            raise ArchiveWorkflowError("Slim export unexpectedly contains matrix payload files")

        release_id = (
            f"evrptw-us11city-{commit[:12]}-"
            f"{stage2.get('runtime_run_id', 'deterministic-run')}-"
            f"{family_count}f-{view_count}v"
        )
        release = {
            "schema": RELEASE_MANIFEST_SCHEMA,
            "release_id": release_id,
            "archive_layout": ARCHIVE_ROOT,
            "code_commit": commit,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "dataset_mode": str(stage2.get("mode", "unknown")),
            "cities": contract["cities"],
            "family_count": family_count,
            "view_count": view_count,
            "matrix_names": contract["matrix_names"],
            "matrix_file_count_omitted": family_count * len(contract["matrix_names"]),
            "matrix_payload_bytes_omitted": int(contract["source_matrix_bytes_omitted"]),
            "cle_payload_bytes": _tree_file_bytes(cle_release),
            "slim_instance_payload_bytes": _tree_file_bytes(instances_release),
            "reference_profile_id": contract["reference_profile"]["profile_id"],
            "file_hash_validation_performed": False,
            "acceptance": {
                "cle_status": acceptance["cle_index"]["status"],
                "verified_cle_count": int(
                    acceptance["cle_index"]["verified_cle_count"]
                ),
                "stage2_passed": True,
                "phase1_all_hard_gates_passed": True,
                "construct_valid_v3_passed": True,
                "full_corpus_feasibility_passed": True,
                "c3_joint_support_passed": True,
                "spatial_and_operational_similarity_diagnostics_are_report_only": True,
            },
        }
        _atomic_write_json(payload / "release_manifest.json", release)
        markdown_fence = chr(96) * 3
        (payload / "README_SLIM.md").write_text(
            "# EVRPTW-DB portable slim dataset\n\n"
            "This archive contains the accepted 11-city CLE and all lightweight "
            "Stage-2 family/view parameters. Dense matrix files are intentionally "
            "omitted and must be reconstructed with the repository code.\n\n"
            "After cloning a compatible repository revision, "
            "run:\n\n"
            f"{markdown_fence}bash\n"
            "./auto.sh archive start --archive /path/to/release.tar.zst "
            "--destination /data --workers 30\n"
            "./auto.sh archive wait --destination /data\n"
            f"{markdown_fence}\n\n"
            "Do not use the restored dataset until archive status is succeeded.\n",
            encoding="utf-8",
        )
        validate_dataset_layout(payload)
        _write_tar_zstd(
            payload,
            archive,
            zstd_bin=str(args.zstd_bin),
            compression_threads=int(args.compression_threads),
            compression_level=int(args.compression_level),
        )
        archive_created = True
        inspection = inspect_archive(archive, str(args.zstd_bin))
        if inspection["release_manifest"] != release:
            raise ArchiveWorkflowError("Archived release manifest differs from staging")
        return {
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "logical_file_bytes": int(inspection["logical_file_bytes"]),
            "family_count": family_count,
            "view_count": view_count,
            "release_id": release_id,
            "matrix_payload_bytes_omitted": int(contract["source_matrix_bytes_omitted"]),
            "file_hash_validation_performed": False,
            "wall_seconds": time.perf_counter() - started,
        }
    except BaseException:
        if archive_created:
            archive.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


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
        release_id = str(config["release_id"])
        expected_identity = dict(config["archive_identity"])
        staging = job_dir / "staging"
        try:
            _update_state(job_dir, "preflight", "Checking stable archive file identity")
            _assert_archive_unchanged(archive, expected_identity)
            if dataset_root.is_symlink():
                raise ArchiveWorkflowError(
                    f"Refusing to replace or reuse a symlinked dataset target: {dataset_root}"
                )
            if dataset_root.exists():
                validate_dataset_provenance(dataset_root, release_id)
            release = dict(config["release_manifest"])
            inspection = {
                "release_manifest": release,
                "member_count": int(config["archive_member_count"]),
                "logical_file_bytes": int(config["archive_logical_bytes"]),
            }
            omitted = int(release.get("matrix_payload_bytes_omitted", -1))
            if omitted <= 0:
                raise ArchiveWorkflowError("Release manifest has no valid omitted matrix size")
            _check_code_revision(repo_root, str(release.get("code_commit", "")))

            existing_matrix_bytes = 0
            if dataset_root.exists():
                # Recheck provenance in the worker. The target may have
                # appeared or been replaced after the launcher initialized the
                # job but before this persistent process acquired its lock.
                validate_dataset_provenance(dataset_root, release_id)
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
                    archive,
                    staging,
                    config["zstd_bin"],
                )
                _assert_archive_unchanged(archive, expected_identity)
                staged_dataset = staging / ARCHIVE_ROOT
                _update_state(job_dir, "validating", "Validating extracted slim dataset")
                validate_dataset_layout(staged_dataset)
                provenance = {
                    "schema": PROVENANCE_SCHEMA,
                    "release_id": release_id,
                    "archive_size": int(expected_identity["size"]),
                    "file_hash_validation_performed": False,
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

            layout = validate_dataset_layout(dataset_root)
            validate_dataset_provenance(dataset_root, release_id)
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
            # Always enter structural and feasibility validation, even when a
            # previous success report exists; file hashes are intentionally disabled.
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
    instance_root = dataset_root / "Instances_v2" / "us_11city"
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
    create = commands.add_parser("create")
    create.add_argument("--cle-root", type=Path, required=True)
    create.add_argument("--instance-root", type=Path, required=True)
    create.add_argument("--profile", type=Path, required=True)
    create.add_argument("--archive", type=Path, required=True)
    create.add_argument("--repo-root", type=Path, required=True)
    create.add_argument("--zstd-bin", required=True)
    create.add_argument("--compression-threads", type=int, default=12)
    create.add_argument("--compression-level", type=int, default=9)

    init = commands.add_parser("init")
    init.add_argument("--archive", type=Path, required=True)
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
        if args.command == "create":
            print(
                json.dumps(
                    create_release_archive(args),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "init":
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
