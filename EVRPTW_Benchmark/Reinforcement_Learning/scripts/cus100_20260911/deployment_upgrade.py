"""One reviewed GPU-placement update for the original Cus100 deployment.

This is deliberately not a general mechanism for accepting changed source.
The old package remains the trust anchor, and every changed source record must
match the committed, finite update descriptor exactly.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from .source_snapshot import digest_file, source_files, source_identity


SCRIPT_ROOT = "EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus100_20260911"
JOB_MANIFEST = SCRIPT_ROOT + "/cus100_seed1234_jobs.jsonl"
DESCRIPTOR = SCRIPT_ROOT + "/deployment_gpu12_upgrade.json"
UPGRADE_ID = "2080ti_3_1_rrnco_gpu12_v1"
SCHEMA = "cus100_gpu12_deployment_upgrade_v1"
BASE_SOURCE_VERSION = "sha256:2ffee534529dbfbe05d7ef8fd2a4742facd5a96e265da929bc863316f3401d44"
BASE_JOB_MANIFEST_SHA256 = "8d1954c3dccd2da20bd3f6ed93bcae0700e13ffca91de0132d276daf2f6c60aa"
ALLOWED_CHANGED_PATHS = frozenset({
    SCRIPT_ROOT + "/launch.py",
    JOB_MANIFEST,
    SCRIPT_ROOT + "/deployment_upgrade.py",
    SCRIPT_ROOT + "/README.md",
    SCRIPT_ROOT + "/CUS100_SCOPE_AND_PREFLIGHT.md",
    SCRIPT_ROOT + "/CUS100_RUN_REGISTRY.md",
    SCRIPT_ROOT + "/CUS100_SMOKE_REPORT.md",
    "EVRPTW_Benchmark/Reinforcement_Learning/tests/test_deployment_upgrade.py",
})


def _fail(message):
    raise RuntimeError("Cus100 deployment upgrade refused: " + message)


def _relative_path(repo, value):
    value = Path(value)
    if value.is_absolute() or ".." in value.parts:
        _fail("expected a repository-relative provenance path")
    path = (repo / value).resolve()
    if not path.is_relative_to(repo):
        _fail("provenance path escapes the repository")
    return path


def _record_map(records):
    if not isinstance(records, list):
        _fail("source records must be a list")
    result = {}
    for record in records:
        name = record.get("path") if isinstance(record, dict) else None
        if not isinstance(name, str) or not name or name in result:
            _fail("invalid or duplicate source record path")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            _fail("source record path is not repository-relative")
        result[name] = record
    return result


def _check_snapshot_identity(snapshot):
    records = snapshot.get("files")
    _record_map(records)
    digest = source_identity(records)
    if snapshot.get("source_sha256") != digest or snapshot.get("source_version") != "sha256:" + digest:
        _fail("source provenance records do not match their identity")


def _load_committed_descriptor(repo):
    path = _relative_path(repo, DESCRIPTOR)
    payload = path.read_bytes()
    try:
        committed = subprocess.check_output(["git", "show", "HEAD:" + DESCRIPTOR], cwd=repo)
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", DESCRIPTOR], cwd=repo)
    except subprocess.CalledProcessError as error:
        _fail("upgrade descriptor is not present in Git HEAD: " + str(error))
    if dirty or committed != payload:
        _fail("upgrade descriptor differs from its committed version")
    descriptor = json.loads(payload)
    expected_keys = {"schema", "upgrade_id", "base_source_version", "base_job_manifest_sha256",
                     "target_job_manifest_sha256", "job_manifest", "changes"}
    if set(descriptor) != expected_keys:
        _fail("unexpected upgrade descriptor fields")
    expected = {"schema": SCHEMA, "upgrade_id": UPGRADE_ID,
                "base_source_version": BASE_SOURCE_VERSION,
                "base_job_manifest_sha256": BASE_JOB_MANIFEST_SHA256,
                "job_manifest": JOB_MANIFEST}
    if any(descriptor.get(key) != value for key, value in expected.items()):
        _fail("upgrade descriptor is not for the known GPU-placement update")
    return descriptor, hashlib.sha256(payload).hexdigest()


def _validate_delta(old_records, new_records, descriptor):
    before = _record_map(old_records)
    after = _record_map(new_records)
    if DESCRIPTOR in before or after.get(DESCRIPTOR, {}).get("kind") != "file":
        _fail("descriptor must be the one declared new regular file")
    actual = {}
    for name in sorted(set(before) | set(after)):
        if name == DESCRIPTOR or before.get(name) == after.get(name):
            continue
        if name not in ALLOWED_CHANGED_PATHS:
            _fail("source change is outside deployment-only scope: " + name)
        actual[name] = {"path": name, "before": before.get(name), "after": after.get(name)}
    declared = {}
    if not isinstance(descriptor["changes"], list):
        _fail("descriptor changes must be a list")
    for change in descriptor["changes"]:
        if not isinstance(change, dict) or set(change) != {"path", "before", "after"}:
            _fail("invalid declared source change")
        name = change["path"]
        if name not in ALLOWED_CHANGED_PATHS or name in declared:
            _fail("unknown or duplicate declared source change")
        declared[name] = change
    if actual != declared:
        _fail("actual source changes differ from the reviewed before/after records")
    if JOB_MANIFEST not in actual or SCRIPT_ROOT + "/launch.py" not in actual:
        _fail("update lacks the required launcher and GPU assignment changes")
    return sorted(actual)


def _old_jobs(repo, old_source, old_records):
    # The archive is located next to its portable provenance manifest. Its
    # original absolute archive path can name a different server and is ignored.
    manifest_path = _relative_path(repo, old_source["source_manifest"])
    archive_path = manifest_path.with_suffix(".tar.gz")
    provenance = json.loads(manifest_path.read_text())
    archive_digest = provenance.get("archive_sha256")
    if not archive_digest or digest_file(archive_path) != archive_digest:
        _fail("original source archive checksum differs")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.name == JOB_MANIFEST]
        if len(members) != 1 or not members[0].isfile():
            _fail("original archive does not contain one regular job manifest")
        stream = archive.extractfile(members[0])
        if stream is None:
            _fail("cannot read original job manifest")
        with stream:
            payload = stream.read()
    digest = hashlib.sha256(payload).hexdigest()
    record = _record_map(old_records).get(JOB_MANIFEST, {})
    if digest != BASE_JOB_MANIFEST_SHA256 or record.get("sha256") != digest:
        _fail("original job manifest checksum differs")
    return payload


def _validate_jobs(old_payload, new_payload):
    old = [json.loads(line) for line in old_payload.splitlines() if line.strip()]
    new = [json.loads(line) for line in new_payload.splitlines() if line.strip()]
    ids = {job.get("experiment_id") for job in old}
    if len(old) != 10 or len(ids) != 10 or not {"TR10", "TR09"}.issubset(ids):
        _fail("original manifest is not the ten declared jobs")
    expected = copy.deepcopy(old)
    for job in expected:
        if job["experiment_id"] in {"TR10", "TR09"}:
            original_gpu, target_gpu = (0, 1) if job["experiment_id"] == "TR10" else (1, 2)
            if (job.get("gpu"), job.get("global_slot"), job.get("server"), job.get("method")) != (
                    original_gpu, original_gpu, "2080ti_3_1", "rrnco"):
                _fail("original RRNCO placement differs")
            job["gpu"] = target_gpu
            job["global_slot"] = target_gpu
    if new != expected:
        _fail("only TR10/TR09 GPU and global_slot may change; training fields must be identical")


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_compatible_deployment(repo, deployed_path, manifest, source_snapshot):
    """Return the deployment, accepting only the one reviewed placement update.

    The caller has already captured the actual current source. An unchanged
    deployment is returned without writing. All upgrade validation happens
    before preserving the old manifest and atomically replacing its metadata.
    """
    repo = Path(repo).resolve()
    deployed_path = Path(deployed_path).resolve()
    manifest = Path(manifest).resolve()
    if not deployed_path.is_relative_to(repo) or manifest != _relative_path(repo, JOB_MANIFEST):
        _fail("unexpected deployment or job manifest location")
    original_bytes = deployed_path.read_bytes()
    deployed = json.loads(original_bytes)
    current_jobs = manifest.read_bytes()
    current_jobs_sha = hashlib.sha256(current_jobs).hexdigest()
    if (deployed.get("source_version") == source_snapshot.get("source_version")
            and deployed.get("job_manifest_sha256") == current_jobs_sha):
        return deployed
    if (deployed.get("schema") != "cus100_portable_deployment_v1"
            or deployed.get("source_version") != BASE_SOURCE_VERSION
            or deployed.get("job_manifest_sha256") != BASE_JOB_MANIFEST_SHA256
            or deployed.get("job_manifest") != JOB_MANIFEST):
        _fail("deployed package is not the known original Cus100 package")
    descriptor, descriptor_sha = _load_committed_descriptor(repo)
    if descriptor["target_job_manifest_sha256"] != current_jobs_sha:
        _fail("current job manifest differs from the reviewed update")
    old_provenance_path = _relative_path(repo, deployed["source_manifest"])
    old_provenance = json.loads(old_provenance_path.read_text())
    _check_snapshot_identity(old_provenance)
    if old_provenance["source_version"] != BASE_SOURCE_VERSION:
        _fail("original source provenance is not the deployed source")
    _check_snapshot_identity(source_snapshot)
    changed_paths = _validate_delta(old_provenance["files"], source_snapshot["files"], descriptor)
    _validate_jobs(_old_jobs(repo, deployed, old_provenance["files"]), current_jobs)
    target_manifest_path = Path(source_snapshot["manifest"]).resolve()
    if not target_manifest_path.is_relative_to(repo):
        _fail("new source provenance must be inside the repository")
    target_provenance = json.loads(target_manifest_path.read_text())
    _check_snapshot_identity(target_provenance)
    if (target_provenance["files"] != source_snapshot["files"]
            or target_provenance["source_version"] != source_snapshot["source_version"]):
        _fail("new saved source provenance differs from the captured snapshot")
    target_archive = target_manifest_path.with_suffix(".tar.gz")
    if digest_file(target_archive) != target_provenance.get("archive_sha256"):
        _fail("new source archive checksum differs")
    if source_files(repo) != source_snapshot["files"]:
        _fail("source changed after snapshot creation")
    # Recheck mutable metadata immediately before creating any receipt.
    if deployed_path.read_bytes() != original_bytes or manifest.read_bytes() != current_jobs:
        _fail("deployment or jobs changed during validation")
    backup = deployed_path.with_name(deployed_path.stem + ".before_" + UPGRADE_ID + deployed_path.suffix)
    if backup.exists():
        if backup.read_bytes() != original_bytes:
            _fail("existing original-deployment backup has different contents")
    else:
        with backup.open("xb") as stream:
            stream.write(original_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    upgraded = copy.deepcopy(deployed)
    upgraded["source_version"] = source_snapshot["source_version"]
    upgraded["source_manifest"] = str(target_manifest_path.relative_to(repo))
    upgraded["job_manifest_sha256"] = current_jobs_sha
    upgraded.setdefault("upgrade_receipts", []).append({
        "schema": "cus100_deployment_upgrade_receipt_v1",
        "upgrade_id": UPGRADE_ID,
        "time": datetime.now(timezone.utc).isoformat(),
        "descriptor": DESCRIPTOR,
        "descriptor_sha256": descriptor_sha,
        "previous_source_version": BASE_SOURCE_VERSION,
        "source_version": source_snapshot["source_version"],
        "previous_job_manifest_sha256": BASE_JOB_MANIFEST_SHA256,
        "job_manifest_sha256": current_jobs_sha,
        "previous_deployment_manifest": str(backup.relative_to(repo)),
        "changed_source_paths": changed_paths,
        "placement_changes": {"TR10": {"from_gpu": 0, "to_gpu": 1},
                              "TR09": {"from_gpu": 1, "to_gpu": 2}},
        "training_parameters_and_data_unchanged": True,
    })
    _atomic_json(deployed_path, upgraded)
    return upgraded
