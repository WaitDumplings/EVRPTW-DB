"""The GPU-placement migration accepts only its reviewed deployment delta."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911 import deployment_upgrade as upgrade
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.source_snapshot import capture_source, digest_file, source_files


def _git(repo, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=repo, stderr=subprocess.STDOUT)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def _json(path, value):
    _write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _jobs_bytes(jobs):
    return "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs).encode()


def _commit(repo):
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "fixture update")


def _refresh_descriptor(case):
    before = {record["path"]: record for record in case.old_source["files"]}
    after = {record["path"]: record for record in source_files(case.repo)}
    changes = [{"path": path, "before": before.get(path), "after": after.get(path)}
               for path in sorted(set(before) | set(after))
               if path != upgrade.DESCRIPTOR and before.get(path) != after.get(path)]
    descriptor = {"schema": upgrade.SCHEMA, "upgrade_id": upgrade.UPGRADE_ID,
                  "base_source_version": upgrade.BASE_SOURCE_VERSION,
                  "base_job_manifest_sha256": upgrade.BASE_JOB_MANIFEST_SHA256,
                  "target_job_manifest_sha256": digest_file(case.manifest),
                  "job_manifest": upgrade.JOB_MANIFEST, "changes": changes}
    _json(case.repo / upgrade.DESCRIPTOR, descriptor)
    _commit(case.repo)
    case.snapshot = capture_source(case.repo, case.repo / "results" / "provenance")
    return descriptor


@pytest.fixture
def case(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "deployment-test@example.invalid")
    _git(repo, "config", "user.name", "Deployment test")
    _write(repo / ".gitignore", "results/\n__pycache__/\n.pytest_cache/\n")
    _write(repo / "scientific_model.py", "parameter = 1\n")
    launch = repo / upgrade.SCRIPT_ROOT / "launch.py"
    _write(launch, "# Original launcher\n")
    jobs = []
    for experiment in ("TR02", "TR01", "TR06", "TR05", "TR04", "TR03", "TR18", "TR17", "TR10", "TR09"):
        rrnco = experiment in {"TR10", "TR09"}
        gpu = 1 if experiment == "TR09" else 0
        jobs.append({"experiment_id": experiment,
                     "server": "2080ti_3_1" if rrnco else "2080ti_4_1",
                     "method": "rrnco" if rrnco else "am_evrptw", "gpu": gpu,
                     "global_slot": gpu, "physical_batch_size": 50, "seed": 1234,
                     "training_stream_path": "results/frozen_stream.jsonl",
                     "training_stream_path_sha256": "unchanged frozen data",
                     "extra_args": ["--graph-mode", "full"]})
    manifest = repo / upgrade.JOB_MANIFEST
    manifest.write_bytes(_jobs_bytes(jobs))
    _commit(repo)
    old_source = capture_source(repo, repo / "results" / "provenance")
    monkeypatch.setattr(upgrade, "BASE_SOURCE_VERSION", old_source["source_version"])
    monkeypatch.setattr(upgrade, "BASE_JOB_MANIFEST_SHA256", digest_file(manifest))
    deployed = repo / "results" / "deployment_manifest.json"
    original = {"schema": "cus100_portable_deployment_v1",
                "source_version": old_source["source_version"],
                "source_manifest": str(Path(old_source["manifest"]).relative_to(repo)),
                "job_manifest": upgrade.JOB_MANIFEST,
                "job_manifest_sha256": digest_file(manifest),
                "synthetic_manifest_sha256": "frozen synthetic corpus",
                "road_data_included": False}
    _json(deployed, original)
    original_bytes = deployed.read_bytes()
    new_jobs = copy.deepcopy(jobs)
    for job in new_jobs:
        if job["experiment_id"] in {"TR10", "TR09"}:
            job["gpu"] += 1
            job["global_slot"] += 1
    manifest.write_bytes(_jobs_bytes(new_jobs))
    _write(launch, "# Launcher using GPUs 1 and 2\n")
    _write(repo / upgrade.SCRIPT_ROOT / "deployment_upgrade.py", "# Reviewed upgrade implementation\n")
    result = SimpleNamespace(repo=repo, manifest=manifest, deployed=deployed,
                             original=original, original_bytes=original_bytes,
                             old_source=old_source, old_jobs=jobs, new_jobs=new_jobs,
                             launch=launch)
    _refresh_descriptor(result)
    return result


def _run(case):
    return upgrade.ensure_compatible_deployment(case.repo, case.deployed, case.manifest, case.snapshot)


def _assert_unchanged(case):
    assert case.deployed.read_bytes() == case.original_bytes
    assert not list(case.deployed.parent.glob("*.before_*.json"))


def test_exact_reviewed_update_preserves_original_and_records_actual_source(case):
    result = _run(case)
    assert result["source_version"] == case.snapshot["source_version"]
    assert result["job_manifest_sha256"] == digest_file(case.manifest)
    assert result["synthetic_manifest_sha256"] == case.original["synthetic_manifest_sha256"]
    receipt = result["upgrade_receipts"][0]
    assert receipt["training_parameters_and_data_unchanged"] is True
    assert receipt["placement_changes"] == {"TR10": {"from_gpu": 0, "to_gpu": 1},
                                             "TR09": {"from_gpu": 1, "to_gpu": 2}}
    assert (case.repo / receipt["previous_deployment_manifest"]).read_bytes() == case.original_bytes
    assert json.loads(case.deployed.read_text()) == result


def test_repeated_update_is_read_only(case):
    expected = _run(case)
    before = case.deployed.read_bytes()
    modified = case.deployed.stat().st_mtime_ns
    assert _run(case) == expected
    assert case.deployed.read_bytes() == before
    assert case.deployed.stat().st_mtime_ns == modified
    assert len(list(case.deployed.parent.glob("*.before_*.json"))) == 1


@pytest.mark.parametrize("field", ["source_version", "job_manifest_sha256"])
def test_unknown_original_identity_is_rejected(case, field):
    value = copy.deepcopy(case.original)
    value[field] = "unknown"
    _json(case.deployed, value)
    before = case.deployed.read_bytes()
    with pytest.raises(RuntimeError, match="not the known original"):
        _run(case)
    assert case.deployed.read_bytes() == before


@pytest.mark.parametrize("mutation", ["modify", "new", "delete"])
def test_undeclared_scientific_source_change_is_rejected(case, mutation):
    if mutation == "modify":
        _write(case.repo / "scientific_model.py", "parameter = 2\n")
    elif mutation == "new":
        _write(case.repo / "new_scientific_model.py", "parameter = 2\n")
    else:
        (case.repo / "scientific_model.py").unlink()
    case.snapshot = capture_source(case.repo, case.repo / "results" / "provenance")
    with pytest.raises(RuntimeError, match="outside deployment-only scope"):
        _run(case)
    _assert_unchanged(case)


def test_descriptor_cannot_authorize_scientific_source_change(case):
    _write(case.repo / "scientific_model.py", "parameter = 2\n")
    _refresh_descriptor(case)
    with pytest.raises(RuntimeError, match="outside deployment-only scope"):
        _run(case)
    _assert_unchanged(case)


def test_changed_allowed_file_must_match_reviewed_after_record(case):
    _write(case.launch, "# An unrelated change in an otherwise allowed path\n")
    case.snapshot = capture_source(case.repo, case.repo / "results" / "provenance")
    with pytest.raises(RuntimeError, match="before/after records"):
        _run(case)
    _assert_unchanged(case)


@pytest.mark.parametrize("staged", [False, True])
def test_dirty_descriptor_is_rejected(case, staged):
    descriptor = case.repo / upgrade.DESCRIPTOR
    descriptor.write_text(descriptor.read_text() + "\n")
    if staged:
        _git(case.repo, "add", upgrade.DESCRIPTOR)
    case.snapshot = capture_source(case.repo, case.repo / "results" / "provenance")
    with pytest.raises(RuntimeError, match="descriptor differs from its committed version"):
        _run(case)
    _assert_unchanged(case)


@pytest.mark.parametrize("experiment,field,value", [
    ("TR10", "physical_batch_size", 51),
    ("TR09", "training_stream_path_sha256", "changed stream"),
    ("TR02", "seed", 999),
    ("TR09", "gpu", 0),
    ("TR10", "extra_args", ["--graph-mode", "node_only"]),
])
def test_even_reviewed_delta_cannot_change_training_contract(case, experiment, field, value):
    jobs = copy.deepcopy(case.new_jobs)
    next(job for job in jobs if job["experiment_id"] == experiment)[field] = value
    case.manifest.write_bytes(_jobs_bytes(jobs))
    _refresh_descriptor(case)
    with pytest.raises(RuntimeError, match="only TR10/TR09 GPU and global_slot"):
        _run(case)
    _assert_unchanged(case)


def test_corrupt_old_source_records_are_rejected(case):
    path = Path(case.old_source["manifest"])
    provenance = json.loads(path.read_text())
    provenance["files"][0]["sha256"] = "tampered"
    _json(path, provenance)
    with pytest.raises(RuntimeError, match="records do not match their identity"):
        _run(case)
    _assert_unchanged(case)


def test_corrupt_old_source_archive_is_rejected(case):
    archive = Path(case.old_source["manifest"]).with_suffix(".tar.gz")
    with archive.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(RuntimeError, match="original source archive checksum"):
        _run(case)
    _assert_unchanged(case)


def test_forged_old_archive_and_metadata_cannot_replace_original_jobs(case):
    archive = Path(case.old_source["manifest"]).with_suffix(".tar.gz")
    jobs = copy.deepcopy(case.old_jobs)
    jobs[0]["seed"] = 999
    payload = _jobs_bytes(jobs)
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(upgrade.JOB_MANIFEST)
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    provenance = json.loads(Path(case.old_source["manifest"]).read_text())
    provenance["archive_sha256"] = digest_file(archive)
    _json(Path(case.old_source["manifest"]), provenance)
    with pytest.raises(RuntimeError, match="original job manifest checksum"):
        _run(case)
    _assert_unchanged(case)


def test_snapshot_must_still_match_current_source(case):
    _write(case.launch, "# Changed after source snapshot was captured\n")
    with pytest.raises(RuntimeError, match="source changed after snapshot"):
        _run(case)
    _assert_unchanged(case)


def test_missing_delta_entry_is_rejected(case):
    path = case.repo / upgrade.DESCRIPTOR
    descriptor = json.loads(path.read_text())
    descriptor["changes"].pop()
    _json(path, descriptor)
    _commit(case.repo)
    case.snapshot = capture_source(case.repo, case.repo / "results" / "provenance")
    with pytest.raises(RuntimeError, match="before/after records"):
        _run(case)
    _assert_unchanged(case)


def test_portable_archive_lookup_ignores_original_absolute_server_path(case):
    path = Path(case.old_source["manifest"])
    provenance = json.loads(path.read_text())
    provenance["archive"] = "/another-server/old-checkout/source.tar.gz"
    _json(path, provenance)
    assert _run(case)["source_version"] == case.snapshot["source_version"]


def test_conflicting_backup_is_never_overwritten(case):
    backup = case.deployed.with_name(case.deployed.stem + ".before_" + upgrade.UPGRADE_ID + ".json")
    backup.write_text("an unrelated pre-existing backup\n")
    with pytest.raises(RuntimeError, match="backup has different contents"):
        _run(case)
    assert backup.read_text() == "an unrelated pre-existing backup\n"
    assert case.deployed.read_bytes() == case.original_bytes
