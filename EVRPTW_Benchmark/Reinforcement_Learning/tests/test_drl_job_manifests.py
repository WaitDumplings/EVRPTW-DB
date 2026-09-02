from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_drl_job_manifests.py"
SPEC = importlib.util.spec_from_file_location("drl_manifest_builder", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _jobs():
    protocol = MODULE.load_protocol(
        ROOT / "configs" / "drl_experiment_protocol_v1.yaml"
    )
    return protocol, *MODULE.build_manifests(protocol)


def test_full_training_matrix_and_a6000_round_robin() -> None:
    _, jobs2080, jobsa6000 = _jobs()
    training = [job for job in jobs2080 + jobsa6000 if job["kind"] == "train"]
    assert len(training) == 48
    assert len({job["job_id"] for job in training}) == 48
    counts = Counter((job["method"], job["scale"]) for job in training)
    assert set(counts.values()) == {3}
    cus1000 = [job for job in jobsa6000 if job["kind"] == "train"]
    assert len(cus1000) == 12
    assert {job["scale"] for job in cus1000} == {"Cus1000"}
    for slot in (0, 1):
        seeds = [
            job["seed"]
            for job in cus1000
            if job["global_slot"] == slot
        ]
        assert seeds == [1234, 1234, 2345, 2345, 3456, 3456]


def test_2080_server_partitions_are_disjoint_and_complete() -> None:
    _, jobs2080, _ = _jobs()
    partitions = []
    for slots in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10)):
        partitions.append({job["job_id"] for job in jobs2080 if job["global_slot"] in slots})
    assert not (partitions[0] & partitions[1])
    assert not (partitions[0] & partitions[2])
    assert not (partitions[1] & partitions[2])
    assert set.union(*partitions) == {job["job_id"] for job in jobs2080}


def test_physical_server_bundles_are_disjoint_complete_and_balanced() -> None:
    _, jobs2080, jobsa6000 = _jobs()
    bundles = MODULE.build_server_bundles(jobs2080, jobsa6000)
    assert set(bundles) == set(MODULE.SERVER_SPECS)
    assigned = [job["job_id"] for jobs in bundles.values() for job in jobs]
    canonical = [job["job_id"] for job in jobs2080 + jobsa6000]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(canonical)
    assert [
        sum(job["kind"] == "train" for job in bundles[server])
        for server in ("2080ti_4_1", "2080ti_4_2", "2080ti_3_1", "a6000_2_1")
    ] == [13, 13, 10, 12]
    assert [
        sum(job["run_mode"] == "pilot" for job in bundles[server])
        for server in ("2080ti_4_1", "2080ti_4_2", "2080ti_3_1", "a6000_2_1")
    ] == [3, 3, 2, 4]
    for server, jobs in bundles.items():
        gpu_count = MODULE.SERVER_SPECS[server]["gpu_count"]
        assert {job["assigned_server"] for job in jobs} == {server}
        assert {job["global_slot"] for job in jobs} == set(range(gpu_count))
        assert all(
            job["canonical_global_slot"]
            in MODULE.SERVER_SPECS[server]["canonical_slots"]
            for job in jobs
            if job["kind"] in {"train", "pilot"}
        )
    locations = {
        job["job_id"]: (server, job["global_slot"], job["hardware"])
        for server, jobs in bundles.items()
        for job in jobs
        if job["kind"] == "train"
    }
    for server, jobs in bundles.items():
        for job in jobs:
            dependency = job.get("checkpoint_job_id")
            if dependency in locations and locations[dependency][2] == job["hardware"]:
                assert (server, job["global_slot"]) == locations[dependency][:2]


def test_scientific_boundaries_are_structural() -> None:
    protocol, jobs2080, jobsa6000 = _jobs()
    jobs = jobs2080 + jobsa6000
    assert all(job["representation"] == "R" for job in jobs)
    assert not any(job["kind"] == "train" and job["scale"] == "Cus2000" for job in jobs)
    assert not any(job["kind"] == "train" and job.get("split") == "test" for job in jobs)
    cus50 = [job for job in jobs if job["kind"] == "eval" and job["scale"] == "Cus50"]
    assert {job["test_id"] for job in cus50} == {"T1"}
    transfer = [job for job in jobs if job["kind"] == "transfer"]
    assert len(transfer) == 48
    assert {job["test_id"] for job in transfer} == {
        "paired_Cus1000",
        "zero_shot_Cus2000",
    }
    assert set(protocol["disabled_tracks"]) == {"E_to_R", "R_to_Inject_to_R"}


def test_candidate_budget_is_matched_across_methods() -> None:
    _, jobs2080, jobsa6000 = _jobs()
    evaluations = [
        job for job in jobs2080 + jobsa6000 if job["kind"] in {"eval", "transfer"}
    ]
    assert {job["candidate_count"] for job in evaluations if job["decode_budget"] == "greedy"} == {1}
    assert {job["candidate_count"] for job in evaluations if job["decode_budget"] == "best_of_50"} == {50}
