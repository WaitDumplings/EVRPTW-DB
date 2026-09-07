from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest
import yaml

from EVRPTW_Benchmark.Reinforcement_Learning.scripts import build_rq_server_manifests as MANIFESTS
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_rq_server_manifests import (
    SERVERS,
    SCRIPT_ROOT,
    build,
    build_a6000_cus1000_priority_queue,
    build_a6000_terran_cus1000_replacement_queue,
    build_a6000_terran_formal_queue,
)


def test_four_server_queues_cover_frozen_24_job_matrix() -> None:
    queues = build()
    assert set(queues) == set(SERVERS)
    rows = [row for queue in queues.values() for row in queue]
    formal = [row for row in rows if row["run_mode"] == "full"]
    assert len(formal) == 24
    assert rows == formal
    assert len({row["job_id"] for row in formal}) == 24
    assert {row["seed"] for row in rows} == {1234}
    assert Counter((row["representation"], row["condition"]) for row in formal) == {
        ("G", "Full-support"): 16,
        ("G", "Random-10%-support"): 2,
        ("G", "Coverage-10%-support"): 2,
        ("E", "Full-support"): 4,
    }
    assert {row["formal_gate_file"] for row in formal} == {MANIFESTS.GATE}
    assert all(not row["training_stream_path"].startswith("/") for row in rows)
    assert {row["scale"] for row in rows} == {"Cus50", "Cus100", "Cus500", "Cus1000"}


def test_method_specific_stream_is_exact_within_job_scope() -> None:
    rows = [row for queue in build().values() for row in queue]
    grouped: dict[tuple[str, str, str, str, int], set[tuple[str, str]]] = {}
    for row in rows:
        snapshot = row["training_stream_contract_snapshot"]
        stream_path = row["training_stream_path"]
        actual = MANIFESTS.load_training_stream_contract(
            MANIFESTS.ROOT.parents[1] / stream_path
        )
        assert actual == snapshot
        assert snapshot["sha256"] == row["training_stream_contract_sha256"]
        assert snapshot["sample_count"] == row["target_environments"]
        assert snapshot["source_index_sha256"]
        assert f"/{row['method']}/{row['scale']}/" in stream_path
        assert row["file_hash_validation_performed"] is True
        key = (
            row["representation"], row["condition"], row["method"],
            row["scale"], row["seed"],
        )
        grouped.setdefault(key, set()).add(
            (stream_path, row["training_stream_contract_sha256"])
        )
    assert all(len(streams) == 1 for streams in grouped.values())

def test_checked_in_formal_decision_is_three_way_consistent_and_scoped() -> None:
    gate = json.loads(
        (MANIFESTS.ROOT.parents[1] / MANIFESTS.GATE).read_text(encoding="utf-8")
    )
    runtime = yaml.safe_load(MANIFESTS.CONFIG.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(
        (MANIFESTS.ROOT / "configs/drl_rq_protocol_frozen_v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    def decision(document):
        statuses = {
            key: value["status"] if isinstance(value, dict) else value
            for key, value in document["formal_launch_gates"].items()
        }
        return (
            document["protocol_id"],
            document["formal_launch_allowed"],
            document["launch_policy"],
            statuses,
        )

    assert decision(gate) == decision(runtime) == decision(protocol)
    assert (
        gate["authorized_job_ids"]
        == runtime["authorized_job_ids"]
        == protocol["authorized_job_ids"]
    )
    assert len(gate["authorized_job_ids"]) == 18
    assert gate["formal_launch_allowed"] is True
    assert gate["launch_policy"] == "reward_contract_v2_formal_user_authorized"
    assert set(gate["formal_launch_gates"]) == {
        f"G{index}" for index in range(1, 9)
    }


def test_checked_in_stream_registry_binds_marker_manifest_and_all_methods() -> None:
    rows = [row for queue in build().values() for row in queue]
    assert rows
    assert len({row["training_stream_registry_path"] for row in rows}) == 1
    registry_path = MANIFESTS.ROOT.parents[1] / rows[0][
        "training_stream_registry_path"
    ]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    canonical = {key: value for key, value in registry.items() if key != "sha256"}
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert registry["sha256"] == digest
    assert {row["training_stream_registry_sha256"] for row in rows} == {digest}
    assert {
        row["artifact_preparation_marker_sha256"] for row in rows
    } == {registry["artifact_preparation_marker_sha256"]}
    for row in rows:
        key = (
            f'{row["representation"]}/{row["condition"]}/'
            f'{row["method"]}/{row["scale"]}/seed_{row["seed"]}'
        )
        assert registry["streams"][key] == {
            "path": row["training_stream_path"],
            "snapshot": row["training_stream_contract_snapshot"],
        }


def test_every_formal_job_uses_the_same_versioned_cost_objective() -> None:
    cfg = yaml.safe_load(MANIFESTS.CONFIG.read_text())
    profile_path = cfg["objective_config_path"]
    expected = json.loads((MANIFESTS.ROOT.parents[1] / profile_path).read_text())["objective"]
    assert expected["mode"] == "energy_vehicle_cost"
    assert expected["electricity_price_usd_per_kwh"] == 0.1341
    assert expected["vehicle_fixed_cost_usd"] == 33.56
    queues = build()
    rows = [row for queue in queues.values() for row in queue]
    assert len(rows) == 24
    for row in rows:
        assert row["objective_config"] == expected
        assert row["objective_config_path"] == profile_path
        assert row["candidate_selection"] == "verifier_feasible_then_min_total_cost_usd"
    frozen = yaml.safe_load((MANIFESTS.ROOT / "configs/drl_rq_protocol_frozen_v1.yaml").read_text())
    assert frozen["objective_config_path"] == profile_path
    assert frozen["objective_revision"] == expected["profile_id"]
    assert frozen["model_selection"]["secondary_metric"] == "mean_verified_cost_usd"
    terran = yaml.safe_load(MANIFESTS.TERRAN_CONFIG.read_text())
    assert terran["objective"] == profile_path


def test_every_formal_job_uses_the_same_adamw_contract() -> None:
    cfg = yaml.safe_load(MANIFESTS.CONFIG.read_text())
    expected = cfg["training_optimizer"]
    assert expected == {"name": "adamw", "weight_decay": 0.01}
    for queue in build().values():
        for row in queue:
            assert row["optimizer_name"] == expected["name"]
            assert row["optimizer_weight_decay"] == expected["weight_decay"]
    frozen = yaml.safe_load(
        (MANIFESTS.ROOT / "configs/drl_rq_protocol_frozen_v1.yaml").read_text()
    )
    assert frozen["training_optimizer"] == expected
    terran = yaml.safe_load(MANIFESTS.TERRAN_CONFIG.read_text())
    assert terran["training"]["optimizer"] == expected["name"]
    assert terran["training"]["weight_decay"] == expected["weight_decay"]


def test_reward_contract_is_shared_by_every_formal_method(tmp_path, monkeypatch) -> None:
    baseline = build()
    cfg = yaml.safe_load(MANIFESTS.CONFIG.read_text(encoding="utf-8"))
    reward_contract = MANIFESTS.load_reward_contract(
        MANIFESTS.ROOT.parents[1] / cfg["reward_contract_config_path"]
    )
    formal = yaml.safe_load(MANIFESTS.TERRAN_CONFIG.read_text(encoding="utf-8"))
    training = formal["training"]
    terran_count = 0
    for queue in baseline.values():
        for row in queue:
            terms = reward_contract.for_scale(row["scale"], row["objective_config"])
            assert (
                row["reward_contract_config_path"]
                == cfg["reward_contract_config_path"]
            )
            assert row["reward_contract_id"] == terms.contract_id
            assert row["reward_contract_sha256"] == terms.digest
            assert row["reward_objective_scale"] == terms.objective_scale
            assert row["reward_failure_base"] == terms.failure_base
            assert row["reward_unserved_coefficient"] == terms.unserved_coefficient
            if row["method"] == "terran":
                terran_count += 1
                assert row["reward_contract_id"] == training["reward_contract_id"]
                assert row["training_gamma"] == training["gamma"] == 1.0
            else:
                assert "training_gamma" not in row
    assert terran_count == 7

    assert set(reward_contract.scales) == {"Cus50", "Cus100", "Cus500", "Cus1000"}
    assert set(cfg["enabled_scales"]) == set(cfg["reward_contract_calibrated_scales"])
    assert cfg["reward_contract_blocked_scales"] == []
    frozen = yaml.safe_load(
        (MANIFESTS.ROOT / "configs/drl_rq_protocol_frozen_v1.yaml").read_text()
    )
    assert frozen["training_scales"] == ["Cus50", "Cus100", "Cus500", "Cus1000"]
    assert set(frozen["reward_contract_calibrated_scales"]) == set(
        reward_contract.scales
    )
    assert set(frozen["reward_contract_launch_enabled_scales"]) == set(
        reward_contract.scales
    )
    assert frozen["reward_contract_blocked_scales"] == []

    # TERRAN keeps a method-specific gamma, but not a method-specific task contract.
    alternate = tmp_path / "terran.yaml"
    alternate.write_text(
        "training:\n  gamma: 0.999\n"
        f"  reward_contract_id: {reward_contract.contract_id}\n"
    )
    monkeypatch.setattr(MANIFESTS, "TERRAN_CONFIG", alternate)
    revised = build()
    for server, baseline_rows in baseline.items():
        for before, after in zip(baseline_rows, revised[server], strict=True):
            if before["method"] != "terran":
                assert before == after
            else:
                assert after["training_gamma"] == 0.999
                assert after["reward_contract_id"] == reward_contract.contract_id
                assert {k: v for k, v in before.items() if k != "training_gamma"} == {
                    k: v for k, v in after.items() if k != "training_gamma"
                }

    alternate.write_text(
        "training:\n  gamma: 1.0\n  reward_contract_id: stale-contract\n"
    )
    with pytest.raises(ValueError, match="common reward contract disagree"):
        build()


def test_manifest_build_fails_closed_for_an_uncalibrated_enabled_scale(
    tmp_path, monkeypatch,
) -> None:
    cfg = yaml.safe_load(MANIFESTS.CONFIG.read_text(encoding="utf-8"))
    cfg["enabled_scales"].append("Cus2000")
    cfg["scale_hardware"]["2080ti"].append("Cus2000")
    alternate = tmp_path / "runtime.yaml"
    alternate.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(MANIFESTS, "CONFIG", alternate)
    with pytest.raises(ValueError, match="no frozen reward calibration.*Cus2000"):
        build()


def test_scale_aware_hardware_assignment_is_strict() -> None:
    queues = build()
    for server, rows in queues.items():
        assert rows
        expected_scales = (
            {"Cus50", "Cus100"}
            if server.startswith("2080ti_")
            else {"Cus500", "Cus1000"}
        )
        assert {row["scale"] for row in rows}.issubset(expected_scales)


def test_full_train_budget_has_exact_epoch_environment_and_exposure_semantics() -> None:
    runtime = yaml.safe_load(MANIFESTS.CONFIG.read_text(encoding="utf-8"))
    formal = [row for queue in build().values() for row in queue]
    for row in formal:
        scale = row["scale"]
        method = row["method"]
        customers = int(scale.removeprefix("Cus"))
        batch = runtime["candidate_logical_batch_by_method_scale"][method][scale]
        epochs = runtime["candidate_logical_epochs"][scale]
        total_environments = epochs * batch
        exposures = total_environments * customers
        assert row["runtime_budget_id"] == runtime["runtime_budget_id"]
        assert row["runtime_budget_id"] in row["training_stream_path"]
        assert row["training_epochs"] == epochs == 10_000
        assert row["planned_logical_epochs"] == epochs
        assert row["logical_environments_per_epoch"] == batch
        assert row["effective_batch_size"] == batch
        assert row["target_environments"] == total_environments
        assert row["customer_exposure_budget"] == exposures
        assert row["physical_batch_size"] <= batch
        assert row["validation_every_epochs"] == 250
        assert row["validation_checkpoints"] == 120
        assert row["minimum_training_epochs"] == 5_000
        assert row["post_minimum_validation_every_epochs"] == 50
        assert row["validation_views"] == 500
        assert row["validation_candidate_count"] == 50
        assert row["test_candidate_count"] == 50
        expected_trajectories = {
            "am_evrptw": 5, "evrptw_rl": 1, "drl_ts": 1, "terran": 50,
        }[method]
        assert row["training_trajectory_count"] == expected_trajectories
        assert row["warm_start_source_commit"] == runtime["warm_start"]["source_commit"]
        assert row["warm_start_scope"] == "exact_job_only"

def test_scale_rollout_limits_match_current_protocol() -> None:
    expected = {
        "Cus50": (65, 98),
        "Cus100": (120, 180),
        "Cus500": (580, 870),
        "Cus1000": (1200, 1800),
    }
    rows = [row for queue in build().values() for row in queue]
    assert rows
    for row in rows:
        training_steps, validation_steps = (
            (1400, 2100)
            if row["method"] == "terran" and row["scale"] == "Cus1000"
            else expected[row["scale"]]
        )
        assert row["training_rollout_steps"] == training_steps
        assert row["validation_rollout_steps"] == validation_steps


def test_2080ti_jobs_match_authorized_small_scale_assignment() -> None:
    queues = build()
    assert {server: len(queues[server]) for server in (
        "2080ti_4_1", "2080ti_4_2", "2080ti_3_1"
    )} == {"2080ti_4_1": 8, "2080ti_4_2": 5, "2080ti_3_1": 3}
    runtime = yaml.safe_load(MANIFESTS.CONFIG.read_text(encoding="utf-8"))
    authorized = set(runtime["authorized_job_ids"])
    for server in ("2080ti_4_1", "2080ti_4_2", "2080ti_3_1"):
        assert {row["job_id"] for row in queues[server]}.issubset(authorized)


def test_a6000_jobs_use_calibrated_even_physical_batches() -> None:
    expected = {
        "am_evrptw": {"Cus500": 8, "Cus1000": 2},
        "evrptw_rl": {"Cus500": 16, "Cus1000": 2},
        "drl_ts": {"Cus500": 8, "Cus1000": 2},
        "terran": {"Cus500": 64, "Cus1000": 2},
    }
    rows = build()["a6000_2_1"]
    assert rows
    for row in rows:
        assert row["physical_batch_size"] == expected[row["method"]][row["scale"]]
        assert row["physical_batch_size"] % 2 == 0
        assert row["validation_views"] == 500


def test_only_terran_has_scale_calibrated_formal_ppo_overrides() -> None:
    rows = [row for queue in build().values() for row in queue]
    overridden = [
        row
        for row in rows
        if "num_minibatches" in row or "ppo_step_chunk_size" in row
    ]
    assert len(overridden) == 2
    assert all(row["method"] == "terran" for row in overridden)
    assert all(row["num_minibatches"] == 1 for row in overridden)
    assert {row["scale"]: row["ppo_step_chunk_size"] for row in overridden} == {
        "Cus500": 36,
        "Cus1000": 720,
    }
    assert {row["scale"]: row["terran_terminal_success_bonus"] for row in overridden} == {
        "Cus500": 0.0,
        "Cus1000": 1.0,
    }

    priority_terran = [
        row
        for row in build_a6000_cus1000_priority_queue()
        if row["method"] == "terran"
    ]
    assert len(priority_terran) == 1
    assert priority_terran[0]["num_minibatches"] == 1
    assert priority_terran[0]["ppo_step_chunk_size"] == 720

    dedicated = {
        row["scale"]: row for row in build_a6000_terran_formal_queue()
    }
    assert dedicated["Cus500"]["num_minibatches"] == 1
    assert dedicated["Cus500"]["ppo_step_chunk_size"] == 36
    assert dedicated["Cus1000"]["num_minibatches"] == 1
    assert dedicated["Cus1000"]["ppo_step_chunk_size"] == 720


def test_a6000_cus1000_priority_queue_uses_approved_two_gpu_order() -> None:
    canonical = build()["a6000_2_1"]
    rows = build_a6000_cus1000_priority_queue()
    assert [
        (row["method"], row["global_slot"], row["queue_position"])
        for row in rows
    ] == [
        ("terran", 1, 0),
        ("drl_ts", 0, 0),
        ("evrptw_rl", 0, 1),
        ("am_evrptw", 0, 2),
    ]
    assert {row["scale"] for row in rows} == {"Cus1000"}
    assert {row["seed"] for row in rows} == {1234}
    assert {row["job_id"] for row in rows} == {
        row["job_id"] for row in canonical if row["scale"] == "Cus1000"
    }

    canonical_by_id = {row["job_id"]: row for row in canonical}
    for row in rows:
        scientific = {
            key: value
            for key, value in row.items()
            if key not in {"global_slot", "queue_position"}
        }
        canonical_scientific = {
            key: value
            for key, value in canonical_by_id[row["job_id"]].items()
            if key not in {"global_slot", "queue_position"}
        }
        assert scientific == canonical_scientific


def test_checked_in_a6000_cus1000_priority_manifest_matches_builder() -> None:
    manifest = SCRIPT_ROOT / "a6000_2_1" / "cus1000_jobs.jsonl"
    checked_in = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert checked_in == build_a6000_cus1000_priority_queue()


def test_a6000_terran_formal_queue_uses_both_gpus_and_exact_authorized_scope() -> None:
    canonical = build()["a6000_2_1"]
    rows = build_a6000_terran_formal_queue()
    assert [
        (row["method"], row["scale"], row["global_slot"], row["queue_position"])
        for row in rows
    ] == [
        ("terran", "Cus500", 0, 0),
        ("terran", "Cus1000", 1, 0),
    ]
    runtime = yaml.safe_load(MANIFESTS.CONFIG.read_text(encoding="utf-8"))
    authorized = set(runtime["authorized_job_ids"])
    assert {row["job_id"] for row in rows} == {
        job_id for job_id in authorized if "__terran__Cus500__" in job_id or "__terran__Cus1000__" in job_id
    }

    canonical_by_id = {row["job_id"]: row for row in canonical}
    for row in rows:
        scientific = {
            key: value
            for key, value in row.items()
            if key not in {"global_slot", "queue_position"}
        }
        canonical_scientific = {
            key: value
            for key, value in canonical_by_id[row["job_id"]].items()
            if key not in {"global_slot", "queue_position"}
        }
        assert scientific == canonical_scientific


def test_checked_in_a6000_terran_formal_manifest_matches_builder() -> None:
    manifest = SCRIPT_ROOT / "a6000_2_1" / "terran_jobs.jsonl"
    checked_in = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert checked_in == build_a6000_terran_formal_queue()
    summary = json.loads(
        (SCRIPT_ROOT / "a6000_2_1" / "terran_assignment_summary.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = yaml.safe_load(MANIFESTS.CONFIG.read_text(encoding="utf-8"))
    assert summary["profile"] == "terran_formal"
    assert summary["formal_jobs"] == 2
    assert summary["formal_launch_allowed"] is runtime["formal_launch_allowed"]
    assert summary["authorized_formal_job_ids"] == sorted(
        row["job_id"] for row in checked_in
    )
    assert summary["slot_queues"] == {
        "0": ["terran/Cus500"],
        "1": ["terran/Cus1000"],
    }


def test_a6000_terran_cus1000_replacement_is_one_gpu1_bound_job() -> None:
    rows = build_a6000_terran_cus1000_replacement_queue()
    assert len(rows) == 1
    row = rows[0]
    assert row["job_id"] == "full__G__Full-support__terran__Cus1000__seed1234"
    assert row["method"] == "terran"
    assert row["scale"] == "Cus1000"
    assert row["training_rollout_steps"] == 1400
    assert row["validation_rollout_steps"] == 2100
    assert row["num_minibatches"] == 1
    assert row["ppo_step_chunk_size"] == 720
    assert row["terran_terminal_success_bonus"] == 1.0
    assert row["global_slot"] == 1
    assert row["queue_position"] == 0
    assert row["required_launcher_id"] == "terran_cus1000_replacement_v1"
    assert row["required_local_gpu"] == 1


def test_checked_in_a6000_terran_cus1000_replacement_matches_builder() -> None:
    destination = SCRIPT_ROOT / "a6000_2_1"
    checked_in = [
        json.loads(line)
        for line in (
            destination / "terran_cus1000_replacement_jobs.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert checked_in == build_a6000_terran_cus1000_replacement_queue()
    summary = json.loads(
        (
            destination
            / "terran_cus1000_replacement_assignment_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["profile"] == "terran_cus1000_reward_replacement_v1"
    assert summary["launcher_id"] == "terran_cus1000_replacement_v1"
    assert summary["formal_jobs"] == 1
    assert summary["slot_gpu_map"] == {"1": 1}
    assert summary["slot_queues"] == {"1": ["terran/Cus1000"]}


def test_checked_in_server_manifests_match_builder() -> None:
    for server, expected in build().items():
        manifest = SCRIPT_ROOT / server / "jobs.jsonl"
        checked_in = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert checked_in == expected


def test_checked_in_assignment_summaries_match_authorized_scope() -> None:
    runtime = yaml.safe_load(MANIFESTS.CONFIG.read_text(encoding="utf-8"))
    for server, rows in build().items():
        summary = json.loads(
            (SCRIPT_ROOT / server / "assignment_summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["formal_jobs"] == len(rows)
        assert summary["formal_launch_allowed"] is (
            bool(rows)
            and bool(runtime["formal_launch_allowed"])
            and {row["job_id"] for row in rows}.issubset(
                set(runtime["authorized_job_ids"])
            )
        )
        assert summary["authorized_formal_job_ids"] == sorted(
            set(runtime["authorized_job_ids"]).intersection(
                row["job_id"] for row in rows
            )
        )
        assert summary["launch_policy"] == (
            runtime["launch_policy"]
            if rows
            else "blocked_no_calibrated_reward_scale"
        )


def test_artifact_preparation_uses_v15_method_specific_exposure_budgets() -> None:
    script = (SCRIPT_ROOT / "prepare_artifacts.sh").read_text(encoding="utf-8")
    assert "drl_rq_runtime_budget_v15_ntraj50_maxbatch_warmstart" in script
    expected = {
        "am_evrptw:Cus50": 1_152_000_000,
        "am_evrptw:Cus100": 800_000_000,
        "evrptw_rl:Cus50": 168_000_000,
        "drl_ts:Cus100": 40_000_000,
        "terran:Cus50": 240_000_000,
        "terran:Cus100": 280_000_000,
    }
    for key, exposure in expected.items():
        assert f"[{key}]={exposure}" in script
    assert 'METHODS=(am_evrptw evrptw_rl drl_ts terran)' in script
    assert 'for method in "${METHODS[@]}"' in script
    assert 'formal/Full-support/$method/$scale/seed_${seed}.parquet' in script
    assert '--customer-exposures "${FORMAL_EXPOSURE[$method:$scale]}"' in script
    assert 'file_hash_validation_performed": True' in script
    assert '"training_stream_contracts": contracts' in script
