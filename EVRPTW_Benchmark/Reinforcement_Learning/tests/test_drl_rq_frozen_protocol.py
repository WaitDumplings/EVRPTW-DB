import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "drl_rq_protocol_frozen_v1.yaml"
RUNTIME_CONFIG = ROOT / "configs" / "drl_rq_runtime_candidates_v2.yaml"
FORMAL_GATE = ROOT / "configs" / "drl_rq_formal_launch_gate_v1.json"


def _protocol():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _runtime():
    return yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))


def _formal_gate():
    return json.loads(FORMAL_GATE.read_text(encoding="utf-8"))


def test_formal_launch_metadata_and_all_eight_gates_match_active_gate_file() -> None:
    protocol = _protocol()
    runtime = _runtime()
    gate_file = _formal_gate()

    assert protocol["status"] == "method_frozen_formal_training_active"
    assert runtime["status"] == "formal_runtime_active"
    assert protocol["formal_launch_allowed"] is True
    assert (
        protocol["formal_launch_allowed"]
        == runtime["formal_launch_allowed"]
        == gate_file["formal_launch_allowed"]
    )
    assert protocol["launch_mode"] == "direct_full"
    assert (
        protocol["launch_policy"]
        == runtime["launch_policy"]
        == gate_file["launch_policy"]
    )
    assert (
        protocol["protocol_id"]
        == runtime["protocol_id"]
        == gate_file["protocol_id"]
    )
    gates = protocol["formal_launch_gates"]
    assert set(gates) == {f"G{index}" for index in range(1, 9)}
    assert {
        gate_id: payload["status"] for gate_id, payload in gates.items()
    } == gate_file["formal_launch_gates"]


def test_rq_training_matrix_and_scientific_boundaries_are_frozen() -> None:
    protocol = _protocol()
    runtime = _runtime()
    questions = protocol["research_questions"]
    assert protocol["methods"] == ["am_evrptw", "evrptw_rl", "drl_ts", "terran"]
    assert protocol["seeds"] == runtime["seeds"] == [1234]
    assert protocol["training_scales"] == runtime["enabled_scales"] == [
        "Cus50",
        "Cus100",
        "Cus500",
        "Cus1000",
    ]
    assert protocol["training_rollout_steps"] == runtime["rollout_steps"] == {
        "Cus50": 65,
        "Cus100": 120,
        "Cus500": 580,
        "Cus1000": 1200,
    }
    assert questions["RQ1"]["same_scale"] == protocol["training_scales"]
    assert questions["RQ1"]["cross_scale_to_cus2000"] == [
        "Cus100",
        "Cus500",
        "Cus1000",
    ]
    assert questions["RQ1"]["main_training_runs"] == 16
    assert questions["RQ2"]["additional_core_training_runs"] == 4
    assert questions["RQ3"]["additional_core_training_runs"] == 4
    assert protocol["core_training_run_count"] == 24
    assert protocol["core_training_run_count"] == sum(
        (
            questions["RQ1"]["main_training_runs"],
            questions["RQ2"]["additional_core_training_runs"],
            questions["RQ3"]["additional_core_training_runs"],
        )
    )
    assert protocol["train_cus2000"] is False
    assert protocol["energy_relaxed_directed_vrptw_model"] == "excluded"
    assert questions["RQ3"]["conditions"] == ["E_to_G", "G_to_G"]
    assert questions["RQ3"]["graph_injection_enabled"] is False


def test_frozen_fairness_numbers_match_v12_runtime_configuration() -> None:
    protocol = _protocol()
    runtime = _runtime()
    fairness = protocol["training_fairness"]

    assert protocol["runtime_budget_id"] == runtime["runtime_budget_id"]
    assert fairness["customer_exposure_budget_by_scale"] == runtime[
        "candidate_customer_exposure_budget"
    ]
    assert fairness["logical_batch_by_scale"] == runtime["candidate_logical_batch"]
    assert fairness["physical_microbatch_by_method_scale"] == runtime[
        "physical_batch_caps"
    ]

    fractions = runtime["formal_candidate"]["exposure_checkpoints_fraction"]
    assert fairness["exposure_checkpoints"] == {
        scale: [int(exposure * float(fraction)) for fraction in fractions]
        for scale, exposure in runtime["candidate_customer_exposure_budget"].items()
    }

    for scale in protocol["training_scales"]:
        customers = int(scale.removeprefix("Cus"))
        expected_exposure = (
            int(runtime["candidate_logical_epochs"][scale])
            * int(runtime["candidate_environments_per_epoch"][scale])
            * customers
        )
        assert fairness["customer_exposure_budget_by_scale"][scale] == expected_exposure
        assert fairness["logical_batch_by_scale"][scale] == runtime[
            "candidate_environments_per_epoch"
        ][scale]

    assert protocol["training_trajectories_per_instance"] == runtime[
        "training_trajectory_count_by_method"
    ]
    assert protocol[
        "training_trajectories_per_instance_by_method_scale"
    ] == runtime["training_trajectory_count_by_method_scale"] == {
        "am_evrptw": {"Cus500": 100, "Cus1000": 100}
    }


def test_reference_and_integrity_claims_are_conservative() -> None:
    protocol = _protocol()
    rq1 = protocol["research_questions"]["RQ1"]
    assert rq1["canonical_lower_bound_available"] is False
    assert rq1["canonical_optimality_gap_available"] is False
    assert rq1["gurobi_bound_semantics"] == "copy_limited_milp_diagnostic_only"
    assert protocol["bks"]["posthoc_only"] is True
    assert protocol["bks"]["checkpoint_selection_use"] == "forbidden"
    assert protocol["ev_integrity"]["reoptimizes_routes"] is False
    assert protocol["ev_integrity"]["preserves_customer_order"] is True


def test_validation_and_test_use_common_best_of_100_sampling_budget() -> None:
    protocol = _protocol()
    runtime = _runtime()
    selection = protocol["model_selection"]
    evaluation = protocol["test_inference"]
    assert selection["validation_decode_type"] == "sampling"
    assert selection["validation_candidate_count"] == 100
    assert selection["early_stopping"] == "enabled_after_minimum_budget"
    assert selection["minimum_training_epochs"] == 5_000
    assert selection["maximum_training_epochs"] == 10_000
    assert selection["validation_every_epochs_through_minimum"] == 250
    assert selection["validation_every_epochs_after_minimum"] == 50
    assert selection["early_stop_patience_validation_checks"] == 10
    assert selection["earliest_possible_early_stop_epoch"] == 5_500
    assert selection["primary_checkpoint"] == "best_overall.ckpt"
    assert selection["minimum_budget_checkpoint"] == "best_within_5000.ckpt"
    assert selection["extended_checkpoint"] == "best_overall.ckpt"
    assert evaluation["decode_type"] == "sampling"
    assert evaluation["candidate_count"] == 100
    assert selection["minimum_training_epochs"] == min(
        runtime["candidate_minimum_logical_epochs"].values()
    )
    assert selection["maximum_training_epochs"] == max(
        runtime["candidate_logical_epochs"].values()
    )
    assert selection["validation_candidate_count"] == runtime["evaluation"][
        "validation_candidate_count"
    ]
    assert evaluation["candidate_count"] == runtime["evaluation"][
        "test_candidate_count"
    ]


def test_support_sampling_and_statistics_use_parent_families() -> None:
    protocol = _protocol()
    rq2 = protocol["research_questions"]["RQ2"]
    assert rq2["support_unit"] == "parent_family"
    assert rq2["support_conditions"] == [
        "Random-10%-support",
        "Coverage-10%-support",
        "Full-support",
    ]
    assert rq2["selection_uses_validation_or_test"] is False
    fairness = protocol["training_fairness"]
    assert fairness["sampling"] == "prefix_stable_full_pool_shuffle_cycle"
    assert fairness["shared_stream_within_scale_seed"] is True
    assert fairness["statistical_unit"] == "parent_family"
