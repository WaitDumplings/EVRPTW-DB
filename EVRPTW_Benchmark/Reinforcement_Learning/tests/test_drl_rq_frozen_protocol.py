from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "drl_rq_protocol_frozen_v1.yaml"


def _protocol():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_formal_launch_is_blocked_until_all_eight_gates_pass() -> None:
    protocol = _protocol()
    assert protocol["formal_launch_allowed"] is False
    gates = protocol["formal_launch_gates"]
    assert set(gates) == {f"G{index}" for index in range(1, 9)}
    assert all(gate["status"] != "PASS" for gate in gates.values())
    assert protocol["training_fairness"]["customer_exposure_budget_by_scale"] is None
    assert protocol["training_fairness"]["logical_batch_by_scale"] is None


def test_rq_training_matrix_and_scientific_boundaries_are_frozen() -> None:
    protocol = _protocol()
    questions = protocol["research_questions"]
    assert questions["RQ1"]["main_training_runs"] == 36
    assert questions["RQ2"]["additional_core_training_runs"] == 12
    assert questions["RQ3"]["additional_core_training_runs"] == 12
    assert protocol["core_training_run_count"] == 60
    assert protocol["train_cus2000"] is False
    assert protocol["energy_relaxed_directed_vrptw_model"] == "excluded"
    assert questions["RQ3"]["conditions"] == ["E_to_G", "G_to_G"]
    assert questions["RQ3"]["graph_injection_enabled"] is False


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


def test_validation_and_test_use_common_best_of_50_sampling_budget() -> None:
    protocol = _protocol()
    selection = protocol["model_selection"]
    evaluation = protocol["test_inference"]
    assert selection["validation_decode_type"] == "sampling"
    assert selection["validation_candidate_count"] == 50
    assert evaluation["decode_type"] == "sampling"
    assert evaluation["candidate_count"] == 50


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
    assert fairness["sampling"] == "stratified_deterministic_shuffle_cycle"
    assert fairness["shared_stream_within_scale_seed"] is True
    assert fairness["statistical_unit"] == "parent_family"
