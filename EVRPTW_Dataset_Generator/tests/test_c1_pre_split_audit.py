from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from evrptw_stage2.connectivity_acceptance import (
    MANUAL_REVIEW_SCHEMA,
    directed_ref_keys,
    h64_rank,
    manual_review_gate,
    select_h64_samples,
)
from evrptw_stage2.release_discipline import quarantine_rate_summary


ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts" / "audit_stage2_terminal_connectivity.py"
SPEC = importlib.util.spec_from_file_location("stage2_c1_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
sys.path.insert(0, str(ROOT / "scripts"))
ACCEPTANCE_SCRIPT_PATH = ROOT / "scripts" / "build_connectivity_audit_acceptance_v2.py"
ACCEPTANCE_SPEC = importlib.util.spec_from_file_location(
    "stage2_connectivity_acceptance", ACCEPTANCE_SCRIPT_PATH
)
assert ACCEPTANCE_SPEC is not None and ACCEPTANCE_SPEC.loader is not None
ACCEPTANCE = importlib.util.module_from_spec(ACCEPTANCE_SPEC)
ACCEPTANCE_SPEC.loader.exec_module(ACCEPTANCE)


def _split(*rows: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["latent_service_location_id", "customer_pool"],
    )


def _stage1_record(source_id: str, community_id: str | None = "city:bg:1:scc:1") -> dict[str, object]:
    return {
        "city_slug": "city",
        "terminal_kind": "customer",
        "source_id": source_id,
        "audit_stage": "stage1_directional",
        "reason_codes": ["stage1_no_reference_scc_inbound_access"],
        "physical_edge_id": "edge",
        "anchor_scc_id": "S0002",
        "directed_edge_ref_count": 1,
        "directed_projection_offsets": "[]",
        "census_block_group_geoid": "1" if community_id else None,
        "community_id": community_id,
        "split_pool": None,
        "split_assignment_status": "excluded_pre_split_connectivity",
        "stage1_generation_eligible": False,
        "family_connectivity_eligible": False,
        "stage1_inbound_access_eligible": False,
        "stage1_outbound_access_eligible": True,
        "stage2_node_outbound_reachable": None,
        "stage2_node_return_reachable": None,
        "stage2_turn_outbound_reachable": None,
        "stage2_turn_return_reachable": None,
        "depot_id": None,
    }


def _stage2_record(source_id: str, pool: str = "heldout") -> dict[str, object]:
    return AUDIT._stage2_ledger_record(
        {
            "terminal_kind": "customer",
            "source_id": source_id,
            "reason_codes": ["turn_unreachable_from_depot"],
            "physical_edge_id": "edge",
            "anchor_scc_id": "S0001",
            "directed_edge_ref_count": 1,
            "directed_projection_offsets": "[]",
        },
        city_slug="city",
        depot_id="depot-1",
        split_lookup={
            source_id: {
                "customer_pool": pool,
                "community_id": "city:bg:1:scc:S0001",
                "census_block_group_geoid": "1",
            }
        },
    )


def test_layered_split_contract_retains_stage2_pool() -> None:
    report = AUDIT._customer_split_contract(
        {"a", "stage2"},
        _split(("a", "train"), ("stage2", "heldout")),
        {"stage1"},
        {"stage2"},
    )
    assert report["passed"]
    assert report["stage2_quarantined_customer_retained_pool_count"] == 1
    assert report["assertions"][
        "every_stage2_only_quarantined_customer_retains_one_frozen_pool"
    ]


def test_stage1_quarantine_in_split_is_rejected() -> None:
    report = AUDIT._customer_split_contract(
        {"a", "stage1"},
        _split(("a", "train"), ("stage1", "heldout")),
        {"stage1"},
        set(),
    )
    assert not report["passed"]
    assert report["stage1_quarantined_customer_ids_in_split"] == ["stage1"]


def test_stage1_and_stage2_ledger_semantics_are_distinct() -> None:
    ledger = AUDIT._aggregate_quarantine_ledger(
        [_stage1_record("stage1"), _stage2_record("stage2")]
    )
    by_id = {item["source_id"]: item for item in ledger}
    assert by_id["stage1"]["split_pool"] is None
    assert by_id["stage1"]["stage1_generation_eligible"] is False
    assert by_id["stage2"]["split_pool"] == "heldout"
    assert by_id["stage2"]["stage1_generation_eligible"] is True
    assert by_id["stage2"]["family_connectivity_eligible"] is False
    assert (
        by_id["stage2"]["split_assignment_status"]
        == "assigned_before_stage2_turn_preflight"
    )
    contract = AUDIT._quarantine_ledger_contract(
        "city", ledger, {"stage1"}, {"stage2"}
    )
    assert contract["passed"]


def test_stage2_customer_without_frozen_pool_is_rejected() -> None:
    with pytest.raises(ValueError, match="lacks a frozen C0 pool"):
        AUDIT._stage2_ledger_record(
            {
                "terminal_kind": "customer",
                "source_id": "q",
                "reason_codes": ["turn_unreachable_from_depot"],
            },
            city_slug="city",
            depot_id="d",
            split_lookup={},
        )


def test_missing_community_stage1_customer_remains_in_r2_numerator() -> None:
    ledger = AUDIT._aggregate_quarantine_ledger(
        [_stage1_record("missing-community", community_id=None)]
    )
    report = AUDIT._quarantine_ledger_contract(
        "city",
        ledger,
        {"missing-community"},
        set(),
        {"missing-community"},
    )
    assert report["passed"]
    assert report["observed_missing_community_customer_count"] == 1


def test_r2_v1_failure_is_preserved_but_not_active_acceptance_rule() -> None:
    report = quarantine_rate_summary(
        ["eligible", "quarantined"],
        stage1_directional_ids=["quarantined"],
        stage2_turn_ids=[],
        rate_limit=0.001,
    )
    assert report["schema"] == "cle_evrptw_unique_terminal_quarantine_rate_v3"
    assert report["rule_id"] == "r2_v1_fixed_rate_stop_review_v1"
    assert not report["passed"]
    assert report["outcome"] == "triggered_stop_and_review"
    assert report["active_acceptance_rule"] is False
    assert report["stage1_or_stage2_union_quarantine"]["rate"] == pytest.approx(0.5)


def test_leakage_contract_never_silently_drops_stage1_eligible_customer() -> None:
    report = AUDIT._customer_split_contract(
        {"a", "missing"},
        _split(("a", "train")),
        set(),
        set(),
    )
    assert not report["passed"]
    assert report["missing_eligible_customer_ids"] == ["missing"]


def test_directed_ref_keys_are_canonical_and_duplicate_sensitive() -> None:
    refs = [
        {"u": 2, "v": 1, "key": 0},
        {"u": 1, "v": 2, "key": 0},
    ]
    assert directed_ref_keys(json.dumps(refs)) == (
        ("1", "2", "0"),
        ("2", "1", "0"),
    )


def test_h64_selection_is_deterministic_and_reports_insufficient_group() -> None:
    frame = pd.DataFrame(
        {
            "city_slug": ["x", "x"],
            "reason_code": ["turn", "turn"],
            "source_id": ["b", "a"],
        }
    )
    first, coverage1 = select_h64_samples(
        frame,
        id_column="source_id",
        group_columns=["city_slug", "reason_code"],
        minimum_per_group=5,
        namespace="test",
    )
    second, coverage2 = select_h64_samples(
        frame.sample(frac=1.0, random_state=9),
        id_column="source_id",
        group_columns=["city_slug", "reason_code"],
        minimum_per_group=5,
        namespace="test",
    )
    assert first["source_id"].tolist() == second["source_id"].tolist()
    assert coverage1 == coverage2
    assert not coverage1[0]["passed"]
    assert h64_rank("test", "x", "turn", "a") == h64_rank("test", "x", "turn", "a")


def test_manual_review_gate_is_pending_then_requires_zero_findings(tmp_path: Path) -> None:
    expected = {"a", "b"}
    assert not manual_review_gate(None, expected)["passed"]
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema": MANUAL_REVIEW_SCHEMA,
                "reviewed_sample_ids": sorted(expected),
                "reviewer_signoff_id": "reviewer-1",
                "code_commit": "b" * 40,
                "findings": {
                    "ignored_valid_access_option_count": 0,
                    "incorrect_road_or_projection_semantics_count": 0,
                    "certificate_replay_disagreement_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    assert manual_review_gate(
        review,
        expected,
        code_commit="b" * 40,
    )["passed"]
    assert not manual_review_gate(
        review,
        expected,
        code_commit="c" * 40,
    )["passed"]


def test_h64_manifest_covers_all_chargers_and_all_available_small_turn_group() -> None:
    rows: list[dict[str, object]] = []

    def add(
        source_id: str,
        *,
        audit_stage: str,
        terminal_kind: str,
        reason: str,
        physical_edge_id: str,
        node_out: bool | None = None,
        node_back: bool | None = None,
        turn_out: bool | None = None,
        turn_back: bool | None = None,
    ) -> None:
        rows.append(
            {
                "city_slug": "fort_worth",
                "source_id": source_id,
                "audit_stage": audit_stage,
                "terminal_kind": terminal_kind,
                "reason_codes": [reason],
                "physical_edge_id": physical_edge_id,
                "node_outbound_reachable": node_out,
                "node_return_reachable": node_back,
                "turn_outbound_reachable": turn_out,
                "turn_return_reachable": turn_back,
            }
        )

    for index in range(5):
        add(
            f"s1-customer-{index}",
            audit_stage="stage1_directional",
            terminal_kind="customer",
            reason="stage1_no_reference_scc_inbound_access",
            physical_edge_id=f"stage1-edge-{index}",
        )
    add(
        "stage1-charger",
        audit_stage="stage1_directional",
        terminal_kind="charging_station",
        reason="stage1_no_reference_scc_outbound_access",
        physical_edge_id="charger-edge-1",
    )
    add(
        "stage2-charger",
        audit_stage="stage2_exact_preflight",
        terminal_kind="charging_station",
        reason="turn_unreachable_from_depot",
        physical_edge_id="charger-edge-2",
    )
    for index in range(2):
        add(
            f"turn-customer-{index}",
            audit_stage="stage2_exact_preflight",
            terminal_kind="customer",
            reason="turn_unreachable_from_depot",
            physical_edge_id=f"major-edge-{index}",
            node_out=True,
            node_back=True,
            turn_out=False,
            turn_back=True,
        )

    sample, report = ACCEPTANCE._sample_manifest(pd.DataFrame(rows))
    chargers = sample.loc[sample["sample_category"].eq("all_quarantined_chargers")]
    assert set(chargers["source_id"]) == {"stage1-charger", "stage2-charger"}
    major = sample.loc[sample["sample_category"].eq("stage2_turn_only_major_edge")]
    assert set(major["physical_edge_id"]) == {"major-edge-0", "major-edge-1"}
    assert all(row["all_major_edges_represented"] for row in report["major_edge_coverage"])
    assert report["coverage_passed"]
    turn = next(row for row in report["coverage_rows"] if row.get("reason_code") == "stage2_turn_only")
    assert turn["requested_minimum_sample_count"] == 5
    assert turn["required_sample_count"] == 2
    assert turn["all_available_selected"]
