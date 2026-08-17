from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from evrptw_stage2.release_discipline import quarantine_rate_summary


ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts" / "audit_stage2_terminal_connectivity.py"
SPEC = importlib.util.spec_from_file_location("stage2_c1_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _split(*rows: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["latent_service_location_id", "customer_pool"],
    )


def _record(
    source_id: str,
    *,
    stage: str = "stage1_directional",
    community_id: str | None = "city:bg:1:scc:1",
    depot_id: str | None = None,
) -> dict[str, object]:
    return {
        "city_slug": "city",
        "terminal_kind": "customer",
        "source_id": source_id,
        "audit_stage": stage,
        "reason_codes": ["stage1_no_reference_scc_inbound_access"],
        "physical_edge_id": "edge",
        "anchor_scc_id": "1",
        "directed_edge_ref_count": 1,
        "directed_projection_offsets": "[]",
        "census_block_group_geoid": "1" if community_id else None,
        "community_id": community_id,
        "split_pool": None,
        "split_assignment_status": "excluded_pre_split_connectivity",
        "generation_eligible": False,
        "stage1_inbound_access_eligible": False if stage == "stage1_directional" else None,
        "stage1_outbound_access_eligible": True if stage == "stage1_directional" else None,
        "stage2_node_outbound_reachable": None,
        "stage2_node_return_reachable": None,
        "stage2_turn_outbound_reachable": None,
        "stage2_turn_return_reachable": None,
        "depot_id": depot_id,
    }


def test_every_generation_eligible_customer_has_exactly_one_pool() -> None:
    report = AUDIT._customer_split_contract(
        {"a", "b"},
        _split(("a", "train"), ("b", "heldout")),
        set(),
    )
    assert report["passed"]
    assert report["assertions"][
        "every_generation_eligible_customer_has_exactly_one_train_or_heldout_pool"
    ]


def test_quarantined_customer_representation_is_null_and_excluded() -> None:
    ledger = AUDIT._aggregate_quarantine_ledger([_record("q")])
    assert ledger[0]["split_pool"] is None
    assert ledger[0]["split_assignment_status"] == "excluded_pre_split_connectivity"
    assert ledger[0]["generation_eligible"] is False
    assert ledger[0]["stage1_inbound_access_eligible"] is False
    assert ledger[0]["stage1_outbound_access_eligible"] is True


def test_no_quarantined_customer_can_appear_in_split_pool() -> None:
    report = AUDIT._customer_split_contract(
        {"a", "q"},
        _split(("a", "train"), ("q", "heldout")),
        {"q"},
    )
    assert not report["passed"]
    assert report["quarantined_customer_ids_in_split"] == ["q"]
    assert not report["assertions"][
        "no_connectivity_quarantined_customer_in_split_family_or_view_pool"
    ]


def test_every_quarantined_customer_appears_once_in_city_ledger() -> None:
    records = [
        _record("q", stage="stage1_directional"),
        _record("q", stage="stage2_exact_preflight", depot_id="d1"),
        _record("q", stage="stage2_exact_preflight", depot_id="d2"),
    ]
    ledger = AUDIT._aggregate_quarantine_ledger(records)
    report = AUDIT._quarantine_ledger_contract("city", ledger, {"q"})
    assert len(ledger) == 1
    assert ledger[0]["depot_ids"] == ["d1", "d2"]
    assert report["passed"]


def test_missing_community_customer_remains_in_r2_numerator() -> None:
    ledger = AUDIT._aggregate_quarantine_ledger(
        [_record("missing-community", community_id=None)]
    )
    report = AUDIT._quarantine_ledger_contract(
        "city",
        ledger,
        {"missing-community"},
        {"missing-community"},
    )
    assert report["passed"]
    assert report["expected_missing_community_customer_count"] == 1
    assert report["observed_missing_community_customer_count"] == 1
    assert report["assertions"][
        "r2_numerator_includes_quarantined_customers_with_missing_community_ids"
    ]


def test_r2_denominator_is_pre_split_unique_customer_universe() -> None:
    report = quarantine_rate_summary(
        ["eligible", "quarantined", "quarantined"],
        stage1_directional_ids=["quarantined"],
        stage2_turn_ids=[],
        rate_limit=0.5,
    )
    assert report["schema"] == "cle_evrptw_unique_terminal_quarantine_rate_v2"
    assert report["rule_id"] == "connectivity_quarantine_precedes_customer_split_v1"
    assert report["audit_input_unique_terminal_count"] == 2
    assert report["stage1_or_stage2_union_quarantine"]["rate"] == pytest.approx(0.5)
    assert "independent of train/heldout" in report["denominator_semantics"]


def test_leakage_contract_never_silently_drops_eligible_customer() -> None:
    report = AUDIT._customer_split_contract(
        {"a", "missing"},
        _split(("a", "train")),
        set(),
    )
    assert not report["passed"]
    assert report["missing_eligible_customer_ids"] == ["missing"]
    assert not report["assertions"][
        "leakage_never_drops_eligible_customer_for_missing_pool"
    ]
