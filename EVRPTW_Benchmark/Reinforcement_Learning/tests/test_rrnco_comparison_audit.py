from __future__ import annotations

import json
from pathlib import Path

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.audit_rrnco_comparison import compare, load_run


def _run(tmp_path: Path, name: str, *, identifiers=("one", "two"), multiplier=1.0, candidates=100, horizon=98):
    path = tmp_path / name
    path.mkdir()
    objective = {"mode": "energy_vehicle_cost", "vehicle_fixed_cost_usd": 10.,
                 "electricity_price_usd_per_kwh": 1., "consumption_kwh_per_km": 1.}
    rows = [{"view_id": identifier, "verifier_passed": True, "environment_success": True,
             "objective_cost_usd": 10 + (index + 1) * multiplier, "objective_distance_km": (index + 1) * multiplier,
             "vehicle_count": 1} for index, identifier in enumerate(identifiers)]
    (path / "validation_summary.json").write_text(json.dumps({"instances": len(rows), "rows": rows,
        "objective_config": objective, "candidate_count": candidates, "validation_rollout_steps": horizon,
        "validation_seed": 123, "decode_type": "sampling", "scale": "Cus50", "split": "validation"}))
    return load_run(name, path)


def test_pairs_ids_independent_of_row_order(tmp_path):
    candidate = _run(tmp_path, "graph", multiplier=0.5)
    baseline = _run(tmp_path, "node")
    baseline["rows"] = dict(reversed(list(baseline["rows"].items())))
    result = compare(candidate, baseline)
    assert result["full_cohort_comparison_verified"]
    assert result["paired_cost"]["wins"] == 2
    assert not result["causal_graph_claim_supported"]


@pytest.mark.parametrize("field,value", [("candidate_count", 50), ("validation_rollout_steps", 65)])
def test_protocol_difference_prevents_certified_comparison(tmp_path, field, value):
    candidate, baseline = _run(tmp_path, "graph"), _run(tmp_path, "node")
    baseline["metadata"][field] = value
    result = compare(candidate, baseline)
    assert field in result["evaluation_audit"]["differences"]
    assert not result["full_cohort_comparison_verified"]


def test_partial_cohort_and_infeasible_rows_stay_explicit(tmp_path):
    candidate = _run(tmp_path, "graph", identifiers=("one", "three"))
    baseline = _run(tmp_path, "node")
    result = compare(candidate, baseline)
    assert result["shared_view_count"] == 1
    assert result["paired_cost"]["scope"] == "jointly_feasible_subset"
    assert not result["full_cohort_comparison_verified"]
    candidate["rows"]["one"]["feasible"] = False
    assert compare(candidate, baseline)["paired_cost"] is None


def test_objective_difference_disables_cost_comparison(tmp_path):
    candidate, baseline = _run(tmp_path, "graph"), _run(tmp_path, "node")
    baseline["metadata"]["objective_config"]["vehicle_fixed_cost_usd"] = 11.
    assert compare(candidate, baseline)["paired_cost"] is None


def test_rejects_duplicate_ids_and_wrong_cost(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        _run(tmp_path, "duplicates", identifiers=("one", "one"))
    _run(tmp_path, "bad_cost")
    path = tmp_path / "bad_cost" / "validation_summary.json"
    summary = json.loads(path.read_text())
    summary["rows"][0]["objective_cost_usd"] = 99.
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="cost disagrees"):
        load_run("bad_cost", path)


def test_self_contained_checkpoint_export_uses_embedded_signature(tmp_path):
    _run(tmp_path, 'export')
    path = tmp_path / 'export' / 'validation_summary.json'
    summary = json.loads(path.read_text())
    summary['resolved_training_signature'] = {
        'effective_batch_size': 24,
        'training_stream_contract_sha256': 'a' * 64,
        'method_specific': {'graph_mode': 'node_only', 'architecture': 'rrnco_ev_stable_aft_v2'},
    }
    path.write_text(json.dumps(summary))
    run = load_run('export', path)
    assert run['metadata']['graph_mode'] == 'node_only'
    assert run['metadata']['effective_batch_size'] == 24
    assert run['metadata']['training_stream_contract_sha256'] == 'a' * 64
