from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import evrptw_stage2.reconstruction as reconstruction_module
from evrptw_stage2.artifacts import verify_materialized_family
from evrptw_stage2.contracts import STAGE2_GENERATION_CONTRACT
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.reconstruction import (
    MATRIX_NAMES,
    ReconstructionContext,
    ReconstructionError,
    export_slim_dataset,
    resolve_family_dirs,
    restore_dataset_matrices,
)

PROFILE_PATH = Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v2.json"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _projection_refs(
    u: str,
    v: str,
    offset_from_u_m: float,
    length_m: float = 100.0,
) -> str:
    return json.dumps(
        [
            {
                "u": u,
                "v": v,
                "key": "0",
                "length_m": length_m,
                "offset_from_u_m": offset_from_u_m,
                "offset_to_v_m": length_m - offset_from_u_m,
            },
            {
                "u": v,
                "v": u,
                "key": "0",
                "length_m": length_m,
                "offset_from_u_m": length_m - offset_from_u_m,
                "offset_to_v_m": offset_from_u_m,
            },
        ]
    )


def _build_tiny_full_dataset(tmp_path: Path) -> tuple[Path, Path]:
    cle_root = tmp_path / "cle"
    city_root = cle_root / "cities" / "tiny-city"
    for relative in ("graph", "service_locations", "infrastructure", "profiles"):
        (city_root / relative).mkdir(parents=True, exist_ok=True)

    graph = nx.MultiDiGraph()
    graph.add_node("a", x=-117.0, y=32.0)
    graph.add_node("b", x=-116.999, y=32.0)
    graph.add_node("c", x=-116.999, y=32.001)
    directed_edges = [
        ("a", "b"),
        ("b", "a"),
        ("b", "c"),
        ("c", "b"),
        ("c", "a"),
        ("a", "c"),
    ]
    for u, v in directed_edges:
        graph.add_edge(u, v, key="0")
    nx.write_graphml(graph, city_root / "graph" / "graph_operational.graphml")

    pd.DataFrame(
        {
            "edge_u": [u for u, _ in directed_edges],
            "edge_v": [v for _, v in directed_edges],
            "edge_key": ["0"] * len(directed_edges),
            "edge_id": [f"{u}:{v}:0" for u, v in directed_edges],
            "physical_segment_id": [
                "ab",
                "ab",
                "bc",
                "bc",
                "ca",
                "ca",
            ],
            "length_m": [100.0] * len(directed_edges),
            "operating_mode": ["U"] * len(directed_edges),
            "legal_speed_kph": [50.0] * len(directed_edges),
            "moves_road_type": ["urban_unrestricted_access"] * len(directed_edges),
            "reference_speed_weekday_kph": [36.0] * len(directed_edges),
            "reference_speed_weekend_kph": [37.0] * len(directed_edges),
        }
    ).to_parquet(city_root / "profiles" / "directed_legal_speeds.parquet", index=False)
    _write_json(
        city_root / "profiles" / "speed_manifest.json",
        {
            "schema": "evrptw_directed_speed_profiles_v6",
            "reference_speed_contract": {"profile_id": "test-moves-profile"},
        },
    )
    pd.DataFrame(
        {
            "latent_service_location_id": ["customer"],
            "cle_default_instance_eligible": [True],
            "customer_release_eligible": [False],
            "protected_inbound_access_eligible": [True],
            "protected_outbound_access_eligible": [True],
            "protected_roundtrip_eligible": [True],
        }
    ).to_parquet(city_root / "service_locations" / "latent_locations.parquet", index=False)
    pd.DataFrame(
        {
            "candidate_id": ["depot"],
            "depot_candidate_eligible": [True],
            "depot_release_eligible": [False],
            "protected_inbound_access_eligible": [True],
            "protected_outbound_access_eligible": [True],
            "protected_roundtrip_eligible": [True],
        }
    ).to_parquet(city_root / "infrastructure" / "depots.parquet", index=False)
    pd.DataFrame(
        {
            "charger_id": ["charger"],
            "charger_candidate_eligible": [True],
            "charger_release_eligible": [False],
            "protected_inbound_access_eligible": [True],
            "protected_outbound_access_eligible": [True],
            "protected_roundtrip_eligible": [True],
        }
    ).to_parquet(city_root / "infrastructure" / "chargers.parquet", index=False)
    _write_json(
        city_root / "manifest.json",
        {
            "schema": "evrptw_city_logistics_environment_v1",
            "connectivity_contract": {"id": "directed_projection_roundtrip_v2"},
            "city_slug": "tiny-city",
            "portable_package_verified": True,
            "release_eligible": False,
            "release_blockers": ["test_fixture"],
            "outputs": {
                "operational_graph": "graph/graph_operational.graphml",
                "latent_locations": "service_locations/latent_locations.parquet",
                "depots": "infrastructure/depots.parquet",
                "chargers": "infrastructure/chargers.parquet",
                "directed_legal_speeds": "profiles/directed_legal_speeds.parquet",
                "speed_manifest": "profiles/speed_manifest.json",
            },
        },
    )

    dataset_root = tmp_path / "full"
    family_root = dataset_root / "materialized" / "families" / "mf-tiny"
    view_root = family_root / "views" / "iv-tiny"
    view_root.mkdir(parents=True)
    terminal_index = pd.DataFrame(
        {
            "terminal_index": [0, 1, 2],
            "terminal_kind": ["depot", "customer", "charging_station"],
            "source_id": ["depot", "customer", "charger"],
            "physical_edge_id": ["ab", "bc", "ca"],
            "directed_projection_offsets": [
                _projection_refs("a", "b", 25.0),
                _projection_refs("b", "c", 60.0),
                _projection_refs("c", "a", 40.0),
            ],
            "connector_length_m": [5.0, 10.0, 15.0],
            "road_projection_node_id": ["rp-depot", "rp-customer", "rp-charger"],
            "access_node_id": ["access-depot", "access-customer", "access-charger"],
        }
    )
    terminal_index.to_parquet(family_root / "terminal_index.parquet", index=False)
    profile = load_reference_profile(PROFILE_PATH)
    matrix_files = {name: f"matrices/{name}.npy" for name in MATRIX_NAMES}
    family_manifest = {
        "schema": "cle_evrptw_materialized_matrix_family_v3",
        "family_id": "mf-tiny",
        "city_slug": "tiny-city",
        "day_type": "weekday",
        "terminal_count": 3,
        "terminal_index": "terminal_index.parquet",
        "matrix_files": matrix_files,
        "view_ids": ["iv-tiny"],
        "view_count": 1,
        "road_state_seed": 123,
        "road_state_report": {
            "moves_road_type_baseline_factors": {"urban_unrestricted_access": 1.0}
        },
        "reference_profile_id": profile["profile_id"],
        "reference_profile_status": profile["profile_status"],
        "generation_mode": "non_release_pilot",
    }
    _write_json(family_root / "family_manifest.json", family_manifest)
    _write_json(
        view_root / "view_manifest.json",
        {
            "schema": "cle_evrptw_materialized_view_v4",
            "view_id": "iv-tiny",
            "family_id": "mf-tiny",
            "scale_id": "cus1",
            "split_id": "test",
            "track_id": "test",
        },
    )
    np.save(view_root / "terminal_parent_indices.npy", np.arange(3, dtype=np.int32))

    matrices = ReconstructionContext(cle_root, profile).route_family(family_root)
    (family_root / "matrices").mkdir()
    for name, matrix in matrices.items():
        np.save(family_root / matrix_files[name], matrix, allow_pickle=False)
    family_manifest["matrix_total_bytes"] = sum(
        (family_root / relative).stat().st_size for relative in matrix_files.values()
    )
    _write_json(family_root / "family_manifest.json", family_manifest)
    return dataset_root, cle_root


def test_slim_export_resolve_view_and_exact_restore(tmp_path: Path) -> None:
    source, cle_root = _build_tiny_full_dataset(tmp_path)
    source_family = source / "materialized" / "families" / "mf-tiny"
    expected = {
        name: np.load(source_family / "matrices" / f"{name}.npy", allow_pickle=False)
        for name in MATRIX_NAMES
    }
    slim = tmp_path / "slim"
    contract = export_slim_dataset(
        source,
        slim,
        cle_root=cle_root,
        profile_path=PROFILE_PATH,
    )
    slim_family = slim / "materialized" / "families" / "mf-tiny"
    assert contract["family_count"] == 1
    registry_contract = contract["reference_profile"]["charging_power_median_registry"]
    assert (slim / registry_contract["relative_path"]).is_file()
    assert not (slim_family / "matrices").exists()
    assert resolve_family_dirs(slim, view_ids=["iv-tiny"]) == [slim_family.resolve()]

    report = restore_dataset_matrices(
        slim,
        cle_root=cle_root,
        view_ids=["iv-tiny"],
        validation="exact",
    )
    assert report["restored_count"] == 1
    for name in MATRIX_NAMES:
        restored_path = slim_family / "matrices" / f"{name}.npy"
        np.testing.assert_array_equal(np.load(restored_path, allow_pickle=False), expected[name])
        manifest = json.loads((slim_family / "family_manifest.json").read_text())
        contract_item = manifest["matrix_reconstruction"]["expected_matrices"][name]
        assert contract_item["shape"] == list(expected[name].shape)
        assert contract_item["dtype"] == "float32"
        assert "npy_sha256" not in contract_item
    assert not list(slim_family.glob(".matrices-rebuild-*"))


def test_slim_restore_rejects_profile_and_cle_mismatch(tmp_path: Path) -> None:
    source, cle_root = _build_tiny_full_dataset(tmp_path)
    slim = tmp_path / "slim"
    export_slim_dataset(source, slim, cle_root=cle_root, profile_path=PROFILE_PATH)

    altered_profile = tmp_path / "profile.json"
    altered_profile.write_text(PROFILE_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    registry_name = load_reference_profile(PROFILE_PATH)["charging"][
        "national_mode_median_registry"
    ]
    (tmp_path / registry_name).write_bytes((PROFILE_PATH.parent / registry_name).read_bytes())
    with pytest.raises(ReconstructionError, match="profile size differs"):
        restore_dataset_matrices(
            slim,
            cle_root=cle_root,
            profile_path=altered_profile,
            view_ids=["iv-tiny"],
        )

    graph_path = cle_root / "cities" / "tiny-city" / "graph" / "graph_operational.graphml"
    graph_path.write_text(graph_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionError, match="operational_graph size differs"):
        restore_dataset_matrices(slim, cle_root=cle_root, view_ids=["iv-tiny"])


def test_failed_exact_restore_leaves_no_partial_matrix_cache(tmp_path: Path) -> None:
    source, cle_root = _build_tiny_full_dataset(tmp_path)
    slim = tmp_path / "slim"
    export_slim_dataset(source, slim, cle_root=cle_root, profile_path=PROFILE_PATH)
    family_root = slim / "materialized" / "families" / "mf-tiny"
    manifest_path = family_root / "family_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["matrix_reconstruction"]["expected_matrices"]["distance_matrix_km"][
        "shape"
    ] = [99, 99]
    _write_json(manifest_path, manifest)

    with pytest.raises(ReconstructionError, match="structural validation failed"):
        restore_dataset_matrices(slim, cle_root=cle_root, view_ids=["iv-tiny"])
    assert not (family_root / "matrices").exists()
    assert not list(family_root.glob(".matrices-rebuild-*"))



def test_current_generation_contract_enables_strict_verifier_fields(
    tmp_path: Path,
) -> None:
    source, _ = _build_tiny_full_dataset(tmp_path)
    family = source / "materialized" / "families" / "mf-tiny"
    manifest_path = family / "family_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stage2_generation_contract"] = STAGE2_GENERATION_CONTRACT
    _write_json(manifest_path, manifest)

    result = verify_materialized_family(family)

    assert result["passed"] is False
    assert any("order_template_id" in error for error in result["errors"])
    assert any("three Phase-1 metric files" in error for error in result["errors"])

def test_reconstruction_propagates_profile_cle_eligibility_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = tmp_path / "mf-official"
    _write_json(
        family / "family_manifest.json",
        {
            "family_id": "mf-official",
            "city_slug": "chicago",
            "generation_mode": "official",
            "reference_profile_id": "release-profile",
        },
    )
    observed: dict[str, object] = {}

    def stop_after_cle_contract(*args: object, **kwargs: object) -> object:
        observed.update(kwargs)
        raise RuntimeError("contract captured")

    monkeypatch.setattr(
        reconstruction_module, "load_portable_cle", stop_after_cle_contract
    )
    context = ReconstructionContext(
        tmp_path / "cle",
        {
            "profile_id": "release-profile",
            "source_contract": {
                "cle_eligibility_contract": "frozen_technical_candidate_v1"
            },
        },
    )

    with pytest.raises(RuntimeError, match="contract captured"):
        context.route_family(family)

    assert observed["mode"] == "official"
    assert observed["official_cle_contract"] == "frozen_technical_candidate_v1"
