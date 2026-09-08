from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.common import training_stream as training_stream_module

from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
    atomic_write_stream,
    build_training_stream,
    load_training_stream_contract,
    read_stream_view_ids,
    training_stream_contract_from_args,
)


def _index() -> pd.DataFrame:
    rows = []
    for city, day, count in (("a", "weekday", 6), ("a", "weekend", 2), ("b", "weekday", 2)):
        for index in range(count):
            rows.append(
                {
                    "view_id": f"{city}-{day}-{index}",
                    "family_id": f"f-{city}-{day}-{index // 2}",
                    "split_id": "train",
                    "track_id": "train",
                    "city_slug": city,
                    "scale_id": "cus100",
                    "customer_count": 100,
                    "day_type": day,
                }
            )
    return pd.DataFrame(rows)


def test_stream_is_deterministic_stratified_shuffle_cycle() -> None:
    first, manifest = build_training_stream(_index(), scale="Cus100", seed=7, sample_count=100)
    replay, _ = build_training_stream(_index(), scale="100", seed=7, sample_count=100)
    assert first.equals(replay)
    assert first.groupby(["city_slug", "day_type"]).size().to_dict() == {
        ("a", "weekday"): 60,
        ("a", "weekend"): 20,
        ("b", "weekday"): 20,
    }
    assert first["view_id"].nunique() == len(_index())
    assert manifest["customer_exposures"] == 10_000
    assert manifest["replacement"] is True
    assert manifest["reuse_policy"] == "only_after_full_eligible_pool_cycle_exhaustion"
    assert manifest["prefix_stable"] is True
    assert manifest["method_independent"] is True

    one_cycle, _ = build_training_stream(
        _index(), scale="Cus100", seed=7, sample_count=len(_index())
    )
    assert set(one_cycle["view_id"]) == set(_index()["view_id"])
    assert one_cycle["view_id"].is_unique


def test_stream_extension_preserves_exact_prefix_and_input_order_invariance() -> None:
    short, _ = build_training_stream(
        _index(), scale="Cus100", seed=13, sample_count=17
    )
    extended, _ = build_training_stream(
        _index().sample(frac=1.0, random_state=99),
        scale="Cus100",
        seed=13,
        sample_count=37,
    )
    assert extended.iloc[: len(short)]["view_id"].tolist() == short["view_id"].tolist()
    assert set(extended.iloc[: len(_index())]["view_id"]) == set(_index()["view_id"])


def test_parent_family_support_is_enforced() -> None:
    allowed = {"f-a-weekday-0", "f-b-weekday-0"}
    stream, manifest = build_training_stream(
        _index(), scale="Cus100", seed=9, sample_count=40, allowed_family_ids=allowed
    )
    assert set(stream["family_id"]) == allowed
    assert manifest["allowed_parent_family_count"] == 2


def test_atomic_stream_round_trip_and_slice(tmp_path) -> None:
    stream, manifest = build_training_stream(_index(), scale="Cus100", seed=11, sample_count=12)
    manifest["source_index_sha256"] = "a" * 64
    manifest["allowed_family_ids_sha256"] = None
    path = tmp_path / "stream.parquet"
    atomic_write_stream(path, stream, manifest)
    assert read_stream_view_ids(path, start=2, stop=5) == stream.iloc[2:5]["view_id"].tolist()
    assert path.with_suffix(".parquet.manifest.json").is_file()
    contract = load_training_stream_contract(path)
    assert contract["sample_count"] == 12
    assert contract["stream_content_sha256"] == manifest["stream_content_sha256"]
    assert contract["manifest_sha256"] == manifest["manifest_sha256"]


def test_preverified_contract_reuses_exact_snapshot_without_loading_artifact(
    monkeypatch,
) -> None:
    snapshot = {
        "schema": "drl_training_stream_contract_v1",
        "sha256": "a" * 64,
        "sample_count": 8,
        "scale": "Cus100",
        "seed": 1234,
    }
    args = SimpleNamespace(
        protocol_id="drl_rq_protocol_frozen_v1",
        training_stream_path="unused.parquet",
        training_stream_contract_sha256=snapshot["sha256"],
        training_stream_contract_snapshot_json=json.dumps(snapshot),
        reuse_preverified_training_streams=True,
    )
    monkeypatch.setattr(
        training_stream_module,
        "load_training_stream_contract",
        lambda *_args, **_kwargs: pytest.fail(
            "preverified reuse must not load or hash the stream artifact"
        ),
    )

    assert training_stream_contract_from_args(args, required=True) == snapshot
    assert args.training_stream_contract_snapshot == snapshot
    assert args.stream_integrity_mode == "reuse_preverified_snapshot_no_rehash"


def test_preverified_contract_fails_closed_without_exact_snapshot() -> None:
    args = SimpleNamespace(
        training_stream_path="unused.parquet",
        training_stream_contract_sha256="a" * 64,
        training_stream_contract_snapshot_json=None,
        reuse_preverified_training_streams=True,
    )
    with pytest.raises(ValueError, match="exact manifest contract snapshot"):
        training_stream_contract_from_args(args, required=True)


def test_stream_contract_rejects_manifest_and_content_tampering(tmp_path) -> None:
    stream, manifest = build_training_stream(
        _index(), scale="Cus100", seed=11, sample_count=12
    )
    manifest["source_index_sha256"] = "b" * 64
    path = tmp_path / "stream.parquet"
    atomic_write_stream(path, stream, manifest)
    manifest_path = path.with_suffix(".parquet.manifest.json")
    original_manifest = manifest_path.read_text(encoding="utf-8")

    corrupted = json.loads(original_manifest)
    corrupted["seed"] = 12
    manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        load_training_stream_contract(path)

    manifest_path.write_text(original_manifest, encoding="utf-8")
    changed = stream.copy()
    changed.loc[0, "view_id"] = "different-view"
    changed.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="logical content SHA256 mismatch"):
        load_training_stream_contract(path)
