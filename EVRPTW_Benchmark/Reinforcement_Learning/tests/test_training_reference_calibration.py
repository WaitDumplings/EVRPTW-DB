from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Dataset_Generator" / "src"))

from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import (  # noqa: E402
    ObjectiveConfig,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import (  # noqa: E402
    RewardContract,
    reward_contract_digest,
)
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.calibrate_training_reference_cost import (  # noqa: E402
    CalibrationError,
    build_contract_payload,
    route_sha256,
    select_fixed_training_cohort,
    stable_cohort_rank,
    summarize_scale_rows,
    validate_complete_results,
    write_calibration_outputs,
)


def objective() -> ObjectiveConfig:
    return ObjectiveConfig(
        mode="energy_vehicle_cost",
        profile_id="test_energy_vehicle_cost_v2",
        electricity_price_usd_per_kwh=0.2,
        consumption_kwh_per_km=0.5,
        vehicle_fixed_cost_usd=3.0,
    )


def synthetic_index(*, rows_per_stratum: int = 37) -> pd.DataFrame:
    rows = []
    for scale, customers in (
        ("Cus50", 50), ("Cus100", 100), ("Cus500", 500), ("Cus1000", 1000)
    ):
        for city_number in range(10):
            city = f"city-{city_number:02d}"
            for day_type in ("weekday", "weekend"):
                for offset in range(rows_per_stratum):
                    token = f"{scale}-{city}-{day_type}-{offset:03d}"
                    rows.append(
                        {
                            "view_id": f"iv_{token}",
                            "family_id": f"mf_{token}",
                            "family_cohort_id": "core/train",
                            "consumer_cohort_id": (
                                "compatibility_cus50/train"
                                if scale == "Cus50"
                                else "core/train"
                            ),
                            "split_id": "train",
                            "track_id": "train",
                            "city_slug": city,
                            "day_type": day_type,
                            "scale_id": scale.lower(),
                            "customer_count": customers,
                            "view_seed": city_number * 10_000 + offset,
                        }
                    )
    return pd.DataFrame(rows)


def test_fixed_cohort_is_hash_ranked_stratified_and_order_independent() -> None:
    frame = synthetic_index()
    first = select_fixed_training_cohort(frame)
    shuffled = select_fixed_training_cohort(
        frame.sample(frac=1.0, random_state=991).reset_index(drop=True)
    )

    assert len(first) == 2_000
    assert first["view_id"].is_unique
    pd.testing.assert_frame_equal(first, shuffled)
    counts = first.groupby(["scale_label", "city_slug", "day_type"]).size()
    assert set(counts.xs("weekday", level="day_type")) == {36}
    assert set(counts.xs("weekend", level="day_type")) == {14}
    assert first.groupby("scale_label").size().to_dict() == {
        "Cus50": 500,
        "Cus100": 500,
        "Cus500": 500,
        "Cus1000": 500,
    }
    changed_seed = select_fixed_training_cohort(frame, seed=17)
    assert set(changed_seed["view_id"]) != set(first["view_id"])


def test_cohort_rank_has_frozen_blake2b_vector() -> None:
    assert stable_cohort_rank(
        seed=20_260_904,
        scale="Cus500",
        city_slug="chicago",
        day_type="weekday",
        view_id="iv_example",
    ) == "e47fc3dec9c79c3bf136bb29f0c6f91a"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns=["day_type"]), "missing columns"),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate view_id",
        ),
        (
            lambda frame: frame.assign(
                city_slug=frame["city_slug"].mask(frame.index == 0, None)
            ),
            "null/empty city_slug",
        ),
        (
            lambda frame: frame.assign(
                split_id=frame["split_id"].mask(frame.index == 0, "val")
            ),
            "training-only",
        ),
        (
            lambda frame: frame.assign(
                consumer_cohort_id=frame["consumer_cohort_id"].mask(
                    frame.index == 0, "core/val"
                )
            ),
            "requires consumer_cohort_id=",
        ),
        (
            lambda frame: frame.assign(
                customer_count=frame["customer_count"].mask(frame.index == 0, 499)
            ),
            "inconsistent customer_count",
        ),
        (
            lambda frame: frame.loc[frame["city_slug"] != "city-09"].copy(),
            "exactly 10 training cities",
        ),
        (
            lambda frame: frame.assign(
                day_type=frame["day_type"].mask(frame.index == 0, "holiday")
            ),
            "weekday/weekend",
        ),
        (
            lambda frame: frame.drop(
                frame.loc[
                    (frame["scale_id"] == "cus500")
                    & (frame["city_slug"] == "city-00")
                    & (frame["day_type"] == "weekday")
                ].index[35:]
            ),
            "requires 36",
        ),
    ],
)
def test_fixed_cohort_rejects_noncanonical_or_incomplete_source(
    mutation, message: str
) -> None:
    with pytest.raises(CalibrationError, match=message):
        select_fixed_training_cohort(mutation(synthetic_index()))


def test_linear_median_and_q99_failure_base_are_frozen() -> None:
    rows = [
        {
            "objective_value": float(value),
            "objective_distance_km": float(value * 10),
            "vehicles_started": int(value % 7 + 1),
        }
        for value in range(1, 501)
    ]
    terms, statistics = summarize_scale_rows(rows)

    assert terms["objective_scale"] == 250.5
    assert statistics["normalized_objective_q99_linear"] == pytest.approx(
        1.9760878243512974, abs=1e-15
    )
    assert terms["failure_base"] == pytest.approx(2.976087824351297, abs=1e-15)
    assert terms["unserved_coefficient"] == 1.0


def verified_row(view_id: str, position: int, *, scale: str = "Cus500") -> dict:
    routes = [[0, 1, 0], [0, 2, 0]]
    fields = objective().fields(10.0, 2)
    return {
        "schema": "drl_training_reference_view_v1",
        "cohort_position": position,
        "view_id": view_id,
        "scale_label": scale,
        "routes": routes,
        "route_sha256": route_sha256(routes),
        "route_validation_passed": True,
        "objective_distance_km": 10.0,
        "vehicles_started": 2,
        **fields,
    }


def test_complete_results_bind_route_hash_D_K_and_C() -> None:
    cohort = pd.DataFrame(
        {"cohort_position": [0], "view_id": ["iv_a"]}
    )
    row = verified_row("iv_a", 0)
    validate_complete_results(cohort, [row], objective())

    bad_hash = dict(row, route_sha256="0" * 64)
    with pytest.raises(CalibrationError, match="route hash mismatch"):
        validate_complete_results(cohort, [bad_hash], objective())
    bad_k = dict(row, vehicles_started=1)
    with pytest.raises(CalibrationError, match="K/routes mismatch"):
        validate_complete_results(cohort, [bad_k], objective())
    bad_c = dict(row, objective_value=float(row["objective_value"]) + 0.01)
    with pytest.raises(CalibrationError, match="objective_value/D/K mismatch"):
        validate_complete_results(cohort, [bad_c], objective())
    with pytest.raises(CalibrationError, match="exactly match"):
        validate_complete_results(cohort, [], objective())


def test_contract_schema_digest_and_per_scale_terms() -> None:
    rows = []
    for scale in ("Cus50", "Cus100", "Cus500", "Cus1000"):
        rows.extend(
            {
                "scale_label": scale,
                "objective_value": float(value),
                "objective_distance_km": float(value),
                "vehicles_started": 1,
            }
            for value in range(1, 501)
        )
    contract = build_contract_payload(
        objective=objective(),
        rows=rows,
        source_index_sha256="1" * 64,
        cohort_file_sha256="2" * 64,
        per_view_file_sha256="3" * 64,
    )

    assert contract["schema"] == "drl_reward_contract_v1"
    assert contract["contract_id"] == "drl_energy_vehicle_reference_scale_v3"
    assert contract["sha256"] == reward_contract_digest(contract)
    assert contract["scales"]["Cus500"]["objective_scale"] == 250.5
    assert contract["scales"]["Cus1000"]["failure_base"] == pytest.approx(
        2.976087824351297
    )
    assert contract["scales"]["Cus500"]["unserved_coefficient"] == 1.0
    assert contract["calibration"]["cohort"]["weighting"] == (
        "approximately_proportional_city_day_stratified_72_28"
    )
    assert contract["calibration"]["cohort"]["day_quota_per_city"] == {
        "weekday": 36,
        "weekend": 14,
    }
    loaded = RewardContract.from_payload(json.loads(json.dumps(contract)))
    assert loaded.digest == contract["sha256"]


def test_invalid_results_do_not_overwrite_existing_artifacts(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    cohort_path = artifact_dir / "cohort.json"
    rows_path = artifact_dir / "per_view.jsonl"
    contract_path = tmp_path / "contract.json"
    for path, content in (
        (cohort_path, b"old cohort\n"),
        (rows_path, b"old rows\n"),
        (contract_path, b"old contract\n"),
    ):
        path.write_bytes(content)
    before = {path: path.read_bytes() for path in (cohort_path, rows_path, contract_path)}
    cohort = pd.DataFrame({"cohort_position": [0], "view_id": ["iv_a"]})
    invalid = verified_row("iv_a", 0)
    invalid["route_validation_passed"] = False

    with pytest.raises(CalibrationError, match="unverified"):
        write_calibration_outputs(
            artifact_dir=artifact_dir,
            contract_path=contract_path,
            cohort=cohort,
            rows=[invalid],
            objective=objective(),
            source_index_sha256=hashlib.sha256(b"index").hexdigest(),
        )
    assert {path: path.read_bytes() for path in before} == before
