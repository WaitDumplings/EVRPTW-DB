from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import json

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.common.method_auxiliary import (
    MethodAuxiliaryProfile,
    assert_checkpoint_method_auxiliary,
    load_method_auxiliary_profile,
    method_auxiliary_digest,
    method_auxiliary_from_args,
)


def _payload(method: str = "evrptw_rl") -> dict:
    payload = {
        "schema": "drl_method_auxiliary_profile_v1",
        "profile_id": "test_station_fraction_v1",
        "method": method,
        "applicability": "formal_training_only",
        "aggregation": "executed_legal_station_visit_count",
        "denominator": "num_customers",
        "step_clip": None,
        "component_clip": None,
        "weights": {"station_visit": 0.3},
    }
    payload["sha256"] = method_auxiliary_digest(payload)
    return payload


def test_method_auxiliary_profile_round_trip_and_args_snapshot(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    profile = load_method_auxiliary_profile(path)
    args = Namespace(method_auxiliary_profile=path)

    resolved = method_auxiliary_from_args(args, expected_method="evrptw_rl")

    assert resolved == profile
    assert args.method_auxiliary_profile_id == profile.profile_id
    assert args.method_auxiliary_sha256 == profile.digest
    assert args.method_auxiliary_denominator == "num_customers"
    assert args.method_auxiliary_weights == {"station_visit": 0.3}
    assert args.method_auxiliary_snapshot == _payload()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(profile_id="changed"),
        lambda payload: payload.update(method="drl_ts"),
        lambda payload: payload["weights"].update(station_visit=float("nan")),
        lambda payload: payload.update(extra=True),
    ],
)
def test_method_auxiliary_profile_rejects_tampering_or_unknown_fields(
    mutation,
) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises((TypeError, ValueError)):
        MethodAuxiliaryProfile.from_payload(payload)


def test_method_auxiliary_profile_rejects_wrong_method() -> None:
    with pytest.raises(ValueError, match="profile is for"):
        MethodAuxiliaryProfile.from_payload(_payload()).require_method("drl_ts")


def test_checkpoint_method_auxiliary_requires_all_exact_snapshots() -> None:
    profile = _payload()
    args = Namespace(method_auxiliary_snapshot=deepcopy(profile))
    payload = {
        "args": {"method_auxiliary_snapshot": deepcopy(profile)},
        "method_auxiliary_profile": deepcopy(profile),
    }
    assert_checkpoint_method_auxiliary(
        payload, args, expected_method="evrptw_rl"
    )

    payload["args"].pop("method_auxiliary_snapshot")
    with pytest.raises(ValueError, match="snapshot mismatch"):
        assert_checkpoint_method_auxiliary(
            payload, args, expected_method="evrptw_rl"
        )


def test_checkpoint_method_auxiliary_rejects_different_digest() -> None:
    current = _payload()
    changed = _payload()
    changed["weights"]["station_visit"] = 0.2
    changed["sha256"] = method_auxiliary_digest(changed)
    with pytest.raises(ValueError, match="digest mismatch"):
        assert_checkpoint_method_auxiliary(
            {
                "args": {"method_auxiliary_snapshot": changed},
                "method_auxiliary_profile": changed,
            },
            Namespace(method_auxiliary_snapshot=current),
            expected_method="evrptw_rl",
        )
