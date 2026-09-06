"""Immutable, method-specific auxiliary-objective profiles.

These profiles are deliberately independent of the shared task reward
contract.  The latter fixes the economic objective and terminal failure cost;
this module freezes optional method-faithful terms that may intentionally
change the ordering between otherwise comparable trajectories.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_AUXILIARY_SCHEMA = "drl_method_auxiliary_profile_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = {
    "schema",
    "profile_id",
    "method",
    "applicability",
    "aggregation",
    "denominator",
    "step_clip",
    "component_clip",
    "weights",
    "sha256",
}


def method_auxiliary_digest(payload: Mapping[str, Any]) -> str:
    """Hash canonical JSON after removing the top-level digest."""

    if not isinstance(payload, Mapping):
        raise TypeError("method auxiliary profile must be a mapping")
    unsigned = deepcopy(dict(payload))
    unsigned.pop("sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"method auxiliary {field} must be a nonempty string")
    return value


def _optional_positive(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"method auxiliary {field} must be numeric or null")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"method auxiliary {field} must be numeric or null"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"method auxiliary {field} must be finite and positive")
    return result


@dataclass(frozen=True)
class MethodAuxiliaryProfile:
    profile_id: str
    method: str
    applicability: str
    aggregation: str
    denominator: str
    step_clip: float | None
    component_clip: float | None
    weights: Mapping[str, float]
    digest: str
    snapshot: Mapping[str, Any]
    source_path: Path | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        source_path: Path | None = None,
    ) -> "MethodAuxiliaryProfile":
        if not isinstance(payload, Mapping):
            raise ValueError("method auxiliary JSON must contain an object")
        snapshot = deepcopy(dict(payload))
        unknown = set(snapshot) - _FIELDS
        missing = _FIELDS - set(snapshot)
        if unknown or missing:
            raise ValueError(
                "method auxiliary profile fields mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if snapshot.get("schema") != METHOD_AUXILIARY_SCHEMA:
            raise ValueError(
                f"unsupported method auxiliary schema: {snapshot.get('schema')!r}"
            )
        supplied_digest = snapshot.get("sha256")
        if not isinstance(supplied_digest, str) or not _SHA256.fullmatch(
            supplied_digest
        ):
            raise ValueError(
                "method auxiliary sha256 must be 64 lowercase hex characters"
            )
        if supplied_digest != method_auxiliary_digest(snapshot):
            raise ValueError(
                "method auxiliary sha256 mismatch; profile was modified"
            )
        raw_weights = snapshot.get("weights")
        if not isinstance(raw_weights, Mapping) or not raw_weights:
            raise ValueError("method auxiliary weights must be a nonempty object")
        weights: dict[str, float] = {}
        for raw_name, raw_value in raw_weights.items():
            name = _nonempty_text(raw_name, "weight name")
            if isinstance(raw_value, bool):
                raise ValueError(f"method auxiliary weight {name} must be numeric")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"method auxiliary weight {name} must be numeric"
                ) from error
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"method auxiliary weight {name} must be finite and non-negative"
                )
            weights[name] = value
        return cls(
            profile_id=_nonempty_text(snapshot.get("profile_id"), "profile_id"),
            method=_nonempty_text(snapshot.get("method"), "method"),
            applicability=_nonempty_text(
                snapshot.get("applicability"), "applicability"
            ),
            aggregation=_nonempty_text(
                snapshot.get("aggregation"), "aggregation"
            ),
            denominator=_nonempty_text(
                snapshot.get("denominator"), "denominator"
            ),
            step_clip=_optional_positive(snapshot.get("step_clip"), "step_clip"),
            component_clip=_optional_positive(
                snapshot.get("component_clip"), "component_clip"
            ),
            weights=weights,
            digest=supplied_digest,
            snapshot=snapshot,
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MethodAuxiliaryProfile":
        return load_method_auxiliary_profile(path)

    def require_method(self, method: str) -> "MethodAuxiliaryProfile":
        if self.method != str(method):
            raise ValueError(
                f"method auxiliary profile is for {self.method}, not {method}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.snapshot))


def load_method_auxiliary_profile(path: str | Path) -> MethodAuxiliaryProfile:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = REPO_ROOT / source
    with source.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return MethodAuxiliaryProfile.from_payload(
        payload, source_path=source.resolve()
    )


def method_auxiliary_from_args(
    args: Any, *, expected_method: str,
) -> MethodAuxiliaryProfile | None:
    """Resolve once and persist a self-verifying snapshot on CLI args."""

    path = getattr(args, "method_auxiliary_profile", None)
    saved_snapshot = getattr(args, "method_auxiliary_snapshot", None)
    if path is None and saved_snapshot is None:
        return None
    profile = (
        load_method_auxiliary_profile(path)
        if path is not None
        else MethodAuxiliaryProfile.from_payload(saved_snapshot)
    ).require_method(expected_method)
    if saved_snapshot is not None:
        frozen = MethodAuxiliaryProfile.from_payload(saved_snapshot)
        if frozen.digest != profile.digest:
            raise ValueError(
                "method auxiliary path and frozen args snapshot disagree"
            )
    args.method_auxiliary_snapshot = profile.to_dict()
    args.method_auxiliary_profile_id = profile.profile_id
    args.method_auxiliary_sha256 = profile.digest
    args.method_auxiliary_method = profile.method
    args.method_auxiliary_applicability = profile.applicability
    args.method_auxiliary_aggregation = profile.aggregation
    args.method_auxiliary_denominator = profile.denominator
    args.method_auxiliary_step_clip = profile.step_clip
    args.method_auxiliary_component_clip = profile.component_clip
    args.method_auxiliary_weights = dict(profile.weights)
    return profile


def assert_checkpoint_method_auxiliary(
    payload: Mapping[str, Any], args: Any, *, expected_method: str,
) -> None:
    """Reject resume across auxiliary semantics, including missing snapshots."""

    current = getattr(args, "method_auxiliary_snapshot", None)
    saved_args = payload.get("args", {}) or {}
    if not isinstance(saved_args, Mapping):
        saved_args = vars(saved_args)
    saved = saved_args.get("method_auxiliary_snapshot")
    top_level = payload.get("method_auxiliary_profile")
    if current is None and saved is None and top_level is None:
        return
    if current is None or saved is None or top_level is None:
        raise ValueError(
            "checkpoint method auxiliary snapshot mismatch; start a fresh run"
        )
    current_profile = MethodAuxiliaryProfile.from_payload(current).require_method(
        expected_method
    )
    saved_profile = MethodAuxiliaryProfile.from_payload(saved).require_method(
        expected_method
    )
    top_profile = MethodAuxiliaryProfile.from_payload(top_level).require_method(
        expected_method
    )
    if not (
        current_profile.digest == saved_profile.digest == top_profile.digest
    ):
        raise ValueError(
            "checkpoint method auxiliary digest mismatch; start a fresh run"
        )


__all__ = [
    "METHOD_AUXILIARY_SCHEMA",
    "MethodAuxiliaryProfile",
    "assert_checkpoint_method_auxiliary",
    "load_method_auxiliary_profile",
    "method_auxiliary_digest",
    "method_auxiliary_from_args",
]
