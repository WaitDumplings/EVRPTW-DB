"""Versioned, self-verifying DRL reward normalization contracts.

The contract deliberately keeps the physical objective, terminal failure cost,
and method-specific shaping as separate quantities.  A single frozen economic
scale is selected by customer-count label and shared by every DRL method.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .objective import ObjectiveConfig, resolve_objective


REPO_ROOT = Path(__file__).resolve().parents[3]
REWARD_CONTRACT_SCHEMA = "drl_reward_contract_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_scale(scale: str | int) -> str:
    value = str(scale).strip().lower().removeprefix("cus")
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"invalid reward-contract scale: {scale}")
    return f"Cus{int(value)}"


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def reward_contract_digest(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 of canonical JSON after removing top-level sha256."""

    if not isinstance(payload, Mapping):
        raise TypeError("reward contract must be a mapping")
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


@dataclass(frozen=True)
class RewardScaleContract:
    contract_id: str
    digest: str
    scale_label: str
    objective_scale: float
    failure_base: float
    unserved_coefficient: float
    objective_config: ObjectiveConfig

    def terminal_failure_cost(
        self, failed: Any, unserved_fraction: Any,
    ) -> Any:
        """One terminal cost: failed * (base + coefficient * unserved fraction)."""

        return failed * (
            self.failure_base + self.unserved_coefficient * unserved_fraction
        )

    def validate_envs(self, envs: Sequence[Any]) -> None:
        """Ensure rollout environments actually use this objective and scale."""

        for env in envs:
            base = env.unwrapped
            active_objective = resolve_objective(
                getattr(base, "objective_config", None)
            )
            if active_objective.to_dict() != self.objective_config.to_dict():
                raise ValueError("rollout environment objective disagrees with reward contract")
            if float(getattr(base, "reward_objective_scale", math.nan)) != float(
                self.objective_scale
            ):
                raise ValueError(
                    "rollout environment does not use the frozen reward objective scale"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "sha256": self.digest,
            "scale": self.scale_label,
            "objective_scale": self.objective_scale,
            "failure_base": self.failure_base,
            "unserved_coefficient": self.unserved_coefficient,
            "objective": self.objective_config.to_dict(),
        }


@dataclass(frozen=True)
class RewardContract:
    contract_id: str
    objective_config: ObjectiveConfig
    scales: Mapping[str, Mapping[str, Any]]
    digest: str
    snapshot: Mapping[str, Any]
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "RewardContract":
        return load_reward_contract(path)

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, source_path: Path | None = None,
    ) -> "RewardContract":
        if not isinstance(payload, Mapping):
            raise ValueError("reward contract JSON must contain an object")
        snapshot = deepcopy(dict(payload))
        if snapshot.get("schema") != REWARD_CONTRACT_SCHEMA:
            raise ValueError(
                f"unsupported reward contract schema: {snapshot.get('schema')!r}"
            )
        contract_id = snapshot.get("contract_id")
        if not isinstance(contract_id, str) or not contract_id.strip():
            raise ValueError("reward contract_id must be a nonempty string")
        supplied_digest = snapshot.get("sha256")
        if not isinstance(supplied_digest, str) or not _SHA256.fullmatch(
            supplied_digest
        ):
            raise ValueError("reward contract sha256 must be 64 lowercase hex characters")
        computed_digest = reward_contract_digest(snapshot)
        if supplied_digest != computed_digest:
            raise ValueError(
                "reward contract sha256 mismatch; calibration/config was modified"
            )

        objective_payload = snapshot.get("objective")
        required_objective = set(ObjectiveConfig.__dataclass_fields__)
        if not isinstance(objective_payload, Mapping) or not required_objective.issubset(
            objective_payload
        ):
            raise ValueError("reward contract requires a complete objective config")
        objective = resolve_objective(dict(objective_payload))

        raw_scales = snapshot.get("scales")
        if not isinstance(raw_scales, Mapping) or not raw_scales:
            raise ValueError("reward contract scales must be a nonempty object")
        scales: dict[str, dict[str, Any]] = {}
        for raw_label, raw_terms in raw_scales.items():
            label = _canonical_scale(raw_label)
            if label != raw_label:
                raise ValueError(
                    f"reward contract scale keys must be canonical (expected {label})"
                )
            if label in scales or not isinstance(raw_terms, Mapping):
                raise ValueError(f"invalid or duplicate reward terms for {label}")
            terms = deepcopy(dict(raw_terms))
            terms["objective_scale"] = _finite_number(
                terms.get("objective_scale"),
                f"scales.{label}.objective_scale",
                positive=True,
            )
            terms["failure_base"] = _finite_number(
                terms.get("failure_base"),
                f"scales.{label}.failure_base",
                positive=True,
            )
            terms["unserved_coefficient"] = _finite_number(
                terms.get("unserved_coefficient"),
                f"scales.{label}.unserved_coefficient",
            )
            scales[label] = terms
        return cls(
            contract_id=contract_id,
            objective_config=objective,
            scales=scales,
            digest=supplied_digest,
            snapshot=snapshot,
            source_path=source_path,
        )

    def for_scale(
        self, scale: str | int, objective: ObjectiveConfig | Mapping[str, Any],
    ) -> RewardScaleContract:
        requested_objective = resolve_objective(objective)
        if requested_objective.to_dict() != self.objective_config.to_dict():
            raise ValueError(
                "reward contract objective mismatch; start a new run with its frozen objective"
            )
        label = _canonical_scale(scale)
        if label not in self.scales:
            raise ValueError(f"reward contract has no calibration for {label}")
        terms = self.scales[label]
        return RewardScaleContract(
            contract_id=self.contract_id,
            digest=self.digest,
            scale_label=label,
            objective_scale=float(terms["objective_scale"]),
            failure_base=float(terms["failure_base"]),
            unserved_coefficient=float(terms["unserved_coefficient"]),
            objective_config=self.objective_config,
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.snapshot))


def load_reward_contract(path: str | Path) -> RewardContract:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = REPO_ROOT / source
    with source.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return RewardContract.from_payload(payload, source_path=source.resolve())


def reward_contract_from_args(
    args: Any,
    *,
    objective: ObjectiveConfig | Mapping[str, Any] | None = None,
    scale: str | int | None = None,
) -> RewardScaleContract | None:
    """Resolve once and persist an immutable contract snapshot on ``args``."""

    path = getattr(args, "reward_contract", None)
    saved_snapshot = getattr(args, "reward_contract_snapshot", None)
    if path is None and saved_snapshot is None:
        return None
    contract = (
        load_reward_contract(path)
        if path is not None
        else RewardContract.from_payload(saved_snapshot)
    )
    if saved_snapshot is not None:
        frozen = RewardContract.from_payload(saved_snapshot)
        if frozen.digest != contract.digest:
            raise ValueError("reward contract path and frozen args snapshot disagree")
    active_objective = resolve_objective(
        objective if objective is not None else getattr(args, "objective", None)
    )
    terms = contract.for_scale(
        scale if scale is not None else getattr(args, "scale"), active_objective
    )
    args.reward_contract_snapshot = contract.to_dict()
    args.reward_contract_sha256 = terms.digest
    args.reward_contract_id = terms.contract_id
    args.reward_contract_scale = terms.scale_label
    args.reward_objective_scale = terms.objective_scale
    args.reward_failure_base = terms.failure_base
    args.reward_unserved_coefficient = terms.unserved_coefficient
    return terms


def assert_checkpoint_reward_contract(payload: Mapping[str, Any], args: Any) -> None:
    """Reject resume across reward semantics, including legacy/missing snapshots."""

    current = getattr(args, "reward_contract_snapshot", None)
    saved_args = payload.get("args", {}) or {}
    if not isinstance(saved_args, Mapping):
        saved_args = vars(saved_args)
    saved = saved_args.get("reward_contract_snapshot")
    top_level = payload.get("reward_contract")
    if current is None and saved is None:
        if top_level is not None:
            raise ValueError("checkpoint reward contract snapshots disagree")
        return
    if current is None or saved is None:
        raise ValueError(
            "checkpoint reward contract mismatch; do not resume legacy and calibrated runs"
        )
    if top_level != saved:
        raise ValueError("checkpoint reward contract snapshots disagree")
    current_contract = RewardContract.from_payload(current)
    saved_contract = RewardContract.from_payload(saved)
    if current_contract.digest != saved_contract.digest:
        raise ValueError("checkpoint reward contract mismatch; start a fresh run")
    requested_scale = _canonical_scale(getattr(args, "scale"))
    saved_scale = saved_args.get("reward_contract_scale")
    if saved_scale != requested_scale:
        raise ValueError("checkpoint reward contract scale mismatch; start a fresh run")
    current_terms = current_contract.for_scale(
        requested_scale, current_contract.objective_config
    )
    saved_terms = saved_contract.for_scale(saved_scale, saved_contract.objective_config)
    expected_fields = {
        "reward_contract_id": saved_terms.contract_id,
        "reward_contract_sha256": saved_terms.digest,
        "reward_objective_scale": saved_terms.objective_scale,
        "reward_failure_base": saved_terms.failure_base,
        "reward_unserved_coefficient": saved_terms.unserved_coefficient,
    }
    if any(saved_args.get(name) != value for name, value in expected_fields.items()):
        raise ValueError(
            "checkpoint reward contract derived fields are missing or inconsistent"
        )
    current_fields = {
        "reward_contract_id": current_terms.contract_id,
        "reward_contract_sha256": current_terms.digest,
        "reward_objective_scale": current_terms.objective_scale,
        "reward_failure_base": current_terms.failure_base,
        "reward_unserved_coefficient": current_terms.unserved_coefficient,
    }
    if any(getattr(args, name, None) != value for name, value in current_fields.items()):
        raise ValueError("requested reward contract derived fields are inconsistent")


def classify_rollout_failure_reasons(
    infos: Sequence[dict[str, Any]],
    *,
    done: np.ndarray,
    success: np.ndarray,
    served_customers: np.ndarray,
    customer_count: np.ndarray,
) -> np.ndarray:
    """Return and persist one mutually exclusive terminal reason per trajectory."""

    reasons = np.full(success.shape, "terminal_failure", dtype=object)
    for row, info in enumerate(infos):
        recorded = np.asarray(
            info.get("failure_reason", np.full(success.shape[1], "in_progress")),
            dtype=object,
        ).reshape(-1)
        for trajectory in range(success.shape[1]):
            if success[row, trajectory]:
                reason = "success"
            elif not done[row, trajectory]:
                reason = (
                    "rollout_budget_exhausted_not_returned"
                    if served_customers[row, trajectory] >= customer_count[row, 0]
                    else "rollout_budget_exhausted"
                )
            elif trajectory < recorded.size and recorded[trajectory] not in {
                None, "", "in_progress", "success"
            }:
                reason = str(recorded[trajectory])
            elif served_customers[row, trajectory] >= customer_count[row, 0]:
                reason = "not_returned_to_depot"
            else:
                reason = "terminal_failure"
            reasons[row, trajectory] = reason
        info["failure_reason"] = reasons[row].copy()
    return reasons


__all__ = [
    "REWARD_CONTRACT_SCHEMA",
    "RewardContract",
    "RewardScaleContract",
    "assert_checkpoint_reward_contract",
    "classify_rollout_failure_reasons",
    "load_reward_contract",
    "reward_contract_digest",
    "reward_contract_from_args",
]
