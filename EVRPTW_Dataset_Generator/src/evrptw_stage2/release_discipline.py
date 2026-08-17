"""Pre-release quarantine and runtime stop rules frozen by advisor review."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


CUSTOMER_QUARANTINE_RATE_LIMIT = 0.001
CHARGER_QUARANTINE_RATE_LIMIT = 0.01
FAMILY_STAGE_LIMIT_S = 7_200.0
SMOKE_GREEN_TERMINAL_SELECTION_S = 3_600.0
PILOT_FIRST_PROJECTION_S = 4 * 3_600.0
PILOT_PROJECTION_INTERVAL_S = 3_600.0
PILOT_PROJECTED_WALL_LIMIT_S = 36 * 3_600.0


def roster_fingerprint(
    terminal_ids: Iterable[object],
    *,
    depot_id: object,
    terminal_kind: str,
) -> str:
    """Hash a set-valued audit roster independently of row ordering."""

    identifiers = sorted(set(map(str, terminal_ids)))
    payload = "\n".join(
        [f"depot={depot_id}", f"kind={terminal_kind}", *identifiers]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def quarantine_rate_summary(
    audit_input_ids: Iterable[object],
    *,
    stage1_directional_ids: Iterable[object],
    stage2_node_ids: Iterable[object] = (),
    stage2_turn_ids: Iterable[object],
    rate_limit: float,
) -> dict[str, Any]:
    """Apply the unique-ID union denominator required by R-2."""

    inputs = set(map(str, audit_input_ids))
    if not inputs:
        raise ValueError("Directional connectivity audit denominator is empty")
    stage1 = set(map(str, stage1_directional_ids))
    stage2_node = set(map(str, stage2_node_ids))
    stage2_turn = set(map(str, stage2_turn_ids))
    for label, values in (
        ("stage1_directional", stage1),
        ("stage2_node", stage2_node),
        ("stage2_turn", stage2_turn),
    ):
        outside = sorted(values - inputs)
        if outside:
            raise ValueError(f"{label} quarantine contains IDs outside audit input: {outside[:5]}")
    union = stage1 | stage2_node | stage2_turn

    def record(values: set[str]) -> dict[str, Any]:
        return {
            "unique_terminal_count": len(values),
            "rate": len(values) / len(inputs),
        }

    return {
        "schema": "cle_evrptw_unique_terminal_quarantine_rate_v2",
        "rule_id": "connectivity_quarantine_precedes_customer_split_v1",
        "denominator_semantics": (
            "city-level unique terminal IDs entering connectivity audit after "
            "non-connectivity eligibility and before Stage-1/Stage-2 filtering; "
            "independent of train/heldout; repeated family/depot checks count once"
        ),
        "audit_input_unique_terminal_count": len(inputs),
        "stage1_directional_quarantine": record(stage1),
        "stage2_node_quarantine": record(stage2_node),
        "stage2_turn_quarantine": record(stage2_turn),
        "stage1_or_stage2_union_quarantine": record(union),
        "rate_limit": float(rate_limit),
        "passed": len(union) / len(inputs) <= float(rate_limit) + 1e-15,
        "failure_semantics": "stop_and_review_not_silent_deletion",
        "scientific_threshold_role": "engineering_bug_detector_only",
    }


def classify_la_smoke(
    *,
    terminal_selection_s: float,
    family_total_s: float,
) -> dict[str, Any]:
    """Return the gap-free GREEN/AMBER/RED classification frozen by R-3."""

    terminal = float(terminal_selection_s)
    total = float(family_total_s)
    if terminal <= SMOKE_GREEN_TERMINAL_SELECTION_S and total <= FAMILY_STAGE_LIMIT_S:
        status = "GREEN"
    elif terminal <= FAMILY_STAGE_LIMIT_S and total <= FAMILY_STAGE_LIMIT_S:
        status = "AMBER"
    else:
        status = "RED"
    return {
        "schema": "cle_evrptw_la_smoke_stop_rule_v1",
        "terminal_selection_s": terminal,
        "family_total_s": total,
        "status": status,
        "pilot_allowed": status in {"GREEN", "AMBER"},
        "exact_performance_optimization_required": status == "AMBER",
        "limits_s": {
            "green_terminal_selection": SMOKE_GREEN_TERMINAL_SELECTION_S,
            "red_terminal_selection": FAMILY_STAGE_LIMIT_S,
            "red_family_total": FAMILY_STAGE_LIMIT_S,
        },
    }


def nonretryable_signature(rejection: Mapping[str, Any]) -> tuple[str, str, str] | None:
    if bool(rejection.get("retryable", True)):
        return None
    family_id = str(rejection.get("family_id", ""))
    fingerprint = str(rejection.get("roster_fingerprint", ""))
    reason_code = str(rejection.get("reason_code", ""))
    if not family_id or not fingerprint or not reason_code:
        return None
    return family_id, fingerprint, reason_code


@dataclass
class PilotStopController:
    """Stop new submissions, then drain already in-flight family tasks."""

    planned_family_count: int
    started_monotonic: float
    completed_family_count: int = 0
    next_projection_check_s: float = PILOT_FIRST_PROJECTION_S
    stop_reasons: list[dict[str, Any]] = field(default_factory=list)
    _nonretryable_seen: set[tuple[str, str, str]] = field(default_factory=set)

    @property
    def stopped(self) -> bool:
        return bool(self.stop_reasons)

    def _stop(self, reason_code: str, **details: Any) -> None:
        if not any(item["reason_code"] == reason_code for item in self.stop_reasons):
            self.stop_reasons.append({"reason_code": reason_code, **details})

    def observe_chunk(self, result: Mapping[str, Any]) -> None:
        materialized = list(result.get("materialized", []))
        unresolved = list(result.get("unresolved_family_ids", []))
        self.completed_family_count += len(materialized) + len(unresolved)
        for item in materialized:
            family_id = str(item.get("family_id", ""))
            total = float(
                item.get("materialization_seconds", item.get("verification_seconds", 0.0))
            )
            if total > FAMILY_STAGE_LIMIT_S:
                self._stop(
                    "family_total_exceeded_7200s",
                    family_id=family_id,
                    observed_seconds=total,
                )
            for stage, seconds in dict(item.get("stage_timings_seconds", {})).items():
                if float(seconds) > FAMILY_STAGE_LIMIT_S:
                    self._stop(
                        "family_stage_exceeded_7200s",
                        family_id=family_id,
                        stage=str(stage),
                        observed_seconds=float(seconds),
                    )
        for rejection in result.get("rejected_attempts", []):
            if float(rejection.get("elapsed_seconds", 0.0)) > FAMILY_STAGE_LIMIT_S:
                self._stop(
                    "rejected_attempt_exceeded_7200s",
                    family_id=str(rejection.get("family_id", "")),
                    observed_seconds=float(rejection["elapsed_seconds"]),
                )
            signature = nonretryable_signature(rejection)
            if signature is None:
                continue
            if signature in self._nonretryable_seen:
                self._stop(
                    "duplicate_nonretryable_signature",
                    family_id=signature[0],
                    roster_fingerprint=signature[1],
                    nonretryable_reason_code=signature[2],
                )
            self._nonretryable_seen.add(signature)

    def poll(self, now_monotonic: float) -> None:
        elapsed = float(now_monotonic) - float(self.started_monotonic)
        if elapsed + 1e-9 < self.next_projection_check_s:
            return
        if self.completed_family_count == 0:
            self._stop(
                "no_completed_family_after_4h",
                elapsed_seconds=elapsed,
            )
        else:
            projected = elapsed * self.planned_family_count / self.completed_family_count
            if projected > PILOT_PROJECTED_WALL_LIMIT_S:
                self._stop(
                    "projected_total_exceeded_36h",
                    elapsed_seconds=elapsed,
                    completed_family_count=self.completed_family_count,
                    planned_family_count=self.planned_family_count,
                    projected_total_seconds=projected,
                )
        while self.next_projection_check_s <= elapsed + 1e-9:
            self.next_projection_check_s += PILOT_PROJECTION_INTERVAL_S

    def report(self) -> dict[str, Any]:
        return {
            "schema": "cle_evrptw_pilot_stop_discipline_v1",
            "planned_family_count": int(self.planned_family_count),
            "completed_family_count": int(self.completed_family_count),
            "stopped": self.stopped,
            "stop_reasons": list(self.stop_reasons),
            "family_stage_limit_s": FAMILY_STAGE_LIMIT_S,
            "projection_first_check_s": PILOT_FIRST_PROJECTION_S,
            "projection_interval_s": PILOT_PROJECTION_INTERVAL_S,
            "projected_total_limit_s": PILOT_PROJECTED_WALL_LIMIT_S,
            "stop_semantics": "stop_new_submissions_then_drain_in_flight_no_family_deletion",
        }

