"""Pure helpers for the certificate-based C1b/R2-v2 connectivity gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


ACCEPTANCE_SCHEMA = "cle_evrptw_connectivity_audit_acceptance_v2"
MANUAL_REVIEW_SCHEMA = "cle_evrptw_connectivity_h64_manual_review_v1"


def json_records(value: object) -> list[dict[str, Any]]:
    """Parse a stored JSON list without silently accepting malformed access state."""

    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, list):
        parsed = value
    else:
        raise ValueError(f"Expected a JSON access-ref list, got {type(value).__name__}")
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("Directed access refs must be a JSON list of objects")
    return [dict(item) for item in parsed]


def directed_ref_keys(value: object) -> tuple[tuple[str, str, str], ...]:
    """Return a canonical, duplicate-sensitive directed-edge reference sequence."""

    keys = [
        (str(item["u"]), str(item["v"]), str(item["key"]))
        for item in json_records(value)
    ]
    return tuple(sorted(keys))


def h64_rank(namespace: str, *parts: object) -> int:
    """Frozen unsigned H64 rank used for reviewer samples."""

    payload = "\x1f".join([namespace, *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def select_h64_samples(
    frame: pd.DataFrame,
    *,
    id_column: str,
    group_columns: list[str],
    minimum_per_group: int,
    namespace: str,
    take_all: bool = False,
    allow_all_when_insufficient: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Select deterministic unique terminals under an explicit availability policy."""

    required = {id_column, *group_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"H64 input lacks columns: {sorted(missing)}")
    selected: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    for key, rows in frame.groupby(group_columns, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        unique = rows.drop_duplicates(id_column).copy()
        unique["h64_rank"] = [
            h64_rank(namespace, *key_tuple, source_id)
            for source_id in unique[id_column].astype(str)
        ]
        unique = unique.sort_values(["h64_rank", id_column])
        requested_minimum = int(minimum_per_group)
        required_count = (
            len(unique)
            if take_all
            or (allow_all_when_insufficient and len(unique) < requested_minimum)
            else requested_minimum
        )
        take = len(unique) if take_all else min(len(unique), required_count)
        selected.append(unique.iloc[:take])
        coverage.append(
            {
                **dict(zip(group_columns, map(str, key_tuple), strict=True)),
                "available_unique_terminal_count": len(unique),
                "requested_minimum_sample_count": requested_minimum,
                "required_sample_count": required_count,
                "selected_sample_count": take,
                "all_available_selected": take == len(unique),
                "passed": take >= required_count,
            }
        )
    result = pd.concat(selected, ignore_index=True) if selected else frame.iloc[:0].copy()
    return result, coverage


def value_distribution(values: pd.Series) -> list[dict[str, Any]]:
    normalized = values.astype("string").fillna("<missing>")
    counts = normalized.value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    return [
        {
            "value": str(value),
            "count": int(count),
            "share": float(count / total) if total else 0.0,
        }
        for value, count in counts.items()
    ]


def concentration_summary(
    audit_input: pd.DataFrame,
    quarantine: pd.DataFrame,
    *,
    id_column: str,
) -> dict[str, Any]:
    """Compare road-access concentration/distributions for bad and eligible terminals."""

    inputs = audit_input.drop_duplicates(id_column).copy()
    bad = quarantine.drop_duplicates(id_column).copy()
    bad_ids = set(bad[id_column].astype(str))
    eligible = inputs.loc[~inputs[id_column].astype(str).isin(bad_ids)].copy()
    edge_counts = (
        bad["physical_edge_id"].astype("string").fillna("<missing>").value_counts()
    )
    total_bad = len(bad)

    def top_share(n: int) -> float:
        return float(edge_counts.iloc[:n].sum() / total_bad) if total_bad else 0.0

    def subset_distributions(frame: pd.DataFrame) -> dict[str, Any]:
        projection = pd.to_numeric(
            frame.get("road_projection_fraction_from_physical_start"),
            errors="coerce",
        ).dropna()
        return {
            "count": len(frame),
            "projection_fraction": {
                "count": len(projection),
                "q00": float(projection.min()) if len(projection) else None,
                "q25": float(projection.quantile(0.25)) if len(projection) else None,
                "q50": float(projection.quantile(0.50)) if len(projection) else None,
                "q75": float(projection.quantile(0.75)) if len(projection) else None,
                "q100": float(projection.max()) if len(projection) else None,
            },
            "highway": value_distribution(frame.get("highway", pd.Series(dtype="string"))),
            "directed_ref_count": value_distribution(
                frame.get("directed_edge_ref_count", pd.Series(dtype="string"))
            ),
        }

    return {
        "audit_input_unique_terminal_count": len(inputs),
        "quarantined_unique_terminal_count": total_bad,
        "eligible_unique_terminal_count": len(eligible),
        "quarantine_rate": float(total_bad / len(inputs)) if len(inputs) else 0.0,
        "unique_physical_edge_count": int(edge_counts.size),
        "physical_edge_terminal_counts": [
            {"physical_edge_id": str(edge), "count": int(count)}
            for edge, count in edge_counts.items()
        ],
        "top_1_physical_edge_share": top_share(1),
        "top_5_physical_edges_share": top_share(5),
        "top_10_physical_edges_share": top_share(10),
        "quarantined_distributions": subset_distributions(bad),
        "eligible_distributions": subset_distributions(eligible),
    }


def manual_review_gate(
    path: Path | None,
    expected_sample_ids: Iterable[str],
    *,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Validate an explicit reviewer artifact; absence remains a hard pending state."""

    expected = set(map(str, expected_sample_ids))
    if path is None or not path.is_file():
        return {
            "schema": MANUAL_REVIEW_SCHEMA,
            "status": "pending_manual_review",
            "passed": False,
            "expected_sample_count": len(expected),
            "reason": "signed --manual-review artifact is required",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    findings = payload.get("findings", {})
    reviewed = set(map(str, payload.get("reviewed_sample_ids", [])))
    assertions = {
        "schema_matches": payload.get("schema") == MANUAL_REVIEW_SCHEMA,
        "all_expected_samples_reviewed_exactly": reviewed == expected,
        "candidate_commit_bound": (
            code_commit is None or payload.get("code_commit") == code_commit
        ),
        "ignored_valid_access_option_count_is_zero": int(
            findings.get("ignored_valid_access_option_count", -1)
        )
        == 0,
        "incorrect_road_or_projection_semantics_count_is_zero": int(
            findings.get("incorrect_road_or_projection_semantics_count", -1)
        )
        == 0,
        "certificate_replay_disagreement_count_is_zero": int(
            findings.get("certificate_replay_disagreement_count", -1)
        )
        == 0,
        "reviewer_signoff_present": bool(
            str(payload.get("reviewer_signoff_id", "")).strip()
        ),
    }
    return {
        "schema": MANUAL_REVIEW_SCHEMA,
        "status": "passed" if all(assertions.values()) else "failed",
        "passed": all(assertions.values()),
        "assertions": assertions,
        "reviewer_signoff_id": payload.get("reviewer_signoff_id"),
        "expected_sample_count": len(expected),
        "reviewed_sample_count": len(reviewed),
    }


def primary_pf2_support(cohort: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck frozen primary family/day support without running C2."""

    rows = []
    for pool in ("GEN-TRAIN", "GEN-EVAL", "METRIC-HOLDOUT"):
        for day_type in ("weekday", "weekend"):
            values = cohort["support"][pool]["by_day_type"][day_type]
            for scale in (100, 500, 1_000):
                structure = int(values[f"single_structure_days_ge_{scale}"])
                order = int(values[f"single_order_days_ge_{scale}"])
                rows.append(
                    {
                        "pool": pool,
                        "day_type": day_type,
                        "customer_count": scale,
                        "single_structure_day_count": structure,
                        "single_order_day_count": order,
                        "passed": structure > 0 and order > 0,
                    }
                )
    return {"passed": bool(rows and all(row["passed"] for row in rows)), "rows": rows}
