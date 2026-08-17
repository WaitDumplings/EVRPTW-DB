"""Frozen Stage-2 V2.1 metric pairing and release-gate utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PAIRING_LEDGER_SCHEMA = "evrptw_metric_pairing_ledger_v1"
STRATUM_COLUMNS = ["day_type", "scale_id", "source_mode"]
PRIMARY_SCALES = frozenset({"cus100", "cus500", "cus1000"})


def _pair_id(*parts: object) -> str:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return "mp_" + hashlib.blake2b(payload, digest_size=12).hexdigest()


def build_metric_pairing_ledger(
    generated: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    metric_components: Iterable[str] = ("M2", "M3_P50", "M3_P90"),
) -> pd.DataFrame:
    """Enumerate all generated/holdout and unordered real/real pairs."""

    generated_required = {
        *STRATUM_COLUMNS,
        "generated_view_id",
        "structure_source_id",
    }
    holdout_required = {*STRATUM_COLUMNS, "holdout_station_day_id", "station_code"}
    if missing := generated_required - set(generated.columns):
        raise ValueError(f"Generated metric inventory lacks: {sorted(missing)}")
    if missing := holdout_required - set(holdout.columns):
        raise ValueError(f"Holdout metric inventory lacks: {sorted(missing)}")

    records: list[dict[str, Any]] = []
    components = tuple(sorted(set(map(str, metric_components))))
    for stratum, generated_group in generated.groupby(STRATUM_COLUMNS, sort=True):
        mask = np.ones(len(holdout), dtype=bool)
        for column, value in zip(STRATUM_COLUMNS, stratum):
            mask &= holdout[column].astype(str).eq(str(value)).to_numpy()
        real_group = holdout.loc[mask].sort_values("holdout_station_day_id", kind="stable")
        for generated_row in generated_group.sort_values(
            "generated_view_id", kind="stable"
        ).itertuples(index=False):
            for real_row in real_group.itertuples(index=False):
                for component in components:
                    records.append(
                        {
                            "schema": PAIRING_LEDGER_SCHEMA,
                            "pair_kind": "generated_to_holdout",
                            "day_type": str(stratum[0]),
                            "scale_id": str(stratum[1]),
                            "source_mode": str(stratum[2]),
                            "metric_component": component,
                            "generated_view_id": str(generated_row.generated_view_id),
                            "structure_source_id": str(generated_row.structure_source_id),
                            "holdout_station_day_left": str(
                                real_row.holdout_station_day_id
                            ),
                            "holdout_station_day_right": None,
                            "station_block": str(real_row.station_code),
                            "pair_id": _pair_id(
                                "generated_to_holdout",
                                generated_row.generated_view_id,
                                real_row.holdout_station_day_id,
                                component,
                            ),
                        }
                    )
        real_rows = list(real_group.itertuples(index=False))
        for left_index, left in enumerate(real_rows):
            for right in real_rows[left_index + 1 :]:
                for component in components:
                    records.append(
                        {
                            "schema": PAIRING_LEDGER_SCHEMA,
                            "pair_kind": "real_to_real",
                            "day_type": str(stratum[0]),
                            "scale_id": str(stratum[1]),
                            "source_mode": str(stratum[2]),
                            "metric_component": component,
                            "generated_view_id": None,
                            "structure_source_id": None,
                            "holdout_station_day_left": str(
                                left.holdout_station_day_id
                            ),
                            "holdout_station_day_right": str(
                                right.holdout_station_day_id
                            ),
                            "station_block": "|".join(
                                sorted({str(left.station_code), str(right.station_code)})
                            ),
                            "pair_id": _pair_id(
                                "real_to_real",
                                left.holdout_station_day_id,
                                right.holdout_station_day_id,
                                component,
                            ),
                        }
                    )
    columns = [
        "schema",
        "pair_id",
        "pair_kind",
        *STRATUM_COLUMNS,
        "metric_component",
        "generated_view_id",
        "structure_source_id",
        "holdout_station_day_left",
        "holdout_station_day_right",
        "station_block",
    ]
    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        [*STRATUM_COLUMNS, "metric_component", "pair_kind", "pair_id"],
        kind="stable",
    ).reset_index(drop=True)


def write_metric_pairing_ledger(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        return float("nan")
    values = values[finite]
    weights = weights[finite]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    threshold = float(quantile) * float(weights.sum())
    index = int(np.searchsorted(np.cumsum(weights), threshold, side="left"))
    return float(values[min(index, len(values) - 1)])


def station_block_bootstrap_q90(
    distances: pd.DataFrame,
    *,
    replicates: int = 1000,
    seed: int = 20260810,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Report station-cluster bootstrap CIs without changing the frozen gate.

    A sampled station's multiplicity weights generated-to-real pairs directly.
    Cross-station real pairs receive the product of the two station
    multiplicities; same-station pairs receive that station's multiplicity.
    """

    required = {
        *STRATUM_COLUMNS,
        "metric_component",
        "pair_kind",
        "distance",
        "station_block",
    }
    if missing := required - set(distances.columns):
        raise ValueError(f"Metric distances lack bootstrap fields: {sorted(missing)}")
    if replicates <= 0:
        raise ValueError("Bootstrap replicate count must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap confidence must be in (0, 1)")

    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    rows: list[dict[str, Any]] = []
    group_columns = [*STRATUM_COLUMNS, "metric_component"]
    for keys, group in distances.groupby(group_columns, sort=True):
        blocks = sorted(
            {
                station
                for value in group["station_block"].dropna().astype(str)
                for station in value.split("|")
                if station
            }
        )
        margins: list[float] = []
        generated_q90: list[float] = []
        real_q90: list[float] = []
        values = pd.to_numeric(group["distance"], errors="coerce").to_numpy(float)
        kinds = group["pair_kind"].astype(str).to_numpy()
        pair_blocks = [str(value).split("|") for value in group["station_block"]]
        for _ in range(replicates):
            sampled = rng.choice(blocks, size=len(blocks), replace=True)
            counts = {block: int(np.count_nonzero(sampled == block)) for block in blocks}
            weights = np.asarray(
                [
                    counts.get(pair[0], 0)
                    if len(set(pair)) == 1
                    else counts.get(pair[0], 0) * counts.get(pair[1], 0)
                    for pair in pair_blocks
                ],
                dtype=float,
            )
            generated = _weighted_quantile(
                values[kinds == "generated_to_holdout"],
                weights[kinds == "generated_to_holdout"],
                0.90,
            )
            real = _weighted_quantile(
                values[kinds == "real_to_real"],
                weights[kinds == "real_to_real"],
                0.90,
            )
            if np.isfinite(generated) and np.isfinite(real):
                generated_q90.append(generated)
                real_q90.append(real)
                margins.append(real - generated)
        rows.append(
            {
                "day_type": str(keys[0]),
                "scale_id": str(keys[1]),
                "source_mode": str(keys[2]),
                "metric_component": str(keys[3]),
                "station_block_count": len(blocks),
                "valid_replicate_count": len(margins),
                "generated_to_holdout_q90_ci": (
                    [
                        float(np.quantile(generated_q90, alpha)),
                        float(np.quantile(generated_q90, 1.0 - alpha)),
                    ]
                    if generated_q90
                    else None
                ),
                "real_to_real_q90_ci": (
                    [
                        float(np.quantile(real_q90, alpha)),
                        float(np.quantile(real_q90, 1.0 - alpha)),
                    ]
                    if real_q90
                    else None
                ),
                "real_minus_generated_q90_margin_ci": (
                    [
                        float(np.quantile(margins, alpha)),
                        float(np.quantile(margins, 1.0 - alpha)),
                    ]
                    if margins
                    else None
                ),
            }
        )
    return {
        "schema": "evrptw_station_block_bootstrap_q90_v1",
        "role": "report_only_not_used_by_release_gate",
        "seed": int(seed),
        "replicates": int(replicates),
        "confidence": float(confidence),
        "weighting": "single_station_multiplicity; cross_station_product",
        "rows": rows,
    }


def evaluate_q90_gate(
    distances: pd.DataFrame,
    *,
    required_primary_strata: Iterable[tuple[str, str, str]],
) -> dict[str, Any]:
    """Evaluate the immutable D-5 Q90 gate; missing primary support fails."""

    required = {*STRATUM_COLUMNS, "metric_component", "pair_kind", "distance"}
    if missing := required - set(distances.columns):
        raise ValueError(f"Metric distances lack: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    required_set = {tuple(map(str, item)) for item in required_primary_strata}
    observed_primary: set[tuple[str, str, str]] = set()
    for keys, group in distances.groupby(
        [*STRATUM_COLUMNS, "metric_component"], sort=True
    ):
        stratum = tuple(map(str, keys[:3]))
        generated = pd.to_numeric(
            group.loc[group["pair_kind"].eq("generated_to_holdout"), "distance"],
            errors="coerce",
        ).dropna()
        real = pd.to_numeric(
            group.loc[group["pair_kind"].eq("real_to_real"), "distance"],
            errors="coerce",
        ).dropna()
        evaluable = bool(len(generated) and len(real))
        generated_q90 = float(generated.quantile(0.90)) if len(generated) else None
        real_q90 = float(real.quantile(0.90)) if len(real) else None
        passed = bool(evaluable and generated_q90 <= real_q90 + 1e-12)
        if evaluable:
            observed_primary.add(stratum)
        rows.append(
            {
                "day_type": keys[0],
                "scale_id": keys[1],
                "source_mode": keys[2],
                "metric_component": keys[3],
                "generated_to_holdout_q90": generated_q90,
                "real_to_real_q90": real_q90,
                "evaluable": evaluable,
                "passed": passed,
            }
        )
    missing_primary = sorted(required_set - observed_primary)
    primary_rows = [
        row for row in rows if tuple(map(str, (row["day_type"], row["scale_id"], row["source_mode"]))) in required_set
    ]
    return {
        "schema": "evrptw_station_block_q90_gate_v1",
        "rows": rows,
        "missing_primary_strata": [list(item) for item in missing_primary],
        "release_calibrated": bool(
            not missing_primary and primary_rows and all(row["passed"] for row in primary_rows)
        ),
        "confidence_intervals_role": "report_only",
        "pair_subsampling": False,
    }
