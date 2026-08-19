"""Amazon Last-Mile 2021 preprocessing and Stage-2 artifact access.

The raw challenge files are intentionally consumed once.  Stage-2 workers read
compact Parquet artifacts containing order templates and station-day spatial
structure; raw coordinates are never transferred to CLE cities.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

AMAZON_ARTIFACT_SCHEMA = "evrptw_amazon_stage2_artifacts_v3"
ALLOWED_PACKAGE_STATUSES = {"DELIVERED", "DELIVERY_ATTEMPTED", "REJECTED"}
STATION_TIMEZONES = {
    "DAU": "America/Chicago",
    "DBO": "America/New_York",
    "DCH": "America/Chicago",
    "DLA": "America/Los_Angeles",
    "DSE": "America/Los_Angeles",
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _stable_rank(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _station_timezone(station_code: str) -> ZoneInfo:
    prefix = station_code[:3].upper()
    if prefix not in STATION_TIMEZONES:
        raise ValueError(f"No frozen timezone mapping for Amazon station {station_code!r}")
    return ZoneInfo(STATION_TIMEZONES[prefix])


def _local_offset_seconds(value: Any, *, station_code: str, route_date: date) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    local = parsed.astimezone(_station_timezone(station_code))
    midnight = datetime.combine(route_date, time.min, tzinfo=local.tzinfo)
    return float((local - midnight).total_seconds())


def _template_record(
    *,
    route_id: str,
    stop_id: str,
    stop: dict[str, Any],
    packages: dict[str, Any],
    station_code: str,
    route_date: date,
    station_to_stop_time_s: float,
    horizon_start_s: float,
    horizon_end_s: float,
    cargo_capacity_cm3: float,
) -> tuple[dict[str, Any] | None, str]:
    if stop.get("type") != "Dropoff":
        return None, "u0_not_dropoff"
    if not isinstance(packages, dict) or not packages:
        return None, "u1_missing_packages"

    package_count = 0
    demand_cm3 = 0.0
    service_time_s = 0.0
    specified_windows: list[tuple[float, float]] = []
    status_counts = {status: 0 for status in sorted(ALLOWED_PACKAGE_STATUSES)}
    for package in packages.values():
        if not isinstance(package, dict):
            return None, "u1_package_parse"
        status = str(package.get("scan_status") or "")
        if status not in ALLOWED_PACKAGE_STATUSES:
            return None, "u2_unknown_status"
        status_counts[status] += 1
        dimensions = package.get("dimensions")
        if not isinstance(dimensions, dict):
            return None, "u1_dimensions_parse"
        dimensions_cm = [_finite(dimensions.get(key)) for key in ("depth_cm", "height_cm", "width_cm")]
        if any(value is None or value <= 0.0 for value in dimensions_cm):
            return None, "u1_dimensions_parse"
        service = _finite(package.get("planned_service_time_seconds"))
        if service is None or service < 0.0:
            return None, "u6_service_time"
        package_count += 1
        demand_cm3 += float(np.prod(np.asarray(dimensions_cm, dtype=float)))
        service_time_s += service

        window = package.get("time_window")
        if isinstance(window, dict):
            start = _local_offset_seconds(
                window.get("start_time_utc"),
                station_code=station_code,
                route_date=route_date,
            )
            end = _local_offset_seconds(
                window.get("end_time_utc"),
                station_code=station_code,
                route_date=route_date,
            )
            if start is not None or end is not None:
                if start is None or end is None or end < start:
                    return None, "u1_time_window_parse"
                specified_windows.append((start, end))

    if demand_cm3 > cargo_capacity_cm3:
        return None, "u5_volume"
    if specified_windows:
        tw_start = max(start for start, _ in specified_windows)
        tw_end = min(end for _, end in specified_windows)
        if tw_end < tw_start:
            return None, "u3_empty_tw_intersection"
    else:
        tw_start, tw_end = horizon_start_s, horizon_end_s
    clipped_start = max(float(tw_start), horizon_start_s)
    clipped_end = min(float(tw_end), horizon_end_s)
    if clipped_end < clipped_start:
        return None, "u4_outside_horizon"

    day_type = "weekend" if route_date.weekday() >= 5 else "weekday"
    station_day_id = f"{station_code}:{route_date.isoformat()}"
    return (
        {
            "template_id": (
                f"{station_code}:{route_date.isoformat()}:{route_id}:{stop_id}"
            ),
            "station_day_id": station_day_id,
            "station_code": station_code,
            "date": route_date.isoformat(),
            "day_type": day_type,
            "route_id": route_id,
            "stop_id": stop_id,
            "package_count": int(package_count),
            "demand_cm3": float(demand_cm3),
            "service_time_s": float(service_time_s),
            "tw_start_s": float(clipped_start),
            "tw_end_s": float(clipped_end),
            "tw_was_specified": bool(specified_windows),
            "tw_was_horizon_clipped": bool(
                specified_windows
                and (clipped_start != float(tw_start) or clipped_end != float(tw_end))
            ),
            "station_to_stop_time_s": float(station_to_stop_time_s),
            **{f"status_{key.lower()}_count": int(value) for key, value in status_counts.items()},
        },
        "usable",
    )


def _quantile(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    try:
        return np.quantile(values, probabilities, method="linear")
    except TypeError:  # NumPy < 1.22 compatibility.
        return np.quantile(values, probabilities, interpolation="linear")


def _assign_deciles(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int8)


def build_amazon_stage2_artifacts(
    *,
    route_data_path: str | Path,
    travel_times_path: str | Path,
    package_data_path: str | Path,
    output_dir: str | Path,
    horizon_start_s: float = 8 * 3600.0,
    horizon_end_s: float = 24 * 3600.0,
    cargo_capacity_cm3: float = 18_500_000.0,
) -> dict[str, Any]:
    """Build compact order and station-day structure artifacts.

    This function deliberately performs no integrity hashing during research
    generation.  Portable upstream object IDs, schema and counts remain in the
    manifest; final release packaging may add archive checks separately.
    """

    route_path = Path(route_data_path)
    travel_path = Path(travel_times_path)
    package_path = Path(package_data_path)
    out = Path(output_dir)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("schema") != AMAZON_ARTIFACT_SCHEMA:
            raise ValueError(f"Existing Amazon artifact uses {existing.get('schema')!r}")
        return existing
    for path in (route_path, travel_path, package_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    route_data = json.loads(route_path.read_text(encoding="utf-8"))
    package_data = json.loads(package_path.read_text(encoding="utf-8"))
    travel_times = json.loads(travel_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    attrition: dict[str, int] = {}
    route_reference: list[dict[str, Any]] = []

    for route_id in sorted(route_data):
        route = route_data[route_id]
        station_code = str(route["station_code"])
        route_date = date.fromisoformat(str(route["date_YYYY_MM_DD"]))
        stops = route.get("stops", {})
        station_ids = sorted(
            str(stop_id) for stop_id, stop in stops.items() if stop.get("type") == "Station"
        )
        if len(station_ids) != 1:
            attrition["route_station_count_not_one"] = attrition.get(
                "route_station_count_not_one", 0
            ) + 1
            continue
        station_id = station_ids[0]
        route_travel = travel_times.get(route_id, {})
        station_travel = route_travel.get(station_id, {})
        route_records: list[dict[str, Any]] = []
        for stop_id in sorted(stops):
            travel = _finite(station_travel.get(stop_id))
            if travel is None or travel < 0.0:
                reason = "u1_station_travel_time"
                attrition[reason] = attrition.get(reason, 0) + 1
                continue
            record, reason = _template_record(
                route_id=route_id,
                stop_id=str(stop_id),
                stop=stops[stop_id],
                packages=package_data.get(route_id, {}).get(stop_id, {}),
                station_code=station_code,
                route_date=route_date,
                station_to_stop_time_s=travel,
                horizon_start_s=horizon_start_s,
                horizon_end_s=horizon_end_s,
                cargo_capacity_cm3=cargo_capacity_cm3,
            )
            attrition[reason] = attrition.get(reason, 0) + 1
            if record is not None:
                route_records.append(record)

        usable_stop_ids = [str(record["stop_id"]) for record in route_records]
        nearest_by_stop = {stop_id: math.inf for stop_id in usable_stop_ids}
        pairwise: list[float] = []
        for left in usable_stop_ids:
            for right in usable_stop_ids:
                if left == right:
                    continue
                forward = _finite(route_travel.get(left, {}).get(right))
                if forward is None or forward < 0.0:
                    continue
                pairwise.append(forward)
                nearest_by_stop[left] = min(nearest_by_stop[left], forward)
        for record in route_records:
            nearest = nearest_by_stop[str(record["stop_id"])]
            record["amazon_route_nearest_neighbor_time_s"] = (
                float(nearest) if math.isfinite(nearest) else np.nan
            )
            records.append(record)
        pair_values = np.asarray(pairwise, dtype=float)
        route_reference.append(
            {
                "station_day_id": f"{station_code}:{route_date.isoformat()}",
                "station_code": station_code,
                "date": route_date.isoformat(),
                "day_type": "weekend" if route_date.weekday() >= 5 else "weekday",
                "route_id": route_id,
                "usable_stop_count": len(route_records),
                "within_route_pair_count": len(pair_values),
                "within_route_pairwise_time_p50_s": (
                    float(np.quantile(pair_values, 0.50)) if len(pair_values) else np.nan
                ),
                "within_route_pairwise_time_p90_s": (
                    float(np.quantile(pair_values, 0.90)) if len(pair_values) else np.nan
                ),
            }
        )

    templates = pd.DataFrame.from_records(records)
    if templates.empty:
        raise ValueError("Amazon preprocessing produced no usable templates")
    all_times = templates["station_to_stop_time_s"].to_numpy(dtype=float)
    t_env = float(_quantile(all_times, np.asarray([0.99]))[0])
    envelope_times = all_times[all_times <= t_env]
    decile_edges = _quantile(envelope_times, np.linspace(0.0, 1.0, 11))
    decile_edges[0] = 0.0
    decile_edges[-1] = t_env
    templates["within_spatial_envelope"] = templates["station_to_stop_time_s"].le(t_env)
    templates["radial_decile"] = -1
    inside = templates["within_spatial_envelope"].to_numpy(dtype=bool)
    templates.loc[inside, "radial_decile"] = _assign_deciles(
        templates.loc[inside, "station_to_stop_time_s"].to_numpy(dtype=float),
        decile_edges,
    )
    templates["radial_decile"] = templates["radial_decile"].astype(np.int8)

    route_frame = pd.DataFrame.from_records(route_reference)
    spatial = templates.loc[templates["within_spatial_envelope"]].copy()
    decile_counts = (
        spatial.groupby(["station_day_id", "station_code", "date", "day_type", "route_id", "radial_decile"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(10), fill_value=0)
        .reset_index()
    )
    decile_counts = decile_counts.rename(columns={index: f"decile_{index}_count" for index in range(10)})
    structure_routes = route_frame.drop(
        columns=["usable_stop_count"], errors="ignore"
    ).merge(
        decile_counts,
        on=["station_day_id", "station_code", "date", "day_type", "route_id"],
        how="right",
        validate="one_to_one",
    )
    count_columns = [f"decile_{index}_count" for index in range(10)]
    structure_routes["structure_usable_stop_count"] = structure_routes[count_columns].sum(axis=1)
    order_day_counts = (
        templates.groupby(["station_day_id", "station_code", "date", "day_type"], observed=True)
        .size()
        .rename("order_usable_stop_count")
        .reset_index()
    )
    structure_day_counts = (
        structure_routes.groupby(["station_day_id", "station_code", "date", "day_type"], observed=True)["structure_usable_stop_count"]
        .sum()
        .reset_index()
    )
    station_days = order_day_counts.merge(
        structure_day_counts,
        on=["station_day_id", "station_code", "date", "day_type"],
        how="left",
        validate="one_to_one",
    )
    station_days["structure_usable_stop_count"] = station_days[
        "structure_usable_stop_count"
    ].fillna(0).astype(int)

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Amazon artifact directory exists without a valid manifest and is not empty: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    templates.to_parquet(out / "templates.parquet", index=False)
    structure_routes.to_parquet(out / "structure_routes.parquet", index=False)
    station_days.to_parquet(out / "station_days.parquet", index=False)
    route_frame.to_parquet(out / "route_spatial_reference.parquet", index=False)
    support = {}
    for day_type in ("weekday", "weekend"):
        day_rows = station_days.loc[station_days["day_type"].eq(day_type)]
        support[day_type] = {
            "single_order_days_ge_1000": int((day_rows["order_usable_stop_count"] >= 1000).sum()),
            "single_order_days_ge_2000": int((day_rows["order_usable_stop_count"] >= 2000).sum()),
            "single_structure_days_ge_1000": int((day_rows["structure_usable_stop_count"] >= 1000).sum()),
            "single_structure_days_ge_2000": int((day_rows["structure_usable_stop_count"] >= 2000).sum()),
        }
    manifest = {
        "schema": AMAZON_ARTIFACT_SCHEMA,
        "artifact_id": "amazon_stationday_stage2_v3",
        "source_scope": "almrrc2021-data-training/model_build_inputs_only",
        "source_files": {
            "route_data": "almrrc2021-data-training/model_build_inputs/route_data.json",
            "travel_times": "almrrc2021-data-training/model_build_inputs/travel_times.json",
            "package_data": "almrrc2021-data-training/model_build_inputs/package_data.json",
        },
        "source_registry_url": (
            "https://registry.opendata.aws/amazon-last-mile-challenges/"
        ),
        "source_license": "CC-BY-NC-4.0",
        "raw_coordinates_exported": False,
        "operating_horizon_s": [float(horizon_start_s), float(horizon_end_s)],
        "cargo_capacity_cm3": float(cargo_capacity_cm3),
        "template_count": len(templates),
        "template_id_contract": "station_code:date:route_id:stop_id",
        "station_day_count": len(station_days),
        "route_count": len(route_frame),
        "t_env_s": t_env,
        "radial_decile_edges_s": decile_edges.tolist(),
        "attrition": {str(key): int(value) for key, value in sorted(attrition.items())},
        "support": support,
        "outputs": {
            "templates": "templates.parquet",
            "structure_routes": "structure_routes.parquet",
            "station_days": "station_days.parquet",
            "route_spatial_reference": "route_spatial_reference.parquet",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


@dataclass(frozen=True)
class AmazonStage2Artifacts:
    root: Path
    manifest: dict[str, Any]
    templates: pd.DataFrame
    structure_routes: pd.DataFrame
    station_days: pd.DataFrame
    route_spatial_reference: pd.DataFrame
    cohort_split: dict[str, Any]
    station_day_pool: dict[str, str]

    def pool_for_track(self, track_id: str) -> str:
        mapping = self.cohort_split["track_to_pool"]
        if track_id not in mapping:
            raise ValueError(f"No frozen Amazon pool for track {track_id!r}")
        pool = str(mapping[track_id])
        if pool == "METRIC-HOLDOUT":
            raise ValueError("METRIC-HOLDOUT is metrics-only and cannot generate instances")
        return pool

    def station_day_ids_for_track(self, track_id: str) -> set[str]:
        pool = self.pool_for_track(track_id)
        rows = self.cohort_split["station_day_assignments"]
        if pool == "GEN-TRAIN":
            return {
                str(row["station_day_id"])
                for row in rows
                if str(row["pool"]) == pool
            }
        if pool == "GEN-EVAL":
            selected = {
                str(row["station_day_id"])
                for row in rows
                if str(row["pool"]) == pool
                and str(row.get("generation_track")) == str(track_id)
            }
            if not selected:
                raise ValueError(f"No frozen GEN-EVAL station-day ledger for {track_id!r}")
            return selected
        raise ValueError(f"Unsupported generation pool: {pool!r}")

    @property
    def t_env_s(self) -> float:
        return float(self.manifest["t_env_s"])

    @property
    def decile_edges_s(self) -> np.ndarray:
        return np.asarray(self.manifest["radial_decile_edges_s"], dtype=float)

    def structure_source(
        self,
        *,
        day_type: str,
        customer_count: int,
        seed: int,
        pool: str,
        track_id: str | None = None,
        allow_composite: bool = False,
        selected_source_ids: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        candidates = self.structure_source_candidates(
            day_type=day_type,
            customer_count=customer_count,
            seed=seed,
            pool=pool,
            track_id=track_id,
            allow_composite=allow_composite,
        )
        if selected_source_ids is None:
            source_days = list(candidates[0]["structure_source_ids"])
            mode = str(candidates[0]["structure_source_mode"])
        else:
            requested = tuple(map(str, selected_source_ids))
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if tuple(map(str, candidate["structure_source_ids"])) == requested
                ),
                None,
            )
            if selected is None:
                raise ValueError(
                    "FROZEN_STRUCTURE_SOURCE_UNAVAILABLE: "
                    f"requested={list(requested)}, day_type={day_type}, N={customer_count}"
                )
            source_days = list(selected["structure_source_ids"])
            mode = str(selected["structure_source_mode"])
        routes = self.structure_routes.loc[
            self.structure_routes["station_day_id"].isin(source_days)
        ].copy()
        targets = self.templates.loc[self.templates["station_day_id"].isin(source_days)].copy()
        source_times = targets["station_to_stop_time_s"].to_numpy(dtype=float)
        source_t_env = float(_quantile(source_times, np.asarray([0.99]))[0])
        within = targets["station_to_stop_time_s"].le(source_t_env).to_numpy(dtype=bool)
        envelope_times = source_times[within]
        source_edges = _quantile(envelope_times, np.linspace(0.0, 1.0, 11))
        source_edges[0] = 0.0
        source_edges[-1] = source_t_env
        targets["within_spatial_envelope"] = within
        targets["radial_decile"] = -1
        targets.loc[within, "radial_decile"] = _assign_deciles(
            envelope_times, source_edges
        )
        targets["radial_decile"] = targets["radial_decile"].astype(np.int8)
        targets = targets.loc[targets["within_spatial_envelope"]].copy()
        return targets, {
            "structure_source_mode": mode,
            "structure_source_ids": sorted(source_days),
            "structure_source_dates": sorted(source_days),
            "source_pool": pool,
            "generation_track": track_id,
            "structure_source_route_count": len(routes),
            "structure_source_stop_count": len(targets),
            "source_t_env_s": source_t_env,
            "source_radial_decile_edges_s": source_edges.tolist(),
        }

    def structure_source_candidates(
        self,
        *,
        day_type: str,
        customer_count: int,
        seed: int,
        pool: str,
        track_id: str | None = None,
        allow_composite: bool = False,
    ) -> list[dict[str, Any]]:
        """Return every legal structure source in its frozen seeded order."""

        days = self.station_days.loc[
            self.station_days["day_type"].eq(day_type)
            & self.station_days["station_day_id"].map(self.station_day_pool).eq(pool)
        ].copy()
        if track_id is not None:
            days = days.loc[
                days["station_day_id"].astype(str).isin(
                    self.station_day_ids_for_track(track_id)
                )
            ].copy()
        source_counts = {}
        for station_day_id, rows in self.templates.loc[
            self.templates["station_day_id"].isin(days["station_day_id"])
        ].groupby("station_day_id", sort=True):
            values = rows["station_to_stop_time_s"].to_numpy(dtype=float)
            source_t_env = float(_quantile(values, np.asarray([0.99]))[0])
            source_counts[str(station_day_id)] = int(np.count_nonzero(values <= source_t_env))
        days["per_source_structure_usable_stop_count"] = (
            days["station_day_id"].astype(str).map(source_counts).fillna(0).astype(int)
        )
        singles = days.loc[
            days["per_source_structure_usable_stop_count"].ge(customer_count)
        ].copy()
        if not singles.empty:
            singles["_rank"] = [
                _stable_rank(seed, "single_structure_day", value)
                for value in singles["station_day_id"]
            ]
            ordered = singles.sort_values(["_rank", "station_day_id"], kind="stable")
            return [
                {
                    "structure_source_mode": "SINGLE_STRUCTURE_DAY",
                    "structure_source_ids": [str(value)],
                    "seeded_rank": int(rank),
                }
                for value, rank in zip(
                    ordered["station_day_id"], ordered["_rank"], strict=True
                )
            ]
        elif allow_composite:
            candidates: list[tuple[int, int, str, list[str]]] = []
            for station_code, group in days.groupby("station_code", sort=True):
                ordered = group.sort_values(
                    ["per_source_structure_usable_stop_count", "date"],
                    ascending=[False, True],
                    kind="stable",
                )
                chosen: list[str] = []
                total = 0
                for row in ordered.itertuples(index=False):
                    chosen.append(str(row.station_day_id))
                    combined_times = self.templates.loc[
                        self.templates["station_day_id"].isin(chosen),
                        "station_to_stop_time_s",
                    ].to_numpy(dtype=float)
                    combined_t_env = float(
                        _quantile(combined_times, np.asarray([0.99]))[0]
                    )
                    total = int(np.count_nonzero(combined_times <= combined_t_env))
                    if total >= customer_count:
                        candidates.append(
                            (
                                len(chosen),
                                _stable_rank(seed, "structure_composite", station_code),
                                str(station_code),
                                chosen.copy(),
                            )
                        )
                        break
            if not candidates:
                raise ValueError(
                    f"PF-2S unsupported for day_type={day_type}, N={customer_count}"
                )
            ordered_candidates = sorted(candidates, key=lambda item: (item[0], item[1], item[2]))
            return [
                {
                    "structure_source_mode": "SAME_STATION_STRUCTURE_COMPOSITE",
                    "structure_source_ids": list(source_days),
                    "seeded_rank": int(rank),
                }
                for _, rank, _, source_days in ordered_candidates
            ]
        else:
            raise ValueError(
                f"PRIMARY_SINGLE_STRUCTURE_DAY_UNSUPPORTED: pool={pool}, "
                f"day_type={day_type}, N={customer_count}"
            )

    def order_sources(
        self,
        *,
        day_type: str,
        customer_count: int,
        seed: int,
        pool: str,
        track_id: str | None = None,
        allow_composite: bool = False,
    ) -> list[dict[str, Any]]:
        days = self.station_days.loc[
            self.station_days["day_type"].eq(day_type)
            & self.station_days["station_day_id"].map(self.station_day_pool).eq(pool)
        ].copy()
        if track_id is not None:
            days = days.loc[
                days["station_day_id"].astype(str).isin(
                    self.station_day_ids_for_track(track_id)
                )
            ].copy()
        singles = days.loc[days["order_usable_stop_count"].ge(customer_count)].copy()
        single_records = [
            {
                "order_source_mode": "SINGLE_ORDER_DAY",
                "station_code": str(row.station_code),
                "station_day_ids": [str(row.station_day_id)],
                "rank": _stable_rank(seed, "order_single", row.station_day_id),
            }
            for row in singles.itertuples(index=False)
        ]
        single_records.sort(key=lambda item: (item["rank"], item["station_day_ids"]))

        composites: list[dict[str, Any]] = []
        for station_code, group in days.groupby("station_code", sort=True):
            ordered = group.sort_values(
                ["order_usable_stop_count", "date"],
                ascending=[False, True],
                kind="stable",
            )
            chosen: list[str] = []
            total = 0
            for row in ordered.itertuples(index=False):
                chosen.append(str(row.station_day_id))
                total += int(row.order_usable_stop_count)
                if total >= customer_count:
                    if len(chosen) > 1:
                        composites.append(
                            {
                                "order_source_mode": "SAME_STATION_ORDER_COMPOSITE",
                                "station_code": str(station_code),
                                "station_day_ids": chosen.copy(),
                                "rank": _stable_rank(
                                    seed,
                                    "order_composite",
                                    station_code,
                                    *chosen,
                                ),
                            }
                        )
                    break
        composites.sort(
            key=lambda item: (len(item["station_day_ids"]), item["rank"], item["station_code"])
        )
        records = [*single_records, *composites] if allow_composite else single_records
        for record in records:
            record["source_pool"] = pool
            record["generation_track"] = track_id
        return records

    def templates_for_source(self, source: dict[str, Any]) -> pd.DataFrame:
        return self.templates.loc[
            self.templates["station_day_id"].isin(source["station_day_ids"])
        ].copy()


def load_amazon_stage2_artifacts(
    root: str | Path,
    *,
    cohort_split_path: str | Path,
) -> AmazonStage2Artifacts:
    artifact_root = Path(root)
    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != AMAZON_ARTIFACT_SCHEMA:
        raise ValueError(f"Unsupported Amazon Stage-2 artifact: {manifest.get('schema')!r}")
    if manifest.get("template_id_contract") != "station_code:date:route_id:stop_id":
        raise ValueError("Stale Amazon v3 artifact lacks the V2.1 template ID contract")
    cohort_split = json.loads(Path(cohort_split_path).read_text(encoding="utf-8"))
    if cohort_split.get("schema") != "evrptw_amazon_cohort_split_v1":
        raise ValueError(f"Unsupported Amazon cohort split: {cohort_split.get('schema')!r}")
    assertions = cohort_split.get("leakage_assertions", {})
    if not assertions or not all(bool(value) for value in assertions.values()):
        raise ValueError("Amazon cohort split leakage assertions are not all true")
    evaluation_allocation = cohort_split.get("evaluation_track_allocation", {})
    if not bool(
        evaluation_allocation.get(
            "station_day_ledgers_pairwise_disjoint_and_exhaustive", False
        )
    ):
        raise ValueError("GEN-EVAL track station-day ledgers are not a frozen partition")
    if bool(evaluation_allocation.get("exact_template_reuse_between_evaluation_tracks", True)):
        raise ValueError("GEN-EVAL track ledger unexpectedly permits exact template reuse")
    evaluation_tracks = set(map(str, evaluation_allocation.get("tracks", [])))
    assigned_evaluation_tracks = {
        str(row.get("generation_track"))
        for row in cohort_split["station_day_assignments"]
        if str(row["pool"]) == "GEN-EVAL"
    }
    if assigned_evaluation_tracks != evaluation_tracks:
        raise ValueError(
            "GEN-EVAL assignment rows do not cover exactly the declared evaluation tracks"
        )
    station_day_pool = {
        str(row["station_day_id"]): str(row["pool"])
        for row in cohort_split["station_day_assignments"]
    }
    outputs = manifest["outputs"]
    return AmazonStage2Artifacts(
        root=artifact_root,
        manifest=manifest,
        templates=pd.read_parquet(artifact_root / outputs["templates"]),
        structure_routes=pd.read_parquet(artifact_root / outputs["structure_routes"]),
        station_days=pd.read_parquet(artifact_root / outputs["station_days"]),
        route_spatial_reference=pd.read_parquet(
            artifact_root / outputs["route_spatial_reference"]
        ),
        cohort_split=cohort_split,
        station_day_pool=station_day_pool,
    )
