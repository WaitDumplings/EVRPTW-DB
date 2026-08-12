"""Extract reviewer-auditable instance-calibration statistics from Amazon ARCD.

Amazon defines planned service time at package level. This script therefore
sums package service times at each physical stop, while separately resolving
package time windows to a one-window-per-stop calibration subset. Raw
coordinates are deliberately not exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-data", type=Path, required=True)
    parser.add_argument("--package-flat-pickle", type=Path, required=True)
    parser.add_argument("--package-json", type=Path)
    parser.add_argument("--flatten-script", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return None
    return text


def _window(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, dict):
        return None
    start = _valid_timestamp(value.get("start_time_utc"))
    end = _valid_timestamp(value.get("end_time_utc"))
    if start is None or end is None:
        return None
    return start, end


def _window_label(window: tuple[str, str]) -> str:
    start = datetime.fromisoformat(window[0])
    end = datetime.fromisoformat(window[1])
    return f"{start.time().isoformat()}--{end.time().isoformat()}"


def _window_duration_hours(window: tuple[str, str]) -> float | None:
    start = datetime.fromisoformat(window[0])
    end = datetime.fromisoformat(window[1])
    hours = (end - start).total_seconds() / 3600.0
    return hours if math.isfinite(hours) and hours >= 0 else None


def _percentile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _summary(values: Iterable[float | int]) -> dict[str, Any]:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        return {"count": 0}
    cleaned.sort()
    return {
        "count": len(cleaned),
        "min": cleaned[0],
        "p10": _percentile(cleaned, 0.10),
        "p25": _percentile(cleaned, 0.25),
        "p50": _percentile(cleaned, 0.50),
        "mean": statistics.fmean(cleaned),
        "p75": _percentile(cleaned, 0.75),
        "p90": _percentile(cleaned, 0.90),
        "p95": _percentile(cleaned, 0.95),
        "p99": _percentile(cleaned, 0.99),
        "max": cleaned[-1],
        "population_std": statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_scale == 0 or y_scale == 0:
        return None
    return covariance / (x_scale * y_scale)


def _top(counter: Counter[str], total: int, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count, "share": count / total if total else 0.0}
        for value, count in counter.most_common(limit)
    ]


def _day_type(date_text: str) -> tuple[str, str]:
    route_date = date.fromisoformat(date_text)
    weekday = route_date.strftime("%A")
    return weekday, "weekend" if route_date.weekday() >= 5 else "weekday"


def _package_count_bin(count: int) -> str:
    if count <= 5:
        return str(count)
    if count <= 10:
        return "6-10"
    return "11+"


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _render_markdown(result: dict[str, Any]) -> str:
    totals = result["totals"]
    stop = result["stop_level"]
    tw = result["time_windows"]
    lines = [
        "# Amazon ARCD training statistics for EVRPTW instance calibration",
        "",
        (
            "> These are aggregate calibration statistics. Raw Amazon coordinates are not "
            "exported, and the results do not claim that any generated city reproduces an "
            "Amazon geography."
        ),
        "",
        "## Dataset inventory",
        "",
        f"- Routes: {totals['route_count']:,}",
        f"- Route stops including stations: {totals['route_stop_count_including_station']:,}",
        f"- Package records: {totals['package_record_count']:,}",
        f"- Delivered package records: {totals['delivered_package_count']:,}",
        f"- Delivered physical stops: {totals['delivered_stop_count']:,}",
        f"- Stations: {totals['station_code_count']:,}",
        f"- Date range: {totals['date_min']} through {totals['date_max']}",
        "",
        "## Stop-level core statistics",
        "",
        (
            f"- Packages per stop: mean {stop['packages_per_stop']['mean']:.3f}, "
            f"median {stop['packages_per_stop']['p50']:.1f}, "
            f"p90 {stop['packages_per_stop']['p90']:.1f}."
        ),
        (
            "- Total planned service time per stop: mean "
            f"{stop['summed_planned_service_time_seconds']['mean']:.3f}s, "
            f"median {stop['summed_planned_service_time_seconds']['p50']:.1f}s, "
            f"p90 {stop['summed_planned_service_time_seconds']['p90']:.1f}s."
        ),
        (
            f"- Package volume per stop: mean {stop['delivered_volume_cm3']['mean']:.3f} cm3, "
            f"median {stop['delivered_volume_cm3']['p50']:.1f} cm3, "
            f"p90 {stop['delivered_volume_cm3']['p90']:.1f} cm3."
        ),
        (
            "- Mean summed service time per route: "
            f"{result['route_level']['summed_planned_service_time_hours']['mean']:.3f}h."
        ),
        "",
        "## Time windows",
        "",
        f"- Delivered packages with a TW: {tw['package_with_tw_share']:.3%}.",
        f"- Delivered stops with at least one TW package: {tw['stop_with_any_tw_share']:.3%}.",
        f"- Stops with exactly one distinct nonmissing TW: {tw['single_tw_stop_share']:.3%}.",
        f"- Stops with multiple distinct nonmissing TWs: {tw['conflicting_tw_stop_share']:.3%}.",
        "",
        "## Intended Stage 2 use",
        "",
        (
            "Use these summaries to calibrate packages per active location, package volume, one "
            "shared time window per active location, and summed package service time. Do not "
            "place package count, time windows, or realized demand in the City Logistics Environment."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    route_data = json.loads(args.route_data.read_text(encoding="utf-8"))
    route_meta: dict[str, dict[str, str]] = {}
    station_codes: Counter[str] = Counter()
    route_scores: Counter[str] = Counter()
    day_names: Counter[str] = Counter()
    day_types: Counter[str] = Counter()
    route_dropoff_counts: list[int] = []
    route_stop_count = 0
    dates: list[str] = []
    route_ids_by_day_type: dict[str, set[str]] = defaultdict(set)
    for route_id, route in route_data.items():
        date_text = str(route["date_YYYY_MM_DD"])
        weekday, day_type = _day_type(date_text)
        route_meta[route_id] = {
            "date": date_text,
            "weekday": weekday,
            "day_type": day_type,
            "route_score": str(route.get("route_score") or "missing"),
        }
        dates.append(date_text)
        day_names[weekday] += 1
        day_types[day_type] += 1
        route_ids_by_day_type[day_type].add(route_id)
        station_codes[str(route.get("station_code") or "missing")] += 1
        route_scores[str(route.get("route_score") or "missing")] += 1
        stops = route.get("stops", {})
        route_stop_count += len(stops)
        route_dropoff_counts.append(
            sum(1 for stop in stops.values() if stop.get("type") == "Dropoff")
        )

    with args.package_flat_pickle.open("rb") as stream:
        package_rows = pickle.load(stream)
    scan_statuses: Counter[str] = Counter()
    package_volume_values: list[float] = []
    invalid_package_dimensions = 0
    delivered_package_count = 0
    delivered_package_with_tw = 0
    stop_aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    route_delivered_packages: Counter[str] = Counter()
    package_service_times: list[float] = []
    route_service_seconds: Counter[str] = Counter()
    for row in package_rows:
        status = str(row.get("scan_status") or "missing")
        scan_statuses[status] += 1
        if status != "DELIVERED":
            continue
        delivered_package_count += 1
        route_id = str(row["route_id"])
        stop_id = str(row["node_name"])
        route_delivered_packages[route_id] += 1
        key = (route_id, stop_id)
        aggregate = stop_aggregates.setdefault(
            key,
            {
                "package_count": 0,
                "volume_cm3": 0.0,
                "valid_volume_count": 0,
                "service_sum": 0.0,
                "valid_service_count": 0,
                "tw_values": set(),
                "tw_missing_count": 0,
            },
        )
        aggregate["package_count"] += 1
        dimensions = row.get("dimensions") or {}
        depth = _finite_number(dimensions.get("depth_cm"))
        height = _finite_number(dimensions.get("height_cm"))
        width = _finite_number(dimensions.get("width_cm"))
        if depth is None or height is None or width is None or min(depth, height, width) <= 0:
            invalid_package_dimensions += 1
        else:
            volume = depth * height * width
            aggregate["volume_cm3"] += volume
            aggregate["valid_volume_count"] += 1
            package_volume_values.append(volume)
        service = _finite_number(row.get("planned_service_time_seconds"))
        if service is not None:
            aggregate["service_sum"] += service
            aggregate["valid_service_count"] += 1
            package_service_times.append(service)
            route_service_seconds[route_id] += service
        tw = _window(row.get("time_window"))
        if tw is None:
            aggregate["tw_missing_count"] += 1
        else:
            delivered_package_with_tw += 1
            aggregate["tw_values"].add(tw)

    del package_rows

    packages_per_stop: list[float] = []
    stop_volumes: list[float] = []
    stop_service_times: list[float] = []
    correlation_packages: list[float] = []
    correlation_volumes: list[float] = []
    correlation_services: list[float] = []
    incomplete_service_stops = 0
    stops_with_any_tw = 0
    stops_with_single_tw = 0
    stops_with_conflicting_tw = 0
    stops_with_mixed_missing_and_tw = 0
    stop_tw_durations: list[float] = []
    package_count_bins: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"service": [], "volume": [], "packages": []}
    )
    by_day_type: dict[str, dict[str, Any]] = {
        day_type: {
            "stop_packages": [],
            "stop_volume": [],
            "stop_service": [],
            "tw_stop_count": 0,
            "single_tw_stop_count": 0,
            "conflicting_tw_stop_count": 0,
            "tw_patterns": Counter(),
        }
        for day_type in ("weekday", "weekend")
    }
    tw_patterns_all: Counter[str] = Counter()
    for (route_id, _), aggregate in stop_aggregates.items():
        package_count = int(aggregate["package_count"])
        volume = float(aggregate["volume_cm3"])
        packages_per_stop.append(package_count)
        stop_volumes.append(volume)
        day_type = route_meta[route_id]["day_type"]
        by_day_type[day_type]["stop_packages"].append(package_count)
        by_day_type[day_type]["stop_volume"].append(volume)
        bin_id = _package_count_bin(package_count)
        package_count_bins[bin_id]["packages"].append(package_count)
        package_count_bins[bin_id]["volume"].append(volume)
        service = float(aggregate["service_sum"])
        if aggregate["valid_service_count"] != package_count:
            incomplete_service_stops += 1
        if aggregate["valid_service_count"]:
            stop_service_times.append(service)
            by_day_type[day_type]["stop_service"].append(service)
            package_count_bins[bin_id]["service"].append(service)
            if aggregate["valid_service_count"] == package_count:
                correlation_packages.append(float(package_count))
                correlation_volumes.append(volume)
                correlation_services.append(float(service))
        tw_values = aggregate["tw_values"]
        if tw_values:
            stops_with_any_tw += 1
            by_day_type[day_type]["tw_stop_count"] += 1
            if aggregate["tw_missing_count"]:
                stops_with_mixed_missing_and_tw += 1
            if len(tw_values) == 1:
                stops_with_single_tw += 1
                by_day_type[day_type]["single_tw_stop_count"] += 1
                tw = next(iter(tw_values))
                label = _window_label(tw)
                tw_patterns_all[label] += 1
                by_day_type[day_type]["tw_patterns"][label] += 1
                duration = _window_duration_hours(tw)
                if duration is not None:
                    stop_tw_durations.append(duration)
            else:
                stops_with_conflicting_tw += 1
                by_day_type[day_type]["conflicting_tw_stop_count"] += 1

    route_package_values = [route_delivered_packages[route_id] for route_id in route_meta]
    route_service_hours = [route_service_seconds[route_id] / 3600.0 for route_id in route_meta]
    total_stops = len(stop_aggregates)
    day_type_result: dict[str, Any] = {}
    for day_type, values in by_day_type.items():
        route_ids = route_ids_by_day_type[day_type]
        route_package_counts = [route_delivered_packages[route_id] for route_id in route_ids]
        day_stop_count = len(values["stop_packages"])
        day_type_result[day_type] = {
            "route_count": len(route_ids),
            "delivered_stop_count": day_stop_count,
            "delivered_package_count": int(sum(route_package_counts)),
            "route_packages": _summary(route_package_counts),
            "packages_per_stop": _summary(values["stop_packages"]),
            "delivered_volume_cm3_per_stop": _summary(values["stop_volume"]),
            "summed_planned_service_time_seconds_per_stop": _summary(
                values["stop_service"]
            ),
            "stop_with_any_tw_share": (
                values["tw_stop_count"] / day_stop_count if day_stop_count else 0.0
            ),
            "single_tw_stop_share": (
                values["single_tw_stop_count"] / day_stop_count if day_stop_count else 0.0
            ),
            "conflicting_tw_stop_share": (
                values["conflicting_tw_stop_count"] / day_stop_count if day_stop_count else 0.0
            ),
            "top_single_tw_patterns": _top(
                values["tw_patterns"], values["single_tw_stop_count"]
            ),
        }

    source_files = {
        "route_data": {"path": str(args.route_data.resolve()), "sha256": _sha256(args.route_data)},
        "package_flat_pickle": {
            "path": str(args.package_flat_pickle.resolve()),
            "sha256": _sha256(args.package_flat_pickle),
        },
    }
    if args.package_json:
        source_files["package_json"] = {
            "path": str(args.package_json.resolve()),
            "sha256": _sha256(args.package_json),
        }
    if args.flatten_script:
        source_files["flatten_script"] = {
            "path": str(args.flatten_script.resolve()),
            "sha256": _sha256(args.flatten_script),
        }

    result = {
        "schema": "evrptw_amazon_arcd_training_statistics_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "stage_role": "Stage_2_instance_calibration_only",
        "coordinate_policy": "raw Amazon coordinates are not exported or used as cle locations",
        "source_files": source_files,
        "totals": {
            "route_count": len(route_meta),
            "route_stop_count_including_station": route_stop_count,
            "route_dropoff_count": int(sum(route_dropoff_counts)),
            "package_record_count": int(sum(scan_statuses.values())),
            "delivered_package_count": delivered_package_count,
            "delivered_stop_count": total_stops,
            "station_code_count": len(station_codes),
            "date_min": min(dates),
            "date_max": max(dates),
        },
        "route_level": {
            "routes_by_weekday": dict(day_names),
            "routes_by_day_type": dict(day_types),
            "routes_by_score": dict(route_scores),
            "routes_by_station_code": dict(station_codes),
            "dropoff_stops_per_route": _summary(route_dropoff_counts),
            "delivered_packages_per_route": _summary(route_package_values),
            "summed_planned_service_time_hours": _summary(route_service_hours),
        },
        "package_level": {
            "scan_status_counts": dict(scan_statuses),
            "invalid_or_nonpositive_dimension_count": invalid_package_dimensions,
            "valid_delivered_package_volume_cm3": _summary(package_volume_values),
            "planned_service_time_seconds": _summary(package_service_times),
        },
        "stop_level": {
            "packages_per_stop": _summary(packages_per_stop),
            "delivered_volume_cm3": _summary(stop_volumes),
            "summed_planned_service_time_seconds": _summary(stop_service_times),
            "service_time_completeness": {
                "incomplete_stop_count": incomplete_service_stops,
                "complete_stop_count": total_stops - incomplete_service_stops,
            },
            "correlations_on_complete_service_stops": {
                "service_vs_package_count_pearson": _pearson(
                    correlation_services, correlation_packages
                ),
                "service_vs_total_volume_pearson": _pearson(
                    correlation_services, correlation_volumes
                ),
            },
            "by_delivered_package_count_bin": {
                bin_id: {
                    "stop_count": len(values["packages"]),
                    "packages_per_stop": _summary(values["packages"]),
                    "delivered_volume_cm3": _summary(values["volume"]),
                    "summed_planned_service_time_seconds": _summary(values["service"]),
                }
                for bin_id, values in sorted(package_count_bins.items())
            },
        },
        "time_windows": {
            "package_with_tw_count": delivered_package_with_tw,
            "package_with_tw_share": delivered_package_with_tw / delivered_package_count,
            "stop_with_any_tw_count": stops_with_any_tw,
            "stop_with_any_tw_share": stops_with_any_tw / total_stops,
            "single_tw_stop_count": stops_with_single_tw,
            "single_tw_stop_share": stops_with_single_tw / total_stops,
            "conflicting_tw_stop_count": stops_with_conflicting_tw,
            "conflicting_tw_stop_share": stops_with_conflicting_tw / total_stops,
            "mixed_missing_and_tw_stop_count": stops_with_mixed_missing_and_tw,
            "single_tw_duration_hours": _summary(stop_tw_durations),
            "top_single_tw_patterns": _top(tw_patterns_all, stops_with_single_tw),
        },
        "weekday_weekend": day_type_result,
        "instance_contract_recommendation": {
            "cus_n": "N distinct activated physical service locations",
            "packages": "sampled at instance generation conditional on active location attributes",
            "time_window": "at most one shared time window per active location in v1",
            "service_time": "generate package-level service requirements and sum them at each active location, or fit an equivalent stop-level conditional model",
            "cle_exclusions": [
                "realized package count",
                "realized package dimensions or volume",
                "time window",
                "realized service time",
            ],
        },
        "known_limitations": [
            "The package_flat_pickle is a local deterministic flattening of package_data.json; both hashes and the flattening-script hash are recorded when supplied.",
            "Amazon stop coordinates are not treated as locations for the eleven-city CLEs.",
            "Amazon ARCD does not label a stop as house or apartment, so building-type-conditioned demand requires a separately disclosed model.",
            "Descriptive weekday/weekend differences are not automatically causal city-transfer parameters.",
        ],
    }
    result = _round_floats(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "amazon_arcd_training_statistics_v1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "amazon_arcd_training_statistics_v1.md").write_text(
        _render_markdown(result), encoding="utf-8"
    )
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
