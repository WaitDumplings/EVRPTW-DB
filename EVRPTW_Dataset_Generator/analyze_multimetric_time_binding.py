from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from evrptw_core.io import iter_instances  # noqa: E402


def offdiag(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    mask = ~np.eye(arr.shape[0], dtype=bool)
    values = arr[mask]
    return values[np.isfinite(values)]


def quantiles(values: np.ndarray, ps: tuple[float, ...] = (0.1, 0.5, 0.9, 0.99)) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(p * 100)}": float("nan") for p in ps}
    return {f"p{int(p * 100)}": float(np.quantile(arr, p)) for p in ps}


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    mask = (~np.eye(av.shape[0], dtype=bool)) & np.isfinite(av) & np.isfinite(bv)
    x = av[mask]
    y = bv[mask]
    if x.size < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def greedy_metrics(instance, time_s: np.ndarray) -> dict[str, float | int | bool]:
    n = instance.num_customers
    service = np.asarray(instance.service_time_s, dtype=np.float64)
    tw = np.asarray(instance.tw_s, dtype=np.float64)
    demand = np.asarray(instance.demands_cm3, dtype=np.float64)
    cap = float(instance.vehicle.get("cargo_capacity_cm3", np.inf))
    working_start = float(instance.working_start_s)
    working_end = float(instance.working_end_s)
    unvisited = np.ones(n, dtype=bool)
    starts: list[float] = []
    ends: list[float] = []
    waits: list[float] = []
    route_lens: list[int] = []
    route_count = 0

    while np.any(unvisited):
        route_count += 1
        cur = 0
        clock = working_start
        load = 0.0
        route_len = 0
        progressed = False
        while True:
            remaining = np.flatnonzero(unvisited)
            if remaining.size == 0:
                break
            nodes = remaining + 1
            travel = time_s[cur, nodes]
            back = time_s[nodes, 0]
            arrival = clock + travel
            service_start = np.maximum(arrival, tw[remaining, 0])
            finish = service_start + service[remaining]
            feasible = (
                (load + demand[remaining] <= cap)
                & np.isfinite(travel)
                & np.isfinite(back)
                & (service_start <= tw[remaining, 1] + 1e-6)
                & (finish + back <= working_end + 1e-6)
            )
            if not np.any(feasible):
                break
            feasible_remaining = remaining[feasible]
            feasible_start = service_start[feasible]
            best = int(feasible_remaining[int(np.argmin(feasible_start))])
            node = best + 1
            raw_arrival = float(clock + time_s[cur, node])
            start = max(raw_arrival, float(tw[best, 0]))
            starts.append(start)
            ends.append(float(tw[best, 1]))
            waits.append(max(0.0, start - raw_arrival))
            clock = start + float(service[best])
            load += float(demand[best])
            unvisited[best] = False
            cur = node
            route_len += 1
            progressed = True
        route_lens.append(route_len)
        if not progressed:
            return {
                "greedy_success": False,
                "route_count": int(route_count),
                "served": int(n - int(np.count_nonzero(unvisited))),
            }

    starts_arr = np.asarray(starts, dtype=np.float64)
    ends_arr = np.asarray(ends, dtype=np.float64)
    waits_arr = np.asarray(waits, dtype=np.float64)
    upper_slack = ends_arr - starts_arr
    return {
        "greedy_success": True,
        "route_count": int(route_count),
        "served": int(n),
        "route_len_p50": float(np.quantile(route_lens, 0.5)),
        "upper_slack_min_min": float(np.min(upper_slack) / 60.0),
        "upper_slack_p10_min": float(np.quantile(upper_slack, 0.1) / 60.0),
        "upper_slack_p50_min": float(np.quantile(upper_slack, 0.5) / 60.0),
        "upper_slack_p90_min": float(np.quantile(upper_slack, 0.9) / 60.0),
        "near_due_5min_share": float(np.mean(upper_slack <= 5.0 * 60.0)),
        "near_due_15min_share": float(np.mean(upper_slack <= 15.0 * 60.0)),
        "near_due_30min_share": float(np.mean(upper_slack <= 30.0 * 60.0)),
        "wait_share": float(np.mean(waits_arr > 1e-6)),
        "wait_p50_min": float(np.quantile(waits_arr, 0.5) / 60.0),
        "wait_p90_min": float(np.quantile(waits_arr, 0.9) / 60.0),
    }


def pairwise_feasible_share(instance, time_s: np.ndarray) -> float:
    n = instance.num_customers
    service = np.asarray(instance.service_time_s, dtype=np.float64)
    tw = np.asarray(instance.tw_s, dtype=np.float64)
    working_start = float(instance.working_start_s)
    feasible: list[bool] = []
    for i in range(n):
        start_i = max(working_start + float(time_s[0, i + 1]), float(tw[i, 0]))
        depart_i = start_i + float(service[i])
        for j in range(n):
            if i == j:
                continue
            feasible.append(depart_i + float(time_s[i + 1, j + 1]) <= float(tw[j, 1]) + 1e-6)
    return float(np.mean(feasible))


def analyze(base_path: Path, multi_path: Path, output_path: Path) -> list[dict[str, object]]:
    base_instances = list(iter_instances(base_path))
    multi_instances = list(iter_instances(multi_path))
    rows: list[dict[str, object]] = []
    for baseline, multi in zip(base_instances, multi_instances):
        distance = np.asarray(multi.distance_matrix_km, dtype=np.float64)
        effective_speed = float(
            baseline.speed_profile.get("effective_speed_kmh")
            or baseline.vehicle.get("design_speed_kmh")
            or 40.0
        )
        baseline_time = distance / max(effective_speed, 1e-9) * 3600.0
        multi_time = np.asarray(multi.raw_travel_time_matrix_s, dtype=np.float64)
        time_ratio = offdiag(multi_time / np.maximum(baseline_time, 1e-9))
        time_absdiff_min = offdiag(multi_time - baseline_time) / 60.0
        tw = np.asarray(multi.tw_s, dtype=np.float64)
        tw_width_h = (tw[:, 1] - tw[:, 0]) / 3600.0
        direct_arrival = float(multi.working_start_s) + multi_time[0, 1 : multi.num_customers + 1]
        direct_service_start = np.maximum(direct_arrival, tw[:, 0])
        direct_upper_slack_min = (tw[:, 1] - direct_service_start) / 60.0
        baseline_greedy = greedy_metrics(baseline, baseline_time)
        multi_greedy = greedy_metrics(multi, multi_time)
        row: dict[str, object] = {
            "instance_id": multi.instance_id,
            "region_id": multi.region_id,
            "day_type": multi.day_type,
            "distance_time_corr": correlation(distance, multi_time),
            "time_ratio_p10": quantiles(time_ratio)["p10"],
            "time_ratio_p50": quantiles(time_ratio)["p50"],
            "time_ratio_p90": quantiles(time_ratio)["p90"],
            "time_ratio_p99": quantiles(time_ratio)["p99"],
            "time_absdiff_p50_min": quantiles(time_absdiff_min)["p50"],
            "time_absdiff_p90_min": quantiles(time_absdiff_min)["p90"],
            "tw_width_p10_h": quantiles(tw_width_h)["p10"],
            "tw_width_p50_h": quantiles(tw_width_h)["p50"],
            "tw_width_p90_h": quantiles(tw_width_h)["p90"],
            "direct_upper_slack_p10_min": quantiles(direct_upper_slack_min)["p10"],
            "direct_upper_slack_p50_min": quantiles(direct_upper_slack_min)["p50"],
            "pairwise_feasible_baseline": pairwise_feasible_share(baseline, baseline_time),
            "pairwise_feasible_multi": pairwise_feasible_share(multi, multi_time),
            "greedy_routes_baseline": baseline_greedy.get("route_count"),
            "greedy_routes_multi": multi_greedy.get("route_count"),
            "greedy_near_due_15m_baseline": baseline_greedy.get("near_due_15min_share"),
            "greedy_near_due_15m_multi": multi_greedy.get("near_due_15min_share"),
            "greedy_near_due_30m_multi": multi_greedy.get("near_due_30min_share"),
            "greedy_upper_slack_p10_multi_min": multi_greedy.get("upper_slack_p10_min"),
            "greedy_upper_slack_p50_multi_min": multi_greedy.get("upper_slack_p50_min"),
            "greedy_wait_share_multi": multi_greedy.get("wait_share"),
        }
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    output_path.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    root = Path("EVRPTW_Dataset/Geo_AC_v1/multimetric_smoke_cus50_5")
    rows = analyze(
        root / "baseline_singlemetric" / "instances.pkl",
        root / "multimetric" / "instances.pkl",
        root / "diagnostics" / "time_binding_summary.csv",
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
