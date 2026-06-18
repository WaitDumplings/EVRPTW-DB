"""Merge redundant routes in solved EVRPTW instances for QA visualization.

This is an analysis utility, not a replacement for the reference solver. It
tries to reduce vehicle/route count while preserving feasibility and keeping
the distance within a small tie-break tolerance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from plot_solution_route_maps import (
    _parse_routes,
    _plot_instance,
    _read_selected_instances,
    _selected_summary_rows,
    _write_contact_sheet,
)


def route_distance_km(route: list[int], distance_matrix_km: np.ndarray) -> float:
    return float(sum(float(distance_matrix_km[a, b]) for a, b in zip(route[:-1], route[1:])))


def total_distance_km(routes: list[list[int]], distance_matrix_km: np.ndarray) -> float:
    return float(sum(route_distance_km(route, distance_matrix_km) for route in routes))


def route_customers(route: list[int], num_customers: int) -> list[int]:
    return [int(node) for node in route if 1 <= int(node) <= num_customers]


def check_route_feasible(route: list[int], instance: dict[str, Any]) -> tuple[bool, str]:
    num_customers = int(np.asarray(instance["customers"]).shape[0])
    distance_matrix_km = np.asarray(instance["distance_matrix_km"], dtype=float)
    demands = np.asarray(instance["demands_cm3"], dtype=float)
    service_time_s = np.asarray(instance["service_time_s"], dtype=float)
    tw_s = np.asarray(instance["tw_s"], dtype=float)
    vehicle = instance["vehicle"]
    speed = float(instance["speed_profile"].get("effective_speed_kmh") or instance["speed_profile"].get("design_speed_kmh"))

    load = sum(float(demands[node - 1]) for node in route if 1 <= int(node) <= num_customers)
    if load > float(vehicle["cargo_capacity_cm3"]) + 1e-6:
        return False, "capacity"

    battery_capacity = float(vehicle["battery_capacity_kwh"])
    consumption = float(vehicle["consumption_kwh_per_km"])
    full_charge_time_s = float(vehicle.get("full_charge_time_s", 0.0))
    remaining_energy = battery_capacity
    current_time = float(instance["working_start_s"])

    for start, end in zip(route[:-1], route[1:]):
        distance = float(distance_matrix_km[int(start), int(end)])
        if not np.isfinite(distance) or distance < -1e-9:
            return False, "distance"

        energy_needed = distance * consumption
        if energy_needed > battery_capacity + 1e-6:
            return False, "single_leg_battery"
        if energy_needed > remaining_energy + 1e-6:
            return False, "battery"
        remaining_energy -= energy_needed

        current_time += distance / speed * 3600.0
        if 1 <= int(end) <= num_customers:
            tw_start, tw_end = tw_s[int(end) - 1]
            if current_time < float(tw_start):
                current_time = float(tw_start)
            if current_time > float(tw_end) + 1e-6:
                return False, "time_window"
            current_time += float(service_time_s[int(end) - 1])
        elif int(end) > num_customers:
            current_time += full_charge_time_s
            remaining_energy = battery_capacity

    if current_time > float(instance["working_end_s"]) + 1e-6:
        return False, "working_horizon"
    return True, "ok"


def best_customer_insertion(
    routes: list[list[int]],
    customer: int,
    instance: dict[str, Any],
) -> tuple[float, int, list[int]] | None:
    distance_matrix_km = np.asarray(instance["distance_matrix_km"], dtype=float)
    best: tuple[float, int, list[int]] | None = None
    for route_idx, route in enumerate(routes):
        base_distance = route_distance_km(route, distance_matrix_km)
        for pos in range(1, len(route)):
            candidate = route[:pos] + [int(customer)] + route[pos:]
            feasible, _ = check_route_feasible(candidate, instance)
            if not feasible:
                continue
            delta = route_distance_km(candidate, distance_matrix_km) - base_distance
            if best is None or delta < best[0]:
                best = (float(delta), route_idx, candidate)
    return best


def try_remove_route(
    routes: list[list[int]],
    donor_idx: int,
    instance: dict[str, Any],
    distance_tolerance_km: float,
) -> tuple[bool, list[list[int]], float]:
    num_customers = int(np.asarray(instance["customers"]).shape[0])
    distance_matrix_km = np.asarray(instance["distance_matrix_km"], dtype=float)
    old_total = total_distance_km(routes, distance_matrix_km)
    donor_customers = route_customers(routes[donor_idx], num_customers)
    remaining_routes = [list(route) for idx, route in enumerate(routes) if idx != donor_idx]

    if not donor_customers:
        new_total = total_distance_km(remaining_routes, distance_matrix_km)
        return True, remaining_routes, new_total - old_total

    depot_dist = distance_matrix_km[0]
    orders = [
        donor_customers,
        sorted(donor_customers, key=lambda node: float(depot_dist[node])),
        sorted(donor_customers, key=lambda node: float(depot_dist[node]), reverse=True),
    ]

    best_solution: tuple[float, list[list[int]]] | None = None
    for order in orders:
        candidate_routes = [list(route) for route in remaining_routes]
        failed = False
        for customer in order:
            insertion = best_customer_insertion(candidate_routes, customer, instance)
            if insertion is None:
                failed = True
                break
            _, route_idx, new_route = insertion
            candidate_routes[route_idx] = new_route
        if failed:
            continue
        new_total = total_distance_km(candidate_routes, distance_matrix_km)
        delta = new_total - old_total
        if delta <= distance_tolerance_km + 1e-9:
            if best_solution is None or delta < best_solution[0]:
                best_solution = (float(delta), candidate_routes)

    if best_solution is None:
        return False, routes, 0.0
    return True, best_solution[1], best_solution[0]


def merge_routes(
    routes: list[list[int]],
    instance: dict[str, Any],
    *,
    abs_tolerance_km: float,
    rel_tolerance: float,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    routes = [list(route) for route in routes if len(route) >= 2]
    distance_matrix_km = np.asarray(instance["distance_matrix_km"], dtype=float)
    history: list[dict[str, Any]] = []

    while len(routes) > 1:
        old_total = total_distance_km(routes, distance_matrix_km)
        tolerance = max(float(abs_tolerance_km), float(rel_tolerance) * old_total)
        best: tuple[float, int, list[list[int]]] | None = None
        for donor_idx in range(len(routes)):
            ok, candidate_routes, delta = try_remove_route(routes, donor_idx, instance, tolerance)
            if not ok:
                continue
            if best is None or delta < best[0]:
                best = (float(delta), donor_idx, candidate_routes)
        if best is None:
            break
        delta, donor_idx, routes = best
        history.append(
            {
                "removed_route_index": int(donor_idx),
                "distance_delta_km": float(delta),
                "remaining_route_count": int(len(routes)),
            }
        )

    for route in routes:
        feasible, reason = check_route_feasible(route, instance)
        if not feasible:
            raise ValueError(f"Merged route became infeasible for {instance['instance_id']}: {reason}")
    return routes, history


def write_html(output_dir: Path, rows: list[dict[str, Any]], contact_sheet: Path) -> None:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Merged route postprocess maps</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;}"
        "img{max-width:100%;border:1px solid #ddd;margin:12px 0;} table{border-collapse:collapse;}"
        "td,th{border-bottom:1px solid #ddd;padding:6px 10px;text-align:left;}</style>",
        "</head><body>",
        "<h1>Cus50/train route-count postprocess</h1>",
        "<p>Routes are greedily merged when feasibility is preserved and distance stays within the configured tie-break tolerance.</p>",
        f"<h2>Contact sheet</h2><img src='{contact_sheet.name}'>",
        "<h2>Summary</h2><table><tr><th>instance</th><th>region</th><th>routes before</th><th>routes after</th><th>distance before</th><th>distance after</th><th>delta km</th><th>map</th></tr>",
    ]
    for row in rows:
        lines.append(
            "<tr>"
            f"<td>{row['instance_id']}</td>"
            f"<td>{row['region_id']}</td>"
            f"<td>{row['route_count_before']}</td>"
            f"<td>{row['route_count_after']}</td>"
            f"<td>{float(row['distance_before_km']):.6f}</td>"
            f"<td>{float(row['distance_after_km']):.6f}</td>"
            f"<td>{float(row['distance_delta_km']):.6f}</td>"
            f"<td><a href='{row['png']}'>{row['png']}</a></td>"
            "</tr>"
        )
    lines.append("</table>")
    for row in rows:
        lines.append(f"<h2>{row['instance_id']}</h2><img src='{row['png']}'>")
    lines.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances-pkl", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-routes", type=int, default=5)
    parser.add_argument("--abs-tolerance-km", type=float, default=0.05)
    parser.add_argument("--rel-tolerance", type=float, default=0.001)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    selected_rows = _selected_summary_rows(args.summary_csv, args.min_routes)
    selected_ids = {row["instance_id"] for row in selected_rows}
    instances = _read_selected_instances(args.instances_pkl, selected_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    for row in selected_rows:
        instance = instances[row["instance_id"]]
        original_routes = _parse_routes(row["routes"])
        distance_matrix_km = np.asarray(instance["distance_matrix_km"], dtype=float)
        merged_routes, history = merge_routes(
            original_routes,
            instance,
            abs_tolerance_km=args.abs_tolerance_km,
            rel_tolerance=args.rel_tolerance,
        )
        distance_before = total_distance_km(original_routes, distance_matrix_km)
        distance_after = total_distance_km(merged_routes, distance_matrix_km)

        merged_row = dict(row)
        merged_row["routes"] = merged_routes
        merged_row["route_count"] = len(merged_routes)
        merged_row["objective_distance_km"] = distance_after

        image_path = args.output_dir / f"{instance['instance_id']}_merged_routes.png"
        plot_row = _plot_instance(instance, merged_row, image_path, args.dpi)
        image_paths.append(image_path)

        summary_rows.append(
            {
                "instance_id": instance["instance_id"],
                "region_id": instance["region_id"],
                "route_count_before": len(original_routes),
                "route_count_after": len(merged_routes),
                "distance_before_km": distance_before,
                "distance_after_km": distance_after,
                "distance_delta_km": distance_after - distance_before,
                "merge_steps": json.dumps(history),
                "merged_routes_json": json.dumps(merged_routes),
                "png": plot_row["png"],
            }
        )

    csv_path = args.output_dir / "merged_route_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id",
                "region_id",
                "route_count_before",
                "route_count_after",
                "distance_before_km",
                "distance_after_km",
                "distance_delta_km",
                "merge_steps",
                "merged_routes_json",
                "png",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    contact_sheet = args.output_dir / "contact_sheet_merged_routes.png"
    _write_contact_sheet(image_paths, contact_sheet)
    write_html(args.output_dir, summary_rows, contact_sheet)

    print(json.dumps({"output_dir": str(args.output_dir), "count": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
