#!/usr/bin/env python3
"""Report 0.85/0.90/0.95 charging derating sensitivity for a Stage-2 pilot."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evrptw_stage2.orders import _certificates_for_view
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.provenance import resolve_git_provenance


FACTORS = (0.85, 0.90, 0.95)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    code_provenance = resolve_git_provenance(
        Path(__file__).resolve().parents[2],
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )

    canonical = load_reference_profile(args.profile)
    rows: list[dict[str, Any]] = []
    family_paths = sorted(
        (args.instance_root / "materialized" / "families").glob("*/family_manifest.json")
    )
    if not family_paths:
        raise FileNotFoundError("No materialized pilot families were found")
    for family_path in family_paths:
        family_dir = family_path.parent
        family = json.loads(family_path.read_text(encoding="utf-8"))
        terminal_index = pd.read_parquet(family_dir / family["terminal_index"])
        time_matrix = np.load(
            family_dir / family["matrix_files"]["running_time_shortest_matrix_s"],
            mmap_mode="r",
            allow_pickle=False,
        )
        distance_matrix = np.load(
            family_dir / family["matrix_files"]["running_time_path_distance_km"],
            mmap_mode="r",
            allow_pickle=False,
        )
        for view_path in sorted((family_dir / "views").glob("*/view_manifest.json")):
            view = json.loads(view_path.read_text(encoding="utf-8"))
            view_dir = view_path.parent
            indices = np.load(
                view_dir / view["terminal_parent_indices"], allow_pickle=False
            ).astype(int)
            customer_count = int(view["customer_count"])
            charger_parent_indices = indices[1 + customer_count :]
            charging_power = pd.to_numeric(
                terminal_index.iloc[charger_parent_indices]["effective_charging_power_kw"],
                errors="raise",
            ).to_numpy(dtype=float)
            attributes = np.load(
                view_dir / view["customer_attributes"], allow_pickle=False
            )
            windows = attributes["time_windows_s"].astype(float)
            service = attributes["service_time_s"].astype(float)
            start_s, end_s = map(float, view["operating_horizon_s"])
            for factor in FACTORS:
                profile = copy.deepcopy(canonical)
                profile["charging"]["charging_power_derating_factor"] = factor
                certificates = _certificates_for_view(
                    customer_count=customer_count,
                    running_time_matrix_s=np.asarray(
                        time_matrix[np.ix_(indices, indices)], dtype=float
                    ),
                    running_time_path_distance_matrix_km=np.asarray(
                        distance_matrix[np.ix_(indices, indices)], dtype=float
                    ),
                    charging_power_kw=charging_power,
                    profile=profile,
                )
                arrival = start_s + certificates.arrival_elapsed_s.astype(float)
                service_start = np.maximum(arrival, windows[:, 0])
                energy_feasible = np.isfinite(arrival) & np.isfinite(
                    certificates.return_duration_s
                )
                time_feasible = (
                    service_start <= windows[:, 1] + 1e-6
                ) & (
                    service_start
                    + service
                    + certificates.return_duration_s.astype(float)
                    <= end_s + 1e-6
                )
                rows.append(
                    {
                        "family_id": str(family["family_id"]),
                        "view_id": str(view["view_id"]),
                        "city_slug": str(view["city_slug"]),
                        "scale_id": str(view["scale_id"]),
                        "day_type": str(view["day_type"]),
                        "charging_power_derating_factor": factor,
                        "canonical_factor": factor == 0.90,
                        "customer_count": customer_count,
                        "energy_feasible_count": int(energy_feasible.sum()),
                        "time_feasible_count": int(time_feasible.sum()),
                        "all_certificates_pass": bool(
                            (energy_feasible & time_feasible).all()
                        ),
                        "requires_charging_count": int(
                            certificates.requires_charging.sum()
                        ),
                        "maximum_charging_visit_count": int(
                            certificates.charging_visit_count.max()
                        ),
                    }
                )
    frame = pd.DataFrame.from_records(rows).sort_values(
        ["charging_power_derating_factor", "city_slug", "view_id"],
        kind="stable",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "charging_sensitivity_views.parquet", index=False)
    summary_rows = []
    for factor, group in frame.groupby("charging_power_derating_factor", sort=True):
        summary_rows.append(
            {
                "charging_power_derating_factor": float(factor),
                "canonical_factor": float(factor) == 0.90,
                "view_count": len(group),
                "passing_view_count": int(group["all_certificates_pass"].sum()),
                "passing_view_share": float(group["all_certificates_pass"].mean()),
                "customer_count": int(group["customer_count"].sum()),
                "energy_feasible_customer_count": int(
                    group["energy_feasible_count"].sum()
                ),
                "time_feasible_customer_count": int(group["time_feasible_count"].sum()),
                "requires_charging_customer_count": int(
                    group["requires_charging_count"].sum()
                ),
            }
        )
    summary = {
        "schema": "evrptw_charging_derating_sensitivity_v1",
        "code_provenance": code_provenance,
        "role": "pilot_report_only_canonical_artifacts_remain_0.90",
        "factors": list(FACTORS),
        "rows": summary_rows,
        "view_ledger": "charging_sensitivity_views.parquet",
    }
    _write_json(args.output_dir / "charging_sensitivity_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
