"""Complete-community construction and deterministic customer partitioning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .reader import PortableCLE


def _stable_fraction(seed: int, namespace: str, value: str) -> float:
    token = f"{seed}|{namespace}|{value}".encode()
    integer = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "big")
    return integer / float(2**64)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _customer_points(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.crs is None:
        raise ValueError("CLE service locations must declare a CRS")
    points = frame.copy()
    geometry_types = set(points.geometry.geom_type.dropna().unique())
    if geometry_types - {"Point", "MultiPoint"}:
        metric = points.to_crs(points.estimate_utm_crs())
        metric.geometry = metric.geometry.representative_point()
        points = metric.to_crs("EPSG:4326")
    else:
        points = points.to_crs("EPSG:4326")
    return points


def _group_features(locations: pd.DataFrame) -> pd.DataFrame:
    types = sorted(map(str, locations["service_location_type"].dropna().unique()))
    bands = sorted(map(str, locations["residential_unit_band"].dropna().unique()))
    records: list[dict[str, Any]] = []
    for community_id, group in locations.groupby("community_id", sort=True):
        record: dict[str, Any] = {
            "community_id": str(community_id),
            "census_block_group_geoid": str(group["census_block_group_geoid"].iloc[0]),
            "road_connectivity_subgroup": str(group["road_connectivity_subgroup"].iloc[0]),
            "location_count": len(group),
            "residential_units": float(
                pd.to_numeric(group["residential_units"], errors="coerce").fillna(0).sum()
            ),
            "centroid_lon": float(group["location_lon"].mean()),
            "centroid_lat": float(group["location_lat"].mean()),
        }
        type_counts = group["service_location_type"].astype(str).value_counts()
        band_counts = group["residential_unit_band"].astype(str).value_counts()
        for value in types:
            record[f"type__{value}"] = int(type_counts.get(value, 0))
        for value in bands:
            record[f"band__{value}"] = int(band_counts.get(value, 0))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _normalized_error(current: np.ndarray, target: np.ndarray) -> float:
    denominator = np.maximum(target, 1.0)
    return float(np.mean(np.abs(current - target) / denominator))


def _assign_group_split(
    communities: pd.DataFrame,
    *,
    heldout_fraction: float,
    seed: int,
    city_slug: str,
) -> pd.DataFrame:
    feature_columns = [
        column
        for column in communities.columns
        if column == "location_count" or column.startswith(("type__", "band__"))
    ]
    matrix = communities[feature_columns].to_numpy(dtype=float)
    target = matrix.sum(axis=0) * float(heldout_fraction)
    location_position = feature_columns.index("location_count")
    target_locations = float(target[location_position])
    total_locations = float(matrix[:, location_position].sum())
    candidates: list[dict[str, Any]] = []
    # A randomized complete-group fill keeps location count as the hard primary
    # objective. Multiple deterministic restarts then let type/unit balance act
    # only among candidates already close to the requested 80/20 ratio.
    for restart in range(256):
        priorities = np.asarray(
            [
                _stable_fraction(
                    seed,
                    f"{city_slug}:community-split:{restart}",
                    str(value),
                )
                for value in communities["community_id"]
            ]
        )
        order = np.argsort(priorities, kind="stable")
        selected_restart: set[int] = set()
        current = np.zeros(len(feature_columns), dtype=float)
        for index_value in order:
            index = int(index_value)
            current_error = abs(current[location_position] - target_locations)
            proposed_error = abs(
                current[location_position] + matrix[index, location_position] - target_locations
            )
            if proposed_error < current_error:
                selected_restart.add(index)
                current += matrix[index]
        count_error = abs(current[location_position] - target_locations) / max(
            total_locations, 1.0
        )
        secondary = np.delete(current, location_position)
        secondary_target = np.delete(target, location_position)
        feature_error = _normalized_error(secondary, secondary_target)
        candidates.append(
            {
                "restart": restart,
                "selected": selected_restart,
                "current": current,
                "priorities": priorities,
                "count_error": count_error,
                "feature_error": feature_error,
            }
        )

    minimum_count_error = min(float(item["count_error"]) for item in candidates)
    count_tolerance = max(0.002, minimum_count_error + 0.0005)
    close_candidates = [
        item for item in candidates if float(item["count_error"]) <= count_tolerance
    ]
    chosen = min(
        close_candidates,
        key=lambda item: (
            float(item["feature_error"]),
            float(item["count_error"]),
            int(item["restart"]),
        ),
    )
    selected = set(chosen["selected"])
    priorities = np.asarray(chosen["priorities"])
    if not selected and len(communities):
        selected.add(int(np.argmin(priorities)))

    result = communities.copy()
    result["customer_pool"] = [
        "heldout" if index in selected else "train" for index in range(len(result))
    ]
    result["training_ineligible"] = result["customer_pool"].eq("heldout")
    result["split_priority"] = priorities
    result["split_restart"] = int(chosen["restart"])
    return result


def build_customer_split(
    cle: PortableCLE,
    *,
    block_groups_path: str | Path,
    output_dir: str | Path,
    split_seed: int,
    heldout_fraction: float = 0.2,
    partition_version: str = "census_block_group_road_scc_80_20_v1",
) -> dict[str, Any]:
    """Build a deterministic complete-community 80/20 location ledger.

    The road-connectivity subgroup is the directed SCC inherited by the
    customer road anchor.  Current CLE packages place default candidates in the
    reference SCC; keeping the field explicit preserves the contract if a later
    portable CLE retains more than one eligible SCC.
    """

    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    block_path = Path(block_groups_path)
    if not block_path.is_file():
        raise FileNotFoundError(f"Census block-group source is missing: {block_path}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "customer_split_manifest.parquet"
    community_path = out / "community_manifest.parquet"
    report_path = out / "customer_split_report.json"
    for path in (ledger_path, community_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen split artifact: {path}")

    locations = gpd.read_parquet(cle.service_locations_path)
    locations = locations.loc[
        locations[cle.customer_eligibility_field].fillna(False).astype(bool)
    ].copy()
    required_location_columns = {
        "latent_service_location_id",
        "service_location_type",
        "residential_unit_band",
        "residential_units",
        "location_lon",
        "location_lat",
        "anchor_scc_id",
        "geometry",
    }
    missing = required_location_columns - set(locations.columns)
    if missing:
        raise ValueError(f"CLE customer layer is missing split columns: {sorted(missing)}")
    points = _customer_points(locations)

    block_groups = gpd.read_file(block_path).to_crs("EPSG:4326")
    geoid_field = next(
        (field for field in ("GEOID", "GEOIDFQ", "geoid", "GEOID20") if field in block_groups),
        None,
    )
    if geoid_field is None:
        raise ValueError("Census block-group source has no recognized GEOID field")
    block_groups = block_groups[[geoid_field, "geometry"]].rename(
        columns={geoid_field: "census_block_group_geoid"}
    )
    block_groups["census_block_group_geoid"] = block_groups[
        "census_block_group_geoid"
    ].astype(str)

    joined = gpd.sjoin(points, block_groups, how="left", predicate="within")
    if joined.index.duplicated().any():
        duplicated = int(joined.index.duplicated(keep=False).sum())
        raise ValueError(f"Block-group join produced {duplicated} duplicate location matches")
    unmatched = joined["census_block_group_geoid"].isna()
    if unmatched.any():
        raise ValueError(
            f"{int(unmatched.sum())} eligible customer locations did not match a Census block group"
        )
    joined["road_connectivity_subgroup"] = joined["anchor_scc_id"].astype(str)
    joined["community_id"] = (
        cle.city_slug
        + ":bg:"
        + joined["census_block_group_geoid"].astype(str)
        + ":scc:"
        + joined["road_connectivity_subgroup"]
    )

    communities = _group_features(joined)
    communities = _assign_group_split(
        communities,
        heldout_fraction=heldout_fraction,
        seed=int(split_seed),
        city_slug=cle.city_slug,
    )
    assignments = communities.set_index("community_id")[["customer_pool", "training_ineligible"]]
    joined = joined.join(assignments, on="community_id", validate="many_to_one")
    ledger_columns = [
        "latent_service_location_id",
        "census_block_group_geoid",
        "road_connectivity_subgroup",
        "community_id",
        "customer_pool",
        "training_ineligible",
        "service_location_type",
        "residential_unit_band",
        "residential_units",
        "location_lon",
        "location_lat",
    ]
    ledger = pd.DataFrame(joined[ledger_columns]).copy()
    ledger.insert(0, "city_slug", cle.city_slug)
    ledger["partition_version"] = partition_version
    ledger["split_seed"] = int(split_seed)
    ledger["source_cle_mode"] = cle.mode
    ledger["non_release_pilot"] = cle.non_release_pilot
    ledger.to_parquet(ledger_path, index=False)
    communities.insert(0, "city_slug", cle.city_slug)
    communities["partition_version"] = partition_version
    communities["split_seed"] = int(split_seed)
    communities["non_release_pilot"] = cle.non_release_pilot
    communities.to_parquet(community_path, index=False)

    pool_counts = ledger["customer_pool"].value_counts().to_dict()
    type_distribution = (
        ledger.groupby(["customer_pool", "service_location_type"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    report = {
        "schema": "cle_evrptw_customer_split_report_v1",
        "city_slug": cle.city_slug,
        "partition_version": partition_version,
        "split_seed": int(split_seed),
        "target_heldout_fraction": float(heldout_fraction),
        "non_release_pilot": cle.non_release_pilot,
        "source_cle_release_eligible": bool(cle.manifest.get("release_eligible", False)),
        "source_customer_eligibility_field": cle.customer_eligibility_field,
        "eligible_location_count": len(ledger),
        "community_count": len(communities),
        "train_location_count": int(pool_counts.get("train", 0)),
        "heldout_location_count": int(pool_counts.get("heldout", 0)),
        "actual_heldout_fraction": float(pool_counts.get("heldout", 0) / len(ledger)),
        "absolute_heldout_fraction_error": float(
            abs(pool_counts.get("heldout", 0) / len(ledger) - heldout_fraction)
        ),
        "assignment_method": "deterministic_randomized_complete_group_balance_v1",
        "assignment_restart": int(communities["split_restart"].iloc[0]),
        "pool_type_counts": {
            str(pool): {str(key): int(value) for key, value in row.items()}
            for pool, row in type_distribution.to_dict(orient="index").items()
        },
        "outputs": {
            "customer_split_manifest": ledger_path.name,
            "community_manifest": community_path.name,
        },
        "warnings": list(cle.warnings),
    }
    _write_json(report_path, report)
    return report
