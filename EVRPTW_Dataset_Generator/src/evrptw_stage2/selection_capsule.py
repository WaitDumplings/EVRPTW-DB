"""Versioned handoff of exact C3 customer activation into materialization."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


SELECTION_CAPSULE_SCHEMA = "cle_evrptw_c3_selection_capsule_v1"
_FAMILY_COLUMN = "_capsule_family_id"
SELECTED_CUSTOMER_COLUMNS = (
    "latent_service_location_id",
    "location_lon",
    "location_lat",
    "physical_edge_id",
    "directed_projection_offsets",
    "connector_length_m",
    "road_projection_node_id",
    "service_access_node_id",
    "anchor_scc_id",
    "community_id",
    "sampling_cluster_id",
    "structure_route_id",
    "activation_decile",
    "service_location_type",
    "residential_unit_band",
    "residential_units",
    "depot_running_time_s",
)


@dataclass(frozen=True)
class FamilySelectionCapsule:
    selected_customers: pd.DataFrame
    radial_baseline: pd.DataFrame
    territory_report: dict[str, Any]
    spatial_activation_metadata: dict[str, Any]


class SelectionCapsuleError(ValueError):
    """A frozen C3 handoff is missing, corrupt, or bound to another family."""

    retryable = False


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def capsule_paths(base: str | Path) -> dict[str, Path]:
    base_path = Path(base)
    return {
        # ``base`` is a task identifier, not a filename with an extension.
        # Parallel C3 task IDs deliberately contain ``.part-XXXX``; using
        # Path.with_suffix() here would strip that partition and make every
        # task for a city overwrite the same three files.
        "metadata": Path(f"{base_path}.metadata.json"),
        "selected_customers": Path(f"{base_path}.selected_customers.parquet"),
        "radial_baseline": Path(f"{base_path}.radial_baseline.parquet"),
    }


def write_task_selection_capsule(
    base: str | Path,
    capsules: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically persist the successful C3 selections for one worker task."""

    if not capsules:
        raise SelectionCapsuleError("A C3 selection capsule task cannot be empty")
    paths = capsule_paths(base)
    metadata_families: dict[str, Any] = {}
    selected_parts: list[pd.DataFrame] = []
    baseline_parts: list[pd.DataFrame] = []
    for capsule in capsules:
        family_id = str(capsule["family_id"])
        if family_id in metadata_families:
            raise SelectionCapsuleError(f"Duplicate family in C3 selection capsule: {family_id}")
        selected_source = capsule["selected_customers"]
        missing_selected = sorted(
            set(SELECTED_CUSTOMER_COLUMNS) - set(selected_source)
        )
        if missing_selected:
            raise SelectionCapsuleError(
                f"C3 selection capsule lacks selected columns for {family_id}: "
                f"{missing_selected}"
            )
        selected = selected_source[list(SELECTED_CUSTOMER_COLUMNS)].copy()
        selected["directed_projection_offsets"] = selected[
            "directed_projection_offsets"
        ].map(str)
        baseline = capsule["radial_baseline"].copy()
        if selected.empty or baseline.empty:
            raise SelectionCapsuleError(f"Empty C3 selection capsule frame for {family_id}")
        if _FAMILY_COLUMN in selected or _FAMILY_COLUMN in baseline:
            raise SelectionCapsuleError(
                f"Reserved capsule column is already present for {family_id}"
            )
        selected.insert(0, _FAMILY_COLUMN, family_id)
        baseline.insert(0, _FAMILY_COLUMN, family_id)
        selected_parts.append(selected)
        baseline_parts.append(baseline)
        metadata_families[family_id] = {
            "binding": dict(capsule["binding"]),
            "territory_report": dict(capsule["territory_report"]),
            "spatial_activation_metadata": dict(
                capsule["spatial_activation_metadata"]
            ),
            "selected_customer_count": int(len(selected)),
            "radial_baseline_count": int(len(baseline)),
        }
    selected_frame = pd.concat(selected_parts, ignore_index=True)
    baseline_frame = pd.concat(baseline_parts, ignore_index=True)
    _atomic_parquet(paths["selected_customers"], selected_frame)
    _atomic_parquet(paths["radial_baseline"], baseline_frame)
    payload = {
        "schema": SELECTION_CAPSULE_SCHEMA,
        "family_count": len(metadata_families),
        "families": metadata_families,
        "selected_customer_row_count": int(len(selected_frame)),
        "radial_baseline_row_count": int(len(baseline_frame)),
        "hash_validation_performed": False,
    }
    _atomic_json(paths["metadata"], payload)
    return payload


def _normalise_source_ids(value: Any) -> list[str]:
    return [str(item) for item in value]


def load_family_selection_capsule(
    output_root: str | Path,
    family: Mapping[str, Any],
    *,
    selected_depot_id: str,
    selected_structure_source_ids: Sequence[str],
) -> FamilySelectionCapsule | None:
    """Load a capsule only after every frozen family binding matches exactly."""

    raw_relpath = family.get("c3_selection_capsule_relpath")
    if raw_relpath is None or (isinstance(raw_relpath, float) and pd.isna(raw_relpath)):
        return None
    relpath = Path(str(raw_relpath))
    if relpath.is_absolute() or ".." in relpath.parts:
        raise SelectionCapsuleError("C3 selection capsule path must be output-root relative")
    root = Path(output_root).resolve()
    base = (root / relpath).resolve()
    if root != base and root not in base.parents:
        raise SelectionCapsuleError("C3 selection capsule escapes the instance output root")
    paths = capsule_paths(base)
    try:
        payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except Exception as error:  # Frozen handoffs never enter seed retry logic.
        raise SelectionCapsuleError(
            f"Cannot read C3 selection capsule metadata: {paths['metadata']}"
        ) from error
    planned_schema = family.get("c3_selection_capsule_schema")
    if planned_schema != SELECTION_CAPSULE_SCHEMA:
        raise SelectionCapsuleError("Plan does not bind the supported C3 selection capsule schema")
    if payload.get("schema") != SELECTION_CAPSULE_SCHEMA:
        raise SelectionCapsuleError("Unsupported C3 selection capsule schema")
    family_id = str(family["family_id"])
    entry = dict(payload.get("families", {}).get(family_id) or {})
    if not entry:
        raise SelectionCapsuleError(f"C3 selection capsule lacks family {family_id}")
    binding = dict(entry.get("binding") or {})
    expected = {
        "family_id": family_id,
        "city_slug": str(family["city_slug"]),
        "day_type": str(family["day_type"]),
        "parent_customer_count": int(family["parent_customer_count"]),
        "customer_superset_seed": int(family["customer_superset_seed"]),
        "road_state_seed": int(family["road_state_seed"]),
        "selected_depot_id": str(selected_depot_id),
        "selected_structure_source_ids": _normalise_source_ids(
            selected_structure_source_ids
        ),
        "joint_support_contract_id": str(family["joint_support_contract_id"]),
        "capacity_contract_fingerprint": str(
            family["capacity_contract_fingerprint"]
        ),
    }
    observed = {
        **binding,
        "parent_customer_count": int(binding.get("parent_customer_count", -1)),
        "customer_superset_seed": int(binding.get("customer_superset_seed", -1)),
        "road_state_seed": int(binding.get("road_state_seed", -1)),
        "selected_structure_source_ids": _normalise_source_ids(
            binding.get("selected_structure_source_ids", [])
        ),
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise SelectionCapsuleError(
            "C3 selection capsule binding mismatch: "
            + json.dumps(mismatches, sort_keys=True, separators=(",", ":"))
        )
    try:
        selected_all = pd.read_parquet(paths["selected_customers"])
        baseline_all = pd.read_parquet(paths["radial_baseline"])
    except Exception as error:  # Frozen handoffs never enter seed retry logic.
        raise SelectionCapsuleError(
            f"Cannot read C3 selection capsule tables under {base}"
        ) from error
    selected = selected_all.loc[
        selected_all[_FAMILY_COLUMN].astype(str).eq(family_id)
    ].drop(columns=_FAMILY_COLUMN).reset_index(drop=True)
    baseline = baseline_all.loc[
        baseline_all[_FAMILY_COLUMN].astype(str).eq(family_id)
    ].drop(columns=_FAMILY_COLUMN).reset_index(drop=True)
    customer_count = int(family["parent_customer_count"])
    if len(selected) != customer_count or len(baseline) != customer_count:
        raise SelectionCapsuleError(
            f"C3 selection capsule count mismatch for {family_id}: "
            f"selected={len(selected)}, baseline={len(baseline)}, expected={customer_count}"
        )
    if selected["latent_service_location_id"].astype(str).duplicated().any():
        raise SelectionCapsuleError(f"C3 selection capsule has duplicate customers for {family_id}")
    required_selected = set(SELECTED_CUSTOMER_COLUMNS)
    missing = sorted(required_selected - set(selected.columns))
    if missing:
        raise SelectionCapsuleError(f"C3 selection capsule lacks selected columns: {missing}")
    required_baseline = {
        "latent_service_location_id",
        "community_id",
        "radial_decile",
        "depot_running_time_s",
    }
    missing_baseline = sorted(required_baseline - set(baseline.columns))
    if missing_baseline:
        raise SelectionCapsuleError(
            f"C3 selection capsule lacks baseline columns: {missing_baseline}"
        )
    return FamilySelectionCapsule(
        selected_customers=selected,
        radial_baseline=baseline,
        territory_report=dict(entry["territory_report"]),
        spatial_activation_metadata=dict(entry["spatial_activation_metadata"]),
    )
