"""Strict reader for self-contained portable CLE packages.

The reader is the only supported Stage-1 -> Stage-2 boundary.  It deliberately
distinguishes an official release input from a non-release engineering pilot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow.parquet as pq

RunMode = Literal["official", "non_release_pilot"]


class CLEEligibilityError(RuntimeError):
    """Raised when a CLE cannot be consumed under the requested run mode."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_inside(root: Path, relative: str, label: str) -> Path:
    declared = Path(relative)
    if declared.is_absolute():
        raise CLEEligibilityError(f"{label} uses an absolute path: {declared}")
    resolved_root = root.resolve()
    resolved = (root / declared).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise CLEEligibilityError(f"{label} escapes the CLE root: {declared}")
    if not resolved.is_file():
        raise CLEEligibilityError(f"{label} is missing: {resolved}")
    return resolved


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _true_count(path: Path, column: str) -> int:
    values = pd.read_parquet(path, columns=[column])[column]
    return int(values.fillna(False).astype(bool).sum())


def _read_eligible_parquet(
    path: Path,
    eligibility_field: str,
    columns: list[str] | None,
) -> pd.DataFrame:
    """Read only eligible rows without forcing callers to request the gate field."""

    requested = None if columns is None else list(dict.fromkeys(columns))
    read_columns = None if requested is None else list(requested)
    if read_columns is not None and eligibility_field not in read_columns:
        read_columns.append(eligibility_field)
    frame = pd.read_parquet(path, columns=read_columns)
    frame = frame.loc[frame[eligibility_field].fillna(False).astype(bool)].copy()
    if requested is not None:
        frame = frame.loc[:, requested]
    return frame


@dataclass(frozen=True)
class PortableCLE:
    root: Path
    city_slug: str
    mode: RunMode
    manifest: dict[str, Any]
    graph_path: Path
    service_locations_path: Path
    depots_path: Path
    chargers_path: Path
    speeds_path: Path
    customer_eligibility_field: str
    depot_eligibility_field: str
    charger_eligibility_field: str
    warnings: tuple[str, ...]

    @property
    def non_release_pilot(self) -> bool:
        return self.mode == "non_release_pilot"

    @property
    def release_blockers(self) -> tuple[str, ...]:
        return tuple(map(str, self.manifest.get("release_blockers", [])))

    def read_service_locations(self, columns: list[str] | None = None) -> pd.DataFrame:
        return _read_eligible_parquet(
            self.service_locations_path, self.customer_eligibility_field, columns
        )

    def read_depots(self, columns: list[str] | None = None) -> pd.DataFrame:
        return _read_eligible_parquet(self.depots_path, self.depot_eligibility_field, columns)

    def read_chargers(self, columns: list[str] | None = None) -> pd.DataFrame:
        return _read_eligible_parquet(
            self.chargers_path, self.charger_eligibility_field, columns
        )

    def eligibility_summary(self) -> dict[str, Any]:
        return {
            "city_slug": self.city_slug,
            "mode": self.mode,
            "non_release_pilot": self.non_release_pilot,
            "cle_release_eligible": bool(self.manifest.get("release_eligible", False)),
            "release_blockers": list(self.release_blockers),
            "customer_eligibility_field": self.customer_eligibility_field,
            "eligible_customers": _true_count(
                self.service_locations_path, self.customer_eligibility_field
            ),
            "depot_eligibility_field": self.depot_eligibility_field,
            "eligible_depots": _true_count(self.depots_path, self.depot_eligibility_field),
            "charger_eligibility_field": self.charger_eligibility_field,
            "eligible_chargers": _true_count(
                self.chargers_path, self.charger_eligibility_field
            ),
            "warnings": list(self.warnings),
        }


def load_portable_cle(
    cle_root: str | Path,
    city_slug: str,
    *,
    mode: RunMode = "official",
    minimum_customers: int = 2_000,
    minimum_depots: int = 1,
    minimum_chargers: int = 50,
) -> PortableCLE:
    if mode not in {"official", "non_release_pilot"}:
        raise ValueError(f"Unsupported Stage-2 run mode: {mode!r}")
    root = Path(cle_root) / "cities" / city_slug
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise CLEEligibilityError(f"Portable CLE manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "evrptw_city_logistics_environment_v1":
        raise CLEEligibilityError(
            f"Unsupported CLE schema for {city_slug}: {manifest.get('schema')!r}"
        )
    if str(manifest.get("city_slug")) != city_slug:
        raise CLEEligibilityError(
            f"CLE city_slug mismatch: requested={city_slug!r}, manifest={manifest.get('city_slug')!r}"
        )
    if not bool(manifest.get("portable_package_verified", False)):
        raise CLEEligibilityError(f"CLE {city_slug} is not a verified portable package")
    if mode == "official" and not bool(manifest.get("release_eligible", False)):
        blockers = ", ".join(map(str, manifest.get("release_blockers", []))) or "unknown"
        raise CLEEligibilityError(
            f"CLE {city_slug} is not release eligible; blockers: {blockers}"
        )

    outputs = manifest.get("outputs", {})
    paths = {
        "graph": _resolve_inside(root, str(outputs["operational_graph"]), "operational_graph"),
        "customers": _resolve_inside(root, str(outputs["latent_locations"]), "latent_locations"),
        "depots": _resolve_inside(root, str(outputs["depots"]), "depots"),
        "chargers": _resolve_inside(root, str(outputs["chargers"]), "chargers"),
        "speeds": _resolve_inside(root, str(outputs["directed_legal_speeds"]), "directed_legal_speeds"),
    }

    official_fields = (
        "customer_release_eligible",
        "depot_release_eligible",
        "charger_release_eligible",
    )
    pilot_fields = (
        "cle_default_instance_eligible",
        "depot_candidate_eligible",
        "charger_candidate_eligible",
    )
    customer_field, depot_field, charger_field = (
        official_fields if mode == "official" else pilot_fields
    )
    required = {
        paths["customers"]: {"latent_service_location_id", customer_field},
        paths["depots"]: {"candidate_id", depot_field},
        paths["chargers"]: {"charger_id", charger_field},
        paths["speeds"]: {
            "edge_u",
            "edge_v",
            "edge_key",
            "length_m",
            "legal_speed_kph",
            "reference_speed_kph",
            "operating_mode",
        },
    }
    for path, expected_columns in required.items():
        missing = expected_columns - _parquet_columns(path)
        if missing:
            raise CLEEligibilityError(f"{path} is missing required columns: {sorted(missing)}")

    counts = {
        "customers": _true_count(paths["customers"], customer_field),
        "depots": _true_count(paths["depots"], depot_field),
        "chargers": _true_count(paths["chargers"], charger_field),
    }
    minima = {
        "customers": int(minimum_customers),
        "depots": int(minimum_depots),
        "chargers": int(minimum_chargers),
    }
    insufficient = [
        f"{name}={counts[name]} < {minimum}"
        for name, minimum in minima.items()
        if counts[name] < minimum
    ]
    if insufficient:
        raise CLEEligibilityError(
            f"CLE {city_slug} cannot support the requested V1 scales: " + "; ".join(insufficient)
        )

    warnings: list[str] = []
    if mode == "non_release_pilot":
        warnings.append(
            "Non-release pilot mode uses candidate/default eligibility fields and must not "
            "produce official train/val/test artifacts."
        )
        if manifest.get("release_blockers"):
            warnings.append(
                "Open CLE release blockers: "
                + ", ".join(map(str, manifest["release_blockers"]))
            )
    return PortableCLE(
        root=root,
        city_slug=city_slug,
        mode=mode,
        manifest=manifest,
        graph_path=paths["graph"],
        service_locations_path=paths["customers"],
        depots_path=paths["depots"],
        chargers_path=paths["chargers"],
        speeds_path=paths["speeds"],
        customer_eligibility_field=customer_field,
        depot_eligibility_field=depot_field,
        charger_eligibility_field=charger_field,
        warnings=tuple(warnings),
    )
