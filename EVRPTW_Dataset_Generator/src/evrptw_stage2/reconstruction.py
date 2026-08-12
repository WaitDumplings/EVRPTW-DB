"""Portable, deterministic reconstruction of Stage-2 routing matrices.

The release representation of an instance is the CLE plus the lightweight
family/view artifacts.  The four dense parent matrices are a cache: they can
be regenerated from the frozen CLE graph and speed layer, the stored
family-level road-state factors, and the terminal edge projections.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .profile import load_reference_profile
from .reader import PortableCLE, load_portable_cle
from .road_state import build_family_road_state
from .routing import PhysicalRoadNetwork, RoutingMatrices

MATRIX_NAMES = (
    "distance_matrix_km",
    "distance_path_travel_time_s",
    "running_time_shortest_matrix_s",
    "running_time_path_distance_km",
)
REQUIRED_TERMINAL_RECONSTRUCTION_COLUMNS = frozenset(
    {
        "terminal_index",
        "terminal_kind",
        "source_id",
        "physical_edge_id",
        "directed_projection_offsets",
        "connector_length_m",
        "road_projection_node_id",
        "access_node_id",
    }
)
ValidationMode = Literal["exact", "allclose", "none"]


class ReconstructionError(RuntimeError):
    """Raised when a slim artifact cannot be reconstructed exactly."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_contract(path: Path) -> dict[str, Any]:
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "relative_path": str(path.name),
        "npy_sha256": sha256_file(path),
        "file_bytes": int(path.stat().st_size),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
    }


def _matrix_payload(matrices: RoutingMatrices) -> dict[str, np.ndarray]:
    return {
        "distance_matrix_km": matrices.distance_matrix_km,
        "distance_path_travel_time_s": matrices.distance_path_travel_time_s,
        "running_time_shortest_matrix_s": matrices.running_time_shortest_matrix_s,
        "running_time_path_distance_km": matrices.running_time_path_distance_km,
    }


def _family_dirs(dataset_root: Path) -> list[Path]:
    root = dataset_root / "materialized" / "families"
    if not root.is_dir():
        raise FileNotFoundError(f"Stage-2 family root is missing: {root}")
    families = sorted(path for path in root.iterdir() if (path / "family_manifest.json").is_file())
    if not families:
        raise FileNotFoundError(f"No materialized families were found under {root}")
    return families


def _build_instance_registry(dataset_root: Path, family_dirs: Sequence[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    plan_root = dataset_root / "generation_plan"
    if plan_root.is_dir():
        for path in sorted(plan_root.rglob("view_index.parquet")):
            frame = pd.read_parquet(path)
            if {"view_id", "family_id"} <= set(frame.columns):
                frame = frame.copy()
                frame["source_view_index"] = str(path.relative_to(dataset_root))
                parts.append(frame)
    if parts:
        registry = pd.concat(parts, ignore_index=True)
        duplicated = registry.loc[registry["view_id"].astype(str).duplicated(keep=False)]
        if not duplicated.empty:
            conflicts = duplicated.groupby("view_id")["family_id"].nunique()
            conflicts = conflicts.loc[conflicts.gt(1)]
            if not conflicts.empty:
                raise ReconstructionError(
                    "A view ID maps to multiple families: " + ", ".join(map(str, conflicts.index))
                )
        return registry.drop_duplicates("view_id").sort_values("view_id").reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for family_dir in family_dirs:
        manifest = _read_json(family_dir / "family_manifest.json")
        for view_id in manifest.get("view_ids", []):
            view_path = family_dir / "views" / str(view_id) / "view_manifest.json"
            view = _read_json(view_path)
            records.append(
                {
                    "view_id": str(view_id),
                    "family_id": str(manifest["family_id"]),
                    "city_slug": str(manifest["city_slug"]),
                    "scale_id": str(view.get("scale_id", "unknown")),
                    "split_id": str(view.get("split_id", "unknown")),
                    "track_id": str(view.get("track_id", "unknown")),
                }
            )
    return pd.DataFrame.from_records(records).sort_values("view_id").reset_index(drop=True)


def _cle_artifact_contract(cle_root: Path, city_slug: str) -> dict[str, Any]:
    city_root = cle_root / "cities" / city_slug
    manifest_path = city_root / "manifest.json"
    manifest = _read_json(manifest_path)
    outputs = manifest["outputs"]
    artifacts: dict[str, dict[str, Any]] = {}
    for label, output_key in (
        ("operational_graph", "operational_graph"),
        ("directed_legal_speeds", "directed_legal_speeds"),
    ):
        relative = Path(str(outputs[output_key]))
        path = city_root / relative
        artifacts[label] = {
            "relative_path": str(Path("cities") / city_slug / relative),
            "sha256": sha256_file(path),
            "file_bytes": int(path.stat().st_size),
        }
    return {
        "city_slug": city_slug,
        "cle_schema": str(manifest.get("schema")),
        "artifacts": artifacts,
    }


def _copy_without_parent_matrices(source_root: Path, output_root: Path) -> None:
    def ignore(path: str, names: list[str]) -> set[str]:
        directory = Path(path)
        ignored: set[str] = set()
        if "matrices" in names and (directory / "family_manifest.json").is_file():
            ignored.add("matrices")
        if directory == source_root and "_reconstruction" in names:
            ignored.add("_reconstruction")
        return ignored

    shutil.copytree(source_root, output_root, ignore=ignore)


def export_slim_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    cle_root: str | Path,
    profile_path: str | Path,
) -> dict[str, Any]:
    """Copy a full Stage-2 tree without matrices and record exact cache checksums."""

    started = time.perf_counter()
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    cle = Path(cle_root).resolve()
    profile_file = Path(profile_path).resolve()
    if output == source or source in output.parents:
        raise ValueError("Slim output must not be inside the source instance tree")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite slim dataset output: {output}")
    families = _family_dirs(source)
    profile = load_reference_profile(profile_file)

    matrix_contracts: dict[str, dict[str, Any]] = {}
    source_matrix_bytes = 0
    for family_dir in families:
        manifest = _read_json(family_dir / "family_manifest.json")
        if str(manifest.get("reference_profile_id")) != str(profile["profile_id"]):
            raise ReconstructionError(
                f"{manifest['family_id']} uses profile {manifest.get('reference_profile_id')!r}, "
                f"not the supplied {profile['profile_id']!r}"
            )
        missing_terminal_columns = REQUIRED_TERMINAL_RECONSTRUCTION_COLUMNS - set(
            pd.read_parquet(family_dir / manifest["terminal_index"]).columns
        )
        if missing_terminal_columns:
            raise ReconstructionError(
                f"{manifest['family_id']} cannot be reconstructed; terminal_index is missing "
                f"{sorted(missing_terminal_columns)}"
            )
        factors = manifest.get("road_state_report", {}).get(
            "moves_road_type_baseline_factors"
        )
        if not factors:
            raise ReconstructionError(
                f"{manifest['family_id']} lacks stored MOVES road-type baseline factors"
            )
        contracts: dict[str, Any] = {}
        for name in MATRIX_NAMES:
            if name not in manifest.get("matrix_files", {}):
                raise ReconstructionError(f"{manifest['family_id']} lacks matrix file {name}")
            matrix_path = family_dir / manifest["matrix_files"][name]
            if not matrix_path.is_file():
                raise FileNotFoundError(matrix_path)
            contracts[name] = _matrix_contract(matrix_path)
            source_matrix_bytes += int(matrix_path.stat().st_size)
        matrix_contracts[str(manifest["family_id"])] = contracts

    _copy_without_parent_matrices(source, output)
    reconstruction_root = output / "_reconstruction"
    reconstruction_root.mkdir(parents=True)
    shutil.copy2(profile_file, reconstruction_root / "reference_profile.json")
    registry = _build_instance_registry(source, families)
    registry.to_parquet(reconstruction_root / "instance_registry.parquet", index=False)

    cities = sorted(
        {
            str(_read_json(family_dir / "family_manifest.json")["city_slug"])
            for family_dir in families
        }
    )
    cle_contract = {city: _cle_artifact_contract(cle, city) for city in cities}
    for family_dir in families:
        family_id = family_dir.name
        copied_manifest_path = (
            output / "materialized" / "families" / family_id / "family_manifest.json"
        )
        manifest = _read_json(copied_manifest_path)
        manifest["matrix_reconstruction"] = {
            "schema": "cle_evrptw_matrix_reconstruction_contract_v1",
            "status": "cache_omitted_reconstruct_before_use",
            "road_state_factor_source": (
                "family_manifest.road_state_report.moves_road_type_baseline_factors"
            ),
            "terminal_access_source": str(manifest["terminal_index"]),
            "expected_matrices": matrix_contracts[family_id],
            "reference_profile_sha256": sha256_file(profile_file),
        }
        _write_json(copied_manifest_path, manifest)

    copied_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    contract = {
        "schema": "cle_evrptw_slim_instances_v1",
        "source_dataset_root_name": source.name,
        "family_count": len(families),
        "view_count": len(registry),
        "cities": cities,
        "matrix_names": list(MATRIX_NAMES),
        "matrix_cache_semantics": "derived_cache_not_release_source_data",
        "source_matrix_bytes_omitted": source_matrix_bytes,
        "slim_dataset_bytes": copied_bytes,
        "reference_profile": {
            "relative_path": "_reconstruction/reference_profile.json",
            "profile_id": str(profile["profile_id"]),
            "sha256": sha256_file(profile_file),
        },
        "instance_registry": "_reconstruction/instance_registry.parquet",
        "cle_requirements": cle_contract,
        "reconstruction_policy": {
            "road_state": "stored_baseline_factors_preferred_over_rng_replay",
            "terminal_access": "exact_directed_edge_projection_and_connector",
            "matrix_dtype": "float32",
        },
        "export_wall_seconds": time.perf_counter() - started,
    }
    _write_json(reconstruction_root / "reconstruction_contract.json", contract)
    return contract


def _load_profile_for_dataset(
    dataset_root: Path, profile_path: str | Path | None
) -> dict[str, Any]:
    path = (
        Path(profile_path)
        if profile_path is not None
        else dataset_root / "_reconstruction" / "reference_profile.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference profile is missing: {path}; provide --profile for a non-slim tree"
        )
    profile = load_reference_profile(path)
    contract_path = dataset_root / "_reconstruction" / "reconstruction_contract.json"
    if contract_path.is_file():
        expected = _read_json(contract_path)["reference_profile"]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ReconstructionError(
                f"Reference profile checksum mismatch: expected={expected}, actual={actual}"
            )
    return profile


def _validate_cle_contract(dataset_root: Path, cle_root: Path, cities: Iterable[str]) -> None:
    contract_path = dataset_root / "_reconstruction" / "reconstruction_contract.json"
    if not contract_path.is_file():
        return
    requirements = _read_json(contract_path).get("cle_requirements", {})
    for city in sorted(set(cities)):
        expected_city = requirements.get(city)
        if expected_city is None:
            raise ReconstructionError(f"Slim contract has no CLE requirement for {city}")
        for label, spec in expected_city["artifacts"].items():
            path = cle_root / spec["relative_path"]
            actual = sha256_file(path)
            if actual != spec["sha256"]:
                raise ReconstructionError(
                    f"CLE {city} {label} checksum mismatch: "
                    f"expected={spec['sha256']}, actual={actual}"
                )


def resolve_family_dirs(
    dataset_root: str | Path,
    *,
    family_ids: Sequence[str] | None = None,
    view_ids: Sequence[str] | None = None,
) -> list[Path]:
    root = Path(dataset_root).resolve()
    all_families = {path.name: path for path in _family_dirs(root)}
    selected = set(map(str, family_ids or ()))
    if view_ids:
        registry_path = root / "_reconstruction" / "instance_registry.parquet"
        if registry_path.is_file():
            registry = pd.read_parquet(registry_path, columns=["view_id", "family_id"])
        else:
            registry = _build_instance_registry(root, list(all_families.values()))[
                ["view_id", "family_id"]
            ]
        mapping = dict(zip(registry["view_id"].astype(str), registry["family_id"].astype(str)))
        missing_views = sorted(set(map(str, view_ids)) - set(mapping))
        if missing_views:
            raise ReconstructionError(f"Unknown view IDs: {missing_views}")
        selected.update(mapping[view_id] for view_id in map(str, view_ids))
    if not selected:
        return list(all_families.values())
    missing_families = sorted(selected - set(all_families))
    if missing_families:
        raise ReconstructionError(f"Unknown family IDs: {missing_families}")
    return [all_families[family_id] for family_id in sorted(selected)]


@dataclass
class ReconstructionContext:
    cle_root: Path
    profile: dict[str, Any]

    def __post_init__(self) -> None:
        self._cle_by_city: dict[str, PortableCLE] = {}
        self._speeds_by_city: dict[str, pd.DataFrame] = {}
        self._topology_by_city: dict[str, PhysicalRoadNetwork] = {}

    def route_family(self, family_dir: Path) -> dict[str, np.ndarray]:
        manifest = _read_json(family_dir / "family_manifest.json")
        family_id = str(manifest["family_id"])
        city = str(manifest["city_slug"])
        if str(manifest["reference_profile_id"]) != str(self.profile["profile_id"]):
            raise ReconstructionError(
                f"{family_id} profile mismatch: manifest={manifest['reference_profile_id']!r}, "
                f"provided={self.profile['profile_id']!r}"
            )
        if city not in self._cle_by_city:
            mode = str(manifest.get("generation_mode", "research"))
            self._cle_by_city[city] = load_portable_cle(
                self.cle_root,
                city,
                mode=mode,  # type: ignore[arg-type]
                minimum_customers=1,
                minimum_depots=1,
                minimum_chargers=1,
            )
            self._speeds_by_city[city] = pd.read_parquet(self._cle_by_city[city].speeds_path)
        baselines = manifest.get("road_state_report", {}).get(
            "moves_road_type_baseline_factors"
        )
        if not baselines:
            raise ReconstructionError(f"{family_id} has no stored road-state baseline factors")
        road_state, report = build_family_road_state(
            self._speeds_by_city[city],
            day_type=str(manifest["day_type"]),
            road_state_seed=int(manifest["road_state_seed"]),
            profile=self.profile,
            moves_road_type_baseline_factors=baselines,
        )
        if report["moves_road_type_baseline_factors"] != {
            str(key): float(value) for key, value in baselines.items()
        }:
            raise ReconstructionError(f"{family_id} did not preserve stored road-state factors")
        cached = self._topology_by_city.get(city)
        if cached is None:
            network = PhysicalRoadNetwork.from_files(
                self._cle_by_city[city].graph_path, road_state, self.profile
            )
            self._topology_by_city[city] = network
        else:
            network = cached.with_road_state(road_state, self.profile)
        terminal_index = pd.read_parquet(family_dir / manifest["terminal_index"])
        missing = REQUIRED_TERMINAL_RECONSTRUCTION_COLUMNS - set(terminal_index.columns)
        if missing:
            raise ReconstructionError(
                f"{family_id} terminal_index is missing reconstruction fields: {sorted(missing)}"
            )
        if len(terminal_index) != int(manifest["terminal_count"]):
            raise ReconstructionError(f"{family_id} terminal count differs from its manifest")
        return _matrix_payload(network.route_terminals(terminal_index))


def _expected_matrix_contracts(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest.get("matrix_reconstruction", {}).get("expected_matrices", {})


def _compare_arrays(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    mode: ValidationMode,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    same_shape = expected.shape == actual.shape
    exact = same_shape and np.array_equal(expected, actual)
    allclose = same_shape and np.allclose(expected, actual, rtol=rtol, atol=atol)
    if same_shape:
        difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        maximum_absolute_error = float(difference.max(initial=0.0))
    else:
        maximum_absolute_error = float("inf")
    if mode == "exact":
        passed = exact
    elif mode == "allclose":
        passed = allclose
    else:
        passed = True
    return {
        "passed": bool(passed),
        "shape_matches": same_shape,
        "array_equal": bool(exact),
        "allclose": bool(allclose),
        "maximum_absolute_error": maximum_absolute_error,
    }


def verify_family_reconstruction(
    family_dir: str | Path,
    *,
    cle_root: str | Path,
    profile_path: str | Path,
    validation: ValidationMode = "exact",
    rtol: float = 1e-6,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Recompute in memory and compare with an existing matrix cache."""

    root = Path(family_dir).resolve()
    manifest = _read_json(root / "family_manifest.json")
    context = ReconstructionContext(Path(cle_root).resolve(), load_reference_profile(profile_path))
    rebuilt = context.route_family(root)
    comparisons: dict[str, Any] = {}
    for name, actual in rebuilt.items():
        expected_path = root / manifest["matrix_files"][name]
        expected = np.load(expected_path, allow_pickle=False)
        comparison = _compare_arrays(
            expected, actual, mode=validation, rtol=rtol, atol=atol
        )
        comparison["existing_npy_sha256"] = sha256_file(expected_path)
        with tempfile.TemporaryDirectory(prefix="evrptw-reconstruction-") as temporary:
            rebuilt_path = Path(temporary) / f"{name}.npy"
            np.save(rebuilt_path, np.asarray(actual, dtype=np.float32), allow_pickle=False)
            comparison["rebuilt_npy_sha256"] = sha256_file(rebuilt_path)
        comparison["npy_checksum_matches"] = (
            comparison["existing_npy_sha256"] == comparison["rebuilt_npy_sha256"]
        )
        comparisons[name] = comparison
    return {
        "schema": "cle_evrptw_family_reconstruction_verification_v1",
        "family_id": str(manifest["family_id"]),
        "validation": validation,
        "passed": all(item["passed"] for item in comparisons.values()),
        "all_npy_checksums_match": all(
            item["npy_checksum_matches"] for item in comparisons.values()
        ),
        "matrices": comparisons,
    }


def restore_family_matrices(
    family_dir: str | Path,
    *,
    context: ReconstructionContext,
    validation: ValidationMode = "exact",
    rtol: float = 1e-6,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Restore a missing parent matrix cache atomically, or verify an existing one."""

    started = time.perf_counter()
    root = Path(family_dir).resolve()
    manifest = _read_json(root / "family_manifest.json")
    family_id = str(manifest["family_id"])
    matrix_dir = root / "matrices"
    expected_contracts = _expected_matrix_contracts(manifest)
    existing_paths = [root / manifest["matrix_files"][name] for name in MATRIX_NAMES]
    if all(path.is_file() for path in existing_paths):
        mismatches = []
        for name, path in zip(MATRIX_NAMES, existing_paths):
            expected_sha = expected_contracts.get(name, {}).get("npy_sha256")
            if validation == "exact" and expected_sha and sha256_file(path) != expected_sha:
                mismatches.append(name)
        if not mismatches:
            return {
                "family_id": family_id,
                "status": "reused_existing_cache",
                "wall_seconds": time.perf_counter() - started,
            }
        raise ReconstructionError(
            f"{family_id} has existing matrices with checksum mismatches: {mismatches}; "
            "the restore command never overwrites an existing cache"
        )
    if any(path.exists() for path in existing_paths) or matrix_dir.exists():
        raise ReconstructionError(
            f"{family_id} has a partial matrix cache; move it aside before deterministic restore"
        )

    rebuilt = context.route_family(root)
    temporary = Path(tempfile.mkdtemp(prefix=".matrices-rebuild-", dir=root))
    try:
        validations: dict[str, Any] = {}
        for name in MATRIX_NAMES:
            path = temporary / f"{name}.npy"
            values = np.asarray(rebuilt[name], dtype=np.float32)
            np.save(path, values, allow_pickle=False)
            expected = expected_contracts.get(name, {})
            checksum_matches = not expected.get("npy_sha256") or (
                sha256_file(path) == expected["npy_sha256"]
            )
            shape_matches = not expected.get("shape") or list(values.shape) == list(
                expected["shape"]
            )
            dtype_matches = not expected.get("dtype") or str(values.dtype) == expected["dtype"]
            if validation == "exact" and not checksum_matches:
                raise ReconstructionError(
                    f"{family_id}/{name} reconstruction checksum mismatch"
                )
            if not shape_matches or not dtype_matches:
                raise ReconstructionError(
                    f"{family_id}/{name} reconstruction shape or dtype mismatch"
                )
            validations[name] = {
                "npy_sha256": sha256_file(path),
                "checksum_matches": checksum_matches,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
        os.replace(temporary, matrix_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "family_id": family_id,
        "status": "restored",
        "validation": validation,
        "matrices": validations,
        "wall_seconds": time.perf_counter() - started,
    }


def _restore_chunk(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = ReconstructionContext(Path(task["cle_root"]), dict(task["profile"]))
    return [
        restore_family_matrices(
            family_dir,
            context=context,
            validation=task["validation"],
            rtol=float(task["rtol"]),
            atol=float(task["atol"]),
        )
        for family_dir in map(Path, task["family_dirs"])
    ]


def restore_dataset_matrices(
    dataset_root: str | Path,
    *,
    cle_root: str | Path,
    profile_path: str | Path | None = None,
    family_ids: Sequence[str] | None = None,
    view_ids: Sequence[str] | None = None,
    workers: int = 1,
    families_per_worker_task: int = 25,
    validation: ValidationMode = "exact",
    rtol: float = 1e-6,
    atol: float = 1e-5,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(dataset_root).resolve()
    cle = Path(cle_root).resolve()
    if workers <= 0 or families_per_worker_task <= 0:
        raise ValueError("workers and families_per_worker_task must be positive")
    if validation not in {"exact", "allclose", "none"}:
        raise ValueError(f"Unsupported validation mode: {validation}")
    selected = resolve_family_dirs(root, family_ids=family_ids, view_ids=view_ids)
    profile = _load_profile_for_dataset(root, profile_path)
    manifests = [_read_json(path / "family_manifest.json") for path in selected]
    _validate_cle_contract(root, cle, [str(item["city_slug"]) for item in manifests])

    chunks: list[list[Path]] = []
    by_city: dict[str, list[Path]] = {}
    for family_dir, manifest in zip(selected, manifests):
        by_city.setdefault(str(manifest["city_slug"]), []).append(family_dir)
    for city in sorted(by_city):
        ordered = sorted(by_city[city])
        chunks.extend(
            ordered[start : start + families_per_worker_task]
            for start in range(0, len(ordered), families_per_worker_task)
        )
    tasks = [
        {
            "cle_root": str(cle),
            "profile": profile,
            "family_dirs": [str(path) for path in chunk],
            "validation": validation,
            "rtol": rtol,
            "atol": atol,
        }
        for chunk in chunks
    ]
    results: list[dict[str, Any]] = []
    if workers == 1:
        for task in tasks:
            results.extend(_restore_chunk(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_restore_chunk, task) for task in tasks]
            for future in as_completed(futures):
                results.extend(future.result())
    results.sort(key=lambda item: item["family_id"])
    return {
        "schema": "cle_evrptw_dataset_matrix_restore_report_v1",
        "dataset_root": str(root),
        "selected_family_count": len(selected),
        "requested_view_ids": list(map(str, view_ids or ())),
        "workers": workers,
        "validation": validation,
        "restored_count": sum(item["status"] == "restored" for item in results),
        "reused_count": sum(item["status"] == "reused_existing_cache" for item in results),
        "families": results,
        "passed": True,
        "wall_seconds": time.perf_counter() - started,
    }
