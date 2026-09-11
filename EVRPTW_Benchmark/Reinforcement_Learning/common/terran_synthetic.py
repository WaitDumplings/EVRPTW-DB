"""Frozen TERRAN-derived XY instances; never use the Road/Haversine adapter.

The upstream generator is executed only by the explicit ``generate`` CLI. All
solvers load exactly the same canonical bytes, using the usual training ID stream.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import difflib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import platform
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd

UPSTREAM_COMMIT = "926665870d08e24b1acea33a57f1412c96376758"
ADAPTER_VERSION = "terran_synthetic_canonical_xy_v1"
INDEX_SCHEMA = "terran_synthetic_index_v1"
CORPUS_SCHEMA = "terran_synthetic_corpus_v1"
LENGTH_KM_PER_UNIT = 1.0
TIME_S_PER_UNIT = 60.0
ENERGY_KWH_PER_UNIT = 100.0 / 257.0
CANONICAL_CARGO_CM3 = 18_500_000.0


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / "view_index.parquet" if candidate.is_dir() else candidate


def is_synthetic_index(dataset_path: str | Path) -> bool:
    path = _index_path(dataset_path)
    sidecar = path.parent / "synthetic_index_manifest.json"
    if not sidecar.is_file():
        return False
    return json.loads(sidecar.read_text()).get("schema") == INDEX_SCHEMA


@lru_cache(maxsize=8)
def _read_index(index_path: str) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    path = Path(index_path).resolve()
    sidecar = json.loads((path.parent / "synthetic_index_manifest.json").read_text())
    if sidecar.get("schema") != INDEX_SCHEMA:
        raise ValueError("not a TERRAN synthetic index")
    if sidecar.get("representation") != "E" or sidecar.get("source_kind") != "terran_synthetic":
        raise ValueError("synthetic source/representation mismatch")
    if sha256_file(path) != sidecar["index_sha256"]:
        raise ValueError("synthetic index content differs from its frozen manifest")
    root = (path.parent / sidecar["corpus_root_relative"]).resolve()
    rows = pd.read_parquet(path).to_dict(orient="records")
    if len(rows) != sidecar["instance_count"]:
        raise ValueError("synthetic index count mismatch")
    if len({str(row["view_id"]) for row in rows}) != len(rows):
        raise ValueError("synthetic index IDs are not unique")
    return root, rows, sidecar


def read_synthetic_tasks(dataset_path: str | Path, **_: Any) -> list[Any]:
    from EVRPTW_Benchmark.MetaHeuristics.benchmark_common import Stage2ViewTask

    path = _index_path(dataset_path).resolve()
    root, rows, _ = _read_index(str(path))
    return [Stage2ViewTask(
        index_path=str(path), family_dir=str(root), view_id=str(row["view_id"]),
        family_id=str(row["family_id"]), consumer_cohort_id="terran_synthetic",
        split_id=str(row["split_id"]), track_id=str(row["track_id"]),
        city_slug="terran_synthetic", scale_id="Cus100", customer_count=100,
        charging_station_count=20, row_position=position,
        family_cohort_id="synthetic_no_city_or_depot_day", terminal_count=121,
        view_seed=int(row["view_seed"]),
    ) for position, row in enumerate(rows)]


def _record(root: Path, row: dict[str, Any], kind: str) -> dict[str, Any]:
    path = (root / row[f"{kind}_relative_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("synthetic record path escapes the corpus root")
    with path.open("rb") as handle:
        handle.seek(int(row[f"{kind}_offset"]))
        data = handle.read(int(row[f"{kind}_length"]))
    if hashlib.sha256(data).hexdigest() != row[f"{kind}_sha256"]:
        raise ValueError(f"{kind} content hash mismatch for {row['view_id']}")
    # These are locally generated, checksum-verified benchmark artifacts.
    return pickle.loads(data)


def load_synthetic_instance(task: Any) -> Any:
    from evrptw_core.schema import EVRPTWInstance

    root, rows, _ = _read_index(str(Path(task.index_path).resolve()))
    row = rows[int(task.row_position)]
    if str(row["view_id"]) != str(task.view_id):
        raise ValueError("synthetic task/index identity mismatch")
    value = _record(root, row, "canonical")
    if value["instance_id"] != task.view_id:
        raise ValueError("synthetic canonical instance ID mismatch")
    instance = EVRPTWInstance.from_dict(value)
    if (instance.num_customers, instance.num_charging_stations) != (100, 20):
        raise ValueError("synthetic corpus is restricted to Cus100 and 20 chargers")
    return instance


def derive_seed(base_seed: int, split: str) -> int:
    name = f"{ADAPTER_VERSION}:base_seed={int(base_seed)}:split={split}"
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def convert_instance(raw: dict[str, Any], *, instance_id: str, split: str,
                     generator_seed: int, sequence_position: int,
                     time_window_probability_actual: int,
                     config: dict[str, Any]) -> dict[str, Any]:
    """Preserve upstream dimensionless time/load/battery/charging relations."""
    count = len(raw["cus_loc"])
    stations = len(raw["rs_loc"])
    if (count, stations) != (100, 20):
        raise ValueError("this experiment accepts only Cus100 / 20 chargers")
    pos_scale = float(config["general"]["pos_scale"])
    native_nodes = np.concatenate([raw["depot_loc"], raw["rs_loc"], raw["cus_loc"]])
    canonical_to_native = np.concatenate(([0], np.arange(1 + stations, 1 + stations + count), np.arange(1, 1 + stations))).astype(np.int32)
    native_to_canonical = np.argsort(canonical_to_native).astype(np.int32)
    coords = np.asarray(native_nodes[canonical_to_native], dtype=np.float64) * pos_scale * LENGTH_KM_PER_UNIT
    native_dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=2) / LENGTH_KM_PER_UNIT
    native_time = native_dist / float(raw["velocity_base"])
    distance = (native_dist * LENGTH_KM_PER_UNIT).astype(np.float32)
    travel_time = (native_time * TIME_S_PER_UNIT).astype(np.float32)
    energy = (native_time * float(raw["energy_consumption"]) * ENERGY_KWH_PER_UNIT).astype(np.float32)
    max_time = float(raw["max_time"])
    horizon_s = max_time * TIME_S_PER_UNIT
    if not float(horizon_s).is_integer():
        raise ValueError("canonical integer horizon requires an exact integral unit conversion")
    native_tw = np.asarray(raw["time_window"], dtype=np.float64) * max_time
    demands_fraction = np.asarray(raw["demand"], dtype=np.float64)[1 + stations:]
    service = np.full(count, float(raw["service_time"]) * max_time * TIME_S_PER_UNIT, dtype=np.float32)
    charging_power = float(raw["charging_rate"]) * ENERGY_KWH_PER_UNIT * 3600.0 / TIME_S_PER_UNIT
    power = np.full(stations, charging_power, dtype=np.float64)
    battery = float(raw["battery_capacity"]) * ENERGY_KWH_PER_UNIT
    cs_time = travel_time[1 + count:, 0].copy()
    vehicle = {
        "profile_id": "terran_derived_type_dependent_battery_fixed_equivalent_cargo_v1",
        "battery_capacity_kwh": battery, "cargo_capacity_cm3": CANONICAL_CARGO_CM3,
        "specific_energy_consumption_kwh_per_km": float(raw["energy_consumption"]) / float(raw["velocity_base"]) * ENERGY_KWH_PER_UNIT / LENGTH_KM_PER_UNIT,
        "full_charge_time_s": float(raw["battery_capacity"]) / float(raw["charging_rate"]) * TIME_S_PER_UNIT,
        "charging_power_derating_factor": 1.0,
    }
    metadata = {
        "source_kind": "terran_synthetic", "coordinate_system": "synthetic_cartesian_xy_km",
        "training_representation": "E", "adapter_version": ADAPTER_VERSION,
        "upstream_commit": UPSTREAM_COMMIT, "upstream_type": str(raw["types"]),
        "generator_seed": int(generator_seed), "generator_sequence_position": int(sequence_position),
        "split_id": split, "track_id": "validation" if split == "val" else "train", "view_id": instance_id,
        "family_id": instance_id, "city_slug": "terran_synthetic",
        "geographic_city": None, "operational_day": None,
        "time_window_probability_actual": int(time_window_probability_actual),
        "rs_random_ratio_actual": 0.5,
        "upstream_values": {key: raw[key] for key in ("max_time", "demand_capacity", "battery_capacity", "velocity_base", "energy_consumption", "charging_rate", "service_time")},
        "canonical_to_upstream_node": canonical_to_native,
        "upstream_to_canonical_node": native_to_canonical,
        "demand_cm3_per_upstream_unit": CANONICAL_CARGO_CM3 / float(raw["demand_capacity"]),
        "metric_contract": {"distance": "explicit_L2_xy_km", "travel_time": "native_distance_over_native_velocity_scaled_to_seconds", "energy": "native_travel_time_times_native_consumption_scaled_to_kwh"},
    }
    return {
        "instance_id": instance_id, "family_id": instance_id,
        "region_id": "terran_synthetic", "mother_board_id": "synthetic_no_parent_road_instance",
        "operating_day_id": "synthetic_not_operational_day", "day_type": "synthetic",
        "working_start_s": 0, "working_end_s": int(horizon_s),
        "depot": coords[0].astype(np.float32), "customers": coords[1:1 + count].astype(np.float32),
        "charging_stations": coords[1 + count:].astype(np.float32),
        "distance_matrix_km": distance,
        "demands_cm3": (demands_fraction * CANONICAL_CARGO_CM3).astype(np.float32),
        "package_counts": np.ones(count, dtype=np.int32),
        "service_time_s": service, "tw_s": (native_tw[1 + stations:] * TIME_S_PER_UNIT).astype(np.float32),
        "cs_time_to_depot_s": cs_time, "vehicle": vehicle,
        "raw_travel_time_matrix_s": travel_time, "ev_transition_time_matrix_s": travel_time,
        "shortest_time_matrix_s": travel_time, "energy_matrix_kwh": energy,
        "speed_profile": {"matrix_source": "terran_synthetic_xy", "effective_speed_kmh": float(raw["velocity_base"]) * LENGTH_KM_PER_UNIT * 3600.0 / TIME_S_PER_UNIT},
        "cs_activation": {"charging_power_kw": power}, "metadata": metadata,
        "charging_power_kw": power,
        "charging_policy": {"charging_power_derating_factor": 1.0, "charging_mode": "station_power_full"},
        "running_time_shortest_matrix_s": travel_time,
        "running_time_path_distance_km": distance, "running_time_path_energy_kwh": energy,
        "distance_path_travel_time_s": travel_time, "full_cs_to_depot_time_s": cs_time,
        "terminal_parent_indices": np.arange(121, dtype=np.int32),
    }


def _write_index(root: Path, name: str, rows: list[dict[str, Any]], *, complete: bool) -> dict[str, Any]:
    destination = root / name
    destination.mkdir(exist_ok=True)
    index = destination / "view_index.parquet"
    pd.DataFrame(rows).to_parquet(index, index=False)
    result = {"schema": INDEX_SCHEMA, "source_kind": "terran_synthetic", "representation": "E",
              "corpus_root_relative": "..", "instance_count": len(rows), "index_sha256": sha256_file(index),
              "split_id": rows[0]["split_id"] if rows else name,
              "complete": complete, "usage": "smoke_only" if name.endswith("smoke") else "formal_corpus",
              "adapter_version": ADAPTER_VERSION, "upstream_commit": UPSTREAM_COMMIT}
    _json(destination / "synthetic_index_manifest.json", result)
    return result


def _rng_snapshot(torch: Any) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}


def generate(args: argparse.Namespace) -> None:
    import torch
    from evrptw_core.schema import EVRPTWInstance
    from evrptw_core.validation import validate_instance_structure

    upstream = Path(args.upstream_root).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"upstream commit mismatch: {commit}")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "instance_generator.py", "configs/config_100c.json"], cwd=upstream, text=True).strip():
        raise ValueError("upstream generator/config has unrecorded edits")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("output must be empty: never overwrite or mix frozen corpora")
    provenance = root / "provenance"
    provenance.mkdir()
    for source, target in [(upstream / "instance_generator.py", "instance_generator.py"), (upstream / "configs/config_100c.json", "upstream_config_100c.json"), (Path(__file__), "terran_synthetic_adapter.py"), (Path(__file__).with_name("terran_synthetic_feasibility.py"), "terran_synthetic_feasibility.py")]:
        shutil.copyfile(source, provenance / target)
    shutil.copyfile(provenance / "upstream_config_100c.json", provenance / "effective_config_100c.json")
    distribution_mode = getattr(args, "distribution_mode", "preserve_upstream_actual")
    generator_path = provenance / "instance_generator.py"
    if distribution_mode in {"reject_inverted_tw_v1", "canonical_candidate_feasibility_v2", "canonical_candidate_feasibility_v3", "canonical_candidate_feasibility_v4"}:
        original = generator_path.read_text()
        needle = "l_i = min(customer_tw_center + customer_tw_width / 2, self.max_time)"
        if original.count(needle) != 2:
            raise ValueError("upstream patch sites differ from the audited commit")
        patched_lines = []
        for line in original.splitlines(True):
            patched_lines.append(line)
            if line.strip() == needle:
                indent = line[:len(line) - len(line.lstrip())]
                if distribution_mode == "reject_inverted_tw_v1":
                    patched_lines.extend([indent + "# TERRAN-derived repair v1: reject an inverted candidate TW.\n", indent + "if e_i > l_i:\n", indent + "    continue\n"])
                else:
                    stops_name = "depot_and_rs" if len(indent) == 20 else "centroids"
                    patched_lines.extend([indent + "# TERRAN-derived v2: include TW waiting and canonical return feasibility.\n", indent + f"if not native_singleton_candidate_feasible(customer_pos, {stops_name}, e_i, l_i, self.velocity, self.energy_consumption, self.battery_capacity, self.charging_rate, self.service_time, self.max_time):\n", indent + "    self.canonical_candidate_rejections = getattr(self, 'canonical_candidate_rejections', 0) + 1\n", indent + "    continue\n"])
        patched = "".join(patched_lines)
        if distribution_mode in {"canonical_candidate_feasibility_v2", "canonical_candidate_feasibility_v3", "canonical_candidate_feasibility_v4"}:
            from .terran_synthetic_feasibility import native_singleton_candidate_feasible
            patched = patched.replace("class Solomon_EVRPTW_Generation:", inspect.getsource(native_singleton_candidate_feasible) + "\n\nclass Solomon_EVRPTW_Generation:")
        if distribution_mode in {"canonical_candidate_feasibility_v3", "canonical_candidate_feasibility_v4"}:
            old_loop = "                cur_customer = 0\n                while True:\n                    cluster_tw_center"
            new_loop = "                cur_customer = 0\n                canonical_cluster_start = len(customer_positions)\n                canonical_candidate_attempts = 0\n                while True:\n                    canonical_candidate_attempts += 1\n                    if canonical_candidate_attempts > 10000:\n                        # The fixed centroid may require a repeated charger, which canonical routes prohibit.\n                        # Retry only this unfinished cluster; preserve preceding clusters and the instance type.\n                        del customer_positions[canonical_cluster_start:]\n                        del customer_time_windows[canonical_cluster_start:]\n                        self.canonical_centroid_restarts = getattr(self, 'canonical_centroid_restarts', 0) + 1\n                        cur_customer = 0\n                        break\n                    cluster_tw_center"
            if patched.count(old_loop) != 1:
                raise ValueError("upstream centroid retry site differs from audited source")
            patched = patched.replace(old_loop, new_loop)
            if distribution_mode == "canonical_candidate_feasibility_v4":
                centroid_anchor = "                cur_customer = 0\n                canonical_cluster_start"
                centroid_guard = "                if not native_singleton_candidate_feasible(centroid_position, depot_and_rs, 0.0, self.max_time, self.velocity, self.energy_consumption, self.battery_capacity, self.charging_rate, self.service_time, self.max_time):\n                    self.canonical_centroid_rejections = getattr(self, 'canonical_centroid_rejections', 0) + 1\n                    continue\n                cur_customer = 0\n                canonical_cluster_start"
                if patched.count(centroid_anchor) != 1:
                    raise ValueError("centroid feasibility guard anchor differs from audited source")
                patched = patched.replace(centroid_anchor, centroid_guard)
        generator_path.write_text(patched)
        (provenance / f"{distribution_mode}.patch").write_text("".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True), fromfile="upstream/instance_generator.py", tofile="derived/instance_generator.py")))
    config_path = provenance / "effective_config_100c.json"
    config = json.loads(config_path.read_text())
    spec = importlib.util.spec_from_file_location("frozen_terran_generator_9266658", provenance / "instance_generator.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    corpus = {
        "schema": CORPUS_SCHEMA, "complete": False, "formal_training_authorized": not bool(args.provisional), "source_kind": "terran_synthetic", "representation": "E",
        "adapter_version": ADAPTER_VERSION, "upstream_commit": UPSTREAM_COMMIT,
        "upstream_url": "git@github.com:NanpengYu/TERRAN.git", "base_training_seed": int(args.seed),
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__},
        "distribution_mode": distribution_mode,
        "generator_lifecycle": "one separately seeded object per split, constructed once before all sequential draws",
        "seed_derivation": "int.from_bytes(sha256(f'{adapter_version}:base_seed={seed}:split={split}').digest()[:4], 'little')",
        "units": {"km_per_native_length": LENGTH_KM_PER_UNIT, "seconds_per_native_time": TIME_S_PER_UNIT,
                  "kwh_per_native_energy": ENERGY_KWH_PER_UNIT, "canonical_equivalent_cargo_cm3": CANONICAL_CARGO_CM3,
                  "demand_rule": "native_normalized_demand * canonical_equivalent_cargo_cm3",
                  "charging_power_kw": "native_charging_rate * kwh_per_native_energy * 3600 / seconds_per_native_time",
                  "charging_power_derating_factor": 1.0},
        "provenance_sha256": {path.name: sha256_file(path) for path in provenance.iterdir() if path.is_file()},
        "requested_counts": {"train": int(args.train_count), "val": int(args.val_count)},
        "splits": {}, "test_data_accessed": False,
    }
    _json(root / "corpus_manifest.json", corpus)
    all_raw_hashes: set[str] = set()
    for split, total in [("val", int(args.val_count)), ("train", int(args.train_count))]:
        split_dir = root / split
        split_dir.mkdir()
        derived_seed = derive_seed(args.seed, split)
        module.set_seed(derived_seed)
        before = _rng_snapshot(torch)
        generator = module.Solomon_EVRPTW_Generation(str(config_path))
        sampled_tw = int(generator.time_window_ratio)
        with (split_dir / "rng_before.pkl").open("wb") as handle:
            pickle.dump(before, handle, protocol=5)
        rows: list[dict[str, Any]] = []
        types: Counter[str] = Counter()
        started = time.monotonic()
        handles: dict[str, Any] = {}
        max_distance = 0.0
        max_tw_s = 0.0
        for index in range(total):
            if index % int(args.shard_size) == 0:
                for handle in handles.values():
                    handle.close()
                shard = index // int(args.shard_size)
                handles = {kind: (split_dir / f"{kind}_{shard:05d}.pkl").open("wb") for kind in ("raw", "canonical")}
            raw = generator._generate_instances()
            id_version = {"preserve_upstream_actual": "v1", "reject_inverted_tw_v1": "twguard1", "canonical_candidate_feasibility_v2": "feasible2", "canonical_candidate_feasibility_v3": "feasible3", "canonical_candidate_feasibility_v4": "feasible4"}[distribution_mode]
            instance_id = f"terran100-{id_version}-seed{args.seed}-{split}-{index:06d}"
            canonical = convert_instance(raw, instance_id=instance_id, split=split, generator_seed=derived_seed,
                                         sequence_position=index, time_window_probability_actual=sampled_tw, config=config)
            canonical["metadata"]["generator_distribution_mode"] = distribution_mode
            if distribution_mode in {"canonical_candidate_feasibility_v2", "canonical_candidate_feasibility_v3", "canonical_candidate_feasibility_v4"}:
                from .terran_synthetic_feasibility import singleton_witnesses
                from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes
                instance = EVRPTWInstance.from_dict(canonical)
                witness, missing = singleton_witnesses(instance)
                replay = validate_routes(instance, witness) if not missing else {"passed": False, "violations": [f"missing singleton witnesses: {missing}"]}
                if not replay["passed"]:
                    (root / "generation_failure_raw.pkl").write_bytes(pickle.dumps(raw, protocol=5))
                    _json(root / "generation_failure.json", {"split": split, "position": index, "replay": replay})
                    raise ValueError(f"canonical candidate witness failed without filtering/resampling: {instance_id}: {replay}")
                canonical["synthetic_singleton_witness_routes"] = witness
                canonical["metadata"]["singleton_witness_independent_replay_passed"] = True
                canonical["metadata"]["singleton_witness_distance_km"] = replay["objective_distance_km"]
                canonical["metadata"]["candidate_rejections_so_far"] = int(getattr(generator, "canonical_candidate_rejections", 0))
                canonical["metadata"]["centroid_restarts_so_far"] = int(getattr(generator, "canonical_centroid_restarts", 0))
                canonical["metadata"]["centroid_rejections_so_far"] = int(getattr(generator, "canonical_centroid_rejections", 0))
            validation = validate_instance_structure(EVRPTWInstance.from_dict(canonical))
            if not validation.success or validation.warnings:
                (root / "generation_failure_raw.pkl").write_bytes(pickle.dumps(raw, protocol=5))
                (root / "generation_failure_canonical.pkl").write_bytes(pickle.dumps(canonical, protocol=5))
                _json(root / "generation_failure.json", {"split": split, "position": index, "errors": validation.errors, "warnings": validation.warnings})
                raise ValueError(f"upstream/canonical structural error without resampling: {instance_id}: {validation}")
            row = {"view_id": instance_id, "family_id": instance_id, "split_id": split, "track_id": "validation" if split == "val" else "train",
                   "city_slug": "terran_synthetic", "scale_id": "Cus100", "customer_count": 100,
                   "charging_station_count": 20, "terminal_count": 121, "day_type": "synthetic",
                   "source_kind": "terran_synthetic", "representation": "E", "view_seed": derived_seed,
                   "upstream_type": str(raw["types"]), "generator_sequence_position": index,
                   "consumer_cohort_id": "terran_synthetic", "family_cohort_id": "synthetic_no_city_or_depot_day"}
            for kind, value in [("raw", raw), ("canonical", canonical)]:
                payload = pickle.dumps(value, protocol=5)
                digest = hashlib.sha256(payload).hexdigest()
                handle = handles[kind]
                row.update({f"{kind}_relative_path": str(Path(handle.name).relative_to(root)),
                            f"{kind}_offset": handle.tell(), f"{kind}_length": len(payload), f"{kind}_sha256": digest})
                handle.write(payload)
                if kind == "raw":
                    if digest in all_raw_hashes:
                        raise ValueError("duplicate raw instance detected; no silent resampling allowed")
                    all_raw_hashes.add(digest)
            rows.append(row)
            types[str(raw["types"])] += 1
            max_distance = max(max_distance, float(np.max(canonical["distance_matrix_km"])))
            max_tw_s = max(max_tw_s, float(np.max(canonical["tw_s"])))
            if split == "train" and index + 1 == min(128, total):
                for handle in handles.values():
                    handle.flush()
                _write_index(root, "train_smoke", rows, complete=False)
            if (index + 1) % 100 == 0 or index + 1 == total:
                elapsed = time.monotonic() - started
                progress = {"split": split, "completed": index + 1, "total": total, "elapsed_s": elapsed,
                            "estimated_remaining_s": elapsed * (total - index - 1) / (index + 1), "types": dict(types)}
                _json(root / "generation_progress.json", progress)
                print(json.dumps(progress), flush=True)
        for handle in handles.values():
            handle.close()
        with (split_dir / "rng_after.pkl").open("wb") as handle:
            pickle.dump(_rng_snapshot(torch), handle, protocol=5)
        index_manifest = _write_index(root, split, rows, complete=True)
        corpus["splits"][split] = {
            "instance_count": len(rows), "derived_seed": derived_seed, "types": dict(types),
            "time_window_probability_actual": sampled_tw, "rc_random_fraction_actual": 0.5,
            "candidate_rejections": int(getattr(generator, "canonical_candidate_rejections", 0)),
            "centroid_restarts": int(getattr(generator, "canonical_centroid_restarts", 0)),
            "centroid_rejections": int(getattr(generator, "canonical_centroid_rejections", 0)),
            "index_relative_path": str((split_dir / "view_index.parquet").relative_to(root)),
            "index_sha256": index_manifest["index_sha256"], "generation_seconds": time.monotonic() - started,
            "max_edge_distance_km": max_distance, "max_customer_tw_s": max_tw_s,
            "file_sha256": {path.name: sha256_file(path) for path in sorted(split_dir.iterdir()) if path.is_file()},
            "schema_validation": "all_instances_passed_without_warnings", "instances_resampled_or_filtered_count": 0,
        }
        _json(root / "corpus_manifest.json", corpus)
    corpus["complete"] = True
    corpus["disjointness"] = {"raw_content_hashes_unique": len(all_raw_hashes), "train_val_overlap": 0,
                              "id_namespaces": "split is part of each immutable ID", "data_rng_seeds_distinct": True}
    _json(root / "corpus_manifest.json", corpus)
    print(json.dumps({"complete": True, "output": str(root), "manifest_sha256": sha256_file(root / "corpus_manifest.json")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate"])
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--distribution-mode", choices=["preserve_upstream_actual", "reject_inverted_tw_v1", "canonical_candidate_feasibility_v2", "canonical_candidate_feasibility_v3", "canonical_candidate_feasibility_v4"], default="preserve_upstream_actual")
    parser.add_argument("--provisional", action="store_true", help="Mark generated data as audit/smoke only, not authorized for formal training")
    parser.add_argument("--train-count", type=int, default=50_000)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--shard-size", type=int, default=1000)
    args = parser.parse_args()
    if args.train_count <= 0 or args.val_count <= 0 or args.shard_size <= 0:
        parser.error("counts and shard size must be positive")
    generate(args)


if __name__ == "__main__":
    main()
