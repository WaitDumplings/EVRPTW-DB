"""Small, reproducible before/after rollout diagnostic; never trains a model.

Uses the same untrained policy, instances and random seeds for both paths.
Synthetic inputs are performance fixtures, not EVRPTW-D release instances.
CUDA timing synchronizes at rollout boundaries, not at every action.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..common.stage2_data import Stage2TaskPool, make_envs
from ..common.route_info import finalize_route_infos
from ..common.evaluation import select_min_verified_distance
from evrptw_core.schema import EVRPTWInstance


def synthetic_instance(customers: int, stations: int, seed: int) -> EVRPTWInstance:
    rng = np.random.default_rng(seed)
    count = 1 + customers + stations
    coords = rng.uniform(0.0, 5.0, size=(count, 2)).astype(np.float32)
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    distance = distance.astype(np.float32)
    travel = distance * np.float32(90.0)
    energy = distance * np.float32(0.5)
    horizon = 100_000
    return EVRPTWInstance(
        instance_id=f"performance_fixture_{customers}_{seed}",
        region_id="synthetic_performance_fixture",
        mother_board_id="synthetic_performance_fixture",
        operating_day_id="synthetic_performance_fixture",
        day_type="weekday",
        working_start_s=0,
        working_end_s=horizon,
        depot=coords[0],
        customers=coords[1 : customers + 1],
        charging_stations=coords[customers + 1 :],
        distance_matrix_km=distance,
        demands_cm3=np.ones(customers, dtype=np.float32),
        package_counts=np.ones(customers, dtype=np.int32),
        service_time_s=np.full(customers, 30, dtype=np.float32),
        tw_s=np.tile(np.asarray([0, horizon], dtype=np.float32), (customers, 1)),
        cs_time_to_depot_s=travel[customers + 1 :, 0],
        vehicle={"battery_capacity_kwh": 100.0, "cargo_capacity_cm3": 20.0},
        shortest_time_matrix_s=travel,
        energy_matrix_kwh=energy,
        raw={
            "charging_power_kw": np.full(stations, 50.0, dtype=np.float32),
            "charging_policy": {"charging_power_derating_factor": 0.9},
        },
    )


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _model(method: str, args: argparse.Namespace) -> Any:
    torch.manual_seed(args.seed)
    if method == "am_evrptw":
        from ..AM_EVRPTW.model import AMEVRPTWPolicy

        policy = AMEVRPTWPolicy(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.embedding_dim,
            n_encode_layers=args.encoder_layers,
            n_heads=8,
        )
    elif method == "evrptw_rl":
        from ..EVRPTW_RL.model import EVRPTWRLPolicy

        policy = EVRPTWRLPolicy(embedding_dim=args.embedding_dim)
    elif method == "drl_ts":
        from ..DRL_TS.model import DRLTSPolicy

        policy = DRLTSPolicy(
            embedding_dim=args.embedding_dim,
            n_encode_layers=args.encoder_layers,
        )
    else:
        from ..TERRAN.models import Agent

        policy = Agent(
            embedding_dim=args.embedding_dim,
            n_encode_layers=args.encoder_layers,
            tanh_clipping=15.0,
            device=args.device,
        )
    return policy.to(args.device).eval()


def _rollout(method, policy, instances, args, *, optimized, seed):
    # Timer includes environment construction, reset, encode, decode and final
    # route export. Verifier time is measured separately outside this function.
    if method == "terran":
        from ..TERRAN.env_factory import make_terran_env
        from ..TERRAN.rollout import rollout_eval_batch

        envs = [
            make_terran_env(
                instance=instance,
                n_traj=args.candidates,
                charging_mode="station_power_full",
                matrix_mode="canonical",
                info_level="full",
                use_jit_mask=not args.no_jit,
            )
            for instance in instances
        ]
        rows = rollout_eval_batch(
            policy, envs, decode_mode="sample", max_steps=args.steps,
            device=args.device, seed=seed, include_routes=True,
            return_final_info=True, cache_static_embeddings=optimized,
            compact_observations=optimized, final_routes_only=optimized,
        )
        return [row["_final_info"] for row in rows]

    if method == "drl_ts":
        from ..DRL_TS.env import DRLTSHardConstraintEnv
        from ..DRL_TS.rollout import rollout

        envs = [
            DRLTSHardConstraintEnv(
                instance=instance, n_traj=args.candidates,
                info_level="light" if optimized else "full",
                use_jit_mask=not args.no_jit,
            )
            for instance in instances
        ]
        options = {"soft_constraints": False}
    else:
        envs = make_envs(
            instances, n_traj=args.candidates,
            info_level="light" if optimized else "full",
            use_jit_mask=not args.no_jit,
        )
        options = {"use_static_cache": optimized}
        if method == "am_evrptw":
            from ..AM_EVRPTW.rollout import rollout

            options["incomplete_penalty_km"] = 10_000.0
        else:
            from ..EVRPTW_RL.rollout import rollout

    result = rollout(
        policy, envs, decode_type="sampling", max_steps=args.steps,
        seed=seed, compute_log_likelihood=not optimized, **options,
    )
    return finalize_route_infos(envs, result.infos) if optimized else result.infos


def compare_infos(reference, optimized) -> dict[str, Any]:
    if len(reference) != len(optimized):
        raise AssertionError("reference/optimized batch sizes differ")
    max_objective_delta = 0.0
    routes_equal = True
    flags_equal = True
    objectives_finite = True
    for left, right in zip(reference, optimized):
        routes_equal &= left["routes"] == right["routes"]
        for field in ("success", "served_customers", "vehicle_count"):
            flags_equal &= np.array_equal(left[field], right[field])
        delta = np.abs(
            np.asarray(left["objective_distance_km"], dtype=np.float64)
            - np.asarray(right["objective_distance_km"], dtype=np.float64)
        )
        objectives_finite &= bool(np.isfinite(delta).all())
        if objectives_finite:
            max_objective_delta = max(max_objective_delta, float(delta.max()))
    return {
        "routes_equal": bool(routes_equal),
        "flags_equal": bool(flags_equal),
        "objectives_finite": bool(objectives_finite),
        "max_objective_delta_km": max_objective_delta,
        "passed": bool(routes_equal and flags_equal and objectives_finite
                       and max_objective_delta <= 1e-6),
    }


def run_benchmark(args) -> dict[str, Any]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    torch.set_num_threads(args.cpu_threads)
    if args.dataset_path:
        pool = Stage2TaskPool(
            dataset_path=args.dataset_path, family_root=args.family_root,
            scale=f"Cus{args.customers}", split_ids="train", track_ids="train",
        )
        instances = list(pool.first(limit=args.batch_size))
        if len(instances) != args.batch_size:
            raise ValueError("training pool smaller than requested diagnostic batch")
        input_kind = "frozen_training_views"
    else:
        instances = [
            synthetic_instance(args.customers, args.stations, args.seed + index)
            for index in range(args.batch_size)
        ]
        input_kind = "synthetic_performance_fixture_not_release_data"
    methods = (
        ["am_evrptw", "evrptw_rl", "drl_ts", "terran"]
        if args.method == "all" else [args.method]
    )
    results = []
    for method in methods:
        policy = _model(method, args)
        durations = {"reference": [], "optimized": []}
        peaks = {"reference": [], "optimized": []}
        comparisons = []
        verifier_checks = []
        for repetition in range(-args.warmup, args.repeats):
            infos = {}
            # Alternate order to avoid consistently favoring the second path.
            order = [False, True] if repetition % 2 == 0 else [True, False]
            for optimized in order:
                seed = args.seed + max(0, repetition) * 1009
                torch.manual_seed(seed)
                if args.device.startswith("cuda"):
                    torch.cuda.manual_seed_all(seed)
                    torch.cuda.reset_peak_memory_stats(args.device)
                name = "optimized" if optimized else "reference"
                _sync(args.device)
                started = time.perf_counter()
                with torch.no_grad():
                    infos[name] = _rollout(
                        method, policy, instances, args, optimized=optimized, seed=seed
                    )
                _sync(args.device)
                elapsed = time.perf_counter() - started
                if repetition >= 0:
                    durations[name].append(elapsed)
                    peaks[name].append(
                        torch.cuda.max_memory_allocated(args.device)
                        if args.device.startswith("cuda") else None
                    )
            if repetition >= 0:
                comparisons.append(compare_infos(infos["reference"], infos["optimized"]))
                started = time.perf_counter()
                left = [select_min_verified_distance(i, x) for i, x in zip(instances, infos["reference"])]
                right = [select_min_verified_distance(i, x) for i, x in zip(instances, infos["optimized"])]
                verifier_checks.append({
                    "selection_and_verifier_equal": all(
                        a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
                        for a, b in zip(left, right)
                    ),
                    "optimized_verified_feasible_count": sum(bool(x[2]["passed"]) for x in right),
                    "both_paths_verifier_wall_s": time.perf_counter() - started,
                })
        ref = statistics.median(durations["reference"])
        opt = statistics.median(durations["optimized"])
        results.append({
            "method": method,
            "reference_rollout_wall_s": durations["reference"],
            "optimized_rollout_wall_s": durations["optimized"],
            "median_speedup": ref / opt,
            "cuda_peak_allocated_bytes": peaks,
            "comparisons": comparisons,
            "verifier_checks": verifier_checks,
            "equivalence_passed": all(x["passed"] for x in comparisons)
            and all(x["selection_and_verifier_equal"] for x in verifier_checks),
        })
        del policy
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], text=True, capture_output=True, check=True
    ).stdout.strip()
    return {
        "schema": "drl_hotpath_diagnostic_v1", "base_commit": commit,
        "working_tree_dirty": bool(dirty), "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device)
        if args.device.startswith("cuda") else platform.processor() or platform.machine(),
        "torch_version": torch.__version__, "cpu_threads": args.cpu_threads,
        "input_kind": input_kind, "instance_ids": [x.instance_id for x in instances],
        "customers": args.customers, "batch_size": args.batch_size,
        "candidates": args.candidates, "max_steps": args.steps,
        "embedding_dim": args.embedding_dim, "encoder_layers": args.encoder_layers,
        "warmup": args.warmup, "repeats": args.repeats, "seed": args.seed,
        "jit_requested": not args.no_jit, "training_performed": False,
        "scope": "untrained_eval_rollout_only_not_end_to_end_training_speedup",
        "results": results, "passed": all(x["equivalence_passed"] for x in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["all", "am_evrptw", "evrptw_rl", "drl_ts", "terran"], default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--stations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    for field in ("customers", "stations", "batch_size", "candidates", "steps", "embedding_dim", "encoder_layers", "repeats", "cpu_threads"):
        if getattr(args, field) <= 0:
            parser.error(f"{field} must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    report = run_benchmark(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(args.output_json),
                      "results": [{"method": x["method"], "median_speedup": x["median_speedup"],
                                   "equivalence_passed": x["equivalence_passed"]} for x in report["results"]]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
