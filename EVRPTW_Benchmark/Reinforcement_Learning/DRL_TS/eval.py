from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.common.evaluation import select_min_verified_distance
from EVRPTW_Benchmark.Reinforcement_Learning.common.candidate_protocol import independent_candidate_batch

from ..common import Stage2TaskPool
from .env import DRLTSHardConstraintEnv
from .model import DRLTSPolicy
from .rollout import rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DRL-TS.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", default="Cus100")
    parser.add_argument("--split-ids", default="test")
    parser.add_argument("--track-ids", default="test1_new_seed")
    parser.add_argument("--city-slugs")
    parser.add_argument(
        "--decode-type",
        choices=("greedy", "sampling"),
        default="sampling",
    )
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--candidate-chunk-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _best_index(info: dict) -> int:
    success = np.asarray(info["success"], dtype=bool)
    objective = np.asarray(info["objective_distance_km"], dtype=float)
    served = np.asarray(info["served_customers"], dtype=int)
    if success.any():
        candidates = np.flatnonzero(success)
        return int(candidates[np.argmin(objective[candidates])])
    candidates = np.flatnonzero(served == served.max())
    return int(candidates[np.argmin(objective[candidates])])


def main() -> None:
    args = parse_args()
    if args.decode_type == "greedy" and args.candidates != 1:
        raise ValueError("greedy evaluation has exactly one candidate")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=args.device,
        weights_only=False,
    )
    model_args = checkpoint.get("args", {})
    policy = DRLTSPolicy(
        embedding_dim=int(model_args.get("embedding_dim", 128)),
        n_encode_layers=int(model_args.get("n_encode_layers", 2)),
        n_heads=int(model_args.get("n_heads", 8)),
        nearest_neighbors=int(model_args.get("nearest_neighbors", 10)),
        tanh_clipping=float(model_args.get("tanh_clipping", 10.0)),
    ).to(args.device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    pool = Stage2TaskPool(
        dataset_path=args.dataset_path,
        family_root=args.family_root,
        scale=args.scale,
        split_ids=args.split_ids,
        track_ids=args.track_ids,
        city_slugs=args.city_slugs,
        seed=args.seed,
    )
    instances = list(pool.first(limit=args.limit))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    routes_path = args.output_dir / "routes.jsonl"
    with routes_path.open("w", encoding="utf-8") as route_stream:
        for batch_start in range(0, len(instances), args.batch_size):
            batch_instances = instances[batch_start : batch_start + args.batch_size]
            def solve_one(instance, candidate_seed):
                envs = [
                    DRLTSHardConstraintEnv(
                        instance=instance,
                        n_traj=1,
                        reward_mode="distance",
                        charging_mode="station_power_full",
                        matrix_mode="canonical",
                        info_level="full",
                    )
                ]
                with torch.no_grad():
                    single = rollout(
                        policy,
                        envs,
                        decode_type=args.decode_type,
                        max_steps=max(env.unwrapped.max_steps for env in envs),
                        seed=candidate_seed,
                        soft_constraints=False,
                        incomplete_penalty=float(
                            model_args.get("incomplete_penalty", 100.0)
                        ),
                    )
                return single.infos[0], single.runtime_s
            result = independent_candidate_batch(
                batch_instances,
                candidate_count=args.candidates,
                candidate_chunk_size=args.candidate_chunk_size,
                base_seed=args.seed,
                instance_offset=batch_start,
                solve_one=solve_one,
            )
            for instance, info in zip(batch_instances, result.infos):
                selected, routes, verification = select_min_verified_distance(
                    instance, info
                )
                row = {
                    "instance_id": instance.instance_id,
                    "solver": "DRL-TS",
                    "decode_type": args.decode_type,
                    "candidate_count": args.candidates,
                    "selected_traj_idx": selected,
                    "environment_success": bool(info["success"][selected]),
                    "verifier_passed": bool(verification["passed"]),
                    "objective_distance_km": float(
                        verification["objective_distance_km"]
                    ),
                    "vehicle_count": len(routes),
                    "charging_visit_count": int(
                        verification["charging_visit_count"]
                    ),
                    "runtime_s": result.runtime_s / max(len(batch_instances), 1),
                }
                rows.append(row)
                route_stream.write(
                    json.dumps(
                        {
                            **row,
                            "routes": routes,
                            "violations": verification["violations"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                "instances": len(rows),
                "verifier_passed": sum(row["verifier_passed"] for row in rows),
                "summary": str(summary_path),
                "routes": str(routes_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
