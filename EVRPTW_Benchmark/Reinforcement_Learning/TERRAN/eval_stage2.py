from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.common.evaluation import select_min_verified_distance
from EVRPTW_Benchmark.Reinforcement_Learning.common.candidate_protocol import independent_candidate_batch

from ..common import Stage2TaskPool
from .env_factory import make_terran_env
from .models import Agent
from .rollout import rollout_eval_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TERRAN on frozen Stage-2 benchmark views."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", default="Cus100")
    parser.add_argument("--split-ids", default="test")
    parser.add_argument("--track-ids", default="test1_new_seed")
    parser.add_argument("--city-slugs")
    parser.add_argument("--representation", choices=("E", "G"), default="G")
    parser.add_argument("--euclidean-manifest", type=Path)
    parser.add_argument("--decode-mode", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--candidate-chunk-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.decode_mode == "greedy" and args.candidates != 1:
        raise ValueError("greedy evaluation has exactly one candidate")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=args.device,
        weights_only=False,
    )
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    agent = Agent(
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        tanh_clipping=float(model_cfg.get("tanh_clipping", 15.0)),
        n_encode_layers=int(model_cfg.get("n_encode_layers", 3)),
        device=args.device,
        use_graph_token=bool(model_cfg.get("use_graph_token", False)),
        use_dynamic_embedding=bool(model_cfg.get("use_dynamic_embedding", False)),
    ).to(args.device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    pool = Stage2TaskPool(
        dataset_path=args.dataset_path,
        family_root=args.family_root,
        scale=args.scale,
        split_ids=args.split_ids,
        track_ids=args.track_ids,
        city_slugs=args.city_slugs,
        seed=args.seed,
        representation=args.representation,
        euclidean_manifest=args.euclidean_manifest,
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
                    make_terran_env(
                        instance=instance,
                        n_traj=1,
                        reward_mode="distance",
                        charging_mode="station_power_full",
                        matrix_mode="canonical",
                        info_level="full",
                    )
                ]
                single_rows = rollout_eval_batch(
                    agent,
                    envs,
                    decode_mode=args.decode_mode,
                    max_steps=max(env.unwrapped.max_steps for env in envs),
                    device=args.device,
                    seed=candidate_seed,
                    include_routes=True,
                    return_final_info=True,
                )
                result = single_rows[0]
                return result.pop("_final_info"), float(result["runtime_s"])
            candidate_batch = independent_candidate_batch(
                batch_instances,
                candidate_count=args.candidates,
                candidate_chunk_size=args.candidate_chunk_size,
                base_seed=args.seed,
                instance_offset=batch_start,
                solve_one=solve_one,
            )
            result_rows = [
                {"_final_info": info, "runtime_s": candidate_batch.runtime_s / max(len(batch_instances), 1)}
                for info in candidate_batch.infos
            ]
            for instance, result in zip(batch_instances, result_rows):
                selected, routes, verification = select_min_verified_distance(
                    instance, result.pop("_final_info")
                )
                row = {
                    "instance_id": instance.instance_id,
                    "solver": "TERRAN",
                    "decode_mode": args.decode_mode,
                    "candidate_count": args.candidates,
                    "selected_traj_idx": selected,
                    "environment_success": bool(verification["passed"]),
                    "verifier_passed": bool(verification["passed"]),
                    "objective_distance_km": float(
                        verification["objective_distance_km"]
                    ),
                    "vehicle_count": len(routes),
                    "charging_visit_count": int(
                        verification["charging_visit_count"]
                    ),
                    "runtime_s": float(result["runtime_s"]),
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
    _write_csv(summary_path, rows)
    print(
        json.dumps(
            {
                "instances": len(rows),
                "verifier_passed": int(
                    np.sum([row["verifier_passed"] for row in rows])
                ),
                "summary": str(summary_path),
                "routes": str(routes_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
