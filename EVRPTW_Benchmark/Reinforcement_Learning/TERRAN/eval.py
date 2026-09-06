from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT))

from evrptw_core.io import iter_instances
from evrptw_core.schema import merge_route_sequences

from ..common.evaluation import select_min_verified_objective
from ..common.objective import objective_from_checkpoint
from ..common.action_constraints import (
    ACTION_CONSTRAINT_CONTRACT_ID, require_checkpoint_action_contract,
)
from .env_factory import make_terran_env
from .models import Agent
from .rollout import rollout_eval_batch


def _eval_instance_batches(
    eval_path: Path,
    num_customers: int,
    num_charging_stations: int,
    batch_size: int,
    limit: int | None = None,
    num_batches_limit: int | None = None,
):
    max_count = None if limit is None else int(limit)
    if num_batches_limit is not None:
        by_batches = max(1, int(batch_size)) * int(num_batches_limit)
        max_count = by_batches if max_count is None else min(max_count, by_batches)
    batch = []
    seen = 0
    for instance in iter_instances(eval_path, num_customers=num_customers, num_charging_stations=num_charging_stations):
        if max_count is not None and seen >= max_count:
            break
        batch.append(instance)
        seen += 1  # noqa: SIM113 - count is checked before accepting each item
        if len(batch) >= max(1, int(batch_size)):
            yield batch
            batch = []
    if batch:
        yield batch


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TERRAN checkpoint with sample best-of-n_traj decoding.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--objective-config", type=Path)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--solver-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-customers", type=int, default=15)
    parser.add_argument("--num-charging-stations", type=int, default=3)
    parser.add_argument("--n-traj", type=int, default=100)
    parser.add_argument("--decode-mode", type=str, default="sample", choices=["sample", "greedy"])
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-num-batches", type=int, default=None)
    parser.add_argument("--info-level", type=str, choices=["light", "full"], default="full")
    parser.add_argument("--save-routes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    require_checkpoint_action_contract(checkpoint)
    cfg = checkpoint.get("config", {})
    objective_config = objective_from_checkpoint(checkpoint, args.objective_config)
    model_cfg = cfg.get("model", {})
    solver_name = args.solver_name or str(cfg.get("run_name", "TERRAN"))
    agent = Agent(
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        tanh_clipping=float(model_cfg.get("tanh_clipping", 15.0)),
        n_encode_layers=int(model_cfg.get("n_encode_layers", 3)),
        device=device,
        use_graph_token=bool(model_cfg.get("use_graph_token", False)),
        use_dynamic_embedding=bool(model_cfg.get("use_dynamic_embedding", False)),
    ).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    rows = []
    batch_size = max(1, int(args.eval_batch_size))
    env_info_level = "full" if args.save_routes else args.info_level
    seen_before_batch = 0
    for instances in _eval_instance_batches(
        args.eval_path,
        args.num_customers,
        args.num_charging_stations,
        batch_size,
        args.limit,
        args.eval_num_batches,
    ):
        eval_env_cfg = dict(cfg.get("env", {}) or {})
        eval_env_cfg["info_level"] = env_info_level
        eval_env_cfg["objective_config"] = objective_config
        envs = [
            make_terran_env(
                instance=instance,
                n_traj=args.n_traj,
                **eval_env_cfg,
            )
            for instance in instances
        ]
        batch_rows = rollout_eval_batch(
            agent,
            envs,
            decode_mode=args.decode_mode,
            max_steps=args.max_steps,
            device=device,
            seed=args.seed + seen_before_batch,
            include_routes=args.save_routes,
            return_final_info=objective_config.is_cost,
        )
        for instance, row in zip(instances, batch_rows):
            if objective_config.is_cost:
                selected, routes, verification = select_min_verified_objective(instance, row.pop("_final_info"), objective_config)
                row.update(verification)
                row.update(selected_traj_idx=selected, feasible=bool(verification["passed"]), vehicle_count=len(routes))
                if args.save_routes:
                    row["routes_json"] = json.dumps(routes)
                    row["route_sequence_json"] = json.dumps(merge_route_sequences(routes))
            row.update(
                {
                    "instance_id": instance.instance_id,
                    "solver_name": solver_name,
                    "seed": args.seed,
                    "checkpoint": str(args.checkpoint_path),
                    "decode_mode": args.decode_mode,
                    "n_traj": args.n_traj,
                    "eval_batch_size": batch_size,
                    "eval_info_level": env_info_level,
                    "save_routes": args.save_routes,
                }
            )
        rows.extend(batch_rows)
        seen_before_batch += len(instances)

    if not rows:
        raise FileNotFoundError(f"No EVRPTW instances found under {args.eval_path}")

    feasible_rows = [row for row in rows if row["feasible"]]
    summary = {
        "action_constraint_contract_id": ACTION_CONSTRAINT_CONTRACT_ID,
        "solver_name": solver_name,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint_path),
        "decode_mode": args.decode_mode,
        "n_traj": args.n_traj,
        "eval_batch_size": max(1, int(args.eval_batch_size)),
        "eval_num_batches": int(np.ceil(len(rows) / max(1, int(args.eval_batch_size)))),
        "eval_info_level": env_info_level,
        "save_routes": args.save_routes,
        "num_instances": len(rows),
        "feasible_rate": float(np.mean([row["feasible"] for row in rows])),
        "avg_objective_distance_km": float(np.mean([row["objective_distance_km"] for row in feasible_rows])) if feasible_rows else float("nan"),
        "avg_vehicle_count": float(np.mean([row["vehicle_count"] for row in feasible_rows])) if feasible_rows else float("nan"),
        "avg_runtime_s": float(np.mean([row["runtime_s"] for row in rows])),
        "objective_mode": objective_config.mode,
        "objective_unit": objective_config.unit,
        "avg_objective": float(np.mean([row.get("objective_value", row["objective_distance_km"]) for row in feasible_rows])) if feasible_rows else float("nan"),
        **{
            f"avg_{key}": float(np.mean([row[key] for row in feasible_rows])) if feasible_rows and objective_config.is_cost else None
            for key in ("objective_cost_usd", "electricity_cost_usd", "vehicle_cost_usd")
        },
    }
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "EVRPTW_Benchmark/results/AC_v1" / f"Cus_{args.num_customers}" / f"CS_{args.num_charging_stations}" / solver_name
    _write_csv(output_dir / "terran_routes.csv", rows)
    _write_csv(output_dir / "terran_summary.csv", [summary])
    print(summary)


if __name__ == "__main__":
    main()
