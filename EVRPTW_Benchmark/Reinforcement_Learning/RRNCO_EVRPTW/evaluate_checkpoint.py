from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from ..AM_EVRPTW.model import AMEVRPTWPolicy
from ..AM_EVRPTW.rollout import rollout as am_rollout
from ..common.objective import objective_from_checkpoint
from ..common.reward_contract import reward_contract_from_args
from ..common.route_info import finalize_route_infos
from ..common.stage2_data import Stage2TaskPool, make_envs
from ..common.training_protocol import atomic_json, verified_validation
from .model import RRNCOEVPolicy
from .rollout import rollout as rrnco_rollout


def _policy(method: str, args: SimpleNamespace) -> torch.nn.Module:
    if method == "RRNCO-EV":
        return RRNCOEVPolicy(
            embedding_dim=args.embedding_dim,
            n_encode_layers=args.n_encode_layers,
            n_heads=args.n_heads,
            feedforward_hidden=args.feedforward_hidden,
            distance_sample_size=args.distance_sample_size,
            tanh_clipping=args.tanh_clipping,
        )
    if method == "AM-EVRPTW":
        return AMEVRPTWPolicy(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.embedding_dim,
            n_encode_layers=args.n_encode_layers,
            n_heads=args.n_heads,
            tanh_clipping=args.tanh_clipping,
        )
    raise ValueError(f"unsupported comparison method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only canonical validation of an AM or RRNCO-EV checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=910001234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()

    payload = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    method = str(payload["method"])
    saved = SimpleNamespace(**payload["args"])
    objective = objective_from_checkpoint(payload)
    reward_contract = reward_contract_from_args(
        saved, objective=objective, scale=saved.scale
    )
    policy = _policy(method, saved).to(cli.device)
    policy.load_state_dict(payload["model"])
    policy.eval()
    pool = Stage2TaskPool(
        dataset_path=cli.dataset_path,
        family_root=cli.family_root,
        scale=saved.scale,
        split_ids="val",
        track_ids="validation",
        seed=cli.seed,
        representation=saved.training_representation,
        euclidean_manifest=saved.euclidean_manifest,
    )
    instances = list(pool.first(limit=cli.limit))
    started = time.perf_counter()

    def solve(instance, seed: int):
        envs = make_envs(
            [instance],
            n_traj=cli.candidates,
            info_level="light",
            reward_distance_scale_km=float(saved.reward_distance_scale_km),
            reward_objective_scale=(
                reward_contract.objective_scale if reward_contract else None
            ),
            invalid_action_penalty=0.0,
            objective_config=objective,
        )
        if method == "RRNCO-EV":
            result = rrnco_rollout(
                policy,
                envs,
                decode_type="sampling",
                max_steps=int(saved.validation_rollout_steps),
                seed=seed,
                incomplete_penalty=float(saved.incomplete_penalty),
                compute_log_likelihood=False,
                reward_contract=reward_contract,
            )
        else:
            result = am_rollout(
                policy,
                envs,
                decode_type="sampling",
                max_steps=int(saved.validation_rollout_steps),
                seed=seed,
                incomplete_penalty_km=float(saved.incomplete_penalty_km),
                compute_log_likelihood=False,
                reward_contract=reward_contract,
            )
        result.infos = finalize_route_infos(envs, result.infos)
        return result.infos[0]

    summary = verified_validation(
        instances, solve, seed=cli.seed, objective_config=objective
    )
    summary.update(
        {
            "schema": "rrnco_ev_checkpoint_validation_v1",
            "method": method,
            "checkpoint": str(cli.checkpoint.resolve()),
            "candidate_count": cli.candidates,
            "validation_seed": cli.seed,
            "validation_wall_time_s": time.perf_counter() - started,
        }
    )
    atomic_json(cli.output, summary)
    print(
        f"{method}: feasible={summary['complete_and_feasible']}/{summary['instances']} "
        f"objective={summary['mean_verified_objective']} "
        f"distance_km={summary['mean_verified_distance_km']}"
    )


if __name__ == "__main__":
    main()
