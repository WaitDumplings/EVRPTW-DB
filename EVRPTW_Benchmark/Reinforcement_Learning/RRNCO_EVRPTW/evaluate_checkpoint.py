from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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



_RRNCO_DEFAULTS = {
    "graph_mode": "full",
    "aft_mode": "legacy",
    "distance_sampling": "random",
    "relation_chunk_size": 0,
    "checkpoint_bias": False,
    "relation_temperature": math.exp(5.0),
    "reinforce_baseline": "paper",
}


def _checkpoint_args(payload: dict[str, Any]) -> SimpleNamespace:
    """Restore declared semantics; old checkpoints retain their legacy defaults."""
    values = dict(payload["args"])
    signature = payload.get("resolved_training_signature") or values.get("resolved_training_signature") or {}
    method_fields = signature.get("method_specific") or {}
    aliases = {"encoder_layers": "n_encode_layers", "heads": "n_heads"}
    for key, value in method_fields.items():
        option = aliases.get(key, key)
        if value is None:
            continue
        if option in _RRNCO_DEFAULTS and option in values and values[option] != value:
            raise ValueError(f"checkpoint args/signature disagree on {option}")
        values.setdefault(option, value)
    aliases = {"training_trajectory_count": "samples_per_instance"}
    for key, value in signature.items():
        if key not in {"method_specific", "schema", "sha256"} and value is not None:
            values.setdefault(aliases.get(key, key), value)
    if payload["method"] == "RRNCO-EV":
        for key, value in _RRNCO_DEFAULTS.items():
            values.setdefault(key, value)
    values.setdefault("training_representation", "G")
    values.setdefault("euclidean_manifest", None)
    return SimpleNamespace(**values)


def _evaluation_metadata(
    payload: dict[str, Any], saved: SimpleNamespace, cli: argparse.Namespace,
) -> dict[str, Any]:
    """Make a standalone export sufficient to audit paired evaluations."""
    signature = payload.get("resolved_training_signature") or getattr(saved, "resolved_training_signature", None)
    stream = payload.get("training_stream_contract") or getattr(saved, "training_stream_contract_snapshot", None)
    reward = payload.get("reward_contract") or getattr(saved, "reward_contract_snapshot", None)
    rrnco = payload["method"] == "RRNCO-EV"
    metadata = {
        "schema": "rrnco_ev_checkpoint_validation_v2",
        "method": str(payload["method"]),
        "checkpoint": str(cli.checkpoint.resolve()),
        "protocol_id": payload.get("protocol_id") or getattr(saved, "protocol_id", None),
        "scale": saved.scale,
        "split": "validation",
        "split_ids": "val",
        "track_ids": "validation",
        "decode_type": "sampling",
        "validation_decode_type": "sampling",
        "candidate_count": cli.candidates,
        "validation_candidates": cli.candidates,
        "validation_limit": cli.limit,
        "validation_seed": cli.seed,
        "validation_rollout_steps": int(saved.validation_rollout_steps),
        "validation_dataset_path": str(cli.dataset_path.resolve()),
        "validation_family_root": str(cli.family_root.resolve()),
        "training_representation": saved.training_representation,
        "training_rollout_steps": getattr(saved, "training_rollout_steps", None),
        "training_seed": getattr(saved, "seed", None),
        "seed": getattr(saved, "seed", None),
        "logical_epoch": payload.get("logical_epoch"),
        "optimizer_steps": payload.get("optimizer_steps"),
        "customer_exposures": payload.get("customer_exposures", payload.get("observed_customer_exposures")),
        "training_epochs": getattr(saved, "training_epochs", None),
        "training_trajectory_count": getattr(saved, "samples_per_instance", None),
        "physical_batch_size": getattr(saved, "physical_batch_size", None),
        "effective_batch_size": getattr(saved, "effective_batch_size", None),
        "reinforce_baseline": getattr(saved, "reinforce_baseline", "paper"),
        "reward_contract_sha256": (reward or {}).get("sha256") or getattr(saved, "reward_contract_sha256", None),
        "training_stream_contract_sha256": (stream or {}).get("sha256") or getattr(saved, "training_stream_contract_sha256", None),
        "training_stream_contract_snapshot": stream,
        "resolved_training_signature": signature,
        "resolved_training_signature_sha256": (signature or {}).get("sha256") or getattr(saved, "resolved_training_signature_sha256", None),
    }
    for key in _RRNCO_DEFAULTS:
        if key != "reinforce_baseline":
            metadata[key] = getattr(saved, key) if rrnco else None
    return metadata


def _policy(method: str, args: SimpleNamespace) -> torch.nn.Module:
    if method == "RRNCO-EV":
        return RRNCOEVPolicy(
            embedding_dim=args.embedding_dim,
            n_encode_layers=args.n_encode_layers,
            n_heads=args.n_heads,
            feedforward_hidden=args.feedforward_hidden,
            distance_sample_size=args.distance_sample_size,
            tanh_clipping=args.tanh_clipping,
            graph_mode=getattr(args, "graph_mode", "full"),
            aft_mode=getattr(args, "aft_mode", "legacy"),
            distance_sampling=getattr(args, "distance_sampling", "random"),
            relation_chunk_size=getattr(args, "relation_chunk_size", 0),
            checkpoint_bias=getattr(args, "checkpoint_bias", False),
            relation_temperature=getattr(args, "relation_temperature", math.exp(5.0)),
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
    saved = _checkpoint_args(payload)
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
    summary.update(_evaluation_metadata(payload, saved, cli))
    summary["validation_wall_time_s"] = time.perf_counter() - started
    atomic_json(cli.output, summary)
    print(
        f"{method}: feasible={summary['complete_and_feasible']}/{summary['instances']} "
        f"objective={summary['mean_verified_objective']} "
        f"distance_km={summary['mean_verified_distance_km']}"
    )


if __name__ == "__main__":
    main()
