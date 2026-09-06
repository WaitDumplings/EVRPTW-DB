from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel

from ..common import Stage2TaskPool
from ..common.protocol_entrypoints import run_drl_ts
from ..common.training_protocol import (
    add_data_pass_arguments,
    build_adamw_optimizer,
    require_adamw,
)
from ..common.objective import objective_from_args
from ..common.method_auxiliary import method_auxiliary_from_args
from ..common.protocol_trainers import prepare_training_objective
from .env import DRLTSHardConstraintEnv
from .model import DRLTSPolicy
from .rollout import rollout
from .soft_env import DRLTSSoftConstraintEnv
from .soft_env import (
    DEFAULT_SOFT_VIOLATION_COMPONENT_CLIP,
    DEFAULT_SOFT_VIOLATION_STEP_CLIP,
    SOFT_VIOLATION_AGGREGATION,
    SOFT_VIOLATION_APPLICABILITY,
    SOFT_VIOLATION_CONTRACT_ID,
    SOFT_VIOLATION_DENOMINATOR,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DRL-TS baseline.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--scale", default="Cus100")
    parser.add_argument("--split-ids", default="train")
    parser.add_argument("--track-ids", default="train")
    parser.add_argument("--city-slugs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--soft-stage-fraction", type=float, default=0.5)
    parser.add_argument("--soft-stage-end-epoch", type=int)
    parser.add_argument("--batches-per-epoch", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-instance", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--n-encode-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--nearest-neighbors", type=int, default=10)
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--capacity-penalty", type=float, default=1.0)
    parser.add_argument("--time-penalty", type=float, default=1.0)
    parser.add_argument("--energy-penalty", type=float, default=1.0)
    parser.add_argument(
        "--soft-violation-contract-id",
        choices=[SOFT_VIOLATION_CONTRACT_ID],
        default=SOFT_VIOLATION_CONTRACT_ID,
    )
    parser.add_argument(
        "--soft-violation-step-clip",
        type=float,
        default=DEFAULT_SOFT_VIOLATION_STEP_CLIP,
    )
    parser.add_argument(
        "--soft-violation-component-clip",
        type=float,
        default=DEFAULT_SOFT_VIOLATION_COMPONENT_CLIP,
    )
    parser.add_argument(
        "--soft-violation-denominator",
        choices=[SOFT_VIOLATION_DENOMINATOR],
        default=SOFT_VIOLATION_DENOMINATOR,
    )
    parser.add_argument("--incomplete-penalty", type=float, default=100.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--baseline-eval-size", type=int, default=64)
    parser.add_argument("--baseline-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    add_data_pass_arguments(parser)
    return parser.parse_args()


def _soft_violation_kwargs(args):
    return {
        "soft_violation_contract_id": args.soft_violation_contract_id,
        "soft_violation_step_clip": args.soft_violation_step_clip,
        "soft_violation_component_clip": args.soft_violation_component_clip,
        "soft_violation_denominator": args.soft_violation_denominator,
    }


def _configure_soft_auxiliary(args) -> None:
    """Make an optional signed profile authoritative over legacy CLI knobs."""

    profile = method_auxiliary_from_args(args, expected_method="drl_ts")
    if profile is None:
        if getattr(args, "reward_contract", None) is not None:
            raise ValueError(
                "formal DRL-TS reward-contract training requires "
                "--method-auxiliary-profile"
            )
        return
    if getattr(args, "reward_contract", None) is None:
        raise ValueError(
            "DRL-TS formal method auxiliary profile requires a reward contract"
        )
    expected = {
        "profile_id": SOFT_VIOLATION_CONTRACT_ID,
        "applicability": SOFT_VIOLATION_APPLICABILITY,
        "aggregation": SOFT_VIOLATION_AGGREGATION,
        "denominator": SOFT_VIOLATION_DENOMINATOR,
        "weights": {"capacity": 1.0, "energy": 1.0, "time_window": 1.0},
    }
    actual = {
        "profile_id": profile.profile_id,
        "applicability": profile.applicability,
        "aggregation": profile.aggregation,
        "denominator": profile.denominator,
        "weights": dict(profile.weights),
    }
    if actual != expected or profile.step_clip is None or profile.component_clip is None:
        raise ValueError(
            f"unsupported DRL-TS soft auxiliary profile: {actual!r}"
        )
    args.soft_violation_contract_id = profile.profile_id
    args.soft_violation_step_clip = float(profile.step_clip)
    args.soft_violation_component_clip = float(profile.component_clip)
    args.soft_violation_denominator = profile.denominator
    args.capacity_penalty = float(profile.weights["capacity"])
    args.time_penalty = float(profile.weights["time_window"])
    args.energy_penalty = float(profile.weights["energy"])


def _make_envs(
    instances,
    *,
    n_traj: int,
    soft: bool,
    info_level: str,
    objective_config=None,
    soft_violation_kwargs=None,
):
    if not soft:
        return [
            DRLTSHardConstraintEnv(
                instance=instance,
                n_traj=n_traj,
                reward_mode="distance",
                charging_mode="station_power_full",
                matrix_mode="canonical",
                info_level=info_level,
                objective_config=objective_config,
            )
            for instance in instances
        ]
    return [
        DRLTSSoftConstraintEnv(
            instance=instance,
            n_traj=n_traj,
            reward_mode="distance",
            charging_mode="station_power_full",
            matrix_mode="canonical",
            info_level=info_level,
            objective_config=objective_config,
            **{
                key: value
                for key, value in (soft_violation_kwargs or {}).items()
                if key != "soft_violation_component_clip"
            },
        )
        for instance in instances
    ]


def _max_steps(envs) -> int:
    return max(env.unwrapped.max_steps for env in envs)


def _greedy_costs(policy, instances, args, *, soft: bool) -> np.ndarray:
    soft_kwargs = _soft_violation_kwargs(args)
    envs = _make_envs(
        instances,
        n_traj=1,
        soft=soft,
        info_level="light",
        objective_config=objective_from_args(args),
        soft_violation_kwargs=soft_kwargs,
    )
    with torch.no_grad():
        result = rollout(
            policy,
            envs,
            decode_type="greedy",
            max_steps=_max_steps(envs),
            seed=args.seed + 30_000,
            soft_constraints=soft,
            capacity_penalty=args.capacity_penalty,
            time_penalty=args.time_penalty,
            energy_penalty=args.energy_penalty,
            incomplete_penalty=args.incomplete_penalty,
            **soft_kwargs,
        )
    return result.training_cost[:, 0].detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    _configure_soft_auxiliary(args)
    objective_config = prepare_training_objective(args)
    args.objective = objective_config.to_dict()
    if not 0.0 <= args.soft_stage_fraction <= 1.0:
        raise ValueError("soft-stage-fraction must be in [0, 1]")
    if args.soft_stage_end_epoch is not None and args.soft_stage_end_epoch < 0:
        raise ValueError("soft-stage-end-epoch cannot be negative")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool = Stage2TaskPool(
        dataset_path=args.dataset_path,
        family_root=args.family_root,
        scale=args.scale,
        split_ids=args.split_ids,
        track_ids=args.track_ids,
        city_slugs=args.city_slugs,
        seed=args.seed,
        representation=args.training_representation,
        euclidean_manifest=args.euclidean_manifest,
    )
    policy = DRLTSPolicy(
        embedding_dim=args.embedding_dim,
        n_encode_layers=args.n_encode_layers,
        n_heads=args.n_heads,
        nearest_neighbors=args.nearest_neighbors,
        tanh_clipping=args.tanh_clipping,
    ).to(args.device)
    baseline = deepcopy(policy).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    optimizer = build_adamw_optimizer(
        policy.parameters(),
        learning_rate=args.learning_rate,
        weight_decay=require_adamw(args),
    )
    if args.data_passes is not None or args.training_epochs is not None:
        run_drl_ts(args, pool, policy, optimizer)
        return
    baseline_instances = pool.sample(args.baseline_eval_size)
    history_path = args.output_dir / "train_history.jsonl"
    soft_epochs = round(args.epochs * args.soft_stage_fraction)

    for epoch in range(args.epochs):
        soft = epoch < soft_epochs
        policy.train()
        epoch_costs: list[float] = []
        epoch_distances: list[float] = []
        epoch_feasible: list[float] = []
        violations = {"capacity": [], "time": [], "energy": []}
        started = time.perf_counter()
        soft_kwargs = _soft_violation_kwargs(args)
        for batch_index in range(args.batches_per_epoch):
            instances = pool.sample(args.batch_size)
            actor_envs = _make_envs(
                instances,
                n_traj=args.samples_per_instance,
                soft=soft,
                info_level="light",
                objective_config=objective_config,
                soft_violation_kwargs=soft_kwargs,
            )
            actor = rollout(
                policy,
                actor_envs,
                decode_type="sampling",
                max_steps=_max_steps(actor_envs),
                seed=args.seed + epoch * args.batches_per_epoch + batch_index,
                soft_constraints=soft,
                capacity_penalty=args.capacity_penalty,
                time_penalty=args.time_penalty,
                energy_penalty=args.energy_penalty,
                incomplete_penalty=args.incomplete_penalty,
                **soft_kwargs,
            )
            baseline_envs = _make_envs(
                instances,
                n_traj=args.samples_per_instance,
                soft=soft,
                info_level="light",
                objective_config=objective_config,
                soft_violation_kwargs=soft_kwargs,
            )
            with torch.no_grad():
                baseline_result = rollout(
                    baseline,
                    baseline_envs,
                    decode_type="greedy",
                    max_steps=_max_steps(baseline_envs),
                    seed=args.seed + epoch * args.batches_per_epoch + batch_index,
                    soft_constraints=soft,
                    capacity_penalty=args.capacity_penalty,
                    time_penalty=args.time_penalty,
                    energy_penalty=args.energy_penalty,
                    incomplete_penalty=args.incomplete_penalty,
                    **soft_kwargs,
                )
            advantage = (actor.training_cost - baseline_result.training_cost).detach()
            loss = (advantage * actor.log_likelihood).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            epoch_costs.append(float(actor.training_cost.mean().detach().cpu()))
            epoch_distances.append(
                float(actor.objective_distance_km.mean().detach().cpu())
            )
            epoch_feasible.append(
                float(actor.feasible.float().mean().detach().cpu())
            )
            violations["capacity"].append(
                float(actor.capacity_violation.mean().detach().cpu())
            )
            violations["time"].append(
                float(actor.time_violation.mean().detach().cpu())
            )
            violations["energy"].append(
                float(actor.energy_violation.mean().detach().cpu())
            )

        policy.eval()
        actor_costs = _greedy_costs(
            policy,
            baseline_instances,
            args,
            soft=soft,
        )
        baseline_costs = _greedy_costs(
            baseline,
            baseline_instances,
            args,
            soft=soft,
        )
        test = ttest_rel(actor_costs, baseline_costs, alternative="less")
        baseline_updated = bool(
            np.mean(actor_costs) < np.mean(baseline_costs)
            and np.isfinite(test.pvalue)
            and float(test.pvalue) < args.baseline_alpha
        )
        if baseline_updated:
            baseline.load_state_dict(policy.state_dict())
            baseline_instances = pool.sample(args.baseline_eval_size)
        row = {
            "epoch": epoch,
            "training_stage": "soft" if soft else "hard",
            "training_cost": float(np.mean(epoch_costs)),
            "objective_mode": objective_config.mode,
            "objective_unit": objective_config.unit,
            "objective_distance_km": float(np.mean(epoch_distances)),
            "feasible_rate": float(np.mean(epoch_feasible)),
            "capacity_violation_normalized": float(
                np.mean(violations["capacity"])
            ),
            "time_violation_normalized": float(np.mean(violations["time"])),
            "energy_violation_normalized": float(
                np.mean(violations["energy"])
            ),
            "paired_t_pvalue": float(test.pvalue),
            "baseline_updated": baseline_updated,
            "epoch_runtime_s": float(time.perf_counter() - started),
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        torch.save(
            {
                "method": "DRL-TS",
                "epoch": epoch,
                "training_stage": row["training_stage"],
                "model": policy.state_dict(),
                "baseline": baseline.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "objective_config": objective_config.to_dict(),
            },
            args.output_dir / "checkpoint_latest.pt",
        )
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
