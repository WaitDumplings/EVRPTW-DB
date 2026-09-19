"""Bounded CPU engineering comparison; never a formal benchmark training run.

Fresh sum/mean policies share initialization, training instances and seeds.
The fixed diagnostic cohort is disjoint from updates and belongs to TRAIN.
Validation/test are never consumed. A short change in cost is not convergence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from diagnose_evrptw_policy import (
    REPO, checkpoint_args, diagnostic_envs, diagnose_policy, file_sha256,
    training_pool, write_report,
)
import torch
from EVRPTW_Benchmark.Reinforcement_Learning.common.method_auxiliary import method_auxiliary_from_args
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import ObjectiveConfig
from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import reward_contract_from_args
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import verified_validation
from EVRPTW_Benchmark.Reinforcement_Learning.common.route_info import finalize_route_infos
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.model import EVRPTWRLPolicy
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.rollout import rollout


def frozen_rewards(values):
    args = SimpleNamespace(**deepcopy(values))
    # Consume the self-verifying embedded contracts, not machine-specific paths.
    args.reward_contract = None
    args.method_auxiliary_profile = None
    reward = reward_contract_from_args(args, objective=values["objective"], scale=values["scale"])
    auxiliary = method_auxiliary_from_args(args, expected_method="evrptw_rl")
    if reward is None or auxiliary is None:
        raise ValueError("probe requires the formal reward and station auxiliary snapshots")
    return reward, auxiliary


def fixed_cohort(policy, instances, values, reward, auxiliary, args):
    policy.eval()
    def solve(instance, seed):
        envs = diagnostic_envs([instance], values, args.n_traj)
        result = rollout(
            policy, envs, decode_type="sampling", max_steps=args.eval_horizon,
            seed=seed, compute_log_likelihood=False, reward_contract=reward,
            method_auxiliary_profile=auxiliary, station_visit_penalty=0.3,
        )
        return finalize_route_infos(envs, result.infos)[0]
    report = verified_validation(
        instances, solve, seed=args.seed + 910_000_000,
        objective_config=ObjectiveConfig(**values["objective"]),
    )
    report.update({"split": "train_diagnostic_held_out_from_probe_updates",
                   "not_formal_validation": True, "n_traj": args.n_traj,
                   "rollout_horizon": args.eval_horizon})
    return report


def run_source(checkpoint, args):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = checkpoint_args(payload)
    pool = training_pool(values)
    count = args.cohort_size + args.updates * args.batch_size
    tasks = pool.tasks[:count]
    if len(tasks) != count:
        raise ValueError("insufficient train instances for disjoint diagnostic/update cohorts")
    instances = [pool.instance(task) for task in tasks]
    cohort = instances[:args.cohort_size]
    training = instances[args.cohort_size:]
    assert not ({item.instance_id for item in cohort} & {item.instance_id for item in training})
    reward, auxiliary = frozen_rewards(values)
    reports = []
    for aggregation in args.aggregations.split(","):
        arm_started = time.perf_counter()
        torch.manual_seed(args.seed)
        policy = EVRPTWRLPolicy(
            embedding_dim=int(values.get("embedding_dim", 128)),
            structure2vec_rounds=int(values.get("structure2vec_rounds", 3)),
            graph_aggregation=aggregation,
        )
        policy.activation_checkpoint_stride = args.activation_checkpoint_stride
        initial = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        optimizer = torch.optim.AdamW(policy.parameters(), lr=float(values.get("learning_rate", 1e-3)),
                                      weight_decay=float(values.get("optimizer_weight_decay", 0.01)))
        before = fixed_cohort(policy, cohort, values, reward, auxiliary, args)
        before_diagnostics = diagnose_policy(
            policy, cohort, values, n_traj=args.diagnostic_n_traj,
            state_steps=(0, 10, 50, 100, 180, 239), max_steps=240, seed=args.seed + 910_000_000,
        )
        ema_cost = None
        rows = []
        for update in range(1, args.updates + 1):
            started = time.perf_counter()
            batch = training[(update - 1) * args.batch_size:update * args.batch_size]
            seed = args.seed + 10_000_000 + update * 1000
            torch.manual_seed(seed)
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            envs = diagnostic_envs(batch, values, args.n_traj)
            actor = rollout(
                policy, envs, decode_type="sampling", max_steps=args.train_horizon,
                seed=seed, reward_contract=reward, method_auxiliary_profile=auxiliary,
                station_visit_penalty=0.3,
            )
            observed = float(actor.training_cost.mean().detach())
            ema_cost = observed if ema_cost is None else 0.9 * ema_cost + 0.1 * observed
            advantage = (actor.training_cost - ema_cost).detach()
            loss = (advantage * actor.log_likelihood).mean()
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0))
            encoder_sq = sum(float(p.grad.square().sum()) for name, p in policy.named_parameters()
                             if p.grad is not None and name.split(".")[0] in {
                                 "local_projection", "global_projection", "neighbor_projection",
                                 "edge_direction", "edge_projection"})
            if not math.isfinite(grad) or not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite loss or gradient in CPU probe")
            optimizer.step()
            row = {"update": update, "loss": float(loss.detach()),
                   "training_cost": observed, "pre_clip_gradient_norm": grad,
                   "post_clip_encoder_gradient_norm": math.sqrt(encoder_sq),
                   "advantage_std": float(advantage.std(unbiased=False)),
                   "training_feasible_rate": float(actor.feasible.float().mean()),
                   "rollout_budget_exhausted_rate": float(actor.rollout_budget_exhausted.float().mean()),
                   "wall_time_s": time.perf_counter() - started}
            rows.append(row)
            print(json.dumps({"source": values["training_representation"], "aggregation": aggregation, **row}), flush=True)
            del actor, loss, advantage
        after = fixed_cohort(policy, cohort, values, reward, auxiliary, args)
        after_diagnostics = diagnose_policy(
            policy, cohort, values, n_traj=args.diagnostic_n_traj,
            state_steps=(0, 10, 50, 100, 180, 239), max_steps=240, seed=args.seed + 910_000_000,
        )
        current = policy.state_dict()
        delta = math.sqrt(sum(float((current[k] - initial[k]).square().sum()) for k in initial))
        # Compare each fixed instance's outcome, not just rounded aggregate cost.
        before_rows = [(r["instance_id"], r["verifier_passed"], r["objective_value"]) for r in before["rows"]]
        after_rows = [(r["instance_id"], r["verifier_passed"], r["objective_value"]) for r in after["rows"]]
        report = {
            "schema": "evrptw_stability_cpu_probe_v1", "device": "cpu",
            "interpretation": "short_engineering_probe_not_convergence_or_benchmark_advantage",
            "initialization": "fresh_identical_seed_weights_not_profile_checkpoint_weights",
            "profile_checkpoint": str(checkpoint), "profile_checkpoint_sha256": file_sha256(checkpoint),
            "graph_aggregation": aggregation, "seed": args.seed,
            "training_index": str(pool.dataset_path), "training_index_sha256": file_sha256(Path(pool.dataset_path)),
            "training_instance_ids": [item.instance_id for item in training],
            "diagnostic_instance_ids": [item.instance_id for item in cohort],
            "cohort_disjoint_from_probe_updates": True, "consumed_validation_or_test": False,
            "batch_size": args.batch_size, "n_traj": args.n_traj,
            "train_horizon": args.train_horizon, "eval_horizon": args.eval_horizon,
            "activation_checkpoint_stride": args.activation_checkpoint_stride,
            "baseline": "EMA_only_fewer_than_native_1000_warmup_updates",
            "reward_contract": reward.to_dict(), "method_auxiliary_profile": auxiliary.to_dict(),
            "updates": rows, "parameter_delta_l2": delta,
            "before": before, "after": after,
            "fixed_cohort_outcomes_changed": before_rows != after_rows,
            "before_diagnostics": before_diagnostics, "after_diagnostics": after_diagnostics,
            "numerical_gate_passed": after_diagnostics["gate"]["passed"],
            "behavior_gate_passed": before_rows != after_rows,
            "model_source_sha256": file_sha256(REPO / "EVRPTW_Benchmark/Reinforcement_Learning/EVRPTW_RL/model.py"),
            "probe_source_sha256": file_sha256(Path(__file__)),
            "wall_time_s": time.perf_counter() - arm_started,
        }
        name = f"{values['training_representation']}_{aggregation}_t{args.n_traj}_u{args.updates}"
        write_report(report, args.output_dir / f"{name}.json")
        reports.append({"name": name, "numerical_gate_passed": report["numerical_gate_passed"],
                        "behavior_gate_passed": report["behavior_gate_passed"], "wall_time_s": report["wall_time_s"]})
        del policy, optimizer
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregations", default="sum,mean")
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-traj", type=int, default=30)
    parser.add_argument("--diagnostic-n-traj", type=int, default=3)
    parser.add_argument("--cohort-size", type=int, default=3)
    parser.add_argument("--train-horizon", type=int, default=240)
    parser.add_argument("--eval-horizon", type=int, default=360)
    parser.add_argument("--activation-checkpoint-stride", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    if (not 1 <= args.updates < 1000 or args.train_horizon != 240 or args.eval_horizon != 360
            or min(args.batch_size, args.n_traj, args.cohort_size, args.diagnostic_n_traj, args.cpu_threads) < 1
            or not set(args.aggregations.split(",")).issubset({"sum", "mean"})):
        parser.error("use 1..999 updates, positive counts and the registered H240/H360")
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for checkpoint in args.profile_checkpoint:
        reports.extend(run_source(checkpoint, args))
    write_report({"arms": reports}, args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
