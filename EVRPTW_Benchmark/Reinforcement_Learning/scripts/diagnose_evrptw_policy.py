"""CPU-only, read-only checkpoint diagnostics on a fixed TRAIN-only cohort.

No optimizer is constructed. Gradients are local single-action sensitivity
probes with the incoming recurrent state detached, not full rollout gradients.
Reports are engineering diagnostics, never validation/test benchmark scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "EVRPTW_Core"))

import numpy as np
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.rollout import stack_observations
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import ObjectiveConfig
from EVRPTW_Benchmark.Reinforcement_Learning.common.stage2_data import Stage2TaskPool, make_envs
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_diagnostics import summarize_values
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.model import EVRPTWRLPolicy, RecurrentState
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.rollout import _normalized_travel_time


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_args(payload: dict[str, Any]) -> dict[str, Any]:
    values = payload.get("args", {})
    return dict(values if isinstance(values, dict) else vars(values))


def training_pool(values: dict[str, Any], *, dataset_path=None, family_root=None,
                  dataset_root=None) -> Stage2TaskPool:
    source = Path(dataset_path or values["dataset_path"])
    families = family_root or values.get("family_root")
    if dataset_root is not None:
        root = Path(dataset_root)
        road = "generation_plan" in Path(values["dataset_path"]).parts
        if dataset_path is None:
            source = root / ("generation_plan/core/train/view_index.parquet" if road
                             else "train/view_index.parquet")
        if family_root is None:
            families = root / "materialized/families" if road else None
    # Guard against accidentally loading a formal evaluation cohort even if
    # a caller overrides an original training checkpoint's data location.
    if any(part.lower() in {"val", "validation", "test", "testing"} for part in source.parts):
        raise ValueError("diagnostics must use a TRAIN index, not validation/test")
    pool = Stage2TaskPool(
        dataset_path=source, family_root=families, scale=values["scale"],
        split_ids="train", track_ids="train", city_slugs=values.get("city_slugs"),
        representation=values.get("training_representation", "G"),
        euclidean_manifest=values.get("euclidean_manifest"),
        seed=int(values.get("seed", 1234)), cache_size=64,
    )
    if any(task.split_id != "train" or task.track_id != "train" for task in pool.tasks):
        raise ValueError("all diagnostic instances must have train split and track")
    return pool


def policy_from_checkpoint(payload: dict[str, Any], aggregation=None) -> EVRPTWRLPolicy:
    values = checkpoint_args(payload)
    policy = EVRPTWRLPolicy(
        embedding_dim=int(values.get("embedding_dim", 128)),
        structure2vec_rounds=int(values.get("structure2vec_rounds", 3)),
        graph_aggregation=aggregation or values.get("graph_aggregation", "sum"),
    )
    policy.load_state_dict(payload["model"], strict=True)
    return policy.cpu().eval()


def diagnostic_envs(instances, values, n_traj, *, use_jit_mask=True):
    return make_envs(
        instances, n_traj=n_traj, info_level="light", use_jit_mask=use_jit_mask,
        objective_config=ObjectiveConfig(**values["objective"]),
        reward_objective_scale=values.get("reward_objective_scale"),
        reward_distance_scale_km=values.get("reward_distance_scale_km"),
        invalid_action_penalty=0.0,
    )


def state_probe(policy, batch, travel, state, active) -> dict[str, Any]:
    projections: dict[str, torch.Tensor] = {}
    handles = []
    for name in ("context_projection", "choice_projection"):
        handles.append(getattr(policy, name).register_forward_hook(
            lambda module, inputs, output, name=name:
                projections.__setitem__(name, output.detach())
        ))
    policy.zero_grad(set_to_none=True)
    detached_state = RecurrentState(state.hidden.detach(), state.cell.detach())
    try:
        with torch.enable_grad():
            logits, _ = policy.logits(batch, travel, detached_state)
            legal = torch.as_tensor(batch["action_mask"], dtype=torch.bool)
            decisions = torch.as_tensor(active, dtype=torch.bool) & (legal.sum(-1) > 1)
            finite = bool(torch.isfinite(logits[legal]).all())
            spans, entropies = [], []
            if finite:
                distribution = torch.distributions.Categorical(logits=logits)
                for b, t in torch.argwhere(decisions).tolist():
                    values = logits[b, t, legal[b, t]]
                    spans.append(float((values.max() - values.min()).detach()))
                    entropies.append(float(distribution.entropy()[b, t].detach())
                                     / math.log(values.numel()))
                if bool(decisions.any()):
                    # First legal action avoids consuming diagnostic rollout RNG.
                    action = legal.long().argmax(dim=-1)
                    (-distribution.log_prob(action)[decisions].mean()).backward()
        gradients = {
            name: None if parameter.grad is None else float(parameter.grad.norm())
            for name, parameter in policy.named_parameters()
        }
        module_squared: dict[str, float] = {}
        for name, norm in gradients.items():
            if norm is not None:
                module = name.split(".")[0]
                module_squared[module] = module_squared.get(module, 0.0) + norm * norm
        encoder = {"local_projection", "global_projection", "neighbor_projection",
                   "edge_direction", "edge_projection"}
        projection_rows = {}
        for name, values in projections.items():
            values = values[decisions]
            if not values.numel():
                projection_rows[name] = {"count": 0}
                continue
            projection_rows[name] = {
                "count": values.numel(), "max_abs": float(values.abs().max()),
                "fraction_abs_gt_5": float((values.abs() > 5).float().mean()),
                "tanh_equal_to_first_node_fraction": float(
                    (torch.tanh(values) == torch.tanh(values[:, :1, :])).float().mean()
                ),
            }
        return {
            "decision_trajectories": int(decisions.sum()),
            "finite_legal_logits": finite,
            "legal_logit_span": summarize_values(spans),
            "normalized_entropy": summarize_values(entropies),
            "gradient_scope": "first_legal_action_negative_logprob_mean_detached_incoming_recurrence",
            "parameter_gradient_norms": gradients,
            "module_gradient_norms": {k: math.sqrt(v) for k, v in module_squared.items()},
            "encoder_gradient_norm": math.sqrt(sum(v for k, v in module_squared.items() if k in encoder)),
            "finite_gradients": all(v is None or math.isfinite(v) for v in gradients.values()),
            "projections": projection_rows,
        }
    finally:
        for handle in handles:
            handle.remove()
        policy.zero_grad(set_to_none=True)


def diagnose_policy(policy, instances, values, *, n_traj=30, state_steps=(0, 10, 50, 100, 180, 239),
                    max_steps=240, seed=910001234, use_jit_mask=True) -> dict[str, Any]:
    previous_training = policy.training
    policy.eval()
    rows = []
    try:
        with torch.random.fork_rng(devices=[]):
            for index, instance in enumerate(instances):
                torch.manual_seed(seed + index)
                envs = diagnostic_envs([instance], values, n_traj, use_jit_mask=use_jit_mask)
                observations = [envs[0].reset(seed=seed + index)[0]]
                travel = _normalized_travel_time(envs)
                state = policy.initial_state(1, n_traj)
                done = np.zeros((1, n_traj), dtype=bool)
                for step in range(min(max_steps, max(state_steps) + 1)):
                    batch = stack_observations(observations)
                    if step in state_steps:
                        row = state_probe(policy, batch, travel, state, ~done)
                        row.update({"instance_id": instance.instance_id, "state_step": step})
                        rows.append(row)
                    with torch.no_grad():
                        logits, state = policy.logits(batch, travel, state)
                        action = torch.distributions.Categorical(logits=logits).sample().numpy()[0]
                    observation, _, terminated, truncated, _ = envs[0].step(action)
                    observations = [observation]
                    done[0] |= terminated | truncated
                    if done.all():
                        break
    finally:
        policy.train(previous_training)
    decision_rows = [row for row in rows if row["decision_trajectories"]]
    effective = [row for row in decision_rows if row["finite_legal_logits"]
                 and row["finite_gradients"] and row["legal_logit_span"]["p50"] > 1e-4
                 and row["encoder_gradient_norm"] > 1e-8]
    coverage = {row["instance_id"] for row in decision_rows}
    fraction = len(effective) / max(1, len(decision_rows))
    finite = all(row["finite_legal_logits"] and row["finite_gradients"] for row in rows)
    return {
        "scope": "TRAIN_only_engineering_diagnostics_not_validation_or_test",
        "instance_ids": [instance.instance_id for instance in instances],
        "n_traj": n_traj, "max_steps": max_steps, "state_steps": list(state_steps),
        "seed": seed, "rows": rows,
        "gate": {
            "passed": finite and len(coverage) == len(instances) and fraction >= 0.8,
            "finite": finite, "covered_instances": len(coverage),
            "decision_states": len(decision_rows), "effective_states": len(effective),
            "effective_state_fraction": fraction,
            "thresholds": {"median_legal_logit_span_gt": 1e-4,
                           "encoder_gradient_norm_gt": 1e-8,
                           "effective_state_fraction_gte": 0.8},
        },
    }


def write_report(report, path):
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(content)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(json.dumps({"report": str(path), "gate": report.get("gate")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--dataset-root", "--data-root", dest="dataset_root", type=Path)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--state-steps", default="0,10,50,100,180,239")
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--n-traj", type=int, default=30)
    parser.add_argument("--seed", type=int, default=910001234)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--graph-aggregation", choices=("sum", "mean"))
    parser.add_argument("--disable-jit-mask", action="store_true")
    parser.add_argument("--require-effective-policy", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    steps = sorted(set(int(value) for value in args.state_steps.split(",")))
    if (min(args.instances, args.n_traj, args.max_steps, args.cpu_threads) < 1
            or args.start_index < 0 or not steps or steps[0] < 0 or steps[-1] >= args.max_steps):
        parser.error("positive counts required; state steps must be within rollout budget")
    if args.output and args.output.resolve() == args.checkpoint.resolve():
        parser.error("output cannot overwrite the checkpoint")
    torch.set_num_threads(args.cpu_threads)
    started = time.perf_counter()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("method") != "EVRPTW-RL":
        raise ValueError("expected an EVRPTW-RL checkpoint")
    values = checkpoint_args(payload)
    pool = training_pool(values, dataset_path=args.dataset_path, family_root=args.family_root,
                         dataset_root=args.dataset_root)
    tasks = pool.tasks[args.start_index:args.start_index + args.instances]
    if len(tasks) != args.instances:
        raise ValueError("requested diagnostic cohort exceeds training pool")
    instances = [pool.instance(task) for task in tasks]
    policy = policy_from_checkpoint(payload, args.graph_aggregation)
    report = diagnose_policy(policy, instances, values, n_traj=args.n_traj,
                             state_steps=steps, max_steps=args.max_steps, seed=args.seed,
                             use_jit_mask=not args.disable_jit_mask)
    report.update({
        "schema": "evrptw_policy_diagnostics_v1", "device": "cpu",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": payload.get("logical_epoch"),
        "checkpoint_graph_aggregation": values.get("graph_aggregation", "sum"),
        "effective_graph_aggregation": policy.graph_aggregation,
        "aggregation_override": args.graph_aggregation,
        "counterfactual": args.graph_aggregation is not None and args.graph_aggregation != values.get("graph_aggregation", "sum"),
        "training_index": str(pool.dataset_path),
        "training_index_sha256": file_sha256(Path(pool.dataset_path)),
        "training_view_ids": [task.view_id for task in tasks],
        "model_source_sha256": file_sha256(REPO / "EVRPTW_Benchmark/Reinforcement_Learning/EVRPTW_RL/model.py"),
        "diagnostic_source_sha256": file_sha256(Path(__file__)),
        "torch_version": torch.__version__, "wall_time_s": time.perf_counter() - started,
    })
    write_report(report, args.output)
    return 2 if args.require_effective_policy and not report["gate"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
