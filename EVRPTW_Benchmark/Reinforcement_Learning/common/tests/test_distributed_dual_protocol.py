"""CPU process groups cover the model-specific dual-GPU training contracts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from EVRPTW_Benchmark.Reinforcement_Learning.common import distributed_protocol as protocol
from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed import DistributedContext
from EVRPTW_Benchmark.Reinforcement_Learning.common.method_auxiliary import method_auxiliary_from_args
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_helpers import init_gloo, run_gloo_workers
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_protocol import _arguments, _Pool, _summary
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import resolved_training_signature_from_args


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right):
            _assert_tree_equal(first, second)
    else:
        assert left == right


def _dual_args(output, common_root, method, baseline_kind, *, resume=False):
    args = _arguments(output, common_root, resume=resume)
    args.training_epochs = 4
    args.validation_checkpoints = 4
    args.customer_exposure_budget = 4 * 4 * 50
    args.early_stop_patience_validations = 0
    args.steps_per_epoch = 2
    args.baseline_warmup_epochs = 1
    args.baseline_eval_size = 3
    args.batches_per_epoch = 2
    args.reinforce_baseline = baseline_kind
    args.learning_rate = 0.01
    args.optimizer = "adamw"
    args.weight_decay = 0.01
    args.resolved_training_method_fields = {"architecture": method, "graph_mode": "full"}
    if method == "DRL-TS":
        args.soft_stage_end_epoch = 2
        args.method_auxiliary_profile = (
            Path(__file__).resolve().parents[2] / "configs/drl_ts_soft_auxiliary_v1.json"
        )
        method_auxiliary_from_args(args, expected_method="drl_ts")
    return args


def _dual_protocol_worker(rank, world_size, rendezvous, output, method, baseline_kind, fail_epoch):
    ctx = init_gloo(rank, world_size, rendezvous)
    output = Path(output)
    try:
        protocol.read_stream_view_ids = lambda _path, *, stop: [f"train-{i}" for i in range(stop)]
        protocol.make_validation_pool = lambda *_args, **_kwargs: _Pool(validation=True)
        protocol.ttest_rel = lambda *_args, **_kwargs: SimpleNamespace(pvalue=0.001)

        def run(name, *, resume=False, interrupt=False, changed=None):
            random.seed(100 + rank)
            np.random.seed(200 + rank)
            torch.manual_seed(300 + rank)
            pool = _Pool(seed=400 + rank)
            policy = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0.01)
            calls = {"epoch": 0, "actors": [], "baselines": [], "validation": []}

            def actor(instances, soft, _seed):
                epoch = instances[0].index // 4 + 1
                calls["epoch"] = epoch
                expected_soft = method == "DRL-TS" and epoch <= 2
                assert soft == expected_soft
                calls["actors"].append((epoch, soft))
                if interrupt and epoch == fail_epoch and rank == 1:
                    raise RuntimeError(f"intentional dual adapter interruption at epoch {epoch}")
                indices = torch.tensor([instance.index + 1 for instance in instances], dtype=torch.float32)
                noise = random.random() + np.random.random() + pool.rng.random() + float(torch.rand(()))
                costs = indices[:, None] + torch.tensor([0.0, 0.5])[None, :]
                if soft:
                    costs = costs + 5.0
                logp = policy.weight.sum() * (indices[:, None] + noise * 0.01) * torch.tensor([[1.0, 2.0]])
                return SimpleNamespace(
                    cost=costs, objective=costs, objective_value=costs,
                    vehicles_started=torch.ones_like(costs), feasible=torch.ones_like(costs, dtype=torch.bool),
                    log_likelihood=logp, environment_transitions=len(instances) * 2,
                    trajectory_steps=torch.ones_like(costs, dtype=torch.int64),
                    rollout_budget_exhausted=torch.zeros_like(costs, dtype=torch.bool),
                )

            def baseline(model, instances, soft, seed):
                assert baseline_kind != "leave_one_out", "LOO must not decode a rollout baseline"
                assert soft == (method == "DRL-TS" and calls["epoch"] <= 2)
                is_probe = seed < 10_000_000
                if is_probe:
                    assert rank == 0
                calls["baselines"].append((calls["epoch"], soft, is_probe))
                offset = 0.0 if model is policy else 10.0
                return SimpleNamespace(cost=torch.tensor([
                    [instance.index + offset, instance.index + offset + 0.5]
                    for instance in instances
                ]))

            def validation(instances, solve, *, seed, objective_config=None, cuda_rng_devices=None):
                assert cuda_rng_devices == []
                rows = []
                for offset, instance in enumerate(instances):
                    solve(instance, seed + offset)
                    rows.append(dict(instance_id=instance.instance_id, verifier_passed=True,
                                     objective_distance_km=20.0 - calls["epoch"] + instance.index))
                return _summary(rows)

            def validation_solve(_model, instance, seed):
                assert seed == 90210 + instance.index
                calls["validation"].append((calls["epoch"], instance.index, seed))
                return {}

            protocol.verified_validation = validation
            args = _dual_args(output / name, output, method, baseline_kind, resume=resume)
            if changed == "model":
                args.resolved_training_method_fields["graph_mode"] = "node_only"
            elif changed == "auxiliary":
                args.method_auxiliary_sha256 = "0" * 64
            elif changed == "stage":
                args.soft_stage_end_epoch = 3
            kwargs = dict(
                method=method, args=args, pool=pool, policy=policy, optimizer=optimizer,
                make_actor=actor, make_baseline=baseline, training_cost=lambda value: value.cost,
                objective_distance=lambda value: value.objective, feasible=lambda value: value.feasible,
                validation_solve=validation_solve, legacy_batch_size=1,
                soft_stage_end_epoch=args.soft_stage_end_epoch if method == "DRL-TS" else None,
            )
            if changed:
                with pytest.raises(RuntimeError, match="signature|auxiliary|stage"):
                    protocol.train_distributed_reinforce_data_passes(**kwargs)
            elif interrupt:
                with pytest.raises(RuntimeError, match="intentional dual adapter interruption"):
                    protocol.train_distributed_reinforce_data_passes(**kwargs)
            else:
                protocol.train_distributed_reinforce_data_passes(**kwargs)
            return calls

        reference_calls = run("uninterrupted")
        run("resumed", interrupt=True)
        before = torch.load(output / "resumed/checkpoint_latest.pt", weights_only=False)
        assert before["logical_epoch"] == fail_epoch - 1
        run("resumed", resume=True, changed="model")
        if method == "DRL-TS":
            run("resumed", resume=True, changed="stage")
            run("resumed", resume=True, changed="auxiliary")
        resumed_calls = run("resumed", resume=True)
        reference = torch.load(output / "uninterrupted/checkpoint_latest.pt", weights_only=False)
        resumed = torch.load(output / "resumed/checkpoint_latest.pt", weights_only=False)
        for key in ("model", "baseline", "optimizer", "rank_rng_states", "baseline_probe_view_ids",
                    "resolved_training_signature", "ema_cost", "baseline_eval_count", "baseline_update_count"):
            _assert_tree_equal(reference[key], resumed[key])
        assert resumed["logical_epoch"] == 4 and resumed["stream_cursor"] == 16
        assert not resumed["early_stopped"]
        assert len(resumed["rank_rng_states"]) == world_size
        if baseline_kind == "leave_one_out":
            assert resumed["ema_cost"] is None
            assert resumed["baseline_eval_count"] == resumed["baseline_update_count"] == 0
            assert not reference_calls["baselines"] and not resumed_calls["baselines"]
        elif method == "RRNCO-EV":
            # Two waves per epoch; EMA is frozen after warmup ends at update 2.
            assert resumed["ema_cost"] == pytest.approx(6.0)
            assert resumed["baseline_eval_count"] == resumed["baseline_update_count"] == 2
        else:
            assert resumed["ema_cost"] is None
            assert resumed["baseline_eval_count"] == resumed["baseline_update_count"] == 2
            assert resumed["soft_stage_contract"]["resolved_soft_stage_end_epoch"] == 2
            assert resumed["method_auxiliary_profile"]["sha256"] == resumed["args"]["method_auxiliary_sha256"]
        Path(output, f"dual_rank_{rank}.json").write_text(json.dumps(reference_calls))
        ctx.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("method,baseline_kind,fail_epoch", [
    ("RRNCO-EV", "paper", 3),
    ("RRNCO-EV", "leave_one_out", 3),
    ("DRL-TS", "paper", 2),
    ("DRL-TS", "paper", 3),
])
def test_two_rank_model_contract_and_exact_resume_across_stage_boundary(tmp_path, method, baseline_kind, fail_epoch):
    run_gloo_workers(_dual_protocol_worker, tmp_path, extra=(method, baseline_kind, fail_epoch))
    for name in ("uninterrupted", "resumed"):
        root = tmp_path / name
        history = [json.loads(line) for line in (root / "logical_epoch_history.jsonl").read_text().splitlines()]
        assert [row["logical_epoch"] for row in history] == [1, 2, 3, 4]
        assert [row["training_stage"] for row in history] == (
            ["soft", "soft", "hard", "hard"] if method == "DRL-TS" else ["hard"] * 4
        )
        expected_baselines = (["same_instance_leave_one_out"] * 4 if baseline_kind == "leave_one_out" else
                              ["greedy_rollout"] * 4 if method == "DRL-TS" else
                              ["paper_ema", "paper_ema", "greedy_rollout", "greedy_rollout"])
        assert [row["baseline_kind"] for row in history] == expected_baselines
        sampled = [json.loads(line) for line in (root / "sampled_view_ids.jsonl").read_text().splitlines()]
        assert [view for row in sampled for view in row["view_ids"]] == [f"train-{i}" for i in range(16)]
        if baseline_kind != "leave_one_out":
            probes = [json.loads(line) for line in (root / "baseline_history.jsonl").read_text().splitlines()]
            assert [row["optimizer_step"] for row in probes] == [2, 4]
            assert [row["probe_training_stage"] for row in probes] == (
                ["soft", "hard"] if method == "DRL-TS" else ["hard", "hard"]
            )
            if method == "DRL-TS":
                assert all(row["schedule_source"] == "native_adapter" for row in probes)
        diagnostics = [json.loads(line) for line in (root / "reward_diagnostics.jsonl").read_text().splitlines()]
        assert [row["training_stage"] for row in diagnostics] == [row["training_stage"] for row in history]
        if baseline_kind == "leave_one_out":
            # Cost pairs are (i+1, i+1.5), so the other-trajectory baseline
            # produces (-0.5,+0.5) for EACH instance, independently of rank.
            for row in diagnostics:
                advantage = row["distributions"]["pre_loss_advantage"]
                assert advantage["mean"] == pytest.approx(0.0)
                assert advantage["min"] == pytest.approx(-0.5)
                assert advantage["max"] == pytest.approx(0.5)


def _actual_model_worker(rank, world_size, rendezvous, output):
    from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
    from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import DRLTSHardConstraintEnv
    from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.model import DRLTSPolicy
    from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout import rollout as ts_rollout
    from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.soft_env import DRLTSSoftConstraintEnv
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnvFast
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.model import RRNCOEVPolicy
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.rollout import rollout as rrnco_rollout

    ctx = init_gloo(rank, world_size, rendezvous)
    try:
        for method, soft in (("RRNCO-EV", False), ("DRL-TS", True), ("DRL-TS", False)):
            torch.manual_seed(123 + rank)
            if method == "RRNCO-EV":
                model = RRNCOEVPolicy(
                    embedding_dim=16, n_encode_layers=1, n_heads=2, feedforward_hidden=32,
                    distance_sample_size=3, graph_mode="full", aft_mode="stable",
                    distance_sampling="nearest", relation_chunk_size=2, checkpoint_bias=True,
                    relation_temperature=5.0,
                ).train()
            else:
                model = DRLTSPolicy(embedding_dim=16, n_encode_layers=1, n_heads=2, nearest_neighbors=2).train()
            ctx.broadcast_model(model)
            reference = deepcopy(model)

            def sample_loss(policy, index):
                torch.manual_seed(1000 + index)
                instance = replace(_instance(), instance_id=f"rank-instance-{index}")
                if method == "RRNCO-EV":
                    env = EVRPTWVectorEnvFast(instance, n_traj=3, use_jit_mask=False)
                    result = rrnco_rollout(policy, [env], decode_type="sampling", max_steps=32 + index, seed=1000 + index)
                else:
                    # Force the soft environment to encounter capacity penalties.
                    if soft:
                        instance = replace(instance, vehicle={"battery_capacity_kwh": 10.0, "cargo_capacity_cm3": 0.5})
                    cls = DRLTSSoftConstraintEnv if soft else DRLTSHardConstraintEnv
                    env = cls(instance, n_traj=3, use_jit_mask=False)
                    result = ts_rollout(policy, [env], decode_type="sampling", max_steps=32 + index,
                                        seed=1000 + index, soft_constraints=soft)
                assert torch.isfinite(result.training_cost).all()
                return (result.training_cost.detach() * result.log_likelihood).mean()

            with ctx.local_phase("actual matrix rollout and backward"):
                (sample_loss(model, rank) / world_size).backward()
            ctx.sum_gradients(model.parameters())
            for index in range(world_size):
                (sample_loss(reference, index) / world_size).backward()
            for (name, actual), expected in zip(model.named_parameters(), reference.parameters()):
                if expected.grad is None:
                    assert actual.grad is None, name
                else:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-6, msg=name)
            active = dict(model.named_parameters())
            key = "encoder.0.row.bias.distance.0.weight" if method == "RRNCO-EV" else "decoder_gru.weight_hh"
            assert active[key].grad is not None and active[key].grad.abs().sum() > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.4)
            torch.optim.AdamW(model.parameters(), lr=0.001).step()
            torch.optim.AdamW(reference.parameters(), lr=0.001).step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
        Path(output, f"actual_rank_{rank}.json").write_text("{}")
    finally:
        dist.destroy_process_group()


def test_two_rank_actual_graph_and_recurrent_rollout_updates_match_serial(tmp_path):
    run_gloo_workers(_actual_model_worker, tmp_path)
    assert len(list(tmp_path.glob("actual_rank_*.json"))) == 2


@pytest.mark.parametrize("method,field,first,second", [
    ("RRNCO-EV", "reinforce_baseline", "paper", "leave_one_out"),
    ("RRNCO-EV", "relation_temperature", 5.0, 8.0),
    ("RRNCO-EV", "feedforward_hidden", 32, 64),
    ("DRL-TS", "batches_per_epoch", 250, 100),
    ("DRL-TS", "nearest_neighbors", 10, 20),
])
def test_dual_method_settings_are_signed_without_losing_architecture_metadata(tmp_path, method, field, first, second):
    args = _dual_args(tmp_path / "run", tmp_path, method, "paper")
    args.resolved_training_method_fields.update(
        relation_channels=["directed_distance", "directed_time", "directed_energy", "angle"],
        ablation_contract="road_inputs_masked_in_ane_encoder_and_decoder_v1",
    )
    context = DistributedContext(rank=0, world_size=2)
    setattr(args, field, first)
    protocol.configure_distributed_contract(args, context, method=method)
    before = resolved_training_signature_from_args(args)
    setattr(args, field, second)
    protocol.configure_distributed_contract(args, context, method=method)
    after = resolved_training_signature_from_args(args)
    assert before["method_specific"]["architecture"] == method
    assert before["method_specific"]["relation_channels"] == ["directed_distance", "directed_time", "directed_energy", "angle"]
    assert before["method_specific"][field] == first
    assert after["method_specific"][field] == second
    assert before["sha256"] != after["sha256"]
