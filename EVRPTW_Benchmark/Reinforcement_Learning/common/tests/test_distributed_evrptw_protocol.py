"""EVRPTW-RL's warmup handoff, signed station auxiliary and exact CPU recovery."""
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
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_dual_protocol import _assert_tree_equal
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_helpers import init_gloo, run_gloo_workers
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_protocol import _arguments, _Pool, _summary
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import resolved_training_signature_from_args


CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _evr_args(output, common_root, *, resume=False):
    args = _arguments(output, common_root, resume=resume)
    args.early_stop_patience_validations = 0
    args.ema_warmup_steps = 2
    args.ema_decay = 0.5
    args.baseline_eval_interval = 2
    args.baseline_eval_size = 3
    args.reinforce_baseline = "paper"
    args.learning_rate = 0.01
    args.optimizer = "adamw"
    args.weight_decay = 0.01
    args.structure2vec_rounds = 2
    args.activation_checkpoint_stride = 1
    args.station_visit_penalty = 0.3
    args.resolved_training_method_fields = {"architecture": "evrptw_rl_native_v1"}
    # The runner fixture isolates synchronization from physical reward scaling;
    # the actual CLI profile/reward pairing is tested separately below.
    args.method_auxiliary_profile = CONFIGS / "evrptw_rl_station_auxiliary_v1.json"
    method_auxiliary_from_args(args, expected_method="evrptw_rl")
    return args


def _warmup_worker(rank, world_size, rendezvous, output, fail_epoch):
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
            calls = {"epoch": 0, "baselines": [], "validation": []}

            def actor(instances, soft, _seed):
                epoch = instances[0].index // 4 + 1
                calls["epoch"] = epoch
                assert not soft, "EVRPTW-RL uses hard environment constraints throughout warmup"
                if interrupt and epoch == fail_epoch and rank == 1:
                    raise RuntimeError(f"intentional EVR interruption at epoch {epoch}")
                indices = torch.tensor([instance.index + 1 for instance in instances], dtype=torch.float32)
                noise = random.random() + np.random.random() + pool.rng.random() + float(torch.rand(()))
                costs = indices[:, None] + torch.tensor([0.0, 0.5])[None, :]
                logp = policy.weight.sum() * (indices[:, None] + noise * 0.01) * torch.tensor([[1.0, 2.0]])
                return SimpleNamespace(
                    cost=costs, objective=costs, objective_value=costs,
                    vehicles_started=torch.ones_like(costs), feasible=torch.ones_like(costs, dtype=torch.bool),
                    log_likelihood=logp, environment_transitions=len(instances) * 2,
                    trajectory_steps=torch.ones_like(costs, dtype=torch.int64),
                    rollout_budget_exhausted=torch.zeros_like(costs, dtype=torch.bool),
                )

            def baseline(model, instances, soft, seed):
                epoch = calls["epoch"]
                assert not soft and epoch > 2
                is_probe = seed < 10_000_000
                if is_probe:
                    assert rank == 0 and epoch in (4, 6)
                if epoch == 3:
                    # No optimizer step has occurred since the warmup handoff.
                    for actual, expected in zip(model.parameters(), policy.parameters()):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                calls["baselines"].append((epoch, is_probe))
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
            args = _evr_args(output / name, output, resume=resume)
            if changed == "warmup":
                args.ema_warmup_steps = 3
            elif changed == "auxiliary":
                args.method_auxiliary_sha256 = "0" * 64
            elif changed == "rounds":
                args.structure2vec_rounds = 3
            elif changed == "checkpoint":
                args.activation_checkpoint_stride = 0
            kwargs = dict(
                method="EVRPTW-RL", args=args, pool=pool, policy=policy, optimizer=optimizer,
                make_actor=actor, make_baseline=baseline, training_cost=lambda value: value.cost,
                objective_distance=lambda value: value.objective, feasible=lambda value: value.feasible,
                validation_solve=validation_solve, legacy_batch_size=1,
            )
            if changed:
                with pytest.raises(RuntimeError, match="signature|auxiliary"):
                    protocol.train_distributed_reinforce_data_passes(**kwargs)
            elif interrupt:
                with pytest.raises(RuntimeError, match="intentional EVR interruption"):
                    protocol.train_distributed_reinforce_data_passes(**kwargs)
            else:
                protocol.train_distributed_reinforce_data_passes(**kwargs)
            return calls

        full = run("uninterrupted")
        partial = run("resumed", interrupt=True)
        before = torch.load(output / "resumed/checkpoint_latest.pt", weights_only=False)
        assert before["logical_epoch"] == fail_epoch - 1
        for change in ("warmup", "auxiliary", "rounds", "checkpoint"):
            run("resumed", resume=True, changed=change)
        rest = run("resumed", resume=True)
        assert partial["validation"] + rest["validation"] == full["validation"]
        reference = torch.load(output / "uninterrupted/checkpoint_latest.pt", weights_only=False)
        resumed = torch.load(output / "resumed/checkpoint_latest.pt", weights_only=False)
        for key in ("model", "baseline", "optimizer", "rank_rng_states", "baseline_probe_view_ids",
                    "resolved_training_signature", "ema_cost", "baseline_eval_count", "baseline_update_count",
                    "method_auxiliary_profile"):
            _assert_tree_equal(reference[key], resumed[key])
        assert resumed["logical_epoch"] == 6 and resumed["stream_cursor"] == 24
        assert resumed["ema_cost"] == pytest.approx(6.0)
        assert resumed["baseline_eval_count"] == resumed["baseline_update_count"] == 2
        assert len(resumed["rank_rng_states"]) == world_size
        assert resumed["method_auxiliary_profile"]["weights"] == {"station_visit": 0.3}
        for name in ("uninterrupted", "resumed"):
            first = torch.load(output / name / "checkpoint_epoch_0001.pt", weights_only=False)
            warmup = torch.load(output / name / "checkpoint_epoch_0002.pt", weights_only=False)
            assert not torch.equal(first["model"]["weight"], first["baseline"]["weight"])
            _assert_tree_equal(warmup["model"], warmup["baseline"])
            assert warmup["baseline_eval_count"] == warmup["baseline_update_count"] == 0
        Path(output, f"evr_rank_{rank}.json").write_text(json.dumps(full))
        ctx.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail_epoch", [2, 3, 5])
def test_two_rank_evr_global_ema_warmup_handoff_probe_and_exact_resume(tmp_path, fail_epoch):
    run_gloo_workers(_warmup_worker, tmp_path, extra=(fail_epoch,))
    for name in ("uninterrupted", "resumed"):
        root = tmp_path / name
        history = [json.loads(line) for line in (root / "logical_epoch_history.jsonl").read_text().splitlines()]
        assert [row["logical_epoch"] for row in history] == list(range(1, 7))
        assert [row["baseline_kind"] for row in history] == ["paper_ema"] * 2 + ["greedy_rollout"] * 4
        assert [row["baseline_warmup_synchronized"] for row in history] == [False, True, False, False, False, False]
        assert all(row["training_stage"] == "hard" for row in history)
        events = [json.loads(line) for line in (root / "baseline_history.jsonl").read_text().splitlines()]
        assert [row["optimizer_step"] for row in events] == [4, 6]
        assert all(row["baseline_updated"] for row in events)
        sampled = [json.loads(line) for line in (root / "sampled_view_ids.jsonl").read_text().splitlines()]
        assert [view for row in sampled for view in row["view_ids"]] == [f"train-{i}" for i in range(24)]


def _actual_evr_worker(rank, world_size, rendezvous, output):
    from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnvFast
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.model import EVRPTWRLPolicy
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.rollout import rollout

    ctx = init_gloo(rank, world_size, rendezvous)
    try:
        torch.manual_seed(125 + rank)
        model = EVRPTWRLPolicy(embedding_dim=16, structure2vec_rounds=2).train()
        model.activation_checkpoint_stride = 1
        ctx.broadcast_model(model)
        reference = deepcopy(model)

        def sample(policy, index):
            torch.manual_seed(1000 + index)
            instance = replace(_instance(), instance_id=f"evr-rank-{index}")
            env = EVRPTWVectorEnvFast(instance, n_traj=3, use_jit_mask=False)
            return rollout(policy, [env], decode_type="sampling", max_steps=32 + index, seed=1000 + index)

        with ctx.local_phase("actual recurrent actor"):
            actual = sample(model, rank)
        total, count = ctx.sum_values([float(actual.training_cost.sum()), actual.training_cost.numel()])
        global_ema = total / count
        with ctx.local_phase("actual recurrent backward"):
            (((actual.training_cost - global_ema).detach() * actual.log_likelihood).mean() / world_size).backward()
        ctx.sum_gradients(model.parameters())
        reference_costs = []
        for index in range(world_size):
            result = sample(reference, index)
            reference_costs.append(result.training_cost.detach())
            (((result.training_cost - global_ema).detach() * result.log_likelihood).mean() / world_size).backward()
        assert global_ema == pytest.approx(float(torch.cat(reference_costs).mean()))
        for (name, actual_parameter), expected in zip(model.named_parameters(), reference.parameters()):
            assert (actual_parameter.grad is None) == (expected.grad is None), name
            if expected.grad is not None:
                torch.testing.assert_close(actual_parameter.grad, expected.grad, rtol=2e-4, atol=2e-6, msg=name)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.4)
        torch.optim.AdamW(model.parameters(), lr=0.001).step()
        torch.optim.AdamW(reference.parameters(), lr=0.001).step()
        for actual_parameter, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual_parameter, expected, rtol=2e-4, atol=2e-6)
        Path(output, f"actual_evr_rank_{rank}.json").write_text("{}")
    finally:
        dist.destroy_process_group()


def test_two_rank_actual_evr_checkpointed_recurrent_update_matches_serial_global_ema(tmp_path):
    run_gloo_workers(_actual_evr_worker, tmp_path)
    assert len(list(tmp_path.glob("actual_evr_rank_*.json"))) == 2


def test_evr_distributed_cli_signs_native_schedule_station_auxiliary_and_checkpoint(tmp_path):
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL import distributed_train as entry

    cli = ["--dataset-path", str(tmp_path / "train.parquet"), "--output-dir", str(tmp_path / "run"),
           "--training-epochs", "6", "--physical-batch-size", "1", "--effective-batch-size", "2",
           "--batch-size", "1", "--device", "cpu", "--expected-world-size", "2",
           "--reward-contract", str(CONFIGS / "drl_reward_contract_energy_vehicle_v3.json"),
           "--method-auxiliary-profile", str(CONFIGS / "evrptw_rl_station_auxiliary_v1.json")]
    signatures = []
    for stride in (0, 1):
        args = entry.parse_args(cli + ["--activation-checkpoint-stride", str(stride), "--instance-cache-size", "2"])
        entry.prepare_method(args)
        assert args.ema_warmup_steps == 1000 and args.baseline_eval_interval == 100
        assert args.station_visit_penalty == 0.3 and len(args.method_auxiliary_sha256) == 64
        policy = entry.build_policy(args)
        assert getattr(policy, "activation_checkpoint_stride", 0) == stride
        protocol.configure_distributed_contract(args, DistributedContext(rank=0, world_size=2), method="EVRPTW-RL")
        signature = resolved_training_signature_from_args(args)
        signatures.append(signature)
        assert signature["method_specific"]["structure2vec_rounds"] == 3
        assert signature["method_specific"]["ema_warmup_steps"] == 1000
        assert signature["method_specific"]["baseline_eval_interval"] == 100
        assert signature["method_auxiliary_sha256"] == args.method_auxiliary_sha256
        args.instance_cache_size = 128
        assert resolved_training_signature_from_args(args) == signature
    assert signatures[0]["sha256"] != signatures[1]["sha256"]
    with pytest.raises(SystemExit):
        entry.parse_args(cli + ["--instance-cache-size", "-1"])
    for option, value in (("--activation-checkpoint-stride", "-1"), ("--ema-warmup-steps", "-1"),
                          ("--baseline-eval-interval", "0")):
        with pytest.raises(ValueError):
            entry.prepare_method(entry.parse_args(cli + [option, value]))


def test_evr_formal_station_auxiliary_cannot_be_dropped_or_replaced(tmp_path):
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL import distributed_train as entry

    base = ["--dataset-path", str(tmp_path / "train.parquet"), "--output-dir", str(tmp_path / "run"),
            "--training-epochs", "6", "--device", "cpu"]
    with pytest.raises(ValueError, match="method-auxiliary-profile"):
        entry.prepare_method(entry.parse_args(base + ["--reward-contract", "signed-reward.json"]))
    with pytest.raises(ValueError, match="reward contract"):
        entry.prepare_method(entry.parse_args(base + ["--method-auxiliary-profile", str(CONFIGS / "evrptw_rl_station_auxiliary_v1.json")]))
    with pytest.raises(ValueError, match="drl_ts|evrptw_rl"):
        entry.prepare_method(entry.parse_args(base + ["--reward-contract", "signed-reward.json",
                                                     "--method-auxiliary-profile", str(CONFIGS / "drl_ts_soft_auxiliary_v1.json")]))
