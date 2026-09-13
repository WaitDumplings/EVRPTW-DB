"""Real adapter option checks for activation recomputation and bounded CPU caches."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.common import distributed_protocol as protocol
from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed import DistributedContext
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_dual_protocol import _assert_tree_equal
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import resolved_training_signature_from_args


@pytest.mark.parametrize("stride", [1, 2])
def test_rrnco_decoder_checkpoint_preserves_graph_routes_full_gradients_rng_and_encoder_calls(stride):
    from unittest.mock import patch
    from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnvFast
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.model import RRNCOEVPolicy
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.rollout import rollout

    torch.manual_seed(401)
    reference = RRNCOEVPolicy(
        embedding_dim=16, n_encode_layers=1, n_heads=2, feedforward_hidden=32,
        distance_sample_size=3, graph_mode="full", aft_mode="stable",
        distance_sampling="nearest", relation_chunk_size=2, checkpoint_bias=True,
        relation_temperature=5.0,
    ).train()
    active = deepcopy(reference)
    active.activation_checkpoint_stride = stride
    assert deepcopy(active).activation_checkpoint_stride == stride
    results, rngs = [], []
    for policy in (reference, active):
        torch.manual_seed(409)
        with patch.object(policy, "encode", wraps=policy.encode) as encode:
            result = rollout(policy, [EVRPTWVectorEnvFast(_instance(), n_traj=4, use_jit_mask=False)],
                             decode_type="sampling", max_steps=32, seed=419)
            (result.training_cost.detach() * result.log_likelihood).mean().backward()
            assert encode.call_count == 1, "Decoder recomputation must not rerun the graph encoder"
        results.append(result)
        rngs.append(torch.random.get_rng_state())
    expected, actual = results
    torch.testing.assert_close(actual.training_cost, expected.training_cost, rtol=0, atol=0)
    torch.testing.assert_close(actual.log_likelihood, expected.log_likelihood, rtol=0, atol=0)
    assert actual.infos[0]["routes"] == expected.infos[0]["routes"]
    assert torch.equal(*rngs)
    for (name, got), want in zip(active.named_parameters(), reference.parameters()):
        assert (got.grad is None) == (want.grad is None), name
        if got.grad is not None:
            torch.testing.assert_close(got.grad, want.grad, rtol=2e-5, atol=2e-5, msg=name)
    assert active.encoder[0].row.bias.distance[0].weight.grad.abs().sum() > 0
    for got, want in zip(active.buffers(), reference.buffers()):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    with torch.no_grad(), patch("EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout.checkpoint") as call:
        rollout(active, [EVRPTWVectorEnvFast(_instance(), n_traj=2, use_jit_mask=False)],
                decode_type="greedy", max_steps=32, seed=421)
        call.assert_not_called()


@pytest.mark.parametrize("method", ["RRNCO-EV", "DRL-TS"])
def test_distributed_cli_preserves_adapter_args_and_signs_checkpoint_mode(tmp_path, method):
    from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS import distributed_train as ts
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW import distributed_train as rr

    module = rr if method == "RRNCO-EV" else ts
    prepare = rr.single_gpu.configure_method_fields if method == "RRNCO-EV" else ts.prepare_method
    cli = ["--dataset-path", str(tmp_path / "train.parquet"), "--output-dir", str(tmp_path / "run"),
           "--training-epochs", "4", "--physical-batch-size", "1", "--effective-batch-size", "2",
           "--batch-size", "1", "--device", "cpu"]
    signatures = []
    for stride in (0, 1):
        args = module.parse_args(cli + ["--activation-checkpoint-stride", str(stride),
                                      "--instance-cache-size", "2", "--expected-world-size", "2",
                                      "--distributed-backend", "gloo"])
        assert args.instance_cache_size == 2
        assert args.expected_world_size == 2 and args.distributed_backend == "gloo"
        prepare(args)
        policy = module.build_policy(args)
        assert getattr(policy, "activation_checkpoint_stride", 0) == stride
        protocol.configure_distributed_contract(args, DistributedContext(rank=0, world_size=2), method=method)
        signatures.append(resolved_training_signature_from_args(args))
        args.instance_cache_size = 100
        assert resolved_training_signature_from_args(args) == signatures[-1], "Host cache does not alter sample or loss semantics"
    assert signatures[0]["sha256"] != signatures[1]["sha256"]
    assert signatures[1]["method_specific"]["activation_checkpoint_stride"] == 1
    with pytest.raises(SystemExit):
        module.parse_args(cli + ["--instance-cache-size", "-1"])
    default = module.parse_args(cli)
    assert default.activation_checkpoint_stride == 0
    negative = module.parse_args(cli + ["--activation-checkpoint-stride", "-1"])
    with pytest.raises(ValueError, match="nonnegative"):
        prepare(negative)


def test_drl_ts_actual_adapter_always_uses_hard_validation_during_soft_training(tmp_path, monkeypatch):
    from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
    from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_entrypoints as entry
    from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS import distributed_train as ts
    from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS import rollout as rollout_module
    from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import DRLTSHardConstraintEnv
    from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.soft_env import DRLTSSoftConstraintEnv

    args = ts.parse_args([
        "--dataset-path", str(tmp_path / "train.parquet"), "--output-dir", str(tmp_path / "run"),
        "--training-epochs", "4", "--soft-stage-end-epoch", "2", "--samples-per-instance", "3",
        "--training-rollout-steps", "20", "--validation-rollout-steps", "30",
        "--validation-decode-type", "sampling", "--validation-candidates", "2", "--device", "cpu",
    ])
    ts.prepare_method(args)
    policy = ts.build_policy(args)
    pool = SimpleNamespace(reward_distance_scale_km=lambda _mode: 2.0, reward_scale_metadata={})
    observed = []

    def fake_rollout(_policy, envs, **kwargs):
        observed.append((type(envs[0]), kwargs["soft_constraints"], kwargs["compute_log_likelihood"],
                         envs[0].n_traj, kwargs["max_steps"]))
        return SimpleNamespace(infos=[{}])

    def fake_train(**callbacks):
        callbacks["make_actor"]([_instance()], True, 123)
        callbacks["make_baseline"](policy, [_instance()], True, 124)
        callbacks["validation_solve"](policy, _instance(), 125)
        callbacks["make_actor"]([_instance()], False, 126)
        assert callbacks["soft_stage_end_epoch"] == 2

    monkeypatch.setattr(rollout_module, "rollout", fake_rollout)
    monkeypatch.setattr(entry, "train_reinforce_data_passes", fake_train)
    monkeypatch.setattr(entry, "_finalize_validation_result", lambda *_args: None)
    entry.run_drl_ts(args, pool, policy, None)
    assert observed == [
        (DRLTSSoftConstraintEnv, True, True, 3, 20),
        (DRLTSSoftConstraintEnv, True, False, 1, 20),
        (DRLTSHardConstraintEnv, False, False, 2, 30),
        (DRLTSHardConstraintEnv, False, True, 3, 20),
    ]


def test_bounded_host_instance_cache_preserves_sampling_and_immutable_payload(tmp_path, monkeypatch):
    from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
    from EVRPTW_Benchmark.Reinforcement_Learning.common import stage2_data
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnvFast

    tasks = [SimpleNamespace(view_id=f"view-{i}", terminal_count=4) for i in range(3)]
    loads = []

    def load(task):
        loads.append(task.view_id)
        return replace(_instance(), instance_id=task.view_id)

    monkeypatch.setattr(stage2_data, "is_synthetic_index", lambda _path: False)
    monkeypatch.setattr(stage2_data, "read_stage2_tasks", lambda *_args, **_kwargs: tasks)
    monkeypatch.setattr(stage2_data, "load_stage2_instance", load)
    disabled = stage2_data.Stage2TaskPool(tmp_path / "train.parquet", seed=901, cache_size=0)
    cached = stage2_data.Stage2TaskPool(tmp_path / "train.parquet", seed=901, cache_size=2)
    for _ in range(5):
        reference, actual = disabled.sample(4), cached.sample(4)
        assert [item.instance_id for item in actual] == [item.instance_id for item in reference]
        for got, want in zip(actual, reference):
            _assert_tree_equal(vars(got), vars(want))
        assert not disabled._cache and len(cached._cache) <= 2
    item = cached.instance(tasks[0])
    before = deepcopy(vars(item))
    env = EVRPTWVectorEnvFast(item, n_traj=2, use_jit_mask=False)
    observation, _ = env.reset(seed=902)
    action = np.asarray([np.flatnonzero(mask)[0] for mask in observation["action_mask"]])
    env.step(action)
    _assert_tree_equal(vars(item), before)
    count = len(loads)
    assert cached.instance(tasks[0]) is item
    assert len(loads) == count, "A repeated view should reuse its immutable CPU instance"
