from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.model import AMEVRPTWPolicy
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.model import DRLTSPolicy
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnv
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.model import EVRPTWRLPolicy


@pytest.mark.parametrize(
    "method,use_static_cache",
    [("AM_EVRPTW", True), ("AM_EVRPTW", False),
     ("EVRPTW_RL", True), ("EVRPTW_RL", False), ("DRL_TS", None)],
)
def test_rollout_timer_includes_setup_but_excludes_reset(
    monkeypatch, method, use_static_cache
):
    module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{method}.rollout"
    )
    if method == "AM_EVRPTW":
        policy = AMEVRPTWPolicy(
            embedding_dim=32, hidden_dim=32, n_encode_layers=1, n_heads=4
        )
        kwargs = {"incomplete_penalty_km": 100.0}
    elif method == "EVRPTW_RL":
        policy = EVRPTWRLPolicy(embedding_dim=32, structure2vec_rounds=2)
        kwargs = {}
    else:
        policy = DRLTSPolicy(embedding_dim=32, n_encode_layers=1, n_heads=4)
        kwargs = {"soft_constraints": False}
    if use_static_cache is not None:
        kwargs["use_static_cache"] = use_static_cache
    policy.eval()
    env = EVRPTWVectorEnv(_instance(), n_traj=2)
    events = []

    def record(target, name, event):
        original = getattr(target, name)

        def wrapped(*args, **kw):
            events.append(event)
            return original(*args, **kw)

        monkeypatch.setattr(target, name, wrapped)

    record(env, "reset", "reset")
    record(env, "step", "step")
    record(module, "stack_observations", "stack")
    record(policy, "logits", "logits")
    for name in ("encode", "encode_static", "initial_state"):
        if hasattr(policy, name):
            record(policy, name, name)
    timer_values = iter((100.0, 140.0))

    def clock():
        events.append("timer")
        return next(timer_values)

    # Replace this module's clock, not the process-wide time module.
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=clock))
    result = module.rollout(
        policy, [env], decode_type="greedy", max_steps=1, seed=181, **kwargs
    )
    assert events[:3] == ["reset", "timer", "stack"]
    assert events.count("timer") == 2
    assert events[-1] == "timer"
    for setup in ("encode", "encode_static", "initial_state"):
        if setup in events:
            assert events.index("timer") < events.index(setup) < events.index("logits")
    assert events.index("logits") < events.index("step")
    assert result.runtime_s == 40.0
