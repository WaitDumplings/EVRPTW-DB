from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import resolve_objective
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models import Agent


@pytest.fixture(scope="module", autouse=True)
def _few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("entrypoint,mode,max_steps", [
    ("eval", "stable_cost_v1", 5),
    ("eval", "legacy", 5),
    ("eval_stage2", "stable_cost_v1", 5),
    ("eval_stage2", "stable_cost_v1", None),
    ("eval_stage2", "legacy", 5),
])
def test_eval_entrypoints_load_checkpoint_mode_and_use_actual_budget(
        monkeypatch, tmp_path, entrypoint, mode, max_steps):
    module = importlib.import_module(f"EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.{entrypoint}")
    model_config = {"embedding_dim": 16, "n_encode_layers": 1}
    if mode != "legacy":
        model_config["critic_mode"] = mode
    agent = Agent(**model_config, popart_beta=0.37, popart_min_std=0.7)
    if mode == "stable_cost_v1":
        agent.critic.popart.update_stats(torch.tensor([1000.0, 1300.0]))
    objective = resolve_objective(
        "EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "model_state_dict": agent.state_dict(),
        "config": {
            "model": model_config, "objective": objective.to_dict(),
            "stable_cost": {"popart_beta": 0.37, "popart_min_std": 0.7},
            "env": {"rollout_horizon_steps": 2},
        },
    }, checkpoint)
    args = SimpleNamespace(
        checkpoint_path=checkpoint, checkpoint=checkpoint, objective_config=None,
        eval_path=tmp_path, dataset_path=tmp_path, output_dir=tmp_path / "evaluation",
        solver_name=None, device="cpu", seed=17, num_customers=2, num_charging_stations=1,
        n_traj=1, decode_mode="greedy", max_steps=max_steps, limit=1,
        eval_batch_size=1, eval_num_batches=None, info_level="full", save_routes=True,
        family_root=None, scale="Cus2", split_ids="test", track_ids="test", city_slugs=None,
        representation="G", euclidean_manifest=None, candidates=1, candidate_chunk_size=1,
        batch_size=1,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    if entrypoint == "eval":
        monkeypatch.setattr(module, "_eval_instance_batches", lambda *_: iter([[_instance()]]))
    else:
        monkeypatch.setattr(module, "Stage2TaskPool", lambda **_: SimpleNamespace(
            first=lambda **_: iter([_instance()])))

    actual_make_env = module.make_terran_env
    env_options = []

    def make_env(**kwargs):
        env_options.append(dict(kwargs))
        return actual_make_env(**{**kwargs, "use_jit_mask": False})

    monkeypatch.setattr(module, "make_terran_env", make_env)
    actual_rollout = module.rollout_eval_batch
    calls = []

    def rollout(loaded, envs, **kwargs):
        calls.append(kwargs)
        assert loaded.critic_mode == mode
        assert not loaded.training
        for key, value in agent.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
        if mode == "stable_cost_v1":
            assert loaded.critic.popart.beta == 0.37
            assert loaded.critic.popart.min_std == 0.7
        for env in envs:
            obs, _ = env.reset()
            if mode == "stable_cost_v1":
                expected = min(kwargs["max_steps"], env.unwrapped.max_steps)
                assert (obs["episode_step_budget"] == expected).all()
                assert (obs["remaining_step_budget"] == expected).all()
                assert "customer_unserved" in obs
            else:
                assert "customer_unserved" not in obs
        return actual_rollout(loaded, envs, **kwargs)

    monkeypatch.setattr(module, "rollout_eval_batch", rollout)
    module.main()
    assert len(calls) == len(env_options) == 1
    if mode == "stable_cost_v1":
        assert env_options[0]["training_mode"] == mode
        assert env_options[0]["rollout_horizon_steps"] == max_steps
    elif entrypoint == "eval":
        assert env_options[0]["rollout_horizon_steps"] == 2
    else:
        assert "training_mode" not in env_options[0]
        assert "rollout_horizon_steps" not in env_options[0]
    summary = "terran_summary.csv" if entrypoint == "eval" else "summary.csv"
    assert (args.output_dir / summary).is_file()
