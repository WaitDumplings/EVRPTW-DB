from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import (
    _instance,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import (
    make_terran_env,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models import Agent
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.pbrs import (
    PotentialRewardConfig,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import protocol as terran_protocol
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import trainer as terran_trainer
from EVRPTW_Benchmark.Reinforcement_Learning.common.data_pass import DataPassState
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import (
    collect_rollout, rollout_eval_batch,
)


def test_canonical_terran_rollout_passes_shared_verifier() -> None:
    agent = Agent(
        embedding_dim=32,
        tanh_clipping=10.0,
        n_encode_layers=1,
        device="cpu",
    )
    env = make_terran_env(
        instance=_instance(),
        n_traj=4,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
    )
    rows = rollout_eval_batch(
        agent,
        [env],
        decode_mode="sample",
        max_steps=32,
        device="cpu",
        seed=41,
        include_routes=True,
    )
    routes = json.loads(rows[0]["routes_json"])
    assert validate_routes(_instance(), routes)["passed"]


def test_terran_rollout_reports_training_budget_exhaustion() -> None:
    agent = Agent(
        embedding_dim=32,
        tanh_clipping=10.0,
        n_encode_layers=1,
        device="cpu",
    )
    env = make_terran_env(
        instance=_instance(),
        n_traj=2,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
    )
    result = collect_rollout(
        agent, [env], rollout_steps=1, decode_mode="sample", device="cpu", seed=42
    )
    assert result.trajectory_steps.tolist() == [[1, 1]]
    assert result.rollout_budget_exhausted.tolist() == [[True, True]]


def test_terran_rollout_horizon_penalizes_remaining_customers() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        rollout_horizon_steps=1,
        pbrs_config=PotentialRewardConfig(
            use_terminal_heuristic=True,
            failure_penalty=0.5,
        ),
    )
    env.reset(seed=43)
    _, _, terminated, truncated, info = env.step(np.asarray([1]))

    assert not terminated[0]
    assert truncated[0]
    assert info["rollout_budget_exhausted"].tolist() == [True]
    assert info["remaining_customers"].tolist() == [1]
    assert info["remaining_customer_fraction"].tolist() == [0.5]
    assert np.isclose(info["reward_components"]["terminal_heuristic"][0], -0.25)


def test_terran_completion_at_rollout_horizon_gets_success_not_failure() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        rollout_horizon_steps=3,
        pbrs_config=PotentialRewardConfig(
            use_terminal_heuristic=True,
            success_bonus=0.1,
            failure_penalty=0.5,
        ),
    )
    env.reset(seed=44)
    env.step(np.asarray([1]))
    env.step(np.asarray([2]))
    _, _, terminated, truncated, info = env.step(np.asarray([0]))

    assert terminated[0]
    assert not truncated[0]
    assert info["rollout_budget_exhausted"].tolist() == [False]
    assert info["remaining_customers"].tolist() == [0]
    assert np.isclose(info["reward_components"]["terminal_heuristic"][0], 0.1)


def test_terran_station_revisit_resets_only_after_depot() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
    )
    observation, _ = env.reset(seed=45)
    station = 3
    assert observation["action_mask"][0, station]

    observation, _, _, _, _ = env.step(np.asarray([station]))
    assert not observation["action_mask"][0, station]
    observation, _, _, _, _ = env.step(np.asarray([1]))
    assert not observation["action_mask"][0, station]
    observation, _, _, _, _ = env.step(np.asarray([0]))
    assert observation["action_mask"][0, station]


def test_pbrs_keeps_distance_as_named_base_reward() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        pbrs_config=PotentialRewardConfig(
            use_customer_pbrs=True,
            use_repair_distance_pbrs=True,
        ),
    )
    observation, _ = env.reset(seed=43)
    customer = int(observation["action_mask"][0].nonzero()[0][0])
    _, reward, _, _, info = env.step([customer])
    components = info["reward_components"]
    assert components["base"][0] < 0.0
    assert reward[0] == components["shaped"][0]
    assert env.unwrapped.objective_distance_km[0] > 0.0


def test_terran_training_env_uses_registered_rollout_horizon(monkeypatch) -> None:
    class Pool:
        def sample(self):
            return _instance()

    captured: list[dict] = []

    monkeypatch.setattr(
        terran_trainer,
        "FixedDatasetInstancePool",
        lambda **_kwargs: Pool(),
    )
    monkeypatch.setattr(
        terran_trainer,
        "make_terran_env",
        lambda **kwargs: captured.append(kwargs) or object(),
    )
    cfg = {
        "data": {
            "train_dataset_path": "unused.jsonl",
            "num_customers": 2,
            "num_charging_stations": 1,
        },
        "training": {"num_envs_per_gpu": 2, "n_traj": 3, "rollout_steps": 140},
        "env": {"use_fast_env": False},
        "pbrs": {"use_terminal_heuristic": True, "failure_penalty": 0.5},
    }

    envs, _ = terran_trainer.make_envs(cfg, seed=46)

    assert len(envs) == 2
    assert [call["rollout_horizon_steps"] for call in captured] == [140, 140]
    assert all(call["pbrs_config"].use_terminal_heuristic for call in captured)
    assert all(call["pbrs_config"].failure_penalty == 0.5 for call in captured)


def test_fixed_epoch_protocol_does_not_expand_to_a_full_data_pass(
    tmp_path: Path, monkeypatch
) -> None:
    class Pool:
        def __len__(self) -> int:
            return 5_000

    monkeypatch.setattr(terran_protocol, "Stage2TaskPool", lambda **_kwargs: Pool())
    args = SimpleNamespace(
        training_epochs=200,
        data_passes=None,
        stage2_dataset_path=Path("train.parquet"),
        stage2_family_root=Path("families"),
        stage2_scale="Cus100",
        stage2_split_ids="train",
        stage2_track_ids="train",
        output_dir=tmp_path,
        resume=False,
        protocol_id="fixed-epoch-test",
        num_envs_per_gpu=1,
        physical_batch_size=1,
        effective_batch_size=2,
        training_rollout_steps=140,
        seed=1234,
        max_batches_per_pass=None,
        validation_every_passes=5,
        validation_every_epochs=50,
        minimum_training_epochs=100,
        post_minimum_validation_every_epochs=10,
        validation_checkpoints=12,
        early_stop_patience_validations=3,
        early_stop_start_epoch=100,
        pilot_mode=False,
    )
    configured, meta = terran_protocol.configure_protocol(args, {})
    assert configured["training"]["epochs"] == 200
    assert configured["training"]["checkpoint_interval"] == 50
    assert configured["protocol"]["validation_checkpoints"] == 12
    assert configured["training"]["validation_epochs"][:3] == [50, 100, 110]
    assert configured["training"]["validation_epochs"][-1] == 200
    assert configured["protocol"]["minimum_training_epochs"] == 100
    assert configured["protocol"]["post_minimum_validation_every_epochs"] == 10
    assert configured["protocol"]["budget_mode"] == "fixed_logical_epochs"
    assert configured["protocol"]["epochs_per_pass"] == 200
    assert configured["protocol"]["logical_environments_per_epoch"] == 2
    assert configured["training"]["num_envs_per_gpu"] == 1
    assert configured["training"]["logical_microbatches_per_epoch"] == 2
    assert configured["protocol"]["views_per_pass"] == 5_000
    assert meta is not None and meta["physical_batch_size"] == 1
    assert meta["effective_batch_size"] == 2


def test_terran_effective_batch_two_accumulates_before_optimizer_step(
    tmp_path: Path, monkeypatch
) -> None:
    class Pool:
        sample_count = 0

        def close(self, terminate: bool = False) -> None:
            del terminate

    pool = Pool()
    collect_calls = []

    def fake_collect(_agent, _envs, **_kwargs):
        pool.sample_count += 1
        collect_calls.append(pool.sample_count)
        return SimpleNamespace(
            actions=torch.zeros((1, 1), dtype=torch.long),
            rewards=torch.ones((1, 1)),
            dones=torch.ones((1, 1), dtype=torch.bool),
            values=torch.zeros((1, 1)),
            valid=torch.ones((1, 1), dtype=torch.bool),
            trajectory_steps=torch.ones((1, 1), dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros((1, 1), dtype=torch.bool),
            final_infos=[
                {
                    "success": np.asarray([True]),
                    "objective_distance_km": np.asarray([1.0]),
                    "vehicle_count": np.asarray([1]),
                    "served_customers": np.asarray([1000]),
                }
            ],
            timings={},
        )

    def fake_loss(agent, *_args, **_kwargs):
        loss = agent.weight.sum()
        metric = loss.detach()
        return loss, metric, metric, metric

    monkeypatch.setattr(
        terran_trainer,
        "Agent",
        lambda **_kwargs: torch.nn.Linear(1, 1, bias=False),
    )
    monkeypatch.setattr(
        terran_trainer, "make_envs", lambda _cfg, _seed: ([object()], pool)
    )
    monkeypatch.setattr(terran_trainer, "collect_rollout", fake_collect)
    monkeypatch.setattr(terran_trainer, "evaluate_policy_loss", fake_loss)

    cfg = {
        "run_name": "batch-two-test",
        "output_dir": str(tmp_path),
        "data": {"num_customers": 1000, "num_charging_stations": 200},
        "model": {},
        "env": {},
        "pbrs": {},
        "evaluation": {"eval_interval": 0},
        "training": {
            "epochs": 1,
            "num_envs_per_gpu": 1,
            "logical_microbatches_per_epoch": 2,
            "rollout_steps": 1,
            "ppo_update_epochs": 1,
            "num_minibatches": 1,
            "gradient_accumulation_steps": 1,
            "checkpoint_interval": 1,
        },
    }
    checkpoint = terran_trainer.train_from_config(
        cfg, seed=1234, device="cpu"
    )

    assert collect_calls == [1, 2]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    optimizer_steps = [
        int(state["step"]) for state in payload["optimizer_state_dict"]["state"].values()
    ]
    assert optimizer_steps == [1]
    with (tmp_path / "logs" / "train_log.csv").open(newline="") as stream:
        row = list(csv.DictReader(stream))[-1]
    assert int(row["samples_seen"]) == 2
    assert int(row["effective_instances_per_optimizer_step"]) == 2


def test_terran_online_selection_publishes_tail_best_as_formal_aliases(
    tmp_path: Path, monkeypatch
) -> None:
    class Pool:
        sample_count = 0

        def close(self, terminate: bool = False) -> None:
            del terminate

    pool = Pool()

    def fake_collect(_agent, _envs, **_kwargs):
        pool.sample_count += 1
        return SimpleNamespace(
            actions=torch.zeros((1, 1), dtype=torch.long),
            rewards=torch.ones((1, 1)),
            dones=torch.ones((1, 1), dtype=torch.bool),
            values=torch.zeros((1, 1)),
            valid=torch.ones((1, 1), dtype=torch.bool),
            trajectory_steps=torch.ones((1, 1), dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros((1, 1), dtype=torch.bool),
            final_infos=[
                {
                    "success": np.asarray([True]),
                    "objective_distance_km": np.asarray([1.0]),
                    "vehicle_count": np.asarray([1]),
                    "served_customers": np.asarray([1000]),
                }
            ],
            timings={},
        )

    def fake_loss(agent, *_args, **_kwargs):
        loss = agent.weight.sum()
        metric = loss.detach()
        return loss, metric, metric, metric

    def fake_evaluate(_agent, _cfg, *, seed, epoch, device):
        del seed, device
        return {
            "eval_avg_objective_distance_km": 10.0 - epoch,
            "eval_feasible_rate": 1.0,
            "eval_num_instances": 2,
            "eval_complete_and_feasible": 2,
            "eval_independent_verifier": True,
            "eval_status": "ok",
        }

    monkeypatch.setattr(
        terran_trainer,
        "Agent",
        lambda **_kwargs: torch.nn.Linear(1, 1, bias=False),
    )
    monkeypatch.setattr(
        terran_trainer, "make_envs", lambda _cfg, _seed: ([object()], pool)
    )
    monkeypatch.setattr(terran_trainer, "collect_rollout", fake_collect)
    monkeypatch.setattr(terran_trainer, "evaluate_policy_loss", fake_loss)
    monkeypatch.setattr(
        terran_trainer, "evaluate_fixed_dataset", fake_evaluate
    )

    cfg = {
        "run_name": "overall-selection-test",
        "output_dir": str(tmp_path),
        "data": {"num_customers": 1000, "num_charging_stations": 200},
        "model": {},
        "env": {},
        "pbrs": {},
        "evaluation": {"eval_interval": 1},
        "training": {
            "epochs": 3,
            "num_envs_per_gpu": 1,
            "logical_microbatches_per_epoch": 1,
            "rollout_steps": 1,
            "ppo_update_epochs": 1,
            "num_minibatches": 1,
            "gradient_accumulation_steps": 1,
            "checkpoint_interval": 1,
            "minimum_training_epochs": 2,
            "validation_epochs": [1, 2, 3],
            "early_stop_patience_validations": 0,
            "early_stop_start_epoch": 2,
        },
    }
    terran_trainer.train_from_config(cfg, seed=1234, device="cpu")

    selected = torch.load(
        tmp_path / "checkpoint_selected.pt", map_location="cpu", weights_only=False
    )
    best = torch.load(
        tmp_path / "best.ckpt", map_location="cpu", weights_only=False
    )
    overall = torch.load(
        tmp_path / "best_overall.ckpt", map_location="cpu", weights_only=False
    )
    within = torch.load(
        tmp_path / "best_within_5000.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert selected["epoch"] == best["epoch"] == overall["epoch"] == 3
    assert within["epoch"] == 2
    assert json.loads((tmp_path / "validation_summary.json").read_text())[
        "logical_epoch"
    ] == 3
    assert json.loads(
        (tmp_path / "validation_summary_within_5000.json").read_text()
    )["logical_epoch"] == 2
    history = [
        json.loads(line)
        for line in (tmp_path / "validation_history.jsonl").read_text().splitlines()
    ]
    assert history[-1]["checkpoint_selected"] is True
    assert history[-1]["best_within_minimum_selected"] is False


def test_terran_finalizer_runs_full_audit_without_reselecting(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run"
    checkpoint_dir = output / "checkpoints"
    log_dir = output / "logs"
    checkpoint_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
    final_checkpoint.write_bytes(b"final")
    (output / "best.ckpt").write_bytes(b"stale-within")
    (output / "best_within_5000.ckpt").write_bytes(b"within")
    (output / "best_overall.ckpt").write_bytes(b"overall")
    (output / "validation_summary.json").write_text(
        json.dumps(
            {
                "logical_epoch": 50,
                "complete_and_feasible_rate": 0.5,
                "mean_verified_distance_km": 10.0,
            }
        )
    )
    (output / "validation_summary_within_5000.json").write_text(
        json.dumps(
            {
                "logical_epoch": 50,
                "complete_and_feasible_rate": 0.5,
                "mean_verified_distance_km": 10.0,
            }
        )
    )
    (output / "validation_summary_overall.json").write_text(
        json.dumps(
            {
                "logical_epoch": 75,
                "complete_and_feasible_rate": 1.0,
                "mean_verified_distance_km": 8.0,
            }
        )
    )
    with (log_dir / "train_log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "samples_seen",
                "environment_transitions_total",
                "optimizer_steps_total",
                "epoch_wall_time_s",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "samples_seen": 2,
                "environment_transitions_total": 10,
                "optimizer_steps_total": 3,
                "epoch_wall_time_s": 1.5,
            }
        )

    observed_commands = []

    def fake_run(command, check):
        assert check is True
        observed_commands.append(command)
        audit_dir = output / "validation" / "final_audit"
        audit_dir.mkdir(parents=True)
        with (audit_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["verifier_passed", "objective_distance_km"],
            )
            writer.writeheader()
            writer.writerow(
                {"verifier_passed": "true", "objective_distance_km": 8.0}
            )
            writer.writerow(
                {"verifier_passed": "false", "objective_distance_km": 9.0}
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(terran_protocol.subprocess, "run", fake_run)
    args = SimpleNamespace(
        output_dir=output,
        training_epochs=1,
        data_passes=None,
        final_validation_limit=2,
        validation_dataset_path=Path("val.parquet"),
        validation_family_root=Path("families"),
        validation_limit=1,
        stage2_scale="Cus1000",
        seed=1234,
        device="cpu",
        training_representation="G",
        euclidean_manifest=None,
        max_batches_per_pass=None,
        protocol_id="final-audit-test",
        training_rollout_steps=1200,
        training_stream_path=Path("stream.parquet"),
        validation_every_epochs=50,
        validation_checkpoints=20,
        pilot_mode=False,
    )
    meta = {
        "views_per_pass": 5000,
        "epochs_per_pass": 1,
        "physical_batch_size": 1,
        "effective_batch_size": 2,
    }
    terran_protocol.finalize_protocol(args, final_checkpoint, meta)

    assert len(observed_commands) == 1
    assert observed_commands[0][observed_commands[0].index("--limit") + 1] == "2"
    audit = json.loads((output / "validation_final_audit.json").read_text())
    assert audit["instances"] == 2
    assert audit["selection_logical_epoch"] == 75
    assert audit["selection_changed"] is False
    assert (output / "checkpoint_selected.pt").read_bytes() == b"overall"
    assert (output / "best.ckpt").read_bytes() == b"overall"
    assert (output / "best_within_5000.ckpt").read_bytes() == b"within"
    assert json.loads((output / "validation_summary.json").read_text())[
        "logical_epoch"
    ] == 75


def test_protocol_resume_carries_exact_optimizer_step_count(
    tmp_path: Path, monkeypatch
) -> None:
    state = DataPassState(
        protocol_id="optimizer-step-test",
        completed_data_passes=1,
        instances_seen=100,
        customer_exposures=10_000,
        optimizer_steps=7,
        environment_transitions=321,
        last_checkpoint="checkpoint_latest.pt",
    )
    state.atomic_write(tmp_path / "data_pass_state.json")
    (tmp_path / "checkpoint_latest.pt").write_bytes(b"checkpoint")

    class Pool:
        def __len__(self) -> int:
            return 100

    monkeypatch.setattr(terran_protocol, "Stage2TaskPool", lambda **_kwargs: Pool())
    args = SimpleNamespace(
        data_passes=2,
        stage2_dataset_path=Path("train.parquet"),
        stage2_family_root=Path("families"),
        stage2_scale="Cus100",
        stage2_split_ids="train",
        stage2_track_ids="train",
        output_dir=tmp_path,
        resume=True,
        protocol_id="optimizer-step-test",
        num_envs_per_gpu=10,
        physical_batch_size=10,
        effective_batch_size=10,
        training_rollout_steps=140,
        seed=1234,
        max_batches_per_pass=None,
        validation_every_passes=5,
    )
    configured, _ = terran_protocol.configure_protocol(args, {})
    assert configured["protocol"]["optimizer_steps"] == 7
    assert configured["protocol"]["completed_samples"] == 100
    assert configured["data"]["stage2_completed_samples"] == 100
    assert configured["training"]["rollout_steps"] == 140
    assert configured["protocol"]["training_rollout_steps"] == 140
