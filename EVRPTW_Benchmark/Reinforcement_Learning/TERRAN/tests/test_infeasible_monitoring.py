from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import rollout
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import trainer


def test_outcome_monitor_separates_success_horizon_and_non_horizon_reasons() -> None:
    infos = [
        {
            "success": np.asarray([True, False, False, False]),
            "failure_reason": np.asarray(
                ["success", "no_feasible_action", "in_progress", "invalid_action"],
                dtype=object,
            ),
        }
    ]
    summary = rollout.summarize_rollout_outcomes(
        infos, np.asarray([[False, False, True, False]])
    )

    assert summary == {
        "trajectory_count": 4,
        "success_count": 1,
        "success_rate": 0.25,
        "rollout_budget_exhausted_count": 1,
        "rollout_budget_exhausted_rate": 0.25,
        "non_horizon_infeasible_count": 2,
        "non_horizon_infeasible_rate": 0.5,
        "non_horizon_infeasible_reason_counts": {
            "invalid_action": 1,
            "no_feasible_action": 1,
        },
    }


def test_outcome_monitor_legacy_environment_limit_reason_is_horizon() -> None:
    summary = rollout.summarize_rollout_outcomes(
        [
            {
                "success": np.asarray([False]),
                "failure_reason": np.asarray(["environment_step_limit"]),
            }
        ]
    )

    assert summary["rollout_budget_exhausted_count"] == 1
    assert summary["non_horizon_infeasible_count"] == 0


class _MixedEndingEnv:
    n_traj = 3
    num_customers = 1

    def __init__(self) -> None:
        self.unwrapped = self
        self.steps = 0

    def reset(self, **_kwargs):
        self.steps = 0
        return {"x": np.zeros((3, 1), dtype=np.float32)}, self._info()

    def step(self, _actions):
        self.steps += 1
        terminated = np.asarray(
            [True, self.steps >= 2, False], dtype=bool
        )
        return (
            {"x": np.zeros((3, 1), dtype=np.float32)},
            np.zeros(3, dtype=np.float32),
            terminated,
            np.zeros(3, dtype=bool),
            self._info(),
        )

    def _info(self):
        reasons = np.asarray(
            [
                "success",
                "no_feasible_action" if self.steps >= 2 else "in_progress",
                "in_progress",
            ],
            dtype=object,
        )
        return {
            "success": np.asarray([True, False, False]),
            "failure_reason": reasons,
            "served_customers": np.asarray([1, 0, 0]),
            "objective_distance_km": np.asarray([1.0, 0.0, 0.0]),
            "objective_value": np.asarray([1.0, 0.0, 0.0]),
            "vehicle_count": np.asarray([1, 0, 0]),
        }


def test_eval_loop_marks_its_own_timeout_as_horizon(monkeypatch) -> None:
    monkeypatch.setattr(
        rollout,
        "sample_eval_actions",
        lambda *_args, **_kwargs: torch.zeros((1, 3), dtype=torch.long),
    )
    rows = rollout.rollout_eval_batch(
        SimpleNamespace(training=True),
        [_MixedEndingEnv()],
        decode_mode="greedy",
        max_steps=2,
        device="cpu",
        compact_observations=False,
    )

    row = rows[0]
    assert row["candidate_trajectory_count"] == 3
    assert row["candidate_success_count"] == 1
    assert row["candidate_rollout_budget_exhausted_count"] == 1
    assert row["candidate_non_horizon_infeasible_count"] == 1
    assert json.loads(row["candidate_non_horizon_infeasible_reason_counts"]) == {
        "no_feasible_action": 1
    }
    assert row["no_success_all_candidates_non_horizon_infeasible"] is False


class _BareEnvironmentStepLimitEnv:
    n_traj = 1
    num_customers = 1
    max_steps = 2

    def __init__(self) -> None:
        self.unwrapped = self
        self.steps = 0

    def reset(self, **_kwargs):
        self.steps = 0
        return {"x": np.zeros((1, 1), dtype=np.float32)}, self._info()

    def step(self, _actions):
        self.steps += 1
        done = self.steps >= self.max_steps
        return (
            {"x": np.zeros((1, 1), dtype=np.float32)},
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=bool),
            np.asarray([done], dtype=bool),
            self._info(),
        )

    def _info(self):
        return {
            "success": np.asarray([False]),
            "failure_reason": np.asarray(
                [
                    "environment_step_limit"
                    if self.steps >= self.max_steps
                    else "in_progress"
                ],
                dtype=object,
            ),
            "served_customers": np.asarray([0]),
            "objective_distance_km": np.asarray([0.0]),
            "objective_value": np.asarray([0.0]),
            "vehicle_count": np.asarray([0]),
        }


def test_bare_environment_step_limit_is_horizon_not_ffp_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        rollout,
        "sample_eval_actions",
        lambda *_args, **_kwargs: torch.zeros((1, 1), dtype=torch.long),
    )
    rows = rollout.rollout_eval_batch(
        SimpleNamespace(training=True),
        [_BareEnvironmentStepLimitEnv()],
        decode_mode="greedy",
        max_steps=2,
        device="cpu",
        compact_observations=False,
    )

    row = rows[0]
    assert row["candidate_rollout_budget_exhausted_count"] == 1
    assert row["candidate_non_horizon_infeasible_count"] == 0
    assert row["no_success_all_candidates_non_horizon_infeasible"] is False


def test_training_csv_and_validation_history_persist_monitor(
    tmp_path, monkeypatch
) -> None:
    class _Pool:
        sample_count = 0

        def close(self, terminate: bool = False) -> None:
            del terminate

    pool = _Pool()

    def fake_collect(_agent, _envs, **_kwargs):
        pool.sample_count += 1
        return SimpleNamespace(
            actions=torch.zeros((1, 1, 3), dtype=torch.long),
            rewards=torch.ones((1, 1, 3)),
            dones=torch.ones((1, 1, 3), dtype=torch.bool),
            values=torch.zeros((1, 1, 3)),
            valid=torch.ones((1, 1, 3), dtype=torch.bool),
            trajectory_steps=torch.ones((1, 3), dtype=torch.int64),
            rollout_budget_exhausted=torch.tensor(
                [[False, False, True]], dtype=torch.bool
            ),
            final_infos=[
                {
                    "success": np.asarray([True, False, False]),
                    "failure_reason": np.asarray(
                        ["success", "no_feasible_action", "rollout_budget_exhausted"]
                    ),
                    "objective_distance_km": np.asarray([1.0, 2.0, 3.0]),
                    "vehicle_count": np.asarray([1, 1, 1]),
                    "served_customers": np.asarray([2, 1, 1]),
                }
            ],
            timings={},
        )

    def fake_loss(agent, *_args, **_kwargs):
        loss = agent.weight.sum()
        metric = loss.detach()
        return loss, metric, metric, metric

    eval_monitor = {
        "eval_candidate_trajectory_count": 6,
        "eval_candidate_success_count": 1,
        "eval_candidate_success_rate": 1 / 6,
        "eval_candidate_rollout_budget_exhausted_count": 2,
        "eval_candidate_rollout_budget_exhausted_rate": 2 / 6,
        "eval_candidate_non_horizon_infeasible_count": 3,
        "eval_candidate_non_horizon_infeasible_rate": 0.5,
        "eval_candidate_non_horizon_infeasible_reason_counts": json.dumps(
            {"no_feasible_action": 3}
        ),
        "eval_no_success_all_candidates_non_horizon_infeasible_instance_count": 1,
        "eval_no_success_all_candidates_non_horizon_infeasible_instance_rate": 0.5,
    }

    monkeypatch.setattr(
        trainer, "Agent", lambda **_kwargs: torch.nn.Linear(1, 1, bias=False)
    )
    monkeypatch.setattr(trainer, "make_envs", lambda _cfg, _seed: ([object()], pool))
    monkeypatch.setattr(trainer, "collect_rollout", fake_collect)
    monkeypatch.setattr(trainer, "evaluate_policy_loss", fake_loss)
    monkeypatch.setattr(
        trainer,
        "evaluate_fixed_dataset",
        lambda *_args, **_kwargs: {
            "eval_avg_objective_distance_km": 1.0,
            "eval_avg_vehicle_count": 1.0,
            "eval_feasible_rate": 0.5,
            "eval_num_instances": 2,
            "eval_complete_and_feasible": 1,
            "eval_independent_verifier": True,
            "eval_status": "ok",
            **eval_monitor,
        },
    )
    cfg = {
        "run_name": "infeasible-monitor-test",
        "output_dir": str(tmp_path),
        "data": {"num_customers": 2, "num_charging_stations": 1},
        "model": {},
        "env": {},
        "pbrs": {},
        "evaluation": {"eval_interval": 1},
        "training": {
            "epochs": 1,
            "num_envs_per_gpu": 1,
            "n_traj": 3,
            "logical_microbatches_per_epoch": 1,
            "rollout_steps": 1,
            "ppo_update_epochs": 1,
            "num_minibatches": 1,
            "gradient_accumulation_steps": 1,
            "checkpoint_interval": 1,
            "minimum_training_epochs": 1,
            "validation_epochs": [1],
            "early_stop_patience_validations": 0,
            "early_stop_start_epoch": 0,
        },
    }
    trainer.train_from_config(cfg, seed=1234, device="cpu")

    with (tmp_path / "logs" / "train_log.csv").open(newline="") as stream:
        train_row = next(csv.DictReader(stream))
    assert int(train_row["successful_trajectory_count"]) == 1
    assert int(train_row["rollout_budget_exhausted_count"]) == 1
    assert int(train_row["non_horizon_infeasible_count"]) == 1
    assert json.loads(train_row["non_horizon_infeasible_reason_counts"]) == {
        "no_feasible_action": 1
    }

    validation = json.loads(
        (tmp_path / "validation_history.jsonl").read_text().splitlines()[0]
    )
    assert validation["candidate_trajectory_count"] == 6
    assert validation["candidate_non_horizon_infeasible_count"] == 3
    assert validation["candidate_non_horizon_infeasible_reason_counts"] == {
        "no_feasible_action": 3
    }
    assert (
        validation[
            "no_success_all_candidates_non_horizon_infeasible_instance_count"
        ]
        == 1
    )


def test_validation_candidate_aggregate_counts_all_dead_end_instances() -> None:
    rows = [
        {
            "candidate_trajectory_count": 3,
            "candidate_success_count": 1,
            "candidate_rollout_budget_exhausted_count": 1,
            "candidate_non_horizon_infeasible_count": 1,
            "candidate_non_horizon_infeasible_reason_counts": json.dumps(
                {"no_feasible_action": 1}
            ),
            "no_success_all_candidates_non_horizon_infeasible": False,
        },
        {
            "candidate_trajectory_count": 2,
            "candidate_success_count": 0,
            "candidate_rollout_budget_exhausted_count": 0,
            "candidate_non_horizon_infeasible_count": 2,
            "candidate_non_horizon_infeasible_reason_counts": json.dumps(
                {"invalid_action": 2}
            ),
            "no_success_all_candidates_non_horizon_infeasible": True,
        },
    ]
    summary = trainer.summarize_eval_candidate_outcomes(rows)
    assert summary["eval_candidate_trajectory_count"] == 5
    assert summary["eval_candidate_non_horizon_infeasible_count"] == 3
    assert summary["eval_candidate_non_horizon_infeasible_rate"] == pytest.approx(0.6)
    assert json.loads(
        summary["eval_candidate_non_horizon_infeasible_reason_counts"]
    ) == {"invalid_action": 2, "no_feasible_action": 1}
    assert (
        summary[
            "eval_no_success_all_candidates_non_horizon_infeasible_instance_count"
        ]
        == 1
    )
    assert summary[
        "eval_no_success_all_candidates_non_horizon_infeasible_instance_rate"
    ] == pytest.approx(0.5)


def test_validation_candidate_reason_counts_must_match_total() -> None:
    with pytest.raises(RuntimeError, match="do not sum"):
        trainer.summarize_eval_candidate_outcomes(
            [
                {
                    "candidate_trajectory_count": 2,
                    "candidate_success_count": 0,
                    "candidate_rollout_budget_exhausted_count": 0,
                    "candidate_non_horizon_infeasible_count": 2,
                    "candidate_non_horizon_infeasible_reason_counts": json.dumps(
                        {"no_feasible_action": 1}
                    ),
                }
            ]
        )
