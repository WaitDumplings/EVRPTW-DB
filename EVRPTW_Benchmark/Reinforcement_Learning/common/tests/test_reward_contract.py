from __future__ import annotations

import importlib
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

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import (
    _instance,
)
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import (
    DRLTSHardConstraintEnv,
)
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.soft_env import (
    DRLTSSoftConstraintEnv,
)
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import (
    EVRPTWVectorEnvFast,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import (
    ObjectiveConfig,
    load_objective,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.method_auxiliary import (
    load_method_auxiliary_profile,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import (
    RewardContract,
    load_reward_contract,
    reward_contract_digest,
    reward_contract_from_args,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_trainers
from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_entrypoints


def _objective() -> ObjectiveConfig:
    return load_objective(
        REPO_ROOT
        / "EVRPTW_Benchmark/Reinforcement_Learning/configs/"
        "rivian_energy_vehicle_cost_v2.json"
    )


def _payload() -> dict:
    payload = {
        "schema": "drl_reward_contract_v1",
        "contract_id": "test_reference_scale_v3",
        "objective": _objective().to_dict(),
        "scales": {
            "Cus2": {
                "objective_scale": 10.0,
                "failure_base": 2.0,
                "unserved_coefficient": 3.0,
                "calibration": {"cohort_size": 8},
            }
        },
        "calibration": {"source_split": "train"},
    }
    payload["sha256"] = reward_contract_digest(payload)
    return payload


def _terms():
    return RewardContract.from_payload(_payload()).for_scale("Cus2", _objective())


def _station_profile():
    return load_method_auxiliary_profile(
        REPO_ROOT
        / "EVRPTW_Benchmark/Reinforcement_Learning/configs/evrptw_rl_station_auxiliary_v1.json"
    )


def _soft_profile():
    return load_method_auxiliary_profile(
        REPO_ROOT
        / "EVRPTW_Benchmark/Reinforcement_Learning/configs/"
        "drl_ts_soft_auxiliary_v1.json"
    )


class _ScriptedPolicy(torch.nn.Module):
    def __init__(self, method: str, actions: tuple[int, ...]):
        super().__init__()
        self.method = method
        self.actions = actions
        self.index = 0
        self.device = torch.device("cpu")
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def encode(self, *_args, **_kwargs):
        return None

    encode_static = encode
    initial_state = encode

    def logits(self, observation, *_args, **_kwargs):
        mask = torch.as_tensor(observation["action_mask"], dtype=torch.bool)
        preferred = torch.zeros(mask.shape[-1])
        preferred[self.actions[self.index]] = 1.0
        self.index += 1
        logits = ((4.0 + self.weight) * preferred).expand(mask.shape)
        logits = logits.masked_fill(~mask, -torch.inf)
        return logits if self.method == "AM_EVRPTW" else (logits, None)


class _ForcedEVRPTWRLPolicy(torch.nn.Module):
    """Ignores the policy mask so the environment can reject the second visit."""

    def __init__(self, station: int):
        super().__init__()
        self.station = station
        self.device = torch.device("cpu")

    def initial_state(self, *_args):
        return None

    def encode_static(self, *_args):
        return None

    def logits(self, observation, *_args, **_kwargs):
        shape = observation["action_mask"].shape
        logits = torch.zeros(shape)
        logits[..., self.station] = 10.0
        return logits, None


def test_contract_hash_objective_and_scale_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    contract = load_reward_contract(path)
    selected = contract.for_scale(2, _objective())
    assert selected.objective_scale == 10.0
    assert selected.terminal_failure_cost(True, 0.25) == 2.75

    tampered = _payload()
    tampered["scales"]["Cus2"]["failure_base"] = 1.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_reward_contract(path)
    with pytest.raises(ValueError, match="objective mismatch"):
        contract.for_scale("Cus2", ObjectiveConfig())
    with pytest.raises(ValueError, match="no calibration"):
        contract.for_scale("Cus500", _objective())


def test_frozen_all_scale_failure_floor_follows_q99_plus_one() -> None:
    """The candidate rule is not a theorem, so freeze its observed safeguard."""

    contract = load_reward_contract(
        REPO_ROOT
        / "EVRPTW_Benchmark/Reinforcement_Learning/configs/"
        "drl_reward_contract_energy_vehicle_v3.json"
    )
    statistics = contract.snapshot["calibration"]["scale_statistics"]
    for scale in ("Cus50", "Cus100", "Cus500", "Cus1000"):
        terms = contract.for_scale(scale, contract.objective_config)
        observed = statistics[scale]
        assert terms.objective_scale == observed["objective_cost_usd"]["median"]
        assert terms.failure_base == pytest.approx(
            observed["normalized_objective_q99_linear"] + 1.0
        )
        # This is the frozen empirical candidate rule, not a theorem that the
        # zero-distance failure floor exceeds every finite cohort outlier.
        assert terms.failure_base > observed["normalized_objective_q99_linear"]


def test_args_freeze_snapshot_and_explicit_environment_scale(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    args = SimpleNamespace(
        reward_contract=path, scale="Cus2", objective=_objective().to_dict()
    )
    terms = reward_contract_from_args(args)
    assert terms is not None
    assert args.reward_contract_snapshot == _payload()
    env = EVRPTWVectorEnvFast(
        _instance(),
        n_traj=1,
        use_jit_mask=False,
        objective_config=_objective(),
        reward_distance_scale_km=999.0,
        reward_objective_scale=terms.objective_scale,
        invalid_action_penalty=0.0,
    )
    assert env.reward_objective_scale == 10.0
    observation, _ = env.reset(seed=1)
    invalid = np.flatnonzero(~observation["action_mask"][0])[0]
    _, reward, _, truncated, info = env.step(np.asarray([invalid]))
    assert truncated[0] and reward[0] == 0.0
    assert info["failure_reason"].tolist() == ["invalid_action"]


def test_checkpoint_resume_requires_exact_contract_snapshot(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_payload()), encoding="utf-8")
    args = SimpleNamespace(
        protocol_id="reward-test",
        reward_contract=contract_path,
        scale="Cus2",
        objective=_objective().to_dict(),
    )
    reward_contract_from_args(args)
    policy = torch.nn.Linear(1, 1, bias=False)
    baseline = torch.nn.Linear(1, 1, bias=False)
    baseline.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters())
    checkpoint = tmp_path / "checkpoint.pt"
    protocol_trainers._save_checkpoint(
        checkpoint,
        method="AM-EVRPTW",
        data_pass=0,
        policy=policy,
        baseline=baseline,
        optimizer=optimizer,
        args=args,
    )
    protocol_trainers._load_checkpoint(
        checkpoint,
        policy=policy,
        baseline=baseline,
        optimizer=optimizer,
        protocol_id=args.protocol_id,
        objective_config=_objective(),
        reward_contract_args=args,
    )

    legacy_args = SimpleNamespace(scale="Cus2")
    with pytest.raises(ValueError, match="checkpoint reward contract mismatch"):
        protocol_trainers._load_checkpoint(
            checkpoint,
            policy=policy,
            baseline=baseline,
            optimizer=optimizer,
            protocol_id=args.protocol_id,
            objective_config=_objective(),
            reward_contract_args=legacy_args,
        )


def test_common_formal_cost_training_requires_reward_contract(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        objective=_objective().to_dict(),
        reward_contract=None,
        scale="Cus2",
        training_epochs=1,
        data_passes=None,
        training_stream_path=None,
        protocol_id="am-formal-cost-test",
        seed=1234,
        output_dir=tmp_path / "fresh",
        resume=False,
    )
    with pytest.raises(
        ValueError, match="formal cost training requires a frozen reward contract"
    ):
        protocol_trainers.prepare_training_objective(args)


def test_evrptw_rl_station_penalty_counts_only_legally_executed_visits() -> None:
    env = EVRPTWVectorEnvFast(
        _instance(), n_traj=1, info_level="full", use_jit_mask=False,
        objective_config=_objective(), reward_objective_scale=10.0,
        invalid_action_penalty=0.0,
    )
    module = importlib.import_module(
        "EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.rollout"
    )
    result = module.rollout(
        _ForcedEVRPTWRLPolicy(env.station_start),
        [env],
        decode_type="greedy",
        max_steps=2,
        seed=1,
        compute_log_likelihood=False,
        reward_contract=_terms(),
        method_auxiliary_profile=_station_profile(),
    )
    assert result.station_visits.item() == 1
    assert result.failure_reasons.tolist() == [["invalid_action"]]
    assert result.training_cost_components["station_visit_auxiliary"].item() == 0.15
    diagnostics = result.method_auxiliary_diagnostics
    assert diagnostics["station_visits_raw"].item() == 1
    assert diagnostics["station_visit_denominator"].item() == 2
    assert diagnostics["station_visits_normalized"].item() == 0.5


def test_evrptw_rl_contract_rollout_requires_method_auxiliary_profile() -> None:
    env = EVRPTWVectorEnvFast(
        _instance(), n_traj=1, info_level="full", use_jit_mask=False,
        objective_config=_objective(), reward_objective_scale=10.0,
        invalid_action_penalty=0.0,
    )
    module = importlib.import_module(
        "EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.rollout"
    )
    with pytest.raises(ValueError, match="requires its frozen method auxiliary"):
        module.rollout(
            _ForcedEVRPTWRLPolicy(env.station_start),
            [env],
            decode_type="greedy",
            max_steps=1,
            seed=1,
            compute_log_likelihood=False,
            reward_contract=_terms(),
        )


@pytest.mark.parametrize(
    ("runner", "module_name"),
    [
        (protocol_entrypoints.run_am, "AM_EVRPTW"),
        (protocol_entrypoints.run_evrptw_rl, "EVRPTW_RL"),
        (protocol_entrypoints.run_drl_ts, "DRL_TS"),
    ],
)
def test_formal_entrypoints_forward_frozen_contract(
    tmp_path: Path, monkeypatch, runner, module_name: str,
) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_payload()), encoding="utf-8")
    rollout_module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{module_name}.rollout"
    )
    observed = []

    def fake_rollout(_policy, envs, **kwargs):
        assert kwargs["reward_contract"].digest == _payload()["sha256"]
        assert all(env.unwrapped.reward_objective_scale == 10.0 for env in envs)
        assert all(env.unwrapped.invalid_action_penalty == 0.0 for env in envs)
        observed.append(kwargs)
        return SimpleNamespace(training_cost=torch.zeros(1, 1))

    def fake_train(**callbacks):
        callbacks["make_actor"]([_instance()], True, 1)

    monkeypatch.setattr(rollout_module, "rollout", fake_rollout)
    monkeypatch.setattr(
        protocol_entrypoints, "train_reinforce_data_passes", fake_train
    )
    args = SimpleNamespace(
        reward_contract=contract_path,
        method_auxiliary_profile=(
            _station_profile().source_path
            if module_name == "EVRPTW_RL"
            else (
                _soft_profile().source_path
                if module_name == "DRL_TS"
                else None
            )
        ),
        scale="Cus2",
        objective=_objective().to_dict(),
        training_rollout_steps=8,
        validation_decode_type="greedy",
        validation_candidates=1,
        samples_per_instance=1,
        incomplete_penalty_km=100.0,
        incomplete_penalty=100.0,
        station_visit_penalty=0.3,
        capacity_penalty=1.0,
        time_penalty=1.0,
        energy_penalty=1.0,
        batch_size=1,
        soft_stage_fraction=0.5,
        soft_stage_end_epoch=None,
    )
    pool = SimpleNamespace(
        reward_distance_scale_km=lambda _mode: 5.0,
        reward_scale_metadata={},
    )
    runner(args, pool, object(), object())
    assert len(observed) == 1
    if module_name == "EVRPTW_RL":
        assert observed[0]["method_auxiliary_profile"].digest == _station_profile().digest
    if module_name == "DRL_TS":
        assert args.capacity_penalty == 1.0
        assert args.time_penalty == 1.0
        assert args.energy_penalty == 1.0
        assert args.soft_violation_step_clip == 1.0
        assert args.soft_violation_component_clip == 1.0


@pytest.mark.parametrize(
    ("with_contract", "with_profile", "message"),
    [
        (True, False, "requires --method-auxiliary-profile"),
        (False, True, "requires a reward contract"),
    ],
)
def test_direct_drl_ts_runner_requires_signed_task_and_auxiliary_together(
    tmp_path: Path,
    monkeypatch,
    with_contract: bool,
    with_profile: bool,
    message: str,
) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_payload()), encoding="utf-8")
    monkeypatch.setattr(
        protocol_entrypoints,
        "train_reinforce_data_passes",
        lambda **_kwargs: pytest.fail("training must not start before the signed gate"),
    )
    args = SimpleNamespace(
        reward_contract=contract_path if with_contract else None,
        method_auxiliary_profile=(
            _soft_profile().source_path if with_profile else None
        ),
        scale="Cus2",
        objective=_objective().to_dict(),
        training_rollout_steps=8,
        validation_decode_type="greedy",
        validation_candidates=1,
        samples_per_instance=1,
        capacity_penalty=99.0,
        time_penalty=98.0,
        energy_penalty=97.0,
        incomplete_penalty=100.0,
        batch_size=1,
        soft_stage_fraction=0.5,
        soft_stage_end_epoch=None,
    )
    pool = SimpleNamespace(
        reward_distance_scale_km=lambda _mode: 5.0,
        reward_scale_metadata={},
    )
    with pytest.raises(ValueError, match=message):
        protocol_entrypoints.run_drl_ts(args, pool, object(), object())


@pytest.mark.parametrize(
    ("method", "soft"),
    [
        ("AM_EVRPTW", False),
        ("EVRPTW_RL", False),
        ("DRL_TS", False),
        ("DRL_TS", True),
    ],
)
def test_common_terminal_failure_cost_applies_once(method: str, soft: bool) -> None:
    env_cls = DRLTSSoftConstraintEnv if soft else DRLTSHardConstraintEnv
    env = (
        env_cls(
            _instance(), n_traj=1, info_level="full", use_jit_mask=False,
            objective_config=_objective(), reward_objective_scale=10.0,
            invalid_action_penalty=0.0,
        )
        if method == "DRL_TS"
        else EVRPTWVectorEnvFast(
            _instance(), n_traj=1, info_level="full", use_jit_mask=False,
            objective_config=_objective(), reward_objective_scale=10.0,
            invalid_action_penalty=0.0,
        )
    )
    module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{method}.rollout"
    )
    kwargs = {"reward_contract": _terms()}
    if method == "AM_EVRPTW":
        kwargs["incomplete_penalty_km"] = 99_999.0
    elif method == "DRL_TS":
        kwargs["soft_constraints"] = soft
    elif method == "EVRPTW_RL":
        kwargs["method_auxiliary_profile"] = _station_profile()
    result = module.rollout(
        _ScriptedPolicy(method, (1,)),
        [env],
        decode_type="greedy",
        max_steps=1,
        seed=1,
        **kwargs,
    )
    expected_terminal = 2.0 + 3.0 * 0.5
    expected = result.objective_value.item() / 10.0 + expected_terminal
    if method == "EVRPTW_RL":
        expected += 0.3 * result.station_visits.item() / 2.0
    if method == "DRL_TS":
        expected += (
            result.capacity_violation.item()
            + result.time_violation.item()
            + result.energy_violation.item()
        )
    assert result.training_cost.item() == pytest.approx(expected)
    components = result.training_cost_components
    assert components["terminal_failure_base"].item() == 2.0
    assert components["terminal_unserved"].item() == 1.5
    assert components["terminal_task_total"].item() == pytest.approx(expected_terminal)
    additive = (
        components["base_objective"]
        + components["terminal_failure_base"]
        + components["terminal_unserved"]
    )
    if method == "EVRPTW_RL":
        additive += components["station_visit_auxiliary"]
    if method == "DRL_TS":
        additive += (
            components["capacity_penalty"]
            + components["time_penalty"]
            + components["energy_penalty"]
        )
    torch.testing.assert_close(additive.float(), result.training_cost.cpu())
    assert result.failure_reasons.tolist() == [["rollout_budget_exhausted"]]


@pytest.mark.parametrize("method", ["AM_EVRPTW", "EVRPTW_RL", "DRL_TS"])
def test_hard_rollout_all_customers_served_without_return_pays_one_failure_base(
    method: str,
) -> None:
    env = (
        DRLTSHardConstraintEnv(
            _instance(), n_traj=1, info_level="full", use_jit_mask=False,
            objective_config=_objective(), reward_objective_scale=10.0,
            invalid_action_penalty=0.0,
        )
        if method == "DRL_TS"
        else EVRPTWVectorEnvFast(
            _instance(), n_traj=1, info_level="full", use_jit_mask=False,
            objective_config=_objective(), reward_objective_scale=10.0,
            invalid_action_penalty=0.0,
        )
    )
    module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{method}.rollout"
    )
    kwargs = {"reward_contract": _terms()}
    if method == "AM_EVRPTW":
        kwargs["incomplete_penalty_km"] = 99_999.0
    elif method == "EVRPTW_RL":
        kwargs["method_auxiliary_profile"] = _station_profile()
    else:
        kwargs["soft_constraints"] = False
    result = module.rollout(
        _ScriptedPolicy(method, (1, 2)),
        [env],
        decode_type="greedy",
        max_steps=2,
        seed=1,
        **kwargs,
    )

    assert result.served_customers.item() == 2
    assert not result.feasible.item()
    components = result.training_cost_components
    assert components["terminal_failure_base"].item() == 2.0
    assert components["terminal_unserved"].item() == 0.0
    assert components["terminal_task_total"].item() == 2.0
    torch.testing.assert_close(
        result.training_cost.cpu(),
        (components["base_objective"] + components["terminal_failure_base"]).float(),
    )
    assert result.failure_reasons.tolist() == [
        ["rollout_budget_exhausted_not_returned"]
    ]
    assert result.infos[0]["failure_reason"].tolist() == [
        "rollout_budget_exhausted_not_returned"
    ]


def test_rollout_rejects_contract_without_explicit_objective_scale() -> None:
    env = EVRPTWVectorEnvFast(
        _instance(), n_traj=1, info_level="full", use_jit_mask=False,
        objective_config=_objective(), reward_distance_scale_km=5.0,
    )
    module = importlib.import_module(
        "EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.rollout"
    )
    with pytest.raises(ValueError, match="frozen reward objective scale"):
        module.rollout(
            _ScriptedPolicy("AM_EVRPTW", (1,)),
            [env],
            decode_type="greedy",
            max_steps=1,
            seed=1,
            incomplete_penalty_km=1.0,
            reward_contract=_terms(),
        )


def test_drl_ts_completed_soft_violation_does_not_pay_failure_base() -> None:
    instance = _instance()
    instance.vehicle["cargo_capacity_cm3"] = 1.5
    env = DRLTSSoftConstraintEnv(
        instance, n_traj=1, info_level="full", use_jit_mask=False,
        objective_config=_objective(), reward_objective_scale=10.0,
        invalid_action_penalty=0.0,
    )
    module = importlib.import_module(
        "EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout"
    )
    result = module.rollout(
        _ScriptedPolicy("DRL_TS", (1, 2, 0)),
        [env],
        decode_type="greedy",
        max_steps=3,
        seed=1,
        soft_constraints=True,
        reward_contract=_terms(),
    )
    assert result.capacity_violation.item() > 0.0
    assert not result.feasible.item()
    assert result.training_cost_components["terminal_failure_base"].item() == 0.0
    assert result.training_cost_components["terminal_unserved"].item() == 0.0
    assert result.training_cost_components["terminal_task_total"].item() == 0.0
    components = result.training_cost_components
    auxiliary = (
        components["capacity_penalty"]
        + components["time_penalty"]
        + components["energy_penalty"]
    )
    torch.testing.assert_close(components["soft_auxiliary_total"], auxiliary)
    torch.testing.assert_close(
        result.training_cost.cpu(),
        (components["base_objective"] + auxiliary).float(),
    )
    assert all(
        0.0 <= components[name].item() <= 1.0
        for name in ("capacity_penalty", "time_penalty", "energy_penalty")
    )
    diagnostics = result.soft_violation_diagnostics
    assert diagnostics is not None
    assert diagnostics["soft_capacity_raw_sum"].item() > 0.0
    assert diagnostics["soft_capacity_applicable_transitions"].item() == 2
    assert result.failure_reasons.tolist() == [["completed_with_soft_violation"]]
