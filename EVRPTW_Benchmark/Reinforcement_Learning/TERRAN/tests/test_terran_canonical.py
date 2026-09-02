from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_fixed_epoch_protocol_does_not_expand_to_a_full_data_pass(
    tmp_path: Path, monkeypatch
) -> None:
    class Pool:
        def __len__(self) -> int:
            return 5_000

    monkeypatch.setattr(terran_protocol, "Stage2TaskPool", lambda **_kwargs: Pool())
    args = SimpleNamespace(
        training_epochs=25,
        data_passes=None,
        stage2_dataset_path=Path("train.parquet"),
        stage2_family_root=Path("families"),
        stage2_scale="Cus100",
        stage2_split_ids="train",
        stage2_track_ids="train",
        output_dir=tmp_path,
        resume=False,
        protocol_id="fixed-epoch-test",
        num_envs_per_gpu=200,
        physical_batch_size=200,
        effective_batch_size=200,
        training_rollout_steps=140,
        seed=1234,
        max_batches_per_pass=None,
        validation_every_passes=5,
        validation_checkpoints=1,
        pilot_mode=False,
    )
    configured, meta = terran_protocol.configure_protocol(args, {})
    assert configured["training"]["epochs"] == 25
    assert configured["training"]["checkpoint_interval"] == 25
    assert configured["protocol"]["budget_mode"] == "fixed_training_epochs"
    assert configured["protocol"]["epochs_per_pass"] == 25
    assert configured["protocol"]["views_per_pass"] == 5_000
    assert meta is not None and meta["physical_batch_size"] == 200


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
    assert configured["training"]["rollout_steps"] == 140
    assert configured["protocol"]["training_rollout_steps"] == 140
