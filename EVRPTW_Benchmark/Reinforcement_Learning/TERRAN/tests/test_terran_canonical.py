from __future__ import annotations

import json
import sys
from pathlib import Path

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
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import (
    rollout_eval_batch,
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
