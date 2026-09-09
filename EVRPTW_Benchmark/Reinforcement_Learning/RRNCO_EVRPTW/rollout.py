from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..common.reward_contract import RewardScaleContract
from ..DRL_TS.rollout import DRLTSRollout, normalized_edge_matrices
from ..DRL_TS.rollout import rollout as _matrix_rollout
from .model import RRNCOEVPolicy


def rollout(
    policy: RRNCOEVPolicy,
    envs: Sequence[Any],
    *,
    decode_type: str,
    max_steps: int,
    seed: int,
    incomplete_penalty: float = 100.0,
    compute_log_likelihood: bool = True,
    reward_contract: RewardScaleContract | None = None,
) -> DRLTSRollout:
    """Use the shared hard EVRPTW rollout; only the policy encoder changes."""

    return _matrix_rollout(
        policy,
        envs,
        decode_type=decode_type,
        max_steps=max_steps,
        seed=seed,
        soft_constraints=False,
        incomplete_penalty=incomplete_penalty,
        compute_log_likelihood=compute_log_likelihood,
        reward_contract=reward_contract,
    )


__all__ = ["normalized_edge_matrices", "rollout"]
