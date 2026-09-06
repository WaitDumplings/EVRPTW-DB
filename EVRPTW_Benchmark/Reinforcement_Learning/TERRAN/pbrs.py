from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gymnasium import Wrapper
import numpy as np


TERMINAL_TASK_REWARD_UNIT = "normalized_objective_cost"


@dataclass(frozen=True)
class PotentialRewardConfig:
    """PBRS controls for TERRAN-style RL training."""

    use_customer_pbrs: bool = False
    use_repair_distance_pbrs: bool = False
    use_feasible_ratio_pbrs: bool = False
    # ``use_terminal_heuristic`` is retained only for replaying legacy runs.
    # New cost training uses the separately named task penalty below; it is not
    # a potential and is never annealed with PBRS.
    use_terminal_heuristic: bool = False
    use_terminal_task_penalty: bool = False
    customer_pbrs_mode: str = "progress"  # strict PBRS: gamma * Phi(s_next) - Phi(s)
    gamma: float = 1.0
    alpha: float = 2.0
    beta: float = 0.5
    customer_pbrs_coef: float = 1.0
    customer_progress_budget: float = 0.5
    customer_progress_mix: float = 0.5
    repair_progress_coef: float = 0.5
    feasible_ratio_coef: float = 0.0
    pbrs_clip: float | None = None
    # Canonical terminal task reward, expressed in the same dimensionless
    # units as normalized economic cost (C / S_N).  Keep this separate from
    # ``success_bonus``, which belongs to the disabled legacy heuristic.
    terminal_success_bonus: float = 0.0
    success_bonus: float = 0.1
    failure_penalty: float = 0.5
    failure_base: float = 0.0
    unserved_coefficient: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.gamma) <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        mode = self.customer_pbrs_mode.lower()
        if mode in {"serve", "served", "ratio_progress"}:
            mode = "progress"
        if mode not in {"progress", "direct_progress"}:
            raise ValueError("customer_pbrs_mode must be progress or direct_progress")
        object.__setattr__(self, "customer_pbrs_mode", mode)
        if self.use_terminal_heuristic and self.use_terminal_task_penalty:
            raise ValueError(
                "legacy terminal heuristic and terminal task penalty are mutually exclusive"
            )
        for name in (
            "terminal_success_bonus",
            "failure_base",
            "unserved_coefficient",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


class PotentialRewardWrapper(Wrapper):
    """Gymnasium wrapper for optional TERRAN PBRS.

    The repair-distance heuristic matches the prior DRL implementation: each
    unserved customer contributes the distance of a one-customer depot-customer-
    depot repair route. Progress is positive when this remaining repair workload
    decreases.
    """

    def __init__(self, env, config: PotentialRewardConfig | None = None, **kwargs: Any) -> None:
        super().__init__(env)
        self.config = config or PotentialRewardConfig(**kwargs)
        self._last_obs: dict[str, np.ndarray] | None = None
        self._last_info: dict[str, Any] | None = None
        self._last_finished: np.ndarray | None = None
        self._single_customer_repair_dist: np.ndarray | None = None
        self._node_to_depot_repair_dist: np.ndarray | None = None
        self._total_customer_repair_dist: float = 1.0
        self.reward_scale = 1.0

    def set_reward_scale(self, reward_scale: float) -> None:
        self.reward_scale = max(float(reward_scale), 0.0)

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._prepare_repair_distance_cache()
        self._last_obs = obs
        self._last_info = info
        self._last_finished = np.zeros_like(info["served_customers"], dtype=bool)
        zero = np.zeros_like(info["served_customers"], dtype=np.float32)
        return obs, self._augment_info(info, zero)

    def step(self, action):
        if self._last_obs is None or self._last_info is None or self._last_finished is None:
            raise RuntimeError("PotentialRewardWrapper.step called before reset.")
        prev_obs = self._last_obs
        prev_info = self._last_info
        prev_finished = self._last_finished.copy()
        prev_repair = self._remaining_repair_ratio()
        prev_objective_distance = np.asarray(
            self.unwrapped.objective_distance_km, dtype=np.float64
        ).copy()

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        distance_delta = np.asarray(
            self.unwrapped.objective_distance_km, dtype=np.float64
        ) - prev_objective_distance
        if bool(getattr(self.unwrapped, "normalize_reward", False)):
            distance_reward = -distance_delta / max(
                float(self.unwrapped.reward_distance_scale_km), 1e-12
            )
        else:
            distance_reward = -distance_delta
        distance_reward = distance_reward.astype(np.float32)
        objective_delta = np.asarray(
            info.get("objective_value", self.unwrapped.objective_distance_km), dtype=np.float64
        ) - np.asarray(
            prev_info.get("objective_value", prev_objective_distance), dtype=np.float64
        )
        objective_scale = (
            max(float(getattr(self.unwrapped, "reward_objective_scale", self.unwrapped.reward_distance_scale_km)), 1e-12)
            if bool(getattr(self.unwrapped, "normalize_reward", False)) else 1.0
        )
        objective_reward = (-objective_delta / objective_scale).astype(np.float32)
        cost_rewards = {}
        for name in ("electricity_cost", "vehicle_cost"):
            current = info.get(f"{name}_usd")
            previous = prev_info.get(f"{name}_usd")
            cost_rewards[name] = (
                -(np.asarray(current, dtype=np.float64) - np.asarray(previous, dtype=np.float64)) / objective_scale
                if current is not None and previous is not None else np.zeros_like(distance_reward)
            ).astype(np.float32)
        # The environment already charged the active objective, including every
        # vehicle departure. Only the residual is an invalid-action/task penalty.
        # Keep historical normalized-km diagnostics separate from USD components.
        base_non_distance = (
            base_reward.astype(np.float32) - objective_reward
        ).astype(np.float32)

        now_finished = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        active = ~prev_finished
        newly_finished = now_finished & active
        customer = self._customer_pbrs(
            prev_info, info, active=active, newly_finished=newly_finished
        )
        repair = self._repair_distance_pbrs(
            prev_repair, active=active, newly_finished=newly_finished
        )
        feasible = self._feasible_ratio_pbrs(
            prev_obs, obs, active=active, newly_finished=newly_finished
        )
        legacy_terminal = self._terminal_heuristic(
            info, terminated, truncated, prev_finished
        )
        (
            terminal,
            terminal_success,
            terminal_base,
            terminal_unserved,
        ) = self._terminal_task_components(
            info, terminated, truncated, prev_finished
        )
        scale = float(self.reward_scale)
        customer = (scale * customer).astype(np.float32)
        repair = (scale * repair).astype(np.float32)
        feasible = (scale * feasible).astype(np.float32)
        # Completion/failure is part of the task objective, not an auxiliary
        # potential.  Annealing it together with PBRS made failures progressively
        # cheaper and caused large-scale policies to collapse to short episodes.
        legacy_terminal = legacy_terminal.astype(np.float32)
        terminal = terminal.astype(np.float32)
        shaped = (
            base_reward.astype(np.float32)
            + customer
            + repair
            + feasible
            + legacy_terminal
            + terminal
        )

        self._last_obs = obs
        self._last_info = info
        self._last_finished = now_finished
        out_info = dict(info)
        out_info["reward_components"] = {
            "base": base_reward.astype(np.float32).copy(),
            "distance": distance_reward.copy(),
            "objective": objective_reward.copy(),
            "electricity_cost": cost_rewards["electricity_cost"].copy(),
            "vehicle_cost": cost_rewards["vehicle_cost"].copy(),
            "base_non_objective": base_non_distance.copy(),
            "base_non_distance": base_non_distance.copy(),
            "pbrs_customer": customer.copy(),
            "pbrs_repair_distance": repair.copy(),
            "pbrs_feasible_ratio": feasible.copy(),
            "terminal_heuristic": legacy_terminal.copy(),
            "terminal_task_total": terminal.copy(),
            "terminal_success_bonus": terminal_success.copy(),
            "terminal_failure_base": terminal_base.copy(),
            "terminal_unserved": terminal_unserved.copy(),
            "shaped": shaped.copy(),
            "pbrs_scale": np.full_like(shaped, scale, dtype=np.float32),
        }
        return obs, shaped, terminated, truncated, out_info

    def _prepare_repair_distance_cache(self) -> None:
        env = self.unwrapped
        distance = np.asarray(env.distance_km, dtype=np.float32)
        n = int(env.num_customers)
        customer_nodes = np.arange(1, 1 + n, dtype=int)
        single = distance[0, customer_nodes] + distance[customer_nodes, 0]
        self._single_customer_repair_dist = single.astype(np.float32)
        self._node_to_depot_repair_dist = distance[:, 0].astype(np.float32)
        self._total_customer_repair_dist = float(max(float(single.sum()), 1e-6))

    def _remaining_repair_ratio(self) -> np.ndarray:
        env = self.unwrapped
        if self._single_customer_repair_dist is None or self._node_to_depot_repair_dist is None:
            self._prepare_repair_distance_cache()
        assert self._single_customer_repair_dist is not None
        assert self._node_to_depot_repair_dist is not None
        unvisited = (~env.visited[:, 1 : 1 + env.num_customers]).astype(np.float32)
        remaining_customer = unvisited @ self._single_customer_repair_dist
        current_to_depot = self._node_to_depot_repair_dist[env.last]
        return np.maximum((remaining_customer + current_to_depot) / self._total_customer_repair_dist, 0.0).astype(np.float32)

    def _repair_distance_pbrs(
        self,
        prev_repair_ratio: np.ndarray,
        *,
        active: np.ndarray,
        newly_finished: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        reward = np.zeros_like(prev_repair_ratio, dtype=np.float32)
        if not cfg.use_repair_distance_pbrs or cfg.repair_progress_coef == 0.0:
            return reward
        post_repair_ratio = self._remaining_repair_ratio()
        # A negative remaining-work potential has Phi=0 on success and avoids
        # the -(1-gamma) constant introduced by the previous positive-progress
        # representation.  Both encode the same ordering only when gamma=1.
        prev_phi = -prev_repair_ratio
        post_phi = -post_repair_ratio
        # Episodic PBRS requires one shared terminal potential.  Set it to zero
        # for every newly finished trajectory, including failed truncations.
        post_phi = np.where(newly_finished, 0.0, post_phi)
        reward = float(cfg.repair_progress_coef) * (float(cfg.gamma) * post_phi - prev_phi)
        reward = np.where(active, reward, 0.0)
        return self._clip(reward.astype(np.float32))

    def _customer_pbrs(
        self,
        prev_info: dict[str, Any],
        next_info: dict[str, Any],
        *,
        active: np.ndarray,
        newly_finished: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        served_prev = np.asarray(prev_info["served_customers"], dtype=np.float32)
        served_next = np.asarray(next_info["served_customers"], dtype=np.float32)
        reward = np.zeros_like(served_next, dtype=np.float32)
        if not cfg.use_customer_pbrs or cfg.customer_pbrs_coef == 0.0:
            return reward
        n = max(float(getattr(self.unwrapped, "num_customers", 1)), 1.0)
        beta = max(float(cfg.beta), 1e-8)
        prev_phi = self._direct_progress_potential(served_prev, n, beta)
        next_phi = self._direct_progress_potential(served_next, n, beta)
        # Scale-invariant progress budget. The linear term prevents early-customer
        # signal from vanishing too aggressively at large scales, while the
        # nonlinear term still gives larger marginal reward near completion.
        served_delta = np.maximum(served_next - served_prev, 0.0) / n
        nonlinear_delta = next_phi - prev_phi
        mix = float(np.clip(cfg.customer_progress_mix, 0.0, 1.0))
        progress_delta = (1.0 - mix) * served_delta + mix * nonlinear_delta
        progress_budget = float(cfg.customer_pbrs_coef) * float(cfg.customer_progress_budget)
        if cfg.customer_pbrs_mode == "progress":
            prev_ratio = (served_prev / n).clip(0.0, 1.0)
            next_ratio = (served_next / n).clip(0.0, 1.0)
            # Use negative remaining work rather than positive completed work.
            # This keeps success at Phi=0 and preserves a positive local signal
            # for serving a customer on long horizons when gamma < 1.
            prev_remaining = (1.0 - mix) * (1.0 - prev_ratio) + mix * (1.0 - prev_phi)
            next_remaining = (1.0 - mix) * (1.0 - next_ratio) + mix * (1.0 - next_phi)
            prev_potential = -prev_remaining
            next_potential = -next_remaining
            next_potential = np.where(newly_finished, 0.0, next_potential)
            reward = progress_budget * (
                float(cfg.gamma) * next_potential - prev_potential
            )
        else:
            reward = progress_budget * progress_delta
        reward = np.where(active, reward, 0.0)
        return reward.astype(np.float32)

    def _feasible_ratio_pbrs(
        self,
        prev_obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        *,
        active: np.ndarray,
        newly_finished: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        reward = np.zeros(prev_obs["action_mask"].shape[0], dtype=np.float32)
        if not cfg.use_feasible_ratio_pbrs or cfg.feasible_ratio_coef == 0.0:
            return reward
        prev_phi = self._feasible_customer_ratio(prev_obs)
        next_phi = self._feasible_customer_ratio(next_obs)
        next_phi = np.where(newly_finished, 0.0, next_phi)
        reward = float(cfg.feasible_ratio_coef) * (float(cfg.gamma) * next_phi - prev_phi)
        reward = np.where(active, reward, 0.0)
        return self._clip(reward.astype(np.float32))

    def _terminal_heuristic(self, next_info: dict[str, Any], terminated: np.ndarray, truncated: np.ndarray, prev_finished: np.ndarray) -> np.ndarray:
        cfg = self.config
        reward = np.zeros_like(np.asarray(next_info["served_customers"], dtype=np.float32))
        if not cfg.use_terminal_heuristic:
            return reward
        now_finished = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        newly_finished = now_finished & (~prev_finished)
        if not np.any(newly_finished):
            return reward
        success = np.asarray(next_info["success"], dtype=bool)
        served_ratio = np.asarray(next_info["served_customers"], dtype=np.float32) / max(float(getattr(self.unwrapped, "num_customers", 1)), 1.0)
        reward[newly_finished & success] += float(cfg.success_bonus)
        fail = newly_finished & (~success)
        reward[fail] -= float(cfg.failure_penalty) * (1.0 - served_ratio[fail])
        return reward.astype(np.float32)

    def _terminal_task_components(
        self,
        next_info: dict[str, Any],
        terminated: np.ndarray,
        truncated: np.ndarray,
        prev_finished: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return one task outcome reward at the first terminal transition.

        A successful completion receives ``+terminal_success_bonus``.  Failure
        components use reward signs (negative costs); in particular, an episode
        that served every customer but did not return to the depot receives
        ``-failure_base`` rather than escaping with zero penalty.  Success and
        failure masks are disjoint and ``newly_finished`` prevents double
        settlement on padded transitions.
        """

        served = np.asarray(next_info["served_customers"], dtype=np.float32)
        zero = np.zeros_like(served, dtype=np.float32)
        if not self.config.use_terminal_task_penalty:
            return zero, zero.copy(), zero.copy(), zero.copy()
        now_finished = np.asarray(terminated, dtype=bool) | np.asarray(
            truncated, dtype=bool
        )
        newly_finished = now_finished & (~prev_finished)
        success = np.asarray(next_info["success"], dtype=bool)
        succeeded = newly_finished & success
        failed = newly_finished & (~success)
        if not np.any(succeeded) and not np.any(failed):
            return zero, zero.copy(), zero.copy(), zero.copy()
        customer_count = max(
            float(getattr(self.unwrapped, "num_customers", 1)), 1.0
        )
        unserved_fraction = np.clip(1.0 - served / customer_count, 0.0, 1.0)
        success_reward = np.zeros_like(zero)
        base = np.zeros_like(zero)
        unserved = np.zeros_like(zero)
        success_reward[succeeded] = float(self.config.terminal_success_bonus)
        base[failed] = -float(self.config.failure_base)
        unserved[failed] = (
            -float(self.config.unserved_coefficient) * unserved_fraction[failed]
        )
        return (
            (success_reward + base + unserved).astype(np.float32),
            success_reward,
            base,
            unserved,
        )

    @staticmethod
    def _direct_progress_potential(served: np.ndarray, n: float, beta: float) -> np.ndarray:
        served_ratio = (served / n).clip(0.0, 1.0)
        return 1.0 - (1.0 - served_ratio).clip(0.0, 1.0) ** beta

    def _feasible_customer_ratio(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        mask = np.asarray(obs["action_mask"], dtype=bool)
        n = int(getattr(self.unwrapped, "num_customers", 0))
        if n <= 0:
            return np.zeros(mask.shape[0], dtype=np.float32)
        return mask[:, 1 : 1 + n].sum(axis=1).astype(np.float32) / float(n)

    def _clip(self, reward: np.ndarray) -> np.ndarray:
        clip_value = self.config.pbrs_clip
        if clip_value is None or clip_value <= 0:
            return reward.astype(np.float32)
        return np.clip(reward, -float(clip_value), float(clip_value)).astype(np.float32)

    @staticmethod
    def _augment_info(info: dict[str, Any], reward: np.ndarray) -> dict[str, Any]:
        info = dict(info)
        info["reward_components"] = {
            "base": reward.copy(),
            "distance": reward.copy(),
            "objective": reward.copy(),
            "electricity_cost": reward.copy(),
            "vehicle_cost": reward.copy(),
            "base_non_objective": reward.copy(),
            "base_non_distance": reward.copy(),
            "pbrs_customer": reward.copy(),
            "pbrs_repair_distance": reward.copy(),
            "pbrs_feasible_ratio": reward.copy(),
            "terminal_heuristic": reward.copy(),
            "terminal_task_total": reward.copy(),
            "terminal_success_bonus": reward.copy(),
            "terminal_failure_base": reward.copy(),
            "terminal_unserved": reward.copy(),
            "shaped": reward.copy(),
            "pbrs_scale": np.ones_like(reward, dtype=np.float32),
        }
        return info


__all__ = [
    "TERMINAL_TASK_REWARD_UNIT",
    "PotentialRewardConfig",
    "PotentialRewardWrapper",
]
