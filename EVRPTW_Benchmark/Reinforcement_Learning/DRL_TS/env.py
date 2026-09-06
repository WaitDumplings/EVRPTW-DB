from __future__ import annotations

import numpy as np

from ..EVRPTW_Env import EVRPTWVectorEnvFast


class DRLTSHardConstraintEnv(EVRPTWVectorEnvFast):
    """Canonical hard environment with the station mask from Chen et al.

    The paper-specific depot-to-station mask is retained in addition to the
    shared no-consecutive-CS rule. The shared environment also disallows
    revisiting a physical station within the same vehicle route; returning to
    the depot resets that record for the next vehicle. These are explicit
    search restrictions, not claims of dominance on directed road matrices.
    """

    def _compute_action_mask(self) -> np.ndarray:
        mask = super()._compute_action_mask()
        for trajectory in range(self.n_traj):
            current = int(self.last[trajectory])
            if current == 0 or self._is_station(current):
                mask[trajectory, self.station_nodes] = False
        return mask


__all__ = ["DRLTSHardConstraintEnv"]
