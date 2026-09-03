from __future__ import annotations

import numpy as np

from ..EVRPTW_Env import EVRPTWVectorEnvFast


class DRLTSHardConstraintEnv(EVRPTWVectorEnvFast):
    """Canonical hard environment with the station mask from Chen et al.

    DRL-TS permits a station to be revisited later in a route, but masks every
    station while the vehicle is at the depot or another station because the
    vehicle is already fully charged in those states. This rules out depot-to-
    station and consecutive station-to-station actions without imposing a
    global one-visit restriction.
    """

    def _compute_action_mask(self) -> np.ndarray:
        mask = super()._compute_action_mask()
        for trajectory in range(self.n_traj):
            current = int(self.last[trajectory])
            if current == 0 or self._is_station(current):
                mask[trajectory, self.station_nodes] = False
        return mask


__all__ = ["DRLTSHardConstraintEnv"]
