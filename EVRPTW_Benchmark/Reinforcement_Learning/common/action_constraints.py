"""Versioned DRL action restrictions, separate from objective and physical data."""
from __future__ import annotations

from typing import Any


ACTION_CONSTRAINT_CONTRACT_ID = "drl_no_consecutive_cs_v1"


def require_checkpoint_action_contract(checkpoint: dict[str, Any]) -> None:
    args = checkpoint.get("args", {}) or {}
    if not isinstance(args, dict):
        args = vars(args)
    config = checkpoint.get("config", {}) or {}
    values = [
        checkpoint.get("action_constraint_contract_id"),
        config.get("action_constraint_contract_id"),
        args.get("action_constraint_contract_id"),
    ]
    present = [value for value in values if value is not None]
    if not present or any(value != ACTION_CONSTRAINT_CONTRACT_ID for value in present):
        raise ValueError(
            "checkpoint action constraint contract mismatch: this DRL revision "
            "forbids consecutive CS visits; start fresh instead of silently "
            "reusing a checkpoint from different or undocumented action rules"
        )


def consecutive_cs_arcs(instance: Any, routes: list[list[int]]) -> list[tuple[int, int, int]]:
    """Return (route index, origin, destination) for adjacent CS terminals only."""
    station_start = int(instance.num_customers) + 1
    station_end = station_start + int(instance.num_charging_stations)
    return [
        (route_index, int(origin), int(destination))
        for route_index, route in enumerate(routes)
        for origin, destination in zip(route, route[1:])
        if station_start <= int(origin) < station_end
        and station_start <= int(destination) < station_end
    ]
