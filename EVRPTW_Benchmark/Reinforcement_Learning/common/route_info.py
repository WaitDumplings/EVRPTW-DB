from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def finalize_route_infos(
    envs: Sequence[Any], infos: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Export routes once after a rollout without losing adapter-specific info.

    Environments may emit light intermediate info while still retaining their
    internal route history. Use this only after the final step, including a
    rollout stopped at its step limit, before independent route verification.
    ``get_routes`` retains the environment's existing partial-route convention.
    """
    if len(envs) != len(infos):
        raise ValueError("one final info is required for each environment")

    from evrptw_core.schema import merge_route_sequences

    result: list[dict[str, Any]] = []
    for env, info in zip(envs, infos):
        routes = env.unwrapped.get_routes()
        result.append(
            {
                **info,
                "routes": routes,
                "route_sequence": [merge_route_sequences(row) for row in routes],
            }
        )
    return result


__all__ = ["finalize_route_infos"]
