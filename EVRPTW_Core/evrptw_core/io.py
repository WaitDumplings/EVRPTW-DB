from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from evrptw_core.schema import EVRPTWInstance, EVRPTWSolution


def _load_pickle_dict(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected pickle dict at {path}, got {type(data)!r}")
    return data


def load_instance(path: str | Path) -> EVRPTWInstance:
    return EVRPTWInstance.from_dict(_load_pickle_dict(path))


def load_solution(path: str | Path) -> EVRPTWSolution:
    return EVRPTWSolution.from_dict(_load_pickle_dict(path))


def save_solution(path: str | Path, solution: EVRPTWSolution) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(solution.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
    return out
