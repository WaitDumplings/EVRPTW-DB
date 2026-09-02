from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def seeded_pass_order(size: int, seed: int, data_pass: int) -> np.ndarray:
    """Return the deterministic permutation for one complete data pass."""

    if size <= 0:
        raise ValueError("data-pass dataset must be non-empty")
    if data_pass < 1:
        raise ValueError("data_pass is one-based and must be positive")
    sequence = np.random.SeedSequence([int(seed), int(data_pass), 0x45565250])
    order = np.arange(int(size), dtype=np.int64)
    np.random.default_rng(sequence).shuffle(order)
    return order


def pass_batches(
    rows: Sequence[T],
    *,
    seed: int,
    data_pass: int,
    physical_batch_size: int,
) -> list[list[T]]:
    """Partition one seeded pass without replacement or dropped rows."""

    if physical_batch_size <= 0:
        raise ValueError("physical_batch_size must be positive")
    order = seeded_pass_order(len(rows), seed, data_pass)
    return [
        [rows[int(index)] for index in order[start : start + physical_batch_size]]
        for start in range(0, len(order), int(physical_batch_size))
    ]


@dataclass
class DataPassState:
    protocol_id: str
    completed_data_passes: int = 0
    instances_seen: int = 0
    customer_exposures: int = 0
    optimizer_steps: int = 0
    environment_transitions: int = 0
    last_checkpoint: str | None = None

    @classmethod
    def load(cls, path: str | Path, *, protocol_id: str) -> "DataPassState":
        source = Path(path)
        if not source.exists():
            return cls(protocol_id=protocol_id)
        payload = json.loads(source.read_text(encoding="utf-8"))
        state = cls(**payload)
        if state.protocol_id != protocol_id:
            raise ValueError(
                f"resume protocol mismatch: {state.protocol_id} != {protocol_id}"
            )
        return state

    def atomic_write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(asdict(self), stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class SeededPassSequence(Generic[T]):
    def __init__(self, rows: Sequence[T], *, seed: int) -> None:
        if not rows:
            raise ValueError("data-pass sequence must be non-empty")
        self.rows = rows
        self.seed = int(seed)

    def batches(self, data_pass: int, physical_batch_size: int) -> list[list[T]]:
        return pass_batches(
            self.rows,
            seed=self.seed,
            data_pass=data_pass,
            physical_batch_size=physical_batch_size,
        )


__all__ = [
    "DataPassState",
    "SeededPassSequence",
    "pass_batches",
    "seeded_pass_order",
]
