from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STREAM_SCHEMA = "drl_training_id_stream_v3"
REQUIRED_INDEX_COLUMNS = {
    "view_id",
    "family_id",
    "split_id",
    "track_id",
    "city_slug",
    "scale_id",
    "customer_count",
    "day_type",
}
STREAM_COLUMNS = [
    "stream_position",
    "view_id",
    "family_id",
    "city_slug",
    "day_type",
    "scale_id",
    "customer_count",
    "source_row_position",
]


def normalize_scale(value: str | int) -> str:
    raw = str(value).strip().lower()
    suffix = raw[3:] if raw.startswith("cus") else raw
    if not suffix.isdigit() or int(suffix) <= 0:
        raise ValueError(f"invalid scale: {value}")
    return f"Cus{int(suffix)}"


def build_training_stream(
    index: pd.DataFrame,
    *,
    scale: str | int,
    seed: int,
    sample_count: int,
    allowed_family_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build one method-independent, prefix-stable training ID stream.

    Each deterministic cycle visits every eligible view exactly once. A longer
    stream with the same source/support/scale/seed therefore preserves the
    complete shorter stream as its prefix. Complete cycles reproduce the exact
    city/day-type composition of the eligible pool; a final partial cycle is a
    seeded sample without replacement. No content hashes are computed.
    """

    missing = sorted(REQUIRED_INDEX_COLUMNS.difference(index.columns))
    if missing:
        raise ValueError(f"training index is missing columns: {missing}")
    if index["view_id"].astype(str).duplicated().any():
        raise ValueError("training index contains duplicate view_id values")
    selected_scale = normalize_scale(scale)
    frame = index.copy()
    normalized = frame["scale_id"].map(normalize_scale)
    frame = frame.loc[
        (normalized == selected_scale)
        & (frame["split_id"].astype(str) == "train")
        & (frame["track_id"].astype(str) == "train")
    ].copy()
    if allowed_family_ids is not None:
        allowed = {str(value) for value in allowed_family_ids}
        if not allowed:
            raise ValueError("allowed parent-family support is empty")
        frame = frame.loc[frame["family_id"].astype(str).isin(allowed)].copy()
    if frame.empty:
        raise ValueError("no rows remain in the requested training support")
    if set(frame["customer_count"].astype(int)) != {
        int(selected_scale.removeprefix("Cus"))
    }:
        raise ValueError("scale/customer_count mismatch in training index")
    if frame["day_type"].isna().any() or frame["city_slug"].isna().any():
        raise ValueError("city_slug and day_type must be populated")
    if int(sample_count) <= 0:
        raise ValueError("stream length must be positive")
    frame = frame.reset_index().rename(columns={"index": "source_row_position"})
    # Canonicalize input ordering so an equivalent parquet index cannot change
    # the registered stream merely by reordering its rows.
    frame = frame.sort_values("view_id", kind="stable").reset_index(drop=True)
    pool_counts = frame.groupby(
        ["city_slug", "day_type"], sort=True, dropna=False
    ).size()
    sampled_parts: list[pd.DataFrame] = []
    remaining = int(sample_count)
    cycle_index = 0
    while remaining:
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(cycle_index), 0x5354524D])
        )
        take = min(remaining, len(frame))
        draws = rng.permutation(len(frame))[:take]
        sampled_parts.append(frame.iloc[draws].copy())
        remaining -= take
        cycle_index += 1
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled.insert(0, "stream_position", np.arange(len(sampled), dtype=np.int64))
    sampled["scale_id"] = selected_scale
    stream = sampled[STREAM_COLUMNS].copy()
    observed = (
        stream.groupby(["city_slug", "day_type"], sort=True)
        .size()
        .to_dict()
    )
    strata = []
    for key in pool_counts.index:
        strata.append(
            {
                "city_slug": str(key[0]),
                "day_type": str(key[1]),
                "pool_views": int(pool_counts.loc[key]),
                "stream_draws": int(observed.get(key, 0)),
            }
        )
    manifest = {
        "schema": STREAM_SCHEMA,
        "sampling": "city_day_type_full_pool_shuffle_cycle_prefix_stable_v3",
        "seed": int(seed),
        "scale": selected_scale,
        "sample_count": len(stream),
        "customer_exposures": len(stream) * int(selected_scale.removeprefix("Cus")),
        "allowed_parent_family_count": int(frame["family_id"].nunique()),
        "pool_view_count": len(frame),
        "realized_unique_family_count": int(stream["family_id"].nunique()),
        "realized_unique_view_count": int(stream["view_id"].nunique()),
        "replacement": len(stream) > len(frame),
        "reuse_policy": "only_after_full_eligible_pool_cycle_exhaustion",
        "prefix_stable": True,
        "prefix_stability_scope": "same_source_support_scale_seed",
        "method_independent": True,
        "strata": strata,
        "file_hash_validation_performed": False,
    }
    return stream, manifest


def atomic_write_stream(output: str | Path, stream: pd.DataFrame, manifest: dict) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".parquet", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_temporary = manifest_path.with_suffix(
        manifest_path.suffix + f".tmp.{os.getpid()}"
    )
    try:
        stream.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
        manifest_temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)


def read_stream_view_ids(
    path: str | Path,
    *,
    start: int = 0,
    stop: int | None = None,
) -> list[str]:
    source = Path(path)
    frame = pd.read_parquet(source, columns=["stream_position", "view_id"])
    if list(frame["stream_position"].astype(int)) != list(range(len(frame))):
        raise ValueError("training stream positions are not contiguous and zero-based")
    begin = int(start)
    end = len(frame) if stop is None else int(stop)
    if begin < 0 or end < begin or end > len(frame):
        raise ValueError("requested training-stream slice is out of range")
    return frame.iloc[begin:end]["view_id"].astype(str).tolist()


__all__ = [
    "STREAM_SCHEMA",
    "atomic_write_stream",
    "build_training_stream",
    "normalize_scale",
    "read_stream_view_ids",
]
