from __future__ import annotations

import json
import hashlib
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


STREAM_SCHEMA = "drl_training_id_stream_v3"
STREAM_CONTRACT_SCHEMA = "drl_training_stream_contract_v1"
STREAM_CONTENT_DIGEST_SCHEME = "sha256_length_prefixed_ordered_view_ids_v1"
STREAM_INTEGRITY_MODE_RUNTIME_REVERIFIED = "runtime_content_rehash"
STREAM_INTEGRITY_MODE_PREVERIFIED = "reuse_preverified_snapshot_no_rehash"
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
    seeded sample without replacement.  The exact ordered stream is frozen by
    :func:`atomic_write_stream`; the logical digest deliberately does not
    depend on a particular Parquet writer version.
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
        "file_hash_validation_performed": True,
    }
    return stream, manifest


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_content_sha256(view_ids: Iterable[str]) -> str:
    """Hash the exact ordered logical stream independently of file encoding."""

    values = [str(value) for value in view_ids]
    digest = hashlib.sha256()
    digest.update((STREAM_CONTENT_DIGEST_SCHEME + "\0").encode("ascii"))
    digest.update(struct.pack(">Q", len(values)))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def training_stream_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Digest scientific manifest content, excluding machine-local locators."""

    canonical = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "manifest_sha256",
            "source_index",
            "allowed_family_ids_source",
        }
    }
    return _canonical_sha256(canonical)


def training_stream_contract_digest(payload: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "sha256"}
    return _canonical_sha256(canonical)


def _validate_stream_frame(frame: pd.DataFrame) -> list[str]:
    required = {"stream_position", "view_id"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"training stream is missing columns: {sorted(required.difference(frame.columns))}"
        )
    if list(frame["stream_position"].astype(int)) != list(range(len(frame))):
        raise ValueError("training stream positions are not contiguous and zero-based")
    values = frame["view_id"].astype(str).tolist()
    if any(not value for value in values):
        raise ValueError("training stream contains an empty view_id")
    return values


def _contract_from_verified_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": STREAM_CONTRACT_SCHEMA,
        "stream_schema": manifest["schema"],
        "content_digest_scheme": manifest["content_digest_scheme"],
        "stream_content_sha256": manifest["stream_content_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": int(manifest["sample_count"]),
        "scale": str(manifest["scale"]),
        "seed": int(manifest["seed"]),
        "source_index_sha256": str(manifest["source_index_sha256"]),
        "allowed_family_ids_sha256": manifest.get("allowed_family_ids_sha256"),
    }
    payload["sha256"] = training_stream_contract_digest(payload)
    return payload


def training_stream_contract_from_preverified_manifest(
    path_or_mapping: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild a contract from its small, previously verified sidecar only.

    This validates the sidecar's self-contained metadata but deliberately does
    not open the Parquet stream or the source index.  Callers must opt into that
    trust boundary explicitly and still validate ordered IDs while consuming
    the stream.
    """

    if isinstance(path_or_mapping, Mapping):
        manifest = dict(path_or_mapping)
    else:
        source = Path(path_or_mapping)
        manifest_path = (
            source
            if source.name.endswith(".manifest.json")
            else source.with_suffix(source.suffix + ".manifest.json")
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read training-stream manifest: {manifest_path}"
            ) from error
    if not isinstance(manifest, dict) or manifest.get("schema") != STREAM_SCHEMA:
        raise ValueError("invalid training-stream manifest schema")
    required = {
        "content_digest_scheme",
        "stream_content_sha256",
        "manifest_sha256",
        "source_index_sha256",
        "sample_count",
        "scale",
        "seed",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"training-stream manifest is missing fields: {missing}")
    if manifest.get("content_digest_scheme") != STREAM_CONTENT_DIGEST_SCHEME:
        raise ValueError("unsupported training-stream content digest scheme")
    if manifest.get("file_hash_validation_performed") is not True:
        raise ValueError("training-stream manifest did not freeze content hashes")
    if manifest.get("manifest_sha256") != training_stream_manifest_digest(manifest):
        raise ValueError("training-stream manifest SHA256 mismatch")
    return _contract_from_verified_manifest(manifest)


def load_training_stream_contract(path: str | Path) -> dict[str, Any]:
    """Load and fully verify a frozen stream and its sibling manifest."""

    source = Path(path)
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read training-stream manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("schema") != STREAM_SCHEMA:
        raise ValueError("invalid training-stream manifest schema")
    required = {
        "content_digest_scheme",
        "stream_content_sha256",
        "manifest_sha256",
        "source_index_sha256",
        "sample_count",
        "scale",
        "seed",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"training-stream manifest is missing fields: {missing}")
    if manifest.get("content_digest_scheme") != STREAM_CONTENT_DIGEST_SCHEME:
        raise ValueError("unsupported training-stream content digest scheme")
    if manifest.get("file_hash_validation_performed") is not True:
        raise ValueError("training-stream manifest did not freeze content hashes")
    expected_manifest_sha = training_stream_manifest_digest(manifest)
    if manifest.get("manifest_sha256") != expected_manifest_sha:
        raise ValueError("training-stream manifest SHA256 mismatch")
    try:
        frame = pd.read_parquet(source, columns=["stream_position", "view_id"])
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read frozen training stream: {source}") from error
    values = _validate_stream_frame(frame)
    if len(values) != int(manifest["sample_count"]):
        raise ValueError("training-stream sample count disagrees with manifest")
    if stream_content_sha256(values) != manifest["stream_content_sha256"]:
        raise ValueError("training-stream logical content SHA256 mismatch")
    return _contract_from_verified_manifest(manifest)


def training_stream_contract_from_args(
    args: Any,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    path = getattr(args, "training_stream_path", None)
    expected = getattr(args, "training_stream_contract_sha256", None)
    reuse_preverified = bool(
        getattr(args, "reuse_preverified_training_streams", False)
    )
    raw_snapshot = getattr(
        args, "training_stream_contract_snapshot_json", None
    )
    if path is None:
        if required or expected is not None or reuse_preverified or raw_snapshot is not None:
            raise ValueError("formal training requires --training-stream-path")
        return None
    if reuse_preverified:
        if not required:
            raise ValueError(
                "--reuse-preverified-training-streams is restricted to the "
                "frozen formal protocol"
            )
        if not expected:
            raise ValueError(
                "preverified stream reuse requires "
                "--training-stream-contract-sha256"
            )
        if raw_snapshot is None:
            raise ValueError(
                "preverified stream reuse requires the exact manifest contract "
                "snapshot"
            )
        if isinstance(raw_snapshot, Mapping):
            contract = dict(raw_snapshot)
        else:
            try:
                decoded = json.loads(str(raw_snapshot))
            except json.JSONDecodeError as error:
                raise ValueError(
                    "training-stream contract snapshot is not valid JSON"
                ) from error
            if not isinstance(decoded, dict):
                raise ValueError(
                    "training-stream contract snapshot must be a JSON object"
                )
            contract = decoded
        if contract.get("schema") != STREAM_CONTRACT_SCHEMA:
            raise ValueError(
                "preverified training-stream contract schema is invalid"
            )
        if contract.get("sha256") != str(expected):
            raise ValueError(
                "preverified training-stream snapshot does not match the "
                "manifest contract"
            )
        setattr(args, "training_stream_contract_snapshot", contract)
        setattr(
            args,
            "stream_integrity_mode",
            STREAM_INTEGRITY_MODE_PREVERIFIED,
        )
        return contract
    if raw_snapshot is not None:
        raise ValueError(
            "a preverified training-stream snapshot may be supplied only with "
            "--reuse-preverified-training-streams"
        )
    try:
        contract = load_training_stream_contract(path)
    except ValueError:
        if required or expected is not None:
            raise
        return None
    if required and not expected:
        raise ValueError(
            "formal training requires --training-stream-contract-sha256"
        )
    if expected is not None and str(expected) != contract["sha256"]:
        raise ValueError(
            "training-stream contract SHA256 does not match the frozen artifact"
        )
    setattr(args, "training_stream_contract_sha256", contract["sha256"])
    setattr(args, "training_stream_contract_snapshot", contract)
    setattr(
        args,
        "stream_integrity_mode",
        STREAM_INTEGRITY_MODE_RUNTIME_REVERIFIED,
    )
    return contract


def assert_checkpoint_training_stream_contract(payload: Mapping[str, Any], args: Any) -> None:
    expected = getattr(args, "training_stream_contract_snapshot", None)
    if expected is None:
        if payload.get("training_stream_contract") is not None:
            raise ValueError("checkpoint unexpectedly contains a training-stream contract")
        return
    if (
        not isinstance(expected, Mapping)
        or expected.get("sha256") != training_stream_contract_digest(expected)
    ):
        raise ValueError("current training-stream contract snapshot is invalid")
    saved_args = payload.get("args", {}) or {}
    if not isinstance(saved_args, Mapping):
        saved_args = vars(saved_args)
    if payload.get("training_stream_contract") != expected:
        raise ValueError("checkpoint training-stream contract mismatch")
    if saved_args.get("training_stream_contract_snapshot") != expected:
        raise ValueError("checkpoint args training-stream snapshot mismatch")
    if saved_args.get("training_stream_contract_sha256") != expected["sha256"]:
        raise ValueError("checkpoint args training-stream SHA256 mismatch")


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
        view_ids = _validate_stream_frame(stream)
        manifest["content_digest_scheme"] = STREAM_CONTENT_DIGEST_SCHEME
        manifest["stream_content_sha256"] = stream_content_sha256(view_ids)
        manifest["file_hash_validation_performed"] = True
        if not manifest.get("source_index_sha256"):
            raise ValueError("training-stream manifest requires source_index_sha256")
        manifest.setdefault("allowed_family_ids_sha256", None)
        manifest["manifest_sha256"] = training_stream_manifest_digest(manifest)
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
    values = _validate_stream_frame(frame)
    begin = int(start)
    end = len(frame) if stop is None else int(stop)
    if begin < 0 or end < begin or end > len(frame):
        raise ValueError("requested training-stream slice is out of range")
    return values[begin:end]


__all__ = [
    "STREAM_SCHEMA",
    "STREAM_CONTRACT_SCHEMA",
    "STREAM_CONTENT_DIGEST_SCHEME",
    "STREAM_INTEGRITY_MODE_PREVERIFIED",
    "STREAM_INTEGRITY_MODE_RUNTIME_REVERIFIED",
    "assert_checkpoint_training_stream_contract",
    "atomic_write_stream",
    "build_training_stream",
    "file_sha256",
    "load_training_stream_contract",
    "normalize_scale",
    "read_stream_view_ids",
    "stream_content_sha256",
    "training_stream_contract_digest",
    "training_stream_contract_from_args",
    "training_stream_contract_from_preverified_manifest",
    "training_stream_manifest_digest",
]
