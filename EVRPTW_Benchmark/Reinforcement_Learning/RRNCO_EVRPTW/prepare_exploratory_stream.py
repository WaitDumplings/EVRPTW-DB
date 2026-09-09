from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from ..common.training_stream import build_training_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic exploratory stream without file hashing."
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite stream: {args.output}")
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    if metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite metadata: {metadata_path}")

    index = pd.read_parquet(args.index)
    stream, manifest = build_training_stream(
        index,
        scale=args.scale,
        seed=args.seed,
        sample_count=args.sample_count,
    )
    metadata = {
        **manifest,
        "schema": "rrnco_ev_exploratory_training_stream_v1",
        "source_index": str(args.index),
        "integrity_mode": "validated_ordered_ids_no_file_hash",
        "file_hash_validation_performed": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".parquet", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    metadata_temporary = metadata_path.with_suffix(
        metadata_path.suffix + f".tmp.{os.getpid()}"
    )
    try:
        stream.to_parquet(temporary, index=False)
        metadata_temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
        os.replace(metadata_temporary, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)

    print(json.dumps({"output": str(args.output), **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
