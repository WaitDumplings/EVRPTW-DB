#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "evrptw-cle/0.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and getattr(response, "status", None) == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
            partial.replace(destination)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)


def replication_timestamp(path: Path) -> str | None:
    osmium = shutil.which("osmium")
    if osmium is None:
        return None
    result = subprocess.run(
        [
            osmium,
            "fileinfo",
            "--get",
            "header.option.osmosis_replication_timestamp",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def pbf_is_readable(path: Path) -> bool:
    osmium = shutil.which("osmium")
    if osmium is None:
        return path.stat().st_size > 0
    result = subprocess.run(
        [osmium, "fileinfo", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the unique frozen-PBF inputs referenced by a city preset."
    )
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Research mode: validate readability and record size/timestamp without hashing the PBF.",
    )
    args = parser.parse_args()

    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    sources: dict[Path, str] = {}
    for city in preset["cities"]:
        if city.get("pbf_file") and city.get("pbf_source_url"):
            path = (args.preset.parent / city["pbf_file"]).resolve()
            sources[path] = city["pbf_source_url"]
    if not sources:
        raise SystemExit("Preset contains no pbf_file/pbf_source_url pairs")

    records = []
    for path, url in sorted(sources.items(), key=lambda item: str(item[0])):
        if path.exists() and not pbf_is_readable(path):
            partial = path.with_suffix(path.suffix + ".part")
            path.replace(partial)
        if args.force or not path.exists():
            print(f"DOWNLOAD {url} -> {path}")
            download(url, path)
        else:
            print(f"REUSE {path}")
        records.append(
            {
                "file": str(path),
                "source_url": url,
                "bytes": path.stat().st_size,
                "sha256": None if args.skip_sha256 else sha256_file(path),
                "pbf_replication_timestamp_utc": replication_timestamp(path),
            }
        )

    manifest = args.manifest or (args.preset.parent / "pbf_source_manifest.json")
    manifest.write_text(
        json.dumps({"preset": str(args.preset), "sources": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {manifest}")


if __name__ == "__main__":
    main()
