#!/usr/bin/env python3
"""Download one official Microsoft USBuildingFootprints state archive."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

BASE_URL = "https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2"


def _download(url: str, destination: Path, *, force: bool, retries: int = 4) -> None:
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"REUSE {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if force:
        partial.unlink(missing_ok=True)
    for attempt in range(1, retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "evrptw-cle/0.4 US-city-adapter"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=600) as response:
                append = offset > 0 and getattr(response, "status", None) == 206
                with partial.open("ab" if append else "wb") as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
            partial.replace(destination)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)


def download_state_buildings(
    *,
    state_file: str,
    output_root: Path,
    source_url: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Download and safely extract one state GeoJSON, without a separate hash pass."""

    output_root = output_root.resolve()
    destination = output_root / state_file
    url = source_url or f"{BASE_URL}/{state_file}.zip"
    archive = output_root / ".downloads" / f"{state_file}.zip"
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"REUSE {destination}", flush=True)
    else:
        _download(url, archive, force=force)
        with zipfile.ZipFile(archive) as bundle:
            matches = [
                item
                for item in bundle.infolist()
                if PurePosixPath(item.filename).name.casefold() == state_file.casefold()
            ]
            if len(matches) != 1:
                names = [item.filename for item in bundle.infolist()[:20]]
                raise RuntimeError(
                    f"Expected one {state_file!r} in {archive}; found {len(matches)}. "
                    f"Archive sample: {names}"
                )
            output_root.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + ".part")
            partial.unlink(missing_ok=True)
            with bundle.open(matches[0]) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            partial.replace(destination)
    manifest = {
        "schema": "evrptw_microsoft_state_buildings_source_v1",
        "source_dataset": "Microsoft USBuildingFootprints",
        "source_url": url,
        "license": "Open Data Commons Open Database License (ODbL)",
        "state_file": state_file,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "integrity_mode": "record_on_extraction",
        "prepared_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = destination.with_suffix(".source.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = download_state_buildings(
        state_file=args.state_file,
        output_root=args.output_root,
        source_url=args.source_url,
        force=args.force,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
