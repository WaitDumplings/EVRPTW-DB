"""Download public TIGER/Line block-group files declared by a source preset."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states", nargs="+")
    parser.add_argument("--timeout-s", type=int, default=180)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    state_map = {str(key): str(value) for key, value in preset["states"].items()}
    selected = args.states or sorted(state_map)
    unknown = sorted(set(selected) - set(state_map))
    if unknown:
        raise ValueError(f"Unknown state keys: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for state in selected:
        state_fips = state_map[state]
        url = str(preset["url_pattern"]).format(state_fips=state_fips)
        destination = args.output_dir / Path(url).name
        if destination.exists():
            status = "reused_existing"
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "EVRPTW-CLE/0.3"})
            with (
                urllib.request.urlopen(request, timeout=args.timeout_s) as response,
                destination.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
            status = "downloaded"
        with zipfile.ZipFile(destination) as archive:
            bad_member = archive.testzip()
            members = archive.namelist()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt member {bad_member!r} in {destination}")
        records.append(
            {
                "state": state,
                "state_fips": state_fips,
                "source_url": url,
                "local_file": destination.name,
                "bytes": destination.stat().st_size,
                "zip_member_count": len(members),
                "status": status,
            }
        )
    manifest = {
        "schema": "cle_census_block_group_download_manifest_v1",
        "source_schema": preset["schema"],
        "vintage": int(preset["vintage"]),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "files": records,
    }
    path = args.output_dir / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
