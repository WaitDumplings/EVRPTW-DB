#!/usr/bin/env python3
"""Export and reconstruct CLE-backed Stage-2 instance caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_stage2.reconstruction import (
    export_slim_dataset,
    resolve_family_dirs,
    restore_dataset_matrices,
    verify_family_reconstruction,
)


def _ids(values: list[str] | None, file_path: Path | None) -> list[str] | None:
    selected = list(values or ())
    if file_path is not None:
        selected.extend(
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return list(dict.fromkeys(selected)) or None


def _write_report(payload: dict[str, object], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Report: {output}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Treat Stage-2 dense matrices as a deterministic cache derived from CLE plus "
            "lightweight family/view artifacts."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser(
        "export-slim", help="copy a complete instance tree while omitting parent matrices"
    )
    export.add_argument("--source-root", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--cle-root", type=Path, required=True)
    export.add_argument("--profile", type=Path, required=True)
    export.add_argument("--report", type=Path)

    restore = commands.add_parser(
        "restore", help="reconstruct all matrices, or the parent family for selected IDs"
    )
    restore.add_argument("--dataset-root", type=Path, required=True)
    restore.add_argument("--cle-root", type=Path, required=True)
    restore.add_argument(
        "--profile",
        type=Path,
        help="optional for a slim export, which embeds the exact reference profile",
    )
    restore.add_argument("--family-id", action="append", dest="family_ids")
    restore.add_argument("--family-id-file", type=Path)
    restore.add_argument("--view-id", action="append", dest="view_ids")
    restore.add_argument("--view-id-file", type=Path)
    restore.add_argument("--workers", type=int, default=1)
    restore.add_argument("--families-per-worker-task", type=int, default=25)
    restore.add_argument("--validation", choices=("exact", "none"), default="exact")
    restore.add_argument("--report", type=Path)

    verify = commands.add_parser(
        "verify-family",
        help="rebuild in memory and compare against one existing family cache",
    )
    verify.add_argument("--family-dir", type=Path, required=True)
    verify.add_argument("--cle-root", type=Path, required=True)
    verify.add_argument("--profile", type=Path, required=True)
    verify.add_argument("--validation", choices=("exact", "allclose"), default="exact")
    verify.add_argument("--rtol", type=float, default=1e-6)
    verify.add_argument("--atol", type=float, default=1e-5)
    verify.add_argument("--report", type=Path)

    resolve = commands.add_parser(
        "resolve-ids", help="show which parent families selected instance/view IDs require"
    )
    resolve.add_argument("--dataset-root", type=Path, required=True)
    resolve.add_argument("--family-id", action="append", dest="family_ids")
    resolve.add_argument("--family-id-file", type=Path)
    resolve.add_argument("--view-id", action="append", dest="view_ids")
    resolve.add_argument("--view-id-file", type=Path)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "export-slim":
        report = export_slim_dataset(
            args.source_root,
            args.output_root,
            cle_root=args.cle_root,
            profile_path=args.profile,
        )
        _write_report(report, args.report)
        return

    if args.command == "verify-family":
        report = verify_family_reconstruction(
            args.family_dir,
            cle_root=args.cle_root,
            profile_path=args.profile,
            validation=args.validation,
            rtol=args.rtol,
            atol=args.atol,
        )
        _write_report(report, args.report)
        if not report["passed"]:
            raise SystemExit(2)
        return

    family_ids = _ids(args.family_ids, args.family_id_file)
    view_ids = _ids(args.view_ids, args.view_id_file)
    if args.command == "resolve-ids":
        paths = resolve_family_dirs(
            args.dataset_root, family_ids=family_ids, view_ids=view_ids
        )
        print("\n".join(path.name for path in paths))
        return

    if args.command == "restore":
        report = restore_dataset_matrices(
            args.dataset_root,
            cle_root=args.cle_root,
            profile_path=args.profile,
            family_ids=family_ids,
            view_ids=view_ids,
            workers=args.workers,
            families_per_worker_task=args.families_per_worker_task,
            validation=args.validation,
        )
        _write_report(report, args.report)
        return

    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
