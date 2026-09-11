#!/usr/bin/env python3
"""Package actual source, synthetic Cus100 and frozen streams; exclude Road data."""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

from .data_contract import inspect_synthetic
from .launch import MANIFEST, REPO, RUN_ROOT, dataset_root, load_jobs, sha256, timestamp, write_json
from .source_snapshot import capture_source, source_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, default=REPO / RUN_ROOT / 'artifacts')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite deployment package: {args.output}')
    jobs = load_jobs(args.manifest)
    if any(not job['enabled'] or job.get('calibration_status') != 'passed' for job in jobs):
        raise RuntimeError('A deployment package requires all ten measured, enabled configurations')
    synthetic = [job for job in jobs if job['source_kind'] == 'terran_synthetic']
    roots = {dataset_root(job) for job in synthetic}
    if len(roots) != 1:
        raise ValueError('All Euclidean methods must share one synthetic corpus')
    data = roots.pop()
    data_contract = inspect_synthetic(data)
    source = capture_source(REPO, REPO / RUN_ROOT / 'provenance')
    files = {}
    for record in source['files']:
        if record['kind'] == 'absent':
            continue
        files[record['path']] = REPO / record['path']
    destination_data = Path(synthetic[0]['dataset_root'])
    if destination_data.is_absolute() or '..' in destination_data.parts:
        raise ValueError('Packaged synthetic destination must be repository-relative')
    for path in sorted(data.rglob('*')):
        if path.is_file():
            files[str(destination_data / path.relative_to(data))] = path
    for path in sorted(args.artifact_root.rglob('*')):
        if path.is_file():
            files[str(path.resolve().relative_to(REPO))] = path
    for key in ('archive', 'manifest'):
        path = Path(source[key])
        files[str(path.relative_to(REPO))] = path
    for job in jobs:
        path = REPO / job['training_stream_path']
        if path != files.get(job['training_stream_path']):
            raise RuntimeError(f'Missing required stream in package: {path}')
        if sha256(path) != job['training_stream_path_sha256']:
            raise RuntimeError(f'Changed stream: {path}')
    forbidden = 'EVRPTW_Dataset/Instances_v2/'
    if any(name.startswith(forbidden) for name in files):
        raise RuntimeError('Road payload unexpectedly entered deployment package')
    package = {'schema': 'cus100_portable_deployment_v1', 'time': timestamp(),
               'source_version': source['source_version'], 'source_manifest': str(Path(source['manifest']).relative_to(REPO)),
               'job_manifest': str(args.manifest.resolve().relative_to(REPO)), 'job_manifest_sha256': sha256(args.manifest),
               'synthetic_manifest_sha256': data_contract['manifest_sha256'],
               'road_data_included': False, 'required_existing_road_release': next(job['dataset_root'] for job in jobs if job['source_kind'] == 'stage2_road'),
               'file_count': len(files), 'total_uncompressed_bytes': sum(path.lstat().st_size for path in files.values()),
               'extraction_destination': 'repository root', 'server_roles': ['2080ti_4_1', '2080ti_4_2', '2080ti_3_1']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f'.tmp.{os.getpid()}')
    try:
        print(json.dumps({'status': 'packaging', **package}), flush=True)
        with tarfile.open(temporary, 'w:gz', compresslevel=1) as archive:
            for index, (name, path) in enumerate(sorted(files.items())):
                archive.add(path, arcname=name, recursive=False)
                if index and index % 100 == 0:
                    print(json.dumps({'status': 'packaging', 'files_written': index, 'files_total': len(files)}), flush=True)
            payload = json.dumps(package, indent=2, sort_keys=True).encode() + b'\n'
            info = tarfile.TarInfo('EVRPTW_Benchmark/results/cus100_20260911/deployment_manifest.json')
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if source_files(REPO) != source['files']:
            raise RuntimeError('Source changed while packaging; retry after edits finish')
        if inspect_synthetic(data)['manifest_sha256'] != data_contract['manifest_sha256']:
            raise RuntimeError('Synthetic corpus changed while packaging')
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    package.update(archive=str(args.output.resolve()), archive_sha256=sha256(args.output), archive_bytes=args.output.stat().st_size)
    write_json(args.output.with_suffix(args.output.suffix + '.json'), package)
    print(json.dumps({'status': 'complete', **package}, indent=2), flush=True)


if __name__ == '__main__':
    main()
