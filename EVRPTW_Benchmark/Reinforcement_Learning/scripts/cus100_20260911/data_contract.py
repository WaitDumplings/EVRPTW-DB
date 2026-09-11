"""Preflight verification for the frozen, non-provisional synthetic corpus."""
from __future__ import annotations

import json
from pathlib import Path

from .source_snapshot import digest_file


def inspect_synthetic(root, *, verify_payloads=True):
    root = Path(root).resolve()
    manifest_path = root / 'corpus_manifest.json'
    corpus = json.loads(manifest_path.read_text())
    if (corpus.get('schema') != 'terran_synthetic_corpus_v1'
            or corpus.get('complete') is not True
            or corpus.get('formal_training_authorized') is not True):
        raise RuntimeError(f'Synthetic corpus is incomplete or provisional: {manifest_path}')
    if corpus.get('source_kind') != 'terran_synthetic' or corpus.get('representation') != 'E':
        raise ValueError('Incorrect synthetic source/representation contract')
    if corpus.get('requested_counts') != {'train': 50000, 'val': 500}:
        raise ValueError('Synthetic corpus must contain 50000 train / 500 validation instances')
    required = {manifest_path: None}
    for name, expected in corpus['provenance_sha256'].items():
        required[root / 'provenance' / name] = expected
    for split, expected_count in (('train', 50000), ('val', 500)):
        entry = corpus['splits'][split]
        sidecar_path = root / split / 'synthetic_index_manifest.json'
        sidecar = json.loads(sidecar_path.read_text())
        if (sidecar.get('schema') != 'terran_synthetic_index_v1' or sidecar.get('complete') is not True
                or sidecar.get('usage') != 'formal_corpus'
                or sidecar.get('instance_count') != expected_count
                or entry.get('instance_count') != expected_count):
            raise ValueError(f'Invalid formal synthetic split: {sidecar_path}')
        index_path = root / entry['index_relative_path']
        if index_path.resolve() != (root / split / 'view_index.parquet').resolve():
            raise ValueError('Unexpected synthetic split index path')
        if entry['index_sha256'] != sidecar['index_sha256']:
            raise ValueError('Synthetic corpus and split index digests disagree')
        required[sidecar_path] = None
        required[index_path] = entry['index_sha256']
        for name, expected in entry['file_sha256'].items():
            required[root / split / name] = expected
    records = {}
    for path, expected in required.items():
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError('Synthetic manifest references a file outside its corpus root')
        if not path.is_file():
            raise FileNotFoundError(path)
        is_payload = path.suffix not in {'.json', '.parquet', '.py', '.patch'}
        measured = digest_file(path) if verify_payloads or not is_payload else None
        if measured and expected and measured != expected:
            raise RuntimeError(f'Synthetic corpus input hash mismatch: {path}')
        records[str(path)] = {'expected_sha256': expected, 'verified_sha256': measured,
                              'size_bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    return {'manifest': str(manifest_path), 'manifest_sha256': digest_file(manifest_path),
            'corpus': corpus, 'files': records, 'all_payload_hashes_verified': verify_payloads}
