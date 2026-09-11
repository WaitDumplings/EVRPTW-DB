"""Capture actual tracked and untracked source bytes, including working edits."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def source_files(repo):
    repo = Path(repo).resolve()
    names = set()
    for arguments in (['ls-files', '-z'], ['ls-files', '--others', '--exclude-standard', '-z']):
        output = subprocess.check_output(['git', *arguments], cwd=repo)
        names.update(name.decode('utf-8') for name in output.split(b'\0') if name)
    records = []
    for name in sorted(names):
        path = repo / name
        if path.is_symlink():
            target = os.readlink(path)
            records.append({'path': name, 'kind': 'symlink', 'target': target,
                            'sha256': hashlib.sha256(target.encode()).hexdigest()})
        elif path.is_file():
            records.append({'path': name, 'kind': 'file', 'size': path.stat().st_size,
                            'mode': 0o755 if path.stat().st_mode & 0o111 else 0o644, 'sha256': digest_file(path)})
        else:
            # A tracked deletion is part of the actual source identity.
            records.append({'path': name, 'kind': 'absent'})
    return records


def source_identity(records):
    payload = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def capture_source(repo, destination):
    repo = Path(repo).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = source_files(repo)
    identity = source_identity(records)
    archive = destination / f'source_{identity}.tar.gz'
    manifest = destination / f'source_{identity}.json'
    if not archive.is_file():
        temporary = archive.with_suffix(archive.suffix + f'.tmp.{os.getpid()}')
        try:
            with temporary.open('wb') as raw:
                with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode='w') as bundle:
                        for record in records:
                            if record['kind'] == 'absent':
                                continue
                            path = repo / record['path']
                            info = bundle.gettarinfo(str(path), arcname=record['path'])
                            info.mtime = 0
                            info.uid = info.gid = 0
                            info.uname = info.gname = ''
                            if record['kind'] == 'file':
                                with path.open('rb') as stream:
                                    bundle.addfile(info, stream)
                            else:
                                bundle.addfile(info)
            if source_files(repo) != records:
                raise RuntimeError('Source files changed during snapshot creation; rerun after edits stop')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    result = {
        'schema': 'cus100_actual_source_snapshot_v1',
        'git_head_reference': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'source_version': 'sha256:' + identity,
        'source_sha256': identity,
        'includes': 'all git-tracked files at current working contents plus git-untracked non-ignored files',
        'mode_identity': 'git-style executable versus non-executable; group/other write bits do not change source identity',
        'files': records,
        'archive': str(archive),
        'archive_sha256': digest_file(archive),
        'manifest': str(manifest),
    }
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result
