#!/usr/bin/env python3
"""Inspect old TR17/TR18; explicitly --stop to back up checkpoints and send TERM."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import signal
import time
from pathlib import Path

MODULE = 'EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.train'
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def process(pid):
    try:
        root = Path('/proc') / str(int(pid))
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': int(stat[19]), 'uid': root.stat().st_uid,
                'argv': [part.decode(errors='replace') for part in (root / 'cmdline').read_bytes().split(b'\0') if part]}
    except (OSError, ValueError, IndexError, TypeError):
        return None


def validate_identity(identity, run):
    argv = identity['argv']
    index = 1
    while index < len(argv) and argv[index] in ('-u', '-B'):
        index += 1
    if identity['uid'] != os.getuid() or argv[index:index + 2] != ['-m', MODULE]:
        raise RuntimeError('PID is not the expected EVRPTW-RL trainer owned by this user')
    try:
        output = argv[argv.index('--output-dir') + 1]
    except (ValueError, IndexError):
        raise RuntimeError('Trainer has no explicit output directory') from None
    path = Path(output)
    if not path.is_absolute():
        path = Path(f"/proc/{identity['pid']}/cwd").resolve() / path
    if path.resolve() != run.resolve():
        raise RuntimeError('PID belongs to a different output directory')


def inspect(root):
    records = []
    for name in ('TR17', 'TR18'):
        run = root / 'runs' / name
        record = json.loads((run / 'launch_record.json').read_text())
        if record.get('experiment_id') != name or Path(record.get('output_dir', '')).resolve() != run.resolve():
            raise RuntimeError(f'{name}: unexpected old launch record identity')
        current = process(record.get('pid'))
        if current:
            validate_identity(current, run)
            for filename in ('best.ckpt', 'checkpoint_latest.pt'):
                checkpoint = run / filename
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    raise RuntimeError(f'{name}: cannot stop without saved {filename}')
        records.append({'experiment_id': name, 'run': str(run), 'process': current,
                        'action': 'TERM_after_backup' if current else 'already_exited'})
    return records


def stop_records(root, records, timeout):
    archive = root / 'repair_backups' / f'before_mean_{time.time_ns()}'
    archive.mkdir(parents=True, exist_ok=False)
    report = {'time_unix': time.time(), 'records': records, 'checkpoint_copies': []}
    # Validate every live PID first; if any identity differs, stop no process.
    for record in records:
        saved = record['process']
        if saved:
            current = process(saved['pid'])
            if current is not None and current['start_ticks'] != saved['start_ticks']:
                raise RuntimeError('PID was reused; no stop is allowed')
            if current:
                validate_identity(current, Path(record['run']))
    for record in records:
        if not record['process']:
            continue
        target = archive / record['experiment_id']
        target.mkdir()
        for filename in ('best.ckpt', 'checkpoint_latest.pt', 'launch_record.json', 'validation_summary.json', 'training_state.json'):
            source = Path(record['run']) / filename
            if source.is_file():
                copied = target / filename
                shutil.copy2(source, copied)
                report['checkpoint_copies'].append({'path': str(copied), 'sha256': hashlib.sha256(copied.read_bytes()).hexdigest()})
    (archive / 'stop_request.json').write_text(json.dumps(report, indent=2) + '\n')
    for record in records:
        saved = record['process']
        if not saved:
            continue
        current = process(saved['pid'])
        if current is None:
            continue
        if current['start_ticks'] != saved['start_ticks']:
            raise RuntimeError('PID changed after backup; refusing TERM')
        validate_identity(current, Path(record['run']))
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            try:
                descriptor = os.pidfd_open(saved['pid'])
            except ProcessLookupError:
                continue
            try:
                current = process(saved['pid'])
                if current is not None:
                    if current['start_ticks'] != saved['start_ticks']:
                        raise RuntimeError('PID changed while opening pidfd')
                    try:
                        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                    except ProcessLookupError:
                        pass  # The verified process may exit before TERM is sent.
            finally:
                os.close(descriptor)
        else:
            try:
                os.kill(saved['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    while True:
        alive = []
        for record in records:
            saved = record['process']
            now = process(saved['pid']) if saved else None
            if now is not None and now['start_ticks'] == saved['start_ticks']:
                alive.append(saved['pid'])
        if not alive or time.monotonic() >= deadline:
            report.update(still_alive=alive, status='stopped' if not alive else 'waiting_for_exit', archive=str(archive))
            (archive / 'stop_result.json').write_text(json.dumps(report, indent=2) + '\n')
            return report
        time.sleep(0.2)


def main():
    default = REPO.parent / 'EVRPTW-DB/EVRPTW_Benchmark/results/cus100_20260911'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-output-root', type=Path, default=Path(os.environ.get('CUS100_OLD_OUTPUT_ROOT', default)))
    parser.add_argument('--stop', action='store_true', help='Back up existing checkpoints, then send TERM only to verified old TR17/TR18 PIDs')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if not 0 <= args.timeout <= 60:
        parser.error('--timeout must be between 0 and 60 seconds')
    root = args.old_output_root.expanduser().resolve()
    records = inspect(root)
    result = stop_records(root, records, args.timeout) if args.stop else {'status': 'inspection_only', 'records': records}
    print(json.dumps(result, indent=2))
    return int(bool(result.get('still_alive')))


if __name__ == '__main__':
    raise SystemExit(main())
