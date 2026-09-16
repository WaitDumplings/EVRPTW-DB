"""Identity checks for resuming one exact-benchmark result directory."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_run_contract(*, solver_config: Any, tasks: list[Any], solver_version: str,
                       algorithm_profile_id: str) -> dict[str, Any]:
    directory = Path(__file__).resolve().parent
    indices = sorted({Path(task.index_path).resolve() for task in tasks})
    # View indices identify the selected instances. Family manifests identify
    # the materialized matrix stores; neither a renamed profile nor changing
    # numerical coefficients may inherit the previous completion markers.
    manifests = sorted({Path(task.family_dir).resolve() / 'family_manifest.json'
                        for task in tasks})
    payload = {
        'schema': 'gurobi_benchmark_run_contract_v1',
        'algorithm_profile_id': algorithm_profile_id,
        'solver_version': solver_version,
        'solver_config': asdict(solver_config),
        'view_indices': {str(path): file_sha256(path) for path in indices},
        'family_manifests': {
            str(path): file_sha256(path) if path.is_file() else None
            for path in manifests
        },
        'source_sha256': {name: file_sha256(directory / name) for name in (
            'gurobi_solver.py', 'run_gurobi.py', 'run_contract.py',
            'stage2_adapter.py', 'route_validator.py',
        )},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {
        'fingerprint': hashlib.sha256(canonical.encode()).hexdigest(),
        'payload': json.loads(canonical),
    }


def ensure_run_contract(save_path: Path, expected: dict[str, Any]) -> None:
    """Refuse to mix legacy or incompatible attempts into existing CSVs."""
    path = save_path / 'run_contract.json'
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing != expected:
            raise ValueError(
                'Gurobi output run contract differs (objective, budget, data, '
                'model or source changed); choose a new --save_path.'
            )
        return
    artifacts = [save_path / name for name in ('gurobi_summary.csv', 'gurobi_time_trace.csv')]
    has_results = any(path.is_file() and path.stat().st_size for path in artifacts)
    has_solutions = any((save_path / 'solutions').rglob('*.pkl'))
    if has_results or has_solutions:
        raise ValueError(
            'Existing Gurobi results have no run contract; their objective and '
            'budget cannot be verified. Choose a new --save_path.'
        )
    save_path.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents silently overwriting another launcher.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(expected, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
