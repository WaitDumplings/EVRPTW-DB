"""Deployment entry must override stale shell/CLI batch choices without launching CUDA."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus1000_evrptw_rl_20260917.common import HERE

DEFAULT_ENTRY = HERE / '2080ti_4_2/full.sh'


def test_shell_fixes_batch12_global48_despite_environment_and_cli(tmp_path):
    entry = Path(os.environ.get('CUS1000_SHELL_UNDER_TEST', DEFAULT_ENTRY))
    fake_python = tmp_path / 'capture-python'
    fake_python.write_text(
        f'#!{sys.executable}\nimport json,sys\n'
        'print(json.dumps(sys.argv[1:]))\n')
    fake_python.chmod(0o755)
    env = {**os.environ, 'CUS1000_PYTHON': str(fake_python), 'CUDA_VISIBLE_DEVICES': '',
           'CUS1000_BATCH_SIZE': '24', 'CUS1000_ACCUMULATION_STEPS': '8',
           'CUS1000_GPUS': '3,2,1,0', 'PYTHONDONTWRITEBYTECODE': '1'}
    # First exercise a stale environment with no explicit batch flags; then contradict
    # the fixed recipe explicitly on the CLI. The real shell executes only our recorder.
    for conflicting_cli in [[], ['--batch-size', '32', '--accumulation-steps', '4',
                                  '--gpus', '0,1', '--model', 'rrnco']]:
        stdout = subprocess.check_output(['bash', str(entry), '--mode', 'status', *conflicting_cli],
                                         env=env, cwd=tmp_path, text=True)
        argv = json.loads(stdout)
        assert Path(argv[0]).name == 'launch.py'
        # Use argparse's actual last-value behavior, including the launcher's env defaults.
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=int(env['CUS1000_BATCH_SIZE']))
        parser.add_argument('--accumulation-steps', type=int, default=int(env['CUS1000_ACCUMULATION_STEPS']))
        parser.add_argument('--gpus')
        parser.add_argument('--model')
        selected, _ = parser.parse_known_args(argv[1:])
        assert selected.batch_size == 12, f'Entry did not pin batch12: {argv}'
        assert selected.accumulation_steps == 1, f'Entry did not pin accumulation1: {argv}'
        assert selected.gpus == '0,1,2,3'
        assert selected.model == 'evrptw_rl'
        assert selected.batch_size * len(selected.gpus.split(',')) * selected.accumulation_steps == 48
