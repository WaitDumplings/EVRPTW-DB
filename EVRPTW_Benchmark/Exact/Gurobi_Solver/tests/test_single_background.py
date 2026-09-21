"""Background lifecycle checks without launching a large optimization."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import pytest
import run_single_cus500 as runner


def test_detached_worker_redirects_output_and_retains_csv(monkeypatch, tmp_path, capsys):
    script = tmp_path/'worker with spaces.py'
    script.write_text('''import argparse, json, os, signal, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output_dir');p.add_argument('--track');p.add_argument('--dry_run',action='store_true')
a=p.parse_args()
out=Path(a.output_dir);out.mkdir()
(out/'identity.json').write_text(json.dumps({'pid':os.getpid(),'sid':os.getsid(0),'stdin':os.read(0,1).decode(),'hup_ignored':signal.getsignal(signal.SIGHUP)==signal.SIG_IGN,'track':a.track,'dry_run':a.dry_run}))
print('solver stdout',flush=True)
print('solver stderr',file=sys.stderr,flush=True)
with (out/'progress.csv').open('w') as f:
 f.write('runtime,mipgap,objective\\n');f.flush()
 time.sleep(.4)
 f.write('1.0,0.2,123.45\\n');f.flush()
''')
    monkeypatch.setattr(runner, '__file__', str(script))
    out = tmp_path/'result with spaces'
    args=argparse.Namespace(background=True,output_dir=out,track='T2',dry_run=True)
    assert runner.launch_background(args)==0
    terminal=capsys.readouterr().out
    assert 'Background PID:' in terminal and str(out/'progress.csv') in terminal
    assert 'solver stdout' not in terminal and 'solver stderr' not in terminal
    metadata=json.loads(Path(str(out)+'.launcher.json').read_text())
    pid=metadata['pid']
    try:
        deadline=time.monotonic()+10
        while not (out/'progress.csv').exists() or '123.45' not in (out/'progress.csv').read_text():
            assert time.monotonic()<deadline
            time.sleep(.05)
        identity=json.loads((out/'identity.json').read_text())
        assert identity['pid']==identity['sid']==pid
        assert identity['sid']!=os.getsid(0)
        assert identity['stdin']=='' and identity['hup_ignored']
        assert identity['track']=='T2' and identity['dry_run']
        assert '--background' not in metadata['command']
        text=Path(str(out)+'.launcher.log').read_text()
        assert 'solver stdout' in text and 'solver stderr' in text
    finally:
        try:os.waitpid(pid,0)
        except ChildProcessError:pass


def test_background_rejects_existing_output_or_reserved_log(tmp_path):
    out=tmp_path/'existing';out.mkdir()
    args=argparse.Namespace(background=True,output_dir=out)
    with pytest.raises(FileExistsError):runner.launch_background(args)
    args.output_dir=tmp_path/'reserved'
    log=Path(str(args.output_dir)+'.launcher.log');log.write_text('keep')
    with pytest.raises(FileExistsError):runner.launch_background(args)
    assert log.read_text()=='keep'
