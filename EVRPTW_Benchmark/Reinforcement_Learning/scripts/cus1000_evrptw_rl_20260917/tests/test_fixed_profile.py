"""A fixed user batch must not silently change after a failed memory probe."""
import json
from pathlib import Path
import pytest
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus1000_evrptw_rl_20260917.common import load_config

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus1000_evrptw_rl_20260917 import autocalibrate as cal

@pytest.mark.parametrize('case',['passed','oom','missing_rank','over_ceiling'])
def test_profile_preserves_fixed_batch_and_blocks_failed_confirmation(tmp_path,monkeypatch,case):
 cfg=load_config(batch=12)
 calls=[]
 peaks={str(i):9.2 for i in range(4)}
 result={'status':'passed','peak_process_gib':peaks}
 if case=='oom':result={'status':'oom','peak_process_gib':{}}
 if case=='missing_rank':del peaks['3']
 if case=='over_ceiling':peaks['3']=10.4
 monkeypatch.setattr(cal,'preflight',lambda *a,**kw:{'gpus':[], 'data':{}})
 monkeypatch.setattr(cal,'lock_gpus',lambda *a:[])
 monkeypatch.setattr(cal,'gpu_inventory',lambda:[])
 monkeypatch.setattr(cal,'gpu_processes',lambda:[])
 monkeypatch.setattr(cal,'validate_gpus',lambda *a:[])
 def probe(*a,**kw):
  calls.append((kw['batch'],kw['confirmation']))
  return result
 monkeypatch.setattr(cal,'run_probe',probe)
 monkeypatch.setattr(cal,'prepare_stream',lambda *a,**kw:{'contract':{'scale':'Cus1000'}})
 if case=='passed':
  path=cal.calibrate(cfg,tmp_path/'road',tmp_path/'out')
  final=json.loads(Path(path).read_text())
  assert final['physical_batch_size']==12 and final['effective_batch_size']==48
  report=json.loads((tmp_path/'out/calibration/report.json').read_text())
  assert report['status']=='passed' and not report['within_target']
 else:
  with pytest.raises(RuntimeError,match='Fixed batch 12'):
   cal.calibrate(cfg,tmp_path/'road',tmp_path/'out')
  assert not (tmp_path/'out/calibrated_config.json').exists()
 assert calls==[(12,True)]
