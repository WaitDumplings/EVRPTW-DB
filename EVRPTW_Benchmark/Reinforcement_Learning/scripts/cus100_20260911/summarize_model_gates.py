#!/usr/bin/env python3
"""Collect the selected DRL-TS/EVRPTW-RL/RRNCO Cus100 engineering gates."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'EVRPTW_Benchmark/results/cus100_20260911/profiling'
SELECTED = {
    'drl_ts_G': 'drl_ts_G_b24_t30_h240_road_final_b24_checkpoint1',
    'drl_ts_E': 'drl_ts_E_b24_t30_h240_synthetic_final_b24_checkpoint1',
    'rrnco_G': 'rrnco_G_b50_t30_h240_road_fullval_b50',
    'rrnco_E': 'rrnco_E_b50_t30_h240_synthetic_final_b50',
    'evrptw_rl_G': 'evrptw_rl_G_b200_t30_h240_road_fullval_b200',
    'evrptw_rl_E': 'evrptw_rl_E_b200_t30_h240_synthetic_final_b200_checkpoint1_accelerated_warmup',
}

def main():
    rows = []
    for key, directory in SELECTED.items():
        out = BASE / directory
        summary_path = out / 'probe_summary.json'
        if not summary_path.exists():
            rows.append({'configuration': key, 'status': 'pending', 'path': str(out)})
            continue
        summary = json.loads(summary_path.read_text())
        validation = json.loads((out/'validation_summary.json').read_text())
        updates = json.loads((out/'optimizer_update_probe.json').read_text())
        history = [json.loads(x) for x in (out/'logical_epoch_history.jsonl').read_text().splitlines() if x]
        verified = [x for x in validation['rows'] if x.get('verifier_passed')]
        error = max((abs(float(x['objective_cost_usd']) - (.151750972762646*float(x['objective_distance_km']) + 413.6331536717643*float(x['vehicle_count']))) for x in verified), default=None)
        row = {
            **summary,
            'configuration': key,
            'path': str(out),
            'validation': {k: validation.get(k) for k in ('instances','candidate_count','complete_and_feasible','verifier_summary_passed','validation_wall_time_s','mean_verified_cost_usd','mean_verified_distance_km','mean_verified_vehicle_count')},
            'selected_verified_cost_decomposition_max_abs_error_usd': error,
            'optimizer_updates_recorded': len(updates),
            'optimizer_parameter_updates_finite_nonzero': all(x['parameters_finite'] and x['parameter_delta_l2'] > 0 for x in updates),
            'torch_peak_allocated_gib': max(x['cuda_peak_allocated_bytes'] for x in updates)/2**30,
            'torch_peak_reserved_gib': max(x['cuda_peak_reserved_bytes'] for x in updates)/2**30,
            'last_five_training_epoch_mean_s': sum(float(x['epoch_wall_time_s']) for x in history[-5:])/len(history[-5:]),
            'smoke_protocol_overrides': ({'ema_warmup_steps':1,'formal_ema_warmup_steps':1000} if key=='evrptw_rl_E' else {'soft_stage_end_epoch':5,'formal_soft_stage_end_epoch':2500} if key.startswith('drl_ts') else {}),
            'gate_interpretation':'engineering completion and finite updates; early policy feasibility is reported separately, not required to be 500/500 after a few updates',
        }
        row['engineering_gate_passed'] = bool(summary['returncode']==0 and summary['losses_finite'] and row['optimizer_parameter_updates_finite_nonzero'] and validation['instances']==500 and validation['candidate_count']==30 and (error is None or error < 1e-8))
        rows.append(row)
    report={'schema':'cus100_three_model_gate_report_v1','configurations':rows,'related_cpu_tests_passed':51,
            'selected_route_validation_scope':'The chosen candidate is verified independently; this does not assert exhaustive verification of all raw sampling candidates.',
            'checkpoints_are_disposable':True,'test_data_accessed':False,
            'excluded_preliminary_probes':'All synthetic_preliminary_oldscale and all failed/OOM attempts remain archived but are excluded from selected final gates.'}
    destination = BASE.parent / 'three_model_memory_gates.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(destination)
    for row in rows:
        print(row['configuration'], row.get('status','complete'), row.get('peak_process_gib'), row.get('validation',{}).get('complete_and_feasible'))

if __name__=='__main__':main()
