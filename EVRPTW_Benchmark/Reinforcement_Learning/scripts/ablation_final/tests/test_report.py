from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('ablation_final_report', Path(__file__).resolve().parents[1] / 'report.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def fixture_root(tmp_path, *, method='am_evrptw', status='running', epochs=300):
    run = tmp_path / 'runs' / method
    run.mkdir(parents=True)
    job = dict(method=method, run_id=method, scale='Cus100', training_epochs=epochs,
               validation_every_epochs=100, validation_views=500, physical_batch_size=4,
               world_size=1, effective_batch_size=4, training_trajectory_count=30,
               soft_stage_end_epoch=300, planned_full_run_soft_stage_end_epoch=2500,
               extra_args=['--ema-warmup-steps', '1000'])
    record = dict(job=job, output_dir=str(run), status=status, returncode=0 if status == 'completed' else None)
    state = dict(status='completed' if status == 'completed' else 'running', jobs=[record])
    (tmp_path / 'status.json').write_text(json.dumps(state))
    return run


def append_jsonl(path, records):
    with path.open('a') as stream:
        for row in records:
            stream.write(json.dumps(row) + '\n')


def validation(epoch, feasible=500, cost=450.1234567890123, instances=500):
    return dict(logical_epoch=epoch, instances=instances, complete_and_feasible=feasible,
                mean_verified_cost_usd=cost)


def test_empty_running_and_pending_epochs_are_not_zero_quality(tmp_path):
    fixture_root(tmp_path)
    _, rows, finals, warnings, done, completed = REPORT.collect(tmp_path)
    assert not done and not completed and not warnings
    assert [r['logical_epoch'] for r in rows] == [100, 200, 300]
    assert all(r['cost_usd'] is None and r['fr_percent'] is None for r in rows)
    assert finals[0]['result_300_status'] == 'pending_300'
    assert finals[0]['last_training_epoch'] == 0


def test_reinforce_epoch_counters_are_summed_not_treated_as_cumulative(tmp_path):
    run = fixture_root(tmp_path)
    append_jsonl(run / 'logical_epoch_history.jsonl', [dict(logical_epoch=e, instances_seen=4) for e in range(1, 4)])
    _, _, finals, _, _, _ = REPORT.collect(tmp_path)
    assert finals[0]['current_instance_exposure'] == 12
    assert finals[0]['current_exposure_evidence'] == 'sum_of_observed_epoch_instances'
    assert finals[0]['current_customer_exposure'] == 1200


def test_terran_reads_cumulative_csv_and_ignores_incomplete_last_line(tmp_path):
    run = fixture_root(tmp_path, method='terran')
    (run / 'logs').mkdir()
    (run / 'logs/train_log.csv').write_text('epoch,samples_seen,train_feasible_rate\n1,4,0.5\n2,8,0.7\n3,12')
    _, _, finals, warnings, _, _ = REPORT.collect(tmp_path)
    assert finals[0]['last_training_epoch'] == 2
    assert finals[0]['current_instance_exposure'] == 8
    assert finals[0]['current_exposure_evidence'] == 'observed_cumulative_samples_seen'
    assert finals[0]['current_train_feasible_rate'] == 0.7
    assert warnings


def test_one_epoch_profile_is_never_exported_as_completed_300(tmp_path):
    run = fixture_root(tmp_path, status='completed', epochs=1)
    (run / 'training_result.json').write_text(json.dumps(dict(status='passed', completed_training_epochs=1)))
    append_jsonl(run / 'validation_history.jsonl', [validation(1)])
    _, rows, finals, _, done, completed = REPORT.collect(tmp_path)
    assert done and not completed
    assert any(r['logical_epoch'] == 1 for r in rows)
    assert finals[0]['result_300_status'] == 'not_a_300_epoch_pilot'
    assert finals[0]['cost_usd'] is None


def test_full_completion_requires_terminal_result_and_three_complete_validations(tmp_path):
    run = fixture_root(tmp_path, status='completed')
    append_jsonl(run / 'validation_history.jsonl', [validation(e) for e in (100, 200, 300)])
    _, _, _, _, done, completed = REPORT.collect(tmp_path)
    assert done and not completed
    (run / 'training_result.json').write_text(json.dumps(dict(status='passed', completed_training_epochs=300)))
    _, _, finals, _, done, completed = REPORT.collect(tmp_path)
    assert done and completed
    assert finals[0]['cost_usd'] == 450.1234567890123
    assert finals[0]['fr_percent'] == 100.0
    assert finals[0]['current_instance_exposure'] == 1200
    assert finals[0]['current_exposure_evidence'].endswith('_estimate')


@pytest.mark.parametrize('bad_validation', [validation(300, instances=499, feasible=499), validation(300)])
def test_bad_cohort_or_duplicate_epoch_is_not_completed(tmp_path, bad_validation):
    run = fixture_root(tmp_path, status='completed')
    (run / 'training_result.json').write_text(json.dumps(dict(status='passed', completed_training_epochs=300)))
    records = [validation(100), validation(200), bad_validation]
    if bad_validation['instances'] == 500:
        records.append(bad_validation)
    append_jsonl(run / 'validation_history.jsonl', records)
    _, _, finals, _, _, completed = REPORT.collect(tmp_path)
    assert not completed and finals[0]['cost_usd'] is None
    assert finals[0]['result_300_status'] == 'incomplete_completion_evidence'


def test_zero_feasible_keeps_zero_fr_but_blank_cost_and_partial_json_is_visible(tmp_path):
    run = fixture_root(tmp_path)
    append_jsonl(run / 'validation_history.jsonl', [validation(100, feasible=0, cost=0)])
    with (run / 'validation_history.jsonl').open('a') as stream:
        stream.write('{"logical_epoch": 200,')
    _, rows, _, warnings, _, _ = REPORT.collect(tmp_path)
    assert rows[0]['fr_percent'] == 0.0 and rows[0]['cost_usd'] is None
    assert rows[1]['validation_status'] == 'missing'
    assert warnings


def test_failed_model_is_preserved_in_output_and_raw_inputs_unchanged(tmp_path):
    run = fixture_root(tmp_path, status='failed', method='evrptw_rl')
    append_jsonl(run / 'validation_history.jsonl', [validation(100, feasible=450)])
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    outcome = REPORT.generate(tmp_path)
    assert outcome['terminal'] and not outcome['completed_300']
    for path, content in before.items():
        assert path.read_bytes() == content
    analysis = tmp_path / 'analysis'
    assert (analysis / 'cost_curve.png').is_file()
    assert (analysis / 'feasibility_curve.png').is_file()
    with (analysis / 'summary_by_epoch.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['cost_usd'] == '450.1234567890123'
    assert rows[1]['cost_usd'] == ''
    assert rows[0]['fr_percent'] == '90.0'
    with (analysis / 'summary_at_300.csv').open() as stream:
        final = next(csv.DictReader(stream))
    assert final['result_300_status'] == 'failed'
    assert 'EMA baseline warmup 1000' in final['notes']
    assert '没有计算五模型共同可行交集' in (analysis / 'RESULTS_SUMMARY.md').read_text()


def test_am_note_records_actual_warmup_units_and_report_explanation(tmp_path):
    fixture_root(tmp_path, method='am_evrptw')
    state, rows, finals, warnings, done, completed = REPORT.collect(tmp_path)
    assert '2500 x 1 = 2500 optimizer updates' in finals[0]['notes']
    assert 'EMA warmup' in finals[0]['notes']
    text = REPORT.markdown(state, rows, finals, warnings, done, completed)
    assert 'steps_per_epoch=2500 × baseline_warmup_epochs=1' in text
    assert '2500 次更新' in text
    overridden = REPORT.stage_note(dict(method='am_evrptw', extra_args=[
        '--steps-per-epoch', '500', '--baseline-warmup-epochs', '2']))
    assert '500 x 2 = 1000 optimizer updates' in overridden
