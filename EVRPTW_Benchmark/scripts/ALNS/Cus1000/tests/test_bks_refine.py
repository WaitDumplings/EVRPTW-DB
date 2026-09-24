"""Fast contract checks; real-data validation is documented in BKS_T1_README.md."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bks_refine import Timeline, select_part


class BKSRefineTests(unittest.TestCase):
    def test_halves_are_disjoint_exhaustive_and_preserve_input_order(self):
        records = [{'instance_id': str(i)} for i in reversed(range(500))]
        upper, lower = select_part(records, 'upper'), select_part(records, 'lower')
        self.assertEqual((len(upper), len(lower)), (250, 250))
        self.assertEqual(upper + lower, records)
        self.assertFalse({r['instance_id'] for r in upper} & {r['instance_id'] for r in lower})

    def test_bad_input_count_or_duplicate_id_is_rejected(self):
        for records in ([{'instance_id': str(i)} for i in range(499)],
                        [{'instance_id': 'duplicate'}] * 500):
            with self.assertRaises(ValueError):
                select_part(records, 'upper')

    def make_timeline(self, directory):
        task = {'output_dir': directory, 'time_limit_s': 4, 'checkpoints_s': [1, 2, 3, 4],
                'bks': {'instance_id': 'example', 'objective': 100,
                        'distance_km': 100, 'solution': [[0, 1, 0]]},
                'contract_fingerprint': 'test'}
        return Timeline(task, {'objective_value': 100})

    def test_delayed_writer_does_not_backfill_late_improvements(self):
        with tempfile.TemporaryDirectory() as directory:
            timeline = self.make_timeline(directory)
            timeline.observe(1.5, [[0, 2, 0]], 90, {'objective_value': 90})
            timeline.observe(2.5, [[0, 3, 0]], 80, {'objective_value': 80})
            timeline.observe(3.5, [[0, 4, 0]], 95, {'objective_value': 95})
            timeline.observe(4.1, [[0, 5, 0]], 70, {'objective_value': 70})
            timeline.write_due(4.2, final=True)
            values = [json.loads((Path(directory)/f'best_at_{t}s.json').read_text())
                      for t in (1, 2, 3, 4)]
            self.assertEqual([v['objective_value'] for v in values], [100, 90, 80, 80])
            self.assertTrue(all(v['incumbent_event_time_s'] <= v['checkpoint_s'] for v in values))
            self.assertEqual(timeline.recorder.best_event['objective_value'], 80)

    def test_only_reached_checkpoints_are_written_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            timeline = self.make_timeline(directory)
            timeline.write_due(1.1)
            first = Path(directory)/'best_at_1s.json'
            before = first.read_bytes()
            self.assertEqual(len(list(Path(directory).glob('best_at*'))), 1)
            timeline.observe(2.0, [[0, 2, 0]], 90, {'objective_value': 90})
            timeline.write_due(4, final=True)
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(json.loads((Path(directory)/'best_at_2s.json').read_text())['objective_value'], 90)


if __name__ == '__main__':
    unittest.main()
