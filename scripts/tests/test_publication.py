import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_site_data import safe, validate, refresh_manifest
from publish_outcome import publish
from run_research import run, test_totals

ROOT = Path(__file__).resolve().parents[2]


class PublicationTests(unittest.TestCase):
    def test_current_snapshot_is_consistent(self):
        self.assertGreater(validate(ROOT / 'site-data')['reviewedIdeas'], 0)

    def test_private_values_and_nonfinite_numbers_are_rejected(self):
        for value in ['/Users/example/private', 'postgresql://example', 'github_pat_example', float('nan')]:
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                safe({'nested': [value]})

    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / 'site-data', root / 'data')
            (root / 'data/latest-jobs.json').write_text('[{}]')
            with self.assertRaisesRegex(AssertionError, 'Checksum mismatch'):
                validate(root / 'data')

    def test_publish_is_idempotent_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / 'site-data', root / 'data')
            record = {'id': '123', 'task': 'tests', 'status': 'failure', 'recordedAt': '2026-10-24',
                      'url': 'https://github.com/cornelpascal/prompt-compression-research/actions/runs/123'}
            source = root / 'outcome.json'
            source.write_text(json.dumps(record))
            publish(source, root / 'data')
            publish(source, root / 'data')
            history = json.loads((root / 'data/latest-jobs.json').read_text())
            self.assertEqual(sum(r['id'] == '123' for r in history), 1)
            self.assertEqual(history[0]['status'], 'failure')
            record['unapproved_raw_records'] = []
            source.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                publish(source, root / 'data')

    def test_infeasible_is_not_a_zero_measurement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / 'site-data', root / 'data')
            path = root / 'data/catalog.json'
            value = json.loads(path.read_text())
            point = next(p for p in value['privacy']['frontier'] if p['status'] == 'infeasible')
            point['leakage_bits'] = 0
            path.write_text(json.dumps(value))
            refresh_manifest(root / 'data')
            with self.assertRaisesRegex(AssertionError, 'Infeasible'):
                validate(root / 'data')

    def test_invalid_task_and_identifier_cannot_launch_processes(self):
        with self.assertRaises(ValueError):
            run('arbitrary-shell', '/unused', '123', 'request-123')
        with self.assertRaises(ValueError):
            run('theory', '/unused', '123', '$(touch example)')

    def test_junit_errors_are_distinct_from_assertion_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'suite.xml'
            path.write_text('<testsuites><testsuite tests="6" failures="1" errors="2" skipped="1"/></testsuites>')
            totals, suites = test_totals([path])
            self.assertEqual(totals, dict(tests=6, failures=1, errors=2, skipped=1))
            self.assertEqual(suites[0]['name'], 'suite')

    def test_killed_suite_cannot_be_counted_as_success(self):
        with tempfile.TemporaryDirectory() as folder, patch('run_research.subprocess.run', return_value=SimpleNamespace(returncode=-9)):
            code = run('tests', folder, '123', 'request-123')
            record = json.loads((Path(folder) / 'public/outcome.json').read_text())
            self.assertNotEqual(code, 0)
            self.assertEqual(record['status'], 'failure')


if __name__ == '__main__':
    unittest.main()
