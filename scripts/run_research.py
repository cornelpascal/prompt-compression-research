"""Fixed CPU presets; publish explicit outcomes even when an executed command fails."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('attention-demo', 'compression-demo', 'privacy-pilot', 'theory', 'tests')


def test_totals(paths):
    totals = dict(tests=0, failures=0, errors=0, skipped=0)
    suites = []
    for path in paths:
        root = ET.parse(path).getroot()
        counts = {key: sum(int(s.attrib.get(key, 0)) for s in root.iter('testsuite')) for key in totals}
        suites.append({'name': path.stem, **counts})
        for key in totals:
            totals[key] += counts[key]
    return totals, suites


def run(task, output, run_id, request_id):
    if task not in TASKS or not re.fullmatch(r'[A-Za-z0-9-]{1,80}', run_id):
        raise ValueError('Invalid CPU task or run identifier')
    if not re.fullmatch(r'[A-Za-z0-9-]{8,80}', request_id):
        raise ValueError('Invalid request identifier')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    public = output / 'public'
    public.mkdir(exist_ok=True)
    python = sys.executable
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(str(ROOT / p) for p in
        ('src', 'research/prompt_compression', 'research/controlled_leakage')),
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MPLBACKEND': 'Agg', 'TOKENIZERS_PARALLELISM': 'false'}
    started = datetime.now(timezone.utc)
    record = {'id': run_id, 'task': task, 'requestId': request_id,
              'recordedAt': started.isoformat(), 'status': 'failure',
              'commit': os.environ.get('GITHUB_SHA', 'local'),
              'url': f'https://github.com/cornelpascal/prompt-compression-research/actions/runs/{run_id}',
              'evidence': 'engineering_fixture' if task in ('attention-demo', 'compression-demo', 'tests') else
                          'finite_source' if task == 'theory' else 'synthetic_pilot'}
    def command(args, suffix='run'):
        with (output / f'{suffix}.log').open('w') as log:
            completed = subprocess.run([python, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return completed.returncode
    try:
        if task == 'tests':
            codes = []
            for name, folder in [('paperkb', 'tests'), ('compression', 'research/prompt_compression/tests'),
                                 ('leakage', 'research/controlled_leakage/tests')]:
                codes.append(command(['-m', 'pytest', '-q', '--tb=short', folder,
                                      f'--junitxml={public / (name + ".xml")}'], name))
            env['PROMPTCOMP_DISTRIBUTED_TEST'] = '1'
            codes.append(command(['-m', 'pytest', '-q', '--tb=short',
                'research/prompt_compression/tests/test_distributed.py',
                f'--junitxml={public / "distributed.xml"}'], 'distributed'))
            record['tests'], record['suites'] = test_totals(sorted(public.glob('*.xml')))
            # Test logs remain local to the runner; JUnit artifacts contain failures if present.
            code = 0 if all(value == 0 for value in codes) else 1
        elif task in ('attention-demo', 'compression-demo'):
            attention = task == 'attention-demo'
            code = command(['-m', 'promptcomp.attention_smoke' if attention else 'promptcomp.smoke',
                            '--output', str(output / 'experiment')])
            if code == 0:
                report = output / 'experiment' / ('test-report' if attention else 'report') / 'summary.json'
                record['summary'] = {'unit': 'receiver tokens', 'rows': json.loads(report.read_text())['summary'],
                    'interpretation': 'Random tiny models and artificial receivers. Validates execution, not real-model quality.'}
        elif task == 'theory':
            code = command(['-m', 'leaklab', 'theory', '--output', str(output / 'frontier.json')])
            if code == 0:
                rows = json.loads((output / 'frontier.json').read_text())
                record['summary'] = {'unit': 'bits', 'rows': rows, 'points': len(rows),
                    'feasible': sum(x['status'] != 'infeasible' for x in rows),
                    'interpretation': 'Finite-source, fixed-decoder optimization. Infeasible points have no leakage value.'}
        else:
            code = command(['-m', 'leaklab', 'generate', '--output', str(output / 'data'), '--size', '12'], 'generate')
            if code == 0:
                code = command(['-m', 'leaklab', 'run', '--config', 'research/controlled_leakage/configs/pilot.json',
                                '--data', str(output / 'data'), '--output', str(output / 'pilot')])
            if code == 0:
                record['summary'] = {'unit': 'UTF-8 bytes', 'rows': json.loads((output / 'pilot/summary.json').read_text()),
                    'interpretation': 'Synthetic rule-based CPU pilot with 12 base records per split. Not a learned privacy guarantee.'}
        record['status'] = 'success' if code == 0 else 'failure'
        record['exitCode'] = code
    except Exception as exc:
        record['errorType'] = type(exc).__name__  # Never publish environment values or exception messages.
        code = 1
    record['elapsedSeconds'] = round((datetime.now(timezone.utc) - started).total_seconds(), 3)
    (public / 'outcome.json').write_text(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: record[k] for k in ('id', 'task', 'status', 'elapsedSeconds')}))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--request-id', required=True)
    args = parser.parse_args()
    sys.exit(run(args.task, args.output, args.run_id, args.request_id))
