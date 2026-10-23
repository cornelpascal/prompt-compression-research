"""Merge a fixed-schema workflow result into the public notebook."""
import argparse
import json
from pathlib import Path
import re
from validate_site_data import safe, refresh_manifest, validate

ALLOWED = {'id', 'task', 'requestId', 'recordedAt', 'status', 'commit', 'url', 'evidence',
           'tests', 'suites', 'summary', 'exitCode', 'errorType', 'elapsedSeconds'}


def publish(source, directory):
    root = Path(directory)
    value = json.loads(Path(source).read_text())
    if set(value) - ALLOWED:
        raise ValueError('Unapproved outcome fields')
    assert re.fullmatch(r'\d+', value['id']), 'Only GitHub run IDs can be published'
    assert value['task'] in ('attention-demo', 'compression-demo', 'privacy-pilot', 'theory', 'tests')
    assert value['status'] in ('success', 'failure')
    if value['status'] == 'success':
        assert value.get('exitCode') == 0, 'Success requires a zero exit code'
        assert not any(value.get('tests', {}).get(k, 0) for k in ('failures', 'errors')), 'Success cannot contain test failures'
    assert value['url'] == f'https://github.com/cornelpascal/prompt-compression-research/actions/runs/{value["id"]}'
    safe(value)
    (root / 'jobs').mkdir(exist_ok=True)
    (root / 'jobs' / f'{value["id"]}.json').write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
    latest = root / 'latest-jobs.json'
    history = json.loads(latest.read_text()) if latest.exists() else []
    concise = {k: v for k, v in value.items() if k != 'summary'}
    concise['resultFile'] = f'jobs/{value["id"]}.json'
    history = [concise] + [r for r in history if r['id'] != value['id']]
    latest.write_text(json.dumps(history[:50], separators=(',', ':')) + '\n')
    refresh_manifest(root)
    validate(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--directory', default='site-data')
    args = parser.parse_args()
    publish(args.source, args.directory)
