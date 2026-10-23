"""Validate portable public data, provenance, relationships and measured units."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def safe(value):
    encoded = json.dumps(value, allow_nan=False)
    for marker in ('/Users/', '/home/', 'postgresql://', 'postgres://', 'ghp_', 'github_pat_', 'Bearer '):
        if marker in encoded:
            raise ValueError('Private infrastructure or credential-like content in public data')
    if isinstance(value, dict):
        for item in value.values():
            safe(item)
    elif isinstance(value, list):
        for item in value:
            safe(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite number in public data')


def validate(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['schemaVersion'] == 1
    for name, expected in manifest['files'].items():
        assert re.fullmatch(r'[a-zA-Z0-9/_-]+\.json', name) and '..' not in name
        raw = (root / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, f'Checksum mismatch: {name}'
    for path in root.rglob('*.json'):
        safe(json.loads(path.read_text()))
        if path.name != 'manifest.json':
            assert path.relative_to(root).as_posix() in manifest['files'], f'Unmanifested data: {path.name}'
    catalog = json.loads((root / 'catalog.json').read_text())
    literature = json.loads((root / 'literature.json').read_text())
    assert catalog['schemaVersion'] == 1 and catalog['author'] == 'Ganea Corneliu'
    assert catalog['privacy']['unit'] == 'UTF-8 bytes'
    papers = {p['id']: p for p in literature['papers']}
    assert len(papers) == len(literature['papers']) == catalog['paperkb']['papers']
    assert len(literature['ideas']) == catalog['paperkb']['reviewedIdeas']
    assert all(papers[i['paper_id']]['status'] == 'complete' for i in literature['ideas'])
    assert all(e['source'] in papers and e['target'] in papers for e in literature['edges'])
    assert catalog['paperkb']['productionParents'] == 0
    for run in catalog['compression']:
        assert run['evidence'] == 'engineering_fixture'
        assert len(run['sourceHash']) == 64
        for row in run['summary']:
            assert 0 <= row['f1']['mean'] <= 1
            assert row['f1']['low'] <= row['f1']['mean'] <= row['f1']['high']
            assert row['cases'] > 0
    for row in catalog['privacy']['frontier']:
        if row['status'] == 'infeasible':
            assert 'leakage_bits' not in row, 'Infeasible points cannot be shown as zero leakage'
    return {'papers': len(papers), 'reviewedIdeas': len(literature['ideas']), 'files': len(manifest['files'])}


def refresh_manifest(root):
    root = Path(root)
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['files'] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(root.rglob('*.json')) if p.name != 'manifest.json'}
    path.write_text(json.dumps(manifest, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='site-data')
    args = parser.parse_args()
    print(json.dumps(validate(args.directory)))
