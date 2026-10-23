"""Export an allowlisted, portable research snapshot. Standard library only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def clean(value):
    """Fail closed on infrastructure identifiers; explicit schemas do the selection."""
    encoded = json.dumps(value, ensure_ascii=False)
    forbidden = ('/Users/', '/home/', 'postgresql://', 'postgres://', 'ghp_', 'github_pat_', 'Bearer ')
    if any(marker in encoded for marker in forbidden):
        raise ValueError('Export contains a private path or credential-like value')
    return value


def literature(root, export_file=None, recorded_at=None):
    status = read(root / '.paperkb/implementation-status.json')
    export = Path(export_file) if export_file else root / status['latest_export']['json']
    data = read(export)['data']
    versions = {v['id']: v for v in data['paper_versions']}
    by_paper = {}
    for v in versions.values():
        by_paper.setdefault(v['paper_id'], []).append(v)
    approved = {(r['version_id'], r['revision']) for r in data['reviews'] if r['accepted']}
    ideas = []
    for idea in data['ideas']:
        version = versions[idea['version_id']]
        if not idea['active'] or version['state'] != 'complete' or (version['id'], idea['revision']) not in approved:
            continue
        ideas.append({**{k: idea[k] for k in ('id', 'title', 'statement', 'kind', 'conditions', 'confidence', 'relevance_score', 'validity_flags')},
                      'paper_id': version['paper_id'], 'version': version['version_label']})
    counts = {}
    for idea in ideas:
        counts[idea['paper_id']] = counts.get(idea['paper_id'], 0) + 1
    papers = []
    for p in data['papers']:
        vs = by_paper.get(p['id'], [])
        complete = bool(vs) and all(v['state'] == 'complete' for v in vs)
        papers.append({**{k: p[k] for k in ('id', 'title', 'arxiv_id', 'doi', 'source_depth')},
            'authors': p['metadata'].get('authors', []), 'published': p['metadata'].get('published'),
            'status': 'complete' if complete else 'in progress', 'ideas': counts.get(p['id'], 0),
            'relevance': max((v['relevance_score'] or 0 for v in vs), default=0),
            'versions': [{'label': v['version_label'], 'status': v['state'], 'date_verified': v['date_verified']} for v in vs],
            'url': 'https://arxiv.org/abs/' + p['arxiv_id'] if p['arxiv_id'] else 'https://doi.org/' + p['doi'] if p['doi'] else None})
    references = {r['id']: r for r in data['paper_references']}
    edges = []
    for n in data['predecessor_nominations']:
        if n['state'] not in ('admitted', 'existing') or not n['target_paper_id']:
            continue
        ref = references[n['reference_id']]
        edges.append({'source': versions[ref['version_id']]['paper_id'], 'target': n['target_paper_id'],
                      'kind': 'reviewed dependency nomination', 'status': n['state']})
    snapshot = {'recordedAt': status['recorded_at'], 'evidence': 'historical_snapshot',
        'papers': sum(p['count'] for p in status['status']['papers']),
        'complete': sum(p['complete'] for p in status['status']['paper_completion']),
        'ideasRecorded': status['status']['ideas'], 'reviewedIdeas': len(ideas),
        'versions': status['status']['versions'], 'productionParents': status['production_parent_links'],
        'sourceHash': sha(export), 'databaseLive': False,
        'note': 'Saved evidence snapshot. Lease and worker records are deliberately excluded. Dependency nominations are not published ancestry.'}
    if export_file:
        if not recorded_at:
            raise ValueError('An explicit snapshot timestamp is required for a replacement export')
        from collections import Counter
        snapshot.update(recordedAt=recorded_at, papers=len(papers), complete=sum(p['status'] == 'complete' for p in papers),
                        ideasRecorded=len(data['ideas']), productionParents=len(data['graph_parents']),
                        versions=[{'state': state, 'count': count} for state, count in sorted(Counter(v['state'] for v in versions.values()).items())])
    return snapshot, clean({'snapshot': snapshot, 'papers': papers, 'ideas': ideas, 'edges': edges})


def compression_run(root, relative, label, section, report_folder='report', evaluation_folder='evaluation'):
    directory = root / relative
    summary = read(directory / report_folder / 'summary.json')
    manifest = read(directory / evaluation_folder / 'manifest.json')
    index = read(directory / evaluation_folder / 'index.json')
    if not manifest['complete']:
        raise ValueError('Cannot publish incomplete evaluation')
    cases = []
    for key in index['keys']:
        path = directory / evaluation_folder / 'cases' / (key + '.json')
        if sha(path) != index['checksums'][key]:
            raise ValueError('Evaluation checksum mismatch')
        r = read(path)
        cases.append({k: r[k] for k in ('id', 'group_id', 'method', 'target', 'ratio', 'seed', 'score', 'budgeted_score', 'compression', 'answer', 'diagnostics')})
    examples = {}
    data_dir = directory / 'data'
    if data_dir.exists():
        for split in ('validation', 'test'):
            for path in data_dir.rglob(split + '.jsonl'):
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    examples[row['id']] = {k: row[k] for k in ('prompt', 'question', 'context', 'answers')}
    for case in cases:
        case['example'] = examples.get(case['id'])
    training = []
    # Only primary model stages, avoiding duplicate CLI verification runs.
    for family in ('rewrite', 'delete', 'attention'):
        for stage in ('sft', 'rl'):
            path = directory / family / stage / 'state.json'
            if path.exists():
                state = read(path)
                training.append({'method': family, 'stage': stage, 'steps': state['step'],
                    'seconds': state['elapsed_seconds'], 'losses': state['losses'], 'memory': state['peak_memory_bytes']})
    return clean({'id': section, 'label': label, 'project': 'compression', 'recordedAt': '2026-10-22',
        'evidence': 'engineering_fixture', 'profile': summary['profile'], 'unit': 'reference tokens',
        'identity': manifest['identity'], 'sourceHash': manifest['inputs']['code'],
        'summary': summary['summary'], 'threshold': summary.get('attention_threshold'),
        'training': training, 'cases': cases, 'limitations': summary['limitations'],
        'source': relative + '/' + report_folder + '/summary.json'})


def privacy(root):
    directory = root / 'research/controlled_leakage/runs/pilot'
    manifest = read(directory / 'manifest.json')
    if sha(directory / 'results.json') != manifest['results_sha256']:
        raise ValueError('Privacy evaluation checksum mismatch')
    with (directory / 'summary.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    comparisons = read(directory / 'comparison.json')
    return clean({'id': 'privacy-pilot', 'label': 'Controlled leakage · CPU pilot',
        'recordedAt': '2026-10-17', 'evidence': 'synthetic_pilot', 'unit': 'UTF-8 bytes',
        'summary': rows, 'comparisons': comparisons,
        'frontier': read(root / 'research/controlled_leakage/runs/frontier.json'),
        'sourceHash': manifest['results_sha256'], 'rows': manifest['evaluation_rows'],
        'limitations': ['Deterministic rule task and synthetic private values.',
            'Repeated baseline seeds are not independently trained models.',
            'Original and redaction-only controls may exceed requested budgets.',
            'Finite certificates apply only to the finite-source, fixed-decoder problem.']})


def export(workspace, output, paperkb_export=None, recorded_at=None):
    root, output = Path(workspace).resolve(), Path(output)
    snapshot, papers = literature(root, paperkb_export, recorded_at)
    attention_validation = read(root / 'research/prompt_compression/attention-validation.json')
    for path, expected in attention_validation['config_sha256'].items():
        if sha(root / 'research/prompt_compression' / path) != expected:
            raise ValueError('Pinned attention configuration changed')
    runs = [compression_run(root, 'research/prompt_compression/runs/cpu-final', 'Rewrite & deletion · CPU fixture', 'compression-fixture'),
        compression_run(root, 'research/prompt_compression/runs/attention-cpu-ready', 'Attention · validation sweep', 'attention-validation', 'validation-report', 'validation'),
        compression_run(root, 'research/prompt_compression/runs/attention-cpu-ready', 'Attention · locked test', 'attention-test', 'test-report', 'test')]
    # The published cases carry only public, artificial examples; no raw source passages.
    data = {'schemaVersion': 1, 'author': 'Ganea Corneliu', 'generatedAt': datetime.now(timezone.utc).isoformat(),
        'repository': 'cornelpascal/prompt-compression-research', 'paperkb': snapshot,
        'compression': runs, 'privacy': privacy(root),
        'datasets': [{'name': 'SQuAD 1.1', 'train': 79100, 'validation': 8499, 'test': 10570,
            'unit': 'questions', 'groups': [397, 45, 48], 'license': 'CC BY-SA 4.0',
            'url': 'https://rajpurkar.github.io/SQuAD-explorer/', 'note': 'Official development set is the study’s held-out test. Article-grouped splits. These counts describe the prepared dataset, not the tiny fixtures plotted here.'}],
        'tests': [
            {'id': 'compression', 'title': 'Prompt compression', 'date': '2026-10-22', 'passed': 30, 'skipped': 2, 'failed': 0, 'status': 'recorded pass', 'scope': 'Package suite; tiny random models', 'command': 'cd research/prompt_compression && python -m pytest -q --tb=short', 'source': 'research/prompt_compression/VALIDATION.md'},
            {'id': 'distributed', 'title': 'Two-process recovery', 'date': '2026-10-22', 'passed': 2, 'skipped': 0, 'failed': 0, 'status': 'recorded pass', 'scope': 'CPU loopback; rewrite and attention', 'command': 'PROMPTCOMP_DISTRIBUTED_TEST=1 python -m pytest -q tests/test_distributed.py', 'source': 'research/prompt_compression/tests/test_distributed.py'},
            {'id': 'leakage', 'title': 'Controlled leakage', 'date': '2026-10-17', 'passed': 29, 'skipped': 0, 'failed': 0, 'status': 'recorded pass', 'scope': 'Synthetic controls, tiny models and real LoRA updates', 'command': 'python -m pytest -q research/controlled_leakage/tests', 'source': 'research/controlled_leakage/docs/VALIDATION.md'},
            {'id': 'paperkb-history', 'title': 'PaperKB · historical suite', 'date': '2026-09-19', 'passed': 164, 'skipped': None, 'failed': None, 'status': 'historical pass', 'scope': 'Database available during recorded validation', 'command': 'python -m pytest -q --tb=short', 'source': 'PAPERKB.md'},
            {'id': 'paperkb-latest', 'title': 'Workspace · later attempt', 'date': '2026-10-22', 'passed': 55, 'skipped': 1, 'failed': 137, 'status': 'infrastructure blocked', 'scope': 'PostgreSQL unavailable; 137 setup errors, not 137 assertion failures. Counts overlap other suites.', 'command': 'python -m pytest -q --tb=short', 'source': 'research/prompt_compression/VALIDATION.md'}],
        'capabilities': {'cpu': True, 'gpu': False, 'productionDatabase': False, 'privateExecutionRepository': None},
        'sources': [{'name': 'PaperKB export', 'sha256': snapshot['sourceHash'], 'date': snapshot['recordedAt']},
            {'name': 'Attention validation record', 'sha256': sha(root / 'research/prompt_compression/attention-validation.json'), 'date': attention_validation['validated_at_utc']}]}
    clean(data)
    write(output / 'catalog.json', data)
    write(output / 'literature.json', papers)
    write(output / 'manifest.json', {'schemaVersion': 1, 'generatedAt': data['generatedAt'],
        'files': {name: sha(output / name) for name in ('catalog.json', 'literature.json')}})
    if not (output / 'latest-jobs.json').exists():
        write(output / 'latest-jobs.json', [])
    from validate_site_data import refresh_manifest
    refresh_manifest(output)
    print(json.dumps({'papers': len(papers['papers']), 'reviewed_ideas': len(papers['ideas']),
        'compression_runs': len(runs), 'privacy_groups': len(data['privacy']['summary']), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', default=str(REPO / 'site-data'))
    parser.add_argument('--paperkb-export', help='Replacement private PaperKB JSON snapshot')
    parser.add_argument('--paperkb-recorded-at', help='ISO-8601 observation time of replacement snapshot')
    args = parser.parse_args()
    export(args.workspace, args.output, args.paperkb_export, args.paperkb_recorded_at)
