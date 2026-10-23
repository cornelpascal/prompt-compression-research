"""Machine-readable command boundary used by the human and Codex workers."""
from __future__ import annotations
import argparse
import json
import hashlib
import sys
from pathlib import Path
from .config import ROOT

_OUTPUT_WRITTEN = object()

def emit(value):
    print(json.dumps(value, ensure_ascii=False, default=str, indent=2))

def payload(path):
    return json.loads(sys.stdin.read() if path=='-' else Path(path).read_text())

def build_parser():
    parser=argparse.ArgumentParser(prog='paperkb',description='Evidence-linked research knowledge base. All results are JSON.')
    parser.add_argument('--json',action='store_true',help='Explicit JSON mode (also the default)')
    commands=parser.add_subparsers(dest='command',required=True)
    db=commands.add_parser('db').add_subparsers(dest='action',required=True)
    for name in ('init-env','up','migrate'): db.add_parser(name)
    p=db.add_parser('backup');p.add_argument('path')
    p=db.add_parser('restore');p.add_argument('path');p.add_argument('--database',required=True)
    corpus=commands.add_parser('corpus').add_subparsers(dest='action',required=True)
    p=corpus.add_parser('import');p.add_argument('--metadata',default=str(ROOT/'arxiv_prompt_compression/papers.json'));p.add_argument('--downloads',default=str(ROOT/'papers/downloads.json'));p.add_argument('--name',default='prompt-compression')
    p=corpus.add_parser('reopen');p.add_argument('corpus_id');p.add_argument('--reason',required=True)
    for name in ('prepare','packet','verify-date','request-revision'):
        p=commands.add_parser(name);p.add_argument('version_id')
        if name=='request-revision':p.add_argument('--reason',required=True)
    p=commands.add_parser('verify-dates');p.add_argument('--limit',type=int,default=20)
    p=commands.add_parser('recover-version');p.add_argument('version_id');p.add_argument('--candidate',required=True)
    p=commands.add_parser('render');p.add_argument('version_id');p.add_argument('page_number',type=int)
    p=commands.add_parser('claim');p.add_argument('--agent',required=True);p.add_argument('--kind',choices=['extract','review']);p.add_argument('--version');p.add_argument('--lease-seconds',type=int,default=1800)
    p=commands.add_parser('retry');p.add_argument('job_id');p.add_argument('--reason',required=True)
    for name in ('heartbeat','checkpoint','submit','review','fail'):
        p=commands.add_parser(name);p.add_argument('job_id');p.add_argument('--agent',required=True);p.add_argument('--token',required=True)
        if name=='heartbeat':p.add_argument('--lease-seconds',type=int,default=1800)
        if name=='checkpoint':p.add_argument('--section',required=True);p.add_argument('--draft-file',help='Persist a complete resumable draft JSON snapshot with the checkpoint')
        if name in ('checkpoint','submit','review'):p.add_argument('--file',required=True,help='Structured JSON file, or - for stdin')
        if name=='fail':p.add_argument('--error',required=True);p.add_argument('--no-retry',action='store_true')
    p=commands.add_parser('embed');p.add_argument('--version');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--initialize',action='store_true')
    p=commands.add_parser('expand');p.add_argument('--limit',type=int)
    p=commands.add_parser('reference-source');p.add_argument('reference_id');p.add_argument('--file',required=True)
    p=commands.add_parser('acquire-version');p.add_argument('nomination_id');p.add_argument('--file',required=True)
    commands.add_parser('expansion-close')
    graph=commands.add_parser('graph').add_subparsers(dest='action',required=True)
    for name in ('barrier','begin'):
        p=graph.add_parser(name);p.add_argument('--corpus')
    p=graph.add_parser('candidates');p.add_argument('idea_id');p.add_argument('--limit',type=int,default=25)
    p=graph.add_parser('relation');p.add_argument('graph_id');p.add_argument('--file',required=True)
    p=graph.add_parser('publish');p.add_argument('graph_id')
    p=commands.add_parser('group');p.add_argument('--label',required=True);p.add_argument('--description',required=True);p.add_argument('--ideas',nargs='+',required=True);p.add_argument('--rationale',required=True)
    commands.add_parser('status')
    p=commands.add_parser('search');p.add_argument('query');p.add_argument('--semantic',action='store_true');p.add_argument('--limit',type=int,default=20);p.add_argument('--min-relevance',type=int,default=0)
    for name in ('idea','lineage'):
        p=commands.add_parser(name);p.add_argument('idea_id')
    p=commands.add_parser('export');p.add_argument('--output')
    p=commands.add_parser('schema');p.add_argument('kind',choices=['extraction','review','relation','reference-source','source-version','rubric'])
    roll=commands.add_parser('rollout').add_subparsers(dest='action',required=True)
    roll.add_parser('init');roll.add_parser('status')
    p=roll.add_parser('next');p.add_argument('--agent',required=True);p.add_argument('--lease-seconds',type=int,default=1800)
    return parser

def run(args):
    from . import db, documents, workflow, expansion, graph, query, infrastructure, embeddings, rollout, source_versions
    if args.command=='schema':
        from .models import ExtractionPayload, ReviewPayload, RELEVANCE_RUBRIC
        return RELEVANCE_RUBRIC if args.kind=='rubric' else {'extraction':ExtractionPayload,'review':ReviewPayload,'relation':graph.RelationPayload,'reference-source':expansion.ReferenceSource,'source-version':source_versions.SourceVersion}[args.kind].model_json_schema()
    if args.command=='db':
        if args.action=='init-env':return infrastructure.init_env()
        if args.action=='up':return infrastructure.up()
        if args.action=='migrate':return {'migrations':db.migrate()}
        if args.action=='backup':return infrastructure.backup(args.path)
        if args.action=='restore':return infrastructure.restore(args.path,args.database)
    with db.connect() as conn:
        cmd=args.command
        if cmd=='corpus':
            if args.action=='import':return documents.import_corpus(conn,args.metadata,args.downloads,args.name)
            return workflow.reopen(conn,args.corpus_id,args.reason)
        if cmd=='prepare':return documents.prepare(conn,args.version_id)
        if cmd=='packet':return documents.packet(conn,args.version_id)
        if cmd=='verify-date':return documents.verify_version_date(conn,args.version_id)
        if cmd=='verify-dates':return documents.verify_version_dates(conn,args.limit)
        if cmd=='recover-version':return documents.recover_version(conn,args.version_id,args.candidate)
        if cmd=='render':return documents.render_page(conn,args.version_id,args.page_number)
        if cmd=='request-revision':return workflow.request_revision(conn,args.version_id,args.reason)
        if cmd=='claim':return workflow.claim(conn,args.agent,args.kind,args.version,args.lease_seconds)
        if cmd=='retry':return workflow.retry_job(conn,args.job_id,args.reason)
        if cmd in ('heartbeat','checkpoint','submit','review','fail'):
            shared=(conn,args.job_id,args.agent,args.token)
            if cmd=='heartbeat':return workflow.heartbeat(*shared,args.lease_seconds)
            if cmd=='checkpoint':
                checkpoint=payload(args.file)
                if args.draft_file:
                    draft=payload(args.draft_file)
                    if not isinstance(checkpoint,dict) or not isinstance(draft,dict):
                        raise ValueError('Checkpoint and draft must be JSON objects')
                    checkpoint['draft_snapshot']=draft
                    checkpoint['draft_sha256']=hashlib.sha256(json.dumps(draft,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                return workflow.checkpoint(*shared,args.section,checkpoint)
            if cmd=='submit':return workflow.submit(*shared,payload(args.file))
            if cmd=='review':return workflow.review(*shared,payload(args.file))
            if cmd=='fail':return workflow.fail(*shared,args.error,not args.no_retry)
        if cmd=='embed':return embeddings.initialize(conn) if args.initialize else embeddings.embed(conn,args.version,args.batch_size)
        if cmd=='expand':return expansion.expand(conn,args.limit)
        if cmd=='reference-source':return expansion.resolve_reference(conn,args.reference_id,payload(args.file))
        if cmd=='acquire-version':return source_versions.acquire(conn,args.nomination_id,payload(args.file))
        if cmd=='expansion-close':return expansion.close_expansion(conn)
        if cmd=='graph':
            if args.action=='barrier':return graph.barrier(conn,args.corpus)
            if args.action=='begin':return graph.begin_build(conn,args.corpus)
            if args.action=='candidates':return graph.candidates(conn,args.idea_id,args.limit)
            if args.action=='relation':return graph.add_relation(conn,args.graph_id,payload(args.file))
            if args.action=='publish':return graph.publish(conn,args.graph_id)
        if cmd=='group':return graph.canonical_group(conn,args.label,args.description,args.ideas,args.rationale)
        if cmd=='status':return workflow.status(conn)
        if cmd=='search':
            vector=None
            if args.semantic:vector,_=embeddings.query_vector(conn,args.query)
            return query.search(conn,args.query,vector,args.limit,args.min_relevance)
        if cmd=='idea':return query.inspect_idea(conn,args.idea_id)
        if cmd=='lineage':return query.lineage(conn,args.idea_id)
        if cmd=='export':
            if args.output:
                return query.export_to_file(conn,args.output)
            query.stream_export(conn,sys.stdout)
            return _OUTPUT_WRITTEN
        if cmd=='rollout':
            if args.action=='init':return rollout.initialize(conn)
            if args.action=='status':return rollout.pilot_status(conn)
            return rollout.next_work(conn,args.agent,args.lease_seconds)
    raise ValueError('Unknown command')

def main(argv=None):
    try:
        result=run(build_parser().parse_args(argv))
        if result is not _OUTPUT_WRITTEN:
            emit(result)
    except (Exception, KeyboardInterrupt) as exc:
        # Never dump a connection string or credentials in a structured CLI error.
        from .config import settings
        message=str(exc).replace(settings()['database_url'],'[database connection]')
        print(json.dumps({'error':type(exc).__name__,'message':message}),file=sys.stderr)
        return 130 if isinstance(exc,KeyboardInterrupt) else 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
