"""Progress of the experiments, reported by every machine to one shared table.

A *unit* is one piece of work of an experiment (utils/experiment.py): a teacher fold, a
hyper-parameter study, or a final fold of a tuned model. Each unit reports its status
(queued / running / done / failed / skipped) with its key metrics. Reports go to:

    local   always: logs/progress/<host>.jsonl, one JSON line per report (append-only)
    Notion  when NOTION_TOKEN (an internal integration's secret) and NOTION_PROGRESS_DB
            (the database id) are set: one row per unit, created or updated in place
            (title = unit key). The database schema is ``NOTION_SCHEMA``; create it once
            and share it with the integration.

Like MLflow tracking, reporting is optional and never stops a run: a failure is warned
about once and the local line is still written. MLflow keeps the detailed record of
every run; this table answers "what is done, what runs where, what is left".
"""
import datetime
import json
import os
import socket
import urllib.request
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTION_VERSION = '2022-06-28'
STATUSES = ('queued', 'running', 'done', 'failed', 'skipped')
# Notion database properties (name -> type) the rows are written to
NOTION_SCHEMA = {
    'Unit': 'title', 'Experiment': 'select', 'Dataset': 'select', 'Model': 'select', 'Stage': 'select',
    'Fold': 'number', 'Status': 'select', 'Machine': 'rich_text', 'Progress': 'rich_text',
    'Val bal. acc': 'number', 'Test acc': 'number', 'Test bal. acc': 'number', 'Test AUC': 'number',
    'Git commit': 'rich_text', 'Started': 'date', 'Updated': 'date', 'Note': 'rich_text',
}
METRIC_PROPS = {'val_balanced_acc': 'Val bal. acc', 'test_acc': 'Test acc',
                'test_balanced_acc': 'Test bal. acc', 'test_auc': 'Test AUC'}

_warned = set()


def machine():
    name = socket.gethostname()
    try:
        import torch
        if torch.cuda.is_available():
            name += f' ({torch.cuda.get_device_name(0)})'
    except Exception:
        pass
    return name


def unit_key(experiment, dataset, stage, model=None, fold=None):
    return '/'.join(str(p) for p in (experiment, dataset, stage, model, None if fold is None else f'fold{fold}')
                    if p is not None)


def _warn(kind, e):
    if kind not in _warned:
        warnings.warn(f'progress reporting ({kind}) failed, the run continues: {e!r}')
        _warned.add(kind)


def report(experiment, dataset, stage, status, model=None, fold=None, metrics=None, progress=None, note=None,
           git_commit=None, started=None):
    """Record one status change of a unit."""
    assert status in STATUSES, status
    now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    rec = {'time': now, 'unit': unit_key(experiment, dataset, stage, model, fold), 'experiment': experiment,
           'dataset': dataset, 'stage': stage, 'model': model, 'fold': fold, 'status': status,
           'machine': machine(), 'progress': progress, 'note': note, 'git_commit': git_commit,
           'started': started, 'metrics': {k: v for k, v in (metrics or {}).items() if v is not None}}
    try:
        d = os.path.join(REPO, 'logs', 'progress')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f'{socket.gethostname()}.jsonl'), 'a') as f:
            f.write(json.dumps(rec) + '\n')
    except Exception as e:
        _warn('local', e)
    if os.environ.get('NOTION_TOKEN') and os.environ.get('NOTION_PROGRESS_DB'):
        try:
            _notion_upsert(rec)
        except Exception as e:
            _warn('notion', e)
    return rec


# ---------------------------------------------------------------------------
# Notion
# ---------------------------------------------------------------------------

def _notion(method, path, body=None):
    req = urllib.request.Request(f'https://api.notion.com/v1/{path}', method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={'Authorization': f"Bearer {os.environ['NOTION_TOKEN']}",
                                          'Notion-Version': NOTION_VERSION, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _text(s):
    return {'rich_text': [{'text': {'content': str(s)[:2000]}}]} if s else {'rich_text': []}


def notion_properties(rec):
    p = {'Unit': {'title': [{'text': {'content': rec['unit']}}]},
         'Experiment': {'select': {'name': rec['experiment']}},
         'Dataset': {'select': {'name': rec['dataset']}},
         'Stage': {'select': {'name': rec['stage']}},
         'Status': {'select': {'name': rec['status']}},
         'Machine': _text(rec['machine']), 'Progress': _text(rec['progress']), 'Note': _text(rec['note']),
         'Git commit': _text((rec['git_commit'] or '')[:12]),
         'Updated': {'date': {'start': rec['time']}}}
    if rec['model']:
        p['Model'] = {'select': {'name': rec['model']}}
    if rec['fold'] is not None:
        p['Fold'] = {'number': rec['fold']}
    if rec['started']:
        p['Started'] = {'date': {'start': rec['started']}}
    for k, prop in METRIC_PROPS.items():
        if k in rec['metrics']:
            p[prop] = {'number': round(float(rec['metrics'][k]), 4)}
    return p


def _notion_upsert(rec):
    db = os.environ['NOTION_PROGRESS_DB']
    found = _notion('POST', f'databases/{db}/query',
                    {'filter': {'property': 'Unit', 'title': {'equals': rec['unit']}}, 'page_size': 1})['results']
    props = notion_properties(rec)
    if found:
        _notion('PATCH', f"pages/{found[0]['id']}", {'properties': props})
    else:
        _notion('POST', 'pages', {'parent': {'database_id': db}, 'properties': props})


def latest(paths=None):
    """Latest record per unit over the local progress logs (all hosts copied into logs/progress/)."""
    import glob
    last = {}
    for p in paths or sorted(glob.glob(os.path.join(REPO, 'logs', 'progress', '*.jsonl'))):
        for line in open(p):
            rec = json.loads(line)
            if rec['unit'] not in last or rec['time'] >= last[rec['unit']]['time']:
                last[rec['unit']] = rec
    return last
