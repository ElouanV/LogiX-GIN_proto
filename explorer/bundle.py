"""Static export of one run for the explorer: everything the web app needs in one JSON (card,
split index, a selection of graph traces), or one self-contained HTML file that opens offline
and can be shared."""
import json
import os

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')


def pick_graphs(rows, n):
    """Up to n graphs, deterministic, spread over what is worth looking at: per true class,
    round-robin over rule errors, network/rule disagreements, predictions no rule backs and
    correct ones, in split order."""
    groups = {}
    for r in rows:
        kind = ('unfaithful' if r['rule'] != r['net'] else 'unbacked' if not r['backed']
                else 'correct' if r['rule'] == r['y'] else 'wrong')
        groups.setdefault((kind, r['y']), []).append(r['i'])
    order = [g for kind in ('wrong', 'unfaithful', 'unbacked', 'correct') for g in sorted(groups) if g[0] == kind]
    picked, j = [], 0
    while len(picked) < n and any(j < len(groups[g]) for g in order):
        for g in order:
            if j < len(groups[g]) and len(picked) < n:
                picked.append(groups[g][j])
        j += 1
    return sorted(picked)


def build_bundle(session, split='test', n_graphs=40, proto_examples=True):
    """card + index of ``split`` + traces of n_graphs of its graphs and, with
    ``proto_examples``, of each prototype's best-matching reference graph."""
    card = session.card()
    index = session.index(split)
    traces = {f'{split}:{i}': session.trace(split, i) for i in pick_graphs(index['rows'], n_graphs)}
    if proto_examples:
        for p in card['protos']:
            for top in p['top']:
                if top:
                    key = f"{card['ref_split']}:{top[0]['graph']}"
                    if key not in traces:
                        traces[key] = session.trace(card['ref_split'], top[0]['graph'])
    return {'meta': session.meta(), 'card': card, 'index': {split: index}, 'traces': traces}


def write_json(bundle, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(bundle, f, separators=(',', ':'))


def _read(path):
    with open(path) as f:
        return f.read()


def write_html(bundle, path):
    """The web app with its stylesheet, script and the bundle inlined."""
    page, css, js = (_read(os.path.join(WEB, n)) for n in ('index.html', 'style.css', 'app.js'))
    data = json.dumps(bundle, separators=(',', ':')).replace('</', '<\\/')
    page = page.replace('<link rel="stylesheet" href="style.css">', f'<style>\n{css}\n</style>')
    page = page.replace('<script src="app.js"></script>',
                        f'<script>window.EXPLORER_BUNDLE = {data};</script>\n<script>\n{js}\n</script>')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write(page)
