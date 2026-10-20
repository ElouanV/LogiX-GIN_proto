"""python -m explorer serve | export (see explorer/__init__.py)."""
import argparse
import hashlib
import os

from explorer.bundle import build_bundle, write_html, write_json
from explorer.session import RunSession, discover, run_label

EXPORTS = 'explorer_exports'


def main():
    ap = argparse.ArgumentParser(prog='python -m explorer', description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--ckpt', default='best.pt')
    common.add_argument('--device', default='cpu', help='cpu by default: inference is light and the GPU is shared')
    common.add_argument('--ref_split', default='train', choices=['train', 'val', 'test'],
                        help='split the observed ranges, firing rates and prototype matches are measured on')
    common.add_argument('--ref_max', type=int, default=None, help='cap on its graphs (default: all)')

    sv = sub.add_parser('serve', parents=[common], help='run the web app on a local server')
    sv.add_argument('--run', nargs='*', default=[], help='run directories (seed dirs or fine-tunes)')
    sv.add_argument('--root', nargs='*', default=[], help='directories searched for runs')
    sv.add_argument('--host', default='127.0.0.1')
    sv.add_argument('--port', type=int, default=8765)

    ex = sub.add_parser('export', parents=[common], help='one run as a static JSON bundle or HTML file')
    ex.add_argument('run_path')
    ex.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ex.add_argument('--graphs', type=int, default=40, help='graphs of the split traced in the export')
    ex.add_argument('--no_proto_examples', action='store_true',
                    help="skip the traces of each prototype's best-matching reference graph")
    ex.add_argument('--json', default=None, help='write the bundle as JSON here (the app opens it)')
    ex.add_argument('--html', default=None, help=f'write a self-contained page here (default: {EXPORTS}/...)')
    a = ap.parse_args()
    kw = dict(ckpt=a.ckpt, device=a.device, ref_split=a.ref_split, ref_max=a.ref_max)

    if a.cmd == 'serve':
        runs = [r for root in a.root for r in discover(root)]
        for p in a.run:
            runs.append({'path': p, 'label': run_label(p, os.path.dirname(os.path.normpath(p))) or p,
                         'dataset': None, 'level': None, 'finetunes': []})
        if not runs:
            ap.error('no run: give --run <dir> or --root <dir>')
        from explorer.server import serve
        serve(runs, a.host, a.port, **kw)
        return

    session = RunSession(a.run_path, **kw)
    bundle = build_bundle(session, a.split, a.graphs, not a.no_proto_examples)
    if a.json:
        write_json(bundle, a.json)
        print(f'written: {a.json}')
    if a.html or not a.json:
        tag = hashlib.sha1(os.path.abspath(a.run_path).encode()).hexdigest()[:8]
        out = a.html or os.path.join(EXPORTS, f'{session.run.dataset}-{tag}-{a.split}.html')
        write_html(bundle, out)
        print(f'written: {out} ({os.path.getsize(out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
