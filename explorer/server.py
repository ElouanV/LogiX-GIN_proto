"""Local server of the explorer: the web app plus a small JSON API over the runs it was given.

Only runs listed at start-up can be opened (a checkpoint is a pickle: loading an arbitrary
path from a request would run arbitrary code). It binds to 127.0.0.1 by default.

    GET /api/runs                         [{id, label, dataset, level, finetunes}]
    GET /api/open?run=<id>                {meta, card}           (first call reads the model)
    GET /api/index?run=<id>&split=test    graph rows + split summary
    GET /api/trace?run=<id>&split=test&i=3
"""
import json
import mimetypes
import os
import threading
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from explorer.bundle import WEB
from explorer.session import RunSession

MAX_OPEN = 4             # runs kept in memory


class Explorer:
    def __init__(self, runs, **session_kw):
        self.runs = runs
        self.session_kw = session_kw
        self.sessions = OrderedDict()
        self.lock = threading.Lock()

    def session(self, rid):
        rid = int(rid)
        if not 0 <= rid < len(self.runs):
            raise KeyError(f'no run #{rid}')
        with self.lock:                  # one model read at a time; later calls hit the cache
            if rid not in self.sessions:
                self.sessions[rid] = RunSession(self.runs[rid]['path'], **self.session_kw)
                while len(self.sessions) > MAX_OPEN:
                    self.sessions.popitem(last=False)
            self.sessions.move_to_end(rid)
            s = self.sessions[rid]
            if s._card is None:
                s.card()
            return s

    def api(self, route, q):
        if route == 'runs':
            return [{'id': i, **{k: r[k] for k in ('label', 'dataset', 'level', 'finetunes')}}
                    for i, r in enumerate(self.runs)]
        s = self.session(q['run'])
        if route == 'open':
            return {'meta': s.meta(), 'card': s.card()}
        if route == 'index':
            return s.index(q.get('split', 'test'))
        if route == 'trace':
            return s.trace(q.get('split', 'test'), int(q['i']))
        raise KeyError(f'unknown route {route}')


def handler(explorer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path.startswith('/api/'):
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                try:
                    body = json.dumps(explorer.api(url.path[5:], q), separators=(',', ':')).encode()
                    self.send(200, body, 'application/json')
                except (KeyError, IndexError, ValueError) as e:
                    self.send(400, json.dumps({'error': str(e)}).encode(), 'application/json')
                except Exception as e:  # noqa: BLE001  (shown in the app, full trace on the console)
                    traceback.print_exc()
                    self.send(500, json.dumps({'error': f'{type(e).__name__}: {e}'}).encode(), 'application/json')
                return
            name = 'index.html' if url.path in ('/', '') else url.path.lstrip('/')
            path = os.path.join(WEB, os.path.basename(name))
            if not os.path.isfile(path):
                self.send(404, b'not found', 'text/plain')
                return
            ctype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
            with open(path, 'rb') as f:
                self.send(200, f.read(), ctype)
    return Handler


def serve(runs, host='127.0.0.1', port=8765, **session_kw):
    server = ThreadingHTTPServer((host, port), handler(Explorer(runs, **session_kw)))
    print(f'LogiX-GIN explorer: http://{host}:{port}/  ({len(runs)} run{"s" * (len(runs) != 1)}; Ctrl-C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
