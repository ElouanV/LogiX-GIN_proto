"""explorer/: the symbolic reading it displays is the one interp_metrics.py scores, and the
facts the web app relies on (counts, witnesses, reasons, constant literals) hold exactly.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_explorer.py -v
"""
import json
import os
import pickle
import sys
import tempfile
import threading
import unittest
import urllib.request

import numpy as np
import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explorer.bundle import build_bundle, pick_graphs, write_html  # noqa: E402
from explorer.reader import SymbolicReader, min_cardinality  # noqa: E402
from explorer.server import Explorer, handler  # noqa: E402
from explorer.session import RunSession, discover  # noqa: E402
from interp_metrics import shortest_explanation, symbolic_head, trunk_states  # noqa: E402
from min_covers import literal_true_intervals  # noqa: E402
from models.model import GINTELL  # noqa: E402
from models_proto.model_proto import MODELS  # noqa: E402

F = 14          # Mutagenicity's feature count, so the registry names the atoms


def graphs(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        k = int(torch.randint(4, 10, (1,), generator=g))
        x = torch.zeros(k, F)
        x[torch.arange(k), torch.randint(0, 5, (k,), generator=g)] = 1
        pairs = [(j, int(torch.randint(0, j, (1,), generator=g))) for j in range(1, k)]
        e = torch.tensor(pairs).t()
        out.append(Data(x=x, edge_index=torch.cat([e, e.flip(0)], 1), y=torch.tensor([i % 2])))
    return out


def models():
    torch.manual_seed(0)
    node = MODELS['node'](F, 2, num_layers=2, hidden_dim=8, num_prototypes=5, proto_mask=True)
    with torch.no_grad():
        node.proto.mask_logits[0] = -3.0          # P0 cares about no bit: its literals are constants
        node.proto.mask_logits[1, ::3] = -3.0
    return {'node': node,
            'node_sum': MODELS['node'](F, 2, num_layers=2, hidden_dim=8, num_prototypes=4, pool_ops=('mean', 'max', 'sum')),
            'graph_sum': MODELS['graph'](F, 2, num_layers=2, hidden_dim=8, num_prototypes=4, pool_ops=('mean', 'max', 'sum')),
            'both': MODELS['both'](F, 2, num_layers=2, hidden_dim=8, num_prototypes=3, num_graph_prototypes=2),
            'base': GINTELL(F, 2, num_layers=2, hidden_dim=8)}


class TestTrace(unittest.TestCase):
    def setUp(self):
        self.data = graphs()

    def test_trace_is_the_symbolic_model(self):
        for name, model in models().items():
            with self.subTest(model=name):
                rd = SymbolicReader(model.eval(), 'Mutagenicity')
                for d in self.data[:5]:
                    t = rd.trace(d, 'test', 0)
                    json.dumps(t, allow_nan=False)
                    xs = trunk_states(model, d.x, d.edge_index)
                    for l, lay in enumerate(t['layers']):
                        ll = model.convs[l].nn[0]
                        out = np.array([[c == '1' for c in r] for r in lay['out']])
                        np.testing.assert_array_equal(out, xs[l].numpy() > 0.5)
                        # a unit is 1 iff its margin is >= 0 (margins are rounded to 1e-3)
                        m = np.array(lay['margin'])
                        clear = np.abs(m) > 1e-3
                        np.testing.assert_array_equal(out[clear], (m >= 0)[clear])
                        # a literal is true iff its count lies in phi's closed-form truth set
                        w, b = ll.phi_in.w.detach().numpy(), ll.phi_in.b.detach().numpy()
                        lit = np.array([[c == '1' for c in r] for r in lay['lit']])
                        for v in range(d.num_nodes):
                            for i in range(lit.shape[1]):
                                c = lay['count'][v][i]
                                ivs = literal_true_intervals(w[i], b[i], c, c)
                                self.assertEqual(lit[v, i], bool(ivs), (name, l, v, i))
                    batch = Batch.from_data_list([d])
                    margin = symbolic_head(model, xs, batch.batch)[0][0].numpy()
                    np.testing.assert_allclose(t['margin'], margin, atol=1e-3)
                    self.assertEqual(t['rule'], int(margin.argmax()))

    def test_counts_are_closed_neighbourhood_counts(self):
        """The app rings the nodes a literal counts: subject holds at w for w in N[v]."""
        rd = SymbolicReader(models()['node'].eval(), 'Mutagenicity')
        for d in self.data[:4]:
            t = rd.trace(d)
            adj = [{v} for v in range(t['n'])]
            for a, b in t['edges']:
                adj[a].add(b)
                adj[b].add(a)
            for l, lay in enumerate(t['layers']):
                n = rd.n_feat if l == 0 else rd.h
                for v in range(t['n']):
                    for i in range(2 * n):
                        def holds(w):
                            h = t['x'][w][i % n] == '1' if l == 0 else t['layers'][l - 1]['out'][w][i % n] == '1'
                            return h != (i >= n)
                        self.assertEqual(lay['count'][v][i], sum(holds(w) for w in adj[v]))

    def test_unit_reasons_are_shortest_sufficient(self):
        model = models()['node'].eval()
        rd = SymbolicReader(model, 'Mutagenicity')
        for d in self.data[:4]:
            t = rd.trace(d)
            for l, lay in enumerate(t['layers']):
                ll = model.convs[l].nn[0]
                W, S = ll.weight.detach().numpy(), (-ll.b).detach().numpy()
                for v in range(t['n']):
                    lit = np.array([c == '1' for c in lay['lit'][v]])
                    for u in range(rd.h):
                        why = lay['why'][v][u]
                        if lay['out'][v][u] == '0':
                            self.assertIsNone(why)
                            continue
                        self.assertTrue(lit[why].all())
                        self.assertGreaterEqual(W[u][why].sum(), S[u] - 1e-5)
                        self.assertEqual(len(why), len(shortest_explanation(W[u], S[u], lit)))


class TestConstants(unittest.TestCase):
    def test_constant_literals_never_change(self):
        """A literal flagged constant has that value for every readout value it can read
        (s on a fine grid of [0, 1], every possible mismatch count for a masked max)."""
        for name, model in models().items():
            rd = SymbolicReader(model.eval(), 'Mutagenicity')
            phi = model.fc.phi_in
            for i, const in enumerate(rd.head_const):
                if const is None:
                    continue
                c, pos = rd.head_literals()[i]
                col = rd.readout_layout()[c]
                if 'k' in col and not bool((rd.proto_layers[col['proto_layer']].care[col['k']] > 0.5).any()):
                    grid = torch.ones(1)                      # similarity is 1 on every input
                elif rd._discrete_max(col):
                    p = rd.proto_layers[col['proto_layer']]
                    d = int((p.care[col['k']] > 0.5).sum())
                    grid = torch.exp(-torch.arange(d + 1.) / p.mask_temp)
                else:
                    grid = torch.linspace(0, 1, 2001)
                inp = grid if pos else 1 - grid
                with torch.no_grad():
                    z = inp[:, None].expand(-1, phi.w.shape[0])
                    truth = (phi(z)[:, i] >= 0.5).numpy()
                with self.subTest(model=name, literal=i):
                    self.assertTrue((truth == const).all())

    def test_no_care_prototype_is_constant(self):
        rd = SymbolicReader(models()['node'].eval(), 'Mutagenicity')
        cols = rd.readout_layout()
        p0 = [i for i, (c, _) in enumerate(rd.head_literals()) if cols[c]['k'] == 0]
        self.assertTrue(p0)
        self.assertTrue(all(rd.head_const[i] is not None for i in p0))

    def test_evidence_is_the_fewest_varying_literals(self):
        rd = SymbolicReader(models()['node'].eval(), 'Mutagenicity')
        rng = np.random.default_rng(0)
        const_true = np.array([v is True for v in rd.head_const])
        varying = np.array([v is None for v in rd.head_const])
        for _ in range(100):
            lits = rng.random(len(rd.head_const)) < 0.5
            lits[const_true] = True
            lits[np.array([v is False for v in rd.head_const])] = False
            for c in range(rd.n_classes):
                ev = rd.evidence(c, lits)
                W, S = rd.W_head[c], rd.S_head[c]
                base = W[const_true].sum()
                if ev is None:
                    self.assertLess(base + W[lits & varying].sum(), S)
                    continue
                self.assertTrue((lits & varying)[ev].all() if ev else True)
                self.assertGreaterEqual(base + W[ev].sum(), S - 1e-6)
                # no shorter set of true varying literals reaches S (heaviest first is the best)
                k = len(ev)
                if k:
                    best = np.sort(W[lits & varying])[::-1][:k - 1].sum()
                    self.assertLess(base + best, S)


class TestCard(unittest.TestCase):
    def test_card_is_plain_json(self):
        for name, model in models().items():
            with self.subTest(model=name):
                rd = SymbolicReader(model.eval(), 'Mutagenicity')
                card = rd.card(graphs(8, seed=1))
                json.dumps(card, allow_nan=False)
                self.assertEqual(len(card['conv']), 2)
                self.assertEqual(len(card['head']['literals']), len(rd.head_literals()))
                for conv in card['conv']:
                    for u in range(rd.h):
                        self.assertEqual(conv['min_card'][u], min_cardinality(conv['W'][u], conv['S'][u]))

    def test_agreement_distributions(self):
        """Graphs whose closest node has 0 mismatches are the exact matches; every node and
        every graph is counted once."""
        data = graphs(10, seed=2)
        rd = SymbolicReader(models()['node'].eval(), 'Mutagenicity')
        card = rd.card(data)
        n_nodes = sum(d.num_nodes for d in data)
        for p in card['protos']:
            for k, dist in enumerate(p['dist']):
                closest = np.array(dist['closest'])
                self.assertEqual(closest.sum(), len(data))
                self.assertEqual(sum(dist['nodes']), n_nodes)
                self.assertEqual(list(closest[:, 0]), p['exact'][k]['class_mix'])
                self.assertEqual(np.array(dist['mean']).sum(), len(data))

    def test_min_cardinality(self):
        self.assertEqual(min_cardinality([3, 1, 1], 4), 2)
        self.assertEqual(min_cardinality([1, 1], 0), 0)
        self.assertIsNone(min_cardinality([1, 1], 5))


class TestRunsAndServer(unittest.TestCase):
    """A fake run directory (the results layout utils/runs.py reads) served over HTTP."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = os.path.join(self.tmp.name, 'results_proto', 'Mutagenicity', 'proto_level=node', 'tcfg', '0')
        os.makedirs(self.run)
        data = graphs(10)
        split = {'train': data[:6], 'val': data[6:8], 'test': data[8:]}
        idx = {'train': list(range(6)), 'val': [6, 7], 'test': [8, 9]}
        with open(os.path.join(self.run, 'data.pkl'), 'wb') as f:
            pickle.dump({**{f'{s}_indices': idx[s] for s in split}, **{f'{s}_dataset': split[s] for s in split}}, f)
        with open(os.path.join(self.run, 'args.json'), 'w') as f:
            json.dump({'proto_level': 'node', 'num_prototypes': 5}, f)
        torch.save(models()['node'], os.path.join(self.run, 'best.pt'))

    def tearDown(self):
        self.tmp.cleanup()

    def test_discover_and_bundle(self):
        runs = discover(os.path.join(self.tmp.name, 'results_proto'))
        self.assertEqual([r['path'] for r in runs], [self.run])
        b = build_bundle(RunSession(self.run), 'test', n_graphs=5)
        self.assertEqual(set(b['traces']) >= {'test:0', 'test:1'}, True)
        out = os.path.join(self.tmp.name, 'x.html')
        write_html(b, out)
        with open(out) as f:
            page = f.read()
        self.assertIn('window.EXPLORER_BUNDLE', page)
        self.assertEqual(page.count('</script>'), 2)          # data and app, nothing broken out of them

    def test_pick_graphs(self):
        rows = [{'i': i, 'y': i % 2, 'rule': (i // 2) % 2, 'net': (i // 2) % 2 if i % 5 else 1 - (i // 2) % 2,
                 'backed': i % 7 != 0} for i in range(40)]
        a, b = pick_graphs(rows, 9), pick_graphs(rows, 9)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 9)
        self.assertTrue(any(rows[i]['rule'] != rows[i]['net'] for i in a))

    def test_http_api(self):
        from http.server import ThreadingHTTPServer
        ex = Explorer(discover(os.path.join(self.tmp.name, 'results_proto')))
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(ex))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_address[1]}'
        try:
            get = lambda p: json.loads(urllib.request.urlopen(base + p).read())  # noqa: E731
            self.assertEqual(len(get('/api/runs')), 1)
            self.assertEqual(get('/api/open?run=0')['card']['level'], 'node')
            self.assertEqual(get('/api/index?run=0&split=test')['summary']['graphs'], 2)
            self.assertEqual(get('/api/trace?run=0&split=test&i=1')['i'], 1)
            self.assertIn(b'LogiX-GIN Explorer', urllib.request.urlopen(base + '/').read())
            for bad in ('/api/open?run=5', '/api/trace?run=0&split=test&i=9'):
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(base + bad)
                self.assertEqual(e.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
