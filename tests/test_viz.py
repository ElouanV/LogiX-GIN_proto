import json
import os
import tempfile
import unittest

from utils.viz import collect_interp, plot_collapse, plot_tradeoff, plot_vocab_sweep, summarize

TEACHER = 'batch_size=128|dropout=0.15|epochs=500|hidden_dim=64|l2=1e-05|lr=0.001|nogumbel=False|num_layers=3'
METRICS = {'acc': 0.8, 'auc': 0.85, 'balanced_acc': 0.8, 'logic_acc': 0.7, 'logic_fidelity': 0.75,
           'rule_backed': 0.9, 'expl_bits_mean': 20.0, 'protos_cited': 4, 'units_cited': 30,
           'gt_precision': 0.6, 'gt_chance': 0.25}


def write_run(root, ds, cfgdir, cfg, seed, sparse=None, **metrics):
    d = os.path.join(root, ds, cfgdir, TEACHER, str(seed))
    if sparse:
        d = os.path.join(d, 'sparse', sparse)
    os.makedirs(d, exist_ok=True)
    if cfg is not None:
        with open(os.path.join(root, ds, cfgdir, 'config.txt'), 'w') as f:
            f.write(cfg)
    with open(os.path.join(d, 'interp_test.json'), 'w') as f:
        json.dump({**METRICS, **metrics}, f)


class TestCollect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = self.tmp.name
        nmp = 'proto_level=node|proto_mask=True|push_every=200|mask_anneal_frac=0.8'
        for k in range(3):
            write_run(r, 'Mutagenicity', 'proto_level=node|push_every=0', None, k, expl_bits_mean=500.0)
            write_run(r, 'Mutagenicity', 'node-aaa', nmp, k)
            write_run(r, 'Mutagenicity', 'node-bbb', nmp + '|vocab_reg=0.1', k, expl_bits_mean=8.0)
            write_run(r, 'Mutagenicity', 'node-aaa', nmp, k, sparse='hoyer_reg=0.0|hoyer_fc=0.0|hard=True',
                      logic_fidelity=1.0)
            write_run(r, 'AIDS', 'proto_level=node|push_every=0', None, k, balanced_acc=0.5)
            write_run(r, 'AIDS', 'node-aaa', nmp, k)
        self.df = collect_interp(r)

    def tearDown(self):
        self.tmp.cleanup()

    def test_labels(self):
        self.assertEqual(sorted(self.df[self.df.ds == 'Mutagenicity'].method.unique()),
                         ['NMP', 'NMP + hard', 'NMP + vocab 0.1', 'node dense'])
        self.assertTrue((self.df.trunk == '64x3').all())

    def test_summary_and_plots(self):
        s = summarize(self.df[self.df.ds == 'Mutagenicity'])
        self.assertEqual(s.loc['NMP + vocab 0.1', 'expl_bits_mean_mean'], 8.0)
        self.assertEqual(s.loc['NMP', 'n'], 3)
        for fig in (plot_tradeoff(self.df), plot_vocab_sweep(self.df), plot_collapse(self.df)):
            self.assertTrue(fig.axes)


if __name__ == '__main__':
    unittest.main()
