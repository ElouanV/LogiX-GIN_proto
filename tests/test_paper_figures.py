import json
import os
import tempfile
import unittest
import warnings

from utils.experiment import final_record
from utils.paper_figures import Figures, load_config


def write_fold(root, exp, ds, model, fold, bacc, interp=True):
    d = os.path.join(root, exp, ds, model, f'fold{fold}')
    os.makedirs(d)
    json.dump({'model': model, 'fold': fold, 'run_dir': f'run{fold}',
               'metrics': {'test_acc': bacc, 'test_balanced_acc': bacc}}, open(os.path.join(d, 'final.json'), 'w'))
    if interp:
        json.dump({'metrics': {'test_interp_logic_acc': bacc - 0.1}}, open(os.path.join(d, 'interp.json'), 'w'))
    return d


class TestPaperFigures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.cfg = {'root': self.root, 'split': 'test', 'fold_rule': 'median_balanced_acc', 'figures': {},
                    'models': {'a': {'label': 'A', 'model': 'm', 'experiment': 'e1',
                                     'by_dataset': {'D2': 'e2'}}}}
        for fold, bacc in enumerate([0.70, 0.90, 0.80, 0.60]):
            write_fold(self.root, 'e1', 'D1', 'm', fold, bacc)
        write_fold(self.root, 'e2', 'D2', 'm', 0, 0.5, interp=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_final_record_merges_interp(self):
        r = final_record(os.path.join(self.root, 'e1', 'D1', 'm', 'fold0'))
        self.assertEqual((r['fold'], r['run_dir']), (0, 'run0'))
        self.assertAlmostEqual(r['test_interp_logic_acc'], 0.6)
        self.assertNotIn('test_interp_logic_acc', final_record(os.path.join(self.root, 'e2', 'D2', 'm', 'fold0')))

    def test_sources_follow_by_dataset(self):
        f = Figures(self.cfg)
        self.assertEqual(f.source('a', 'D1'), ('e1', 'm'))
        self.assertEqual(f.source('a', 'D2'), ('e2', 'm'))
        self.assertEqual(len(f.folds('a', 'D1')), 4)
        self.assertEqual(len(f.folds('a', 'D2')), 1)

    def test_missing_source_is_empty(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            self.assertTrue(Figures(self.cfg).folds('a', 'D3').empty)

    def test_representative_is_lower_median(self):
        # balanced accuracies 0.60 0.70 0.80 0.90 -> lower median 0.70 = fold 0, not the best fold
        self.assertEqual(int(Figures(self.cfg).representative('a', 'D1')['fold']), 0)

    def test_unknown_figure(self):
        with self.assertRaises(ValueError):
            Figures(self.cfg).draw('nope')

    def test_repository_config_is_complete(self):
        cfg = load_config()
        for name in Figures.NAMES:
            self.assertIn(name, cfg['figures'])
        keys = set(cfg['models'])
        for spec in cfg['figures'].values():
            used = spec.get('models', []) + spec.get('finetunes', []) + [spec.get('model'), spec.get('start')]
            self.assertTrue({k for k in used if k} <= keys)


if __name__ == '__main__':
    unittest.main()
