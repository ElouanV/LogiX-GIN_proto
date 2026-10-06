import glob
import os
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np

from utils import splits
from utils.splits import N_FOLDS, split_indices


class TestSplits(unittest.TestCase):
    y = np.array([0] * 70 + [1] * 30)

    def test_random_is_upstream(self):
        """Same indices as the data.pkl the existing Mutagenicity teachers were trained on."""
        pkls = sorted(glob.glob('results/Mutagenicity/*nogumbel=False*/[0-9]/data.pkl'))
        if not pkls:
            self.skipTest('no Mutagenicity teacher run')
        from utils.utils import get_dataset
        y = get_dataset('Mutagenicity').data.y
        for p in pkls[:3]:
            seed = int(os.path.basename(os.path.dirname(p)))
            data = pickle.load(open(p, 'rb'))
            got = split_indices(y, seed, 'random')
            for name, idx in zip(('train', 'val', 'test'), got):
                self.assertEqual(list(idx), list(data[f'{name}_indices']), (p, name))

    def test_kfold_partition(self):
        tests = []
        for k in range(N_FOLDS):
            tr, va, te = split_indices(self.y, k, 'kfold')
            self.assertEqual(len(set(tr) | set(va) | set(te)), len(self.y))
            self.assertFalse(set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
            self.assertAlmostEqual(self.y[te].mean(), 0.3, delta=0.05)   # stratified
            tests += te
        self.assertEqual(sorted(tests), list(range(len(self.y))))        # each graph tested once

    def test_kfold_deterministic_and_bounded(self):
        self.assertEqual(split_indices(self.y, 4, 'kfold'), split_indices(self.y, 4, 'kfold'))
        with self.assertRaises(ValueError):
            split_indices(self.y, N_FOLDS, 'kfold')


    def test_saved_file_is_the_reference(self):
        """split_indices(dataset=...) writes the file once, then reads it; other labels refuse it."""
        with tempfile.TemporaryDirectory() as d, mock.patch.object(splits, 'SPLIT_DIR', d):
            got = split_indices(self.y, 3, 'kfold', 'Toy')
            self.assertTrue(os.path.exists(os.path.join(d, 'Toy_kfold.json')))
            self.assertEqual(got, splits.draw_split(self.y, 3, 'kfold'))
            self.assertEqual(split_indices(self.y, 3, 'kfold', 'Toy'), got)
            data = dict(zip(('train_indices', 'val_indices', 'test_indices'), got))
            self.assertEqual(len(splits.same_split('Toy', 3, data)), 64)
            with self.assertRaises(RuntimeError):
                splits.same_split('Toy', 4, data)
            with self.assertRaises(RuntimeError):
                split_indices(self.y[::-1], 3, 'kfold', 'Toy')

    def test_saved_files_match_a_fresh_draw(self):
        """Committed splits/*.json are the folds the k-fold teachers were trained on."""
        if not glob.glob(os.path.join(splits.SPLIT_DIR, 'MUTAG_kfold.json')):
            self.skipTest('no saved MUTAG split')
        from utils.utils import get_dataset
        y = get_dataset('MUTAG').data.y
        for k in range(N_FOLDS):
            f = splits.load_split('MUTAG', k, y=y)
            self.assertEqual((f['train'], f['val'], f['test']), splits.draw_split(y, k, 'kfold'))


if __name__ == '__main__':
    unittest.main()
