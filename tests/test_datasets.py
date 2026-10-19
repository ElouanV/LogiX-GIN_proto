import os
import pickle
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets import REGIONS, MolOneHot, registry, spec  # noqa: E402


class TestRegistry(unittest.TestCase):
    def test_every_entry_is_well_formed(self):
        for name, s in registry().items():
            self.assertIn(s.loader, ('tu', 'moleculenet', 'synthetic', 'bamultishapes', 'spmotif', 'planetoid',
                                     'ogb_mag'), name)
            if s.ground_truth:
                self.assertIn(s.ground_truth.region, REGIONS, name)
                self.assertTrue(s.ground_truth.classes, name)

    def test_bbbp_names_follow_the_transform(self):
        self.assertEqual(list(spec('BBBP').features), MolOneHot.FEATURES)

    def test_old_bbbp_pickles_still_load(self):
        # data.pkl files written before fc1d730 reference utils.utils.MolOneHot
        self.assertIsInstance(pickle.loads(b'cutils.utils\nMolOneHot\n)\x81.'), MolOneHot)

    def test_unlisted_dataset_is_a_tudataset_with_generic_names(self):
        s = spec('NCI1')
        self.assertEqual((s.loader, s.use_node_attr, s.ground_truth), ('tu', True, None))
        self.assertEqual(s.feature_names(2), ['f0', 'f1'])
        self.assertEqual(s.class_name(1), 'class 1')

    def test_mutagenicity_names(self):
        s = spec('Mutagenicity')
        self.assertEqual(s.class_name(0), 'mutagen')
        self.assertEqual(s.feature_name(4), 'N')
        self.assertEqual(s.feature_name(20), 'f20')
        self.assertEqual(s.ground_truth.classes, (0,))


if __name__ == '__main__':
    unittest.main()
