"""The two claims unpack_rules.py relies on beyond min_covers.py.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_unpack_rules.py -v
"""
import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from min_covers import iter_min_covers
    from unpack_rules import fmt_counts, greedy_cover, patterns, shortest_per_pattern, simplify
except Exception as e:                                   # torch_geometric missing etc.
    raise unittest.SkipTest(f'unpack_rules not importable: {e}')


class TestReductions(unittest.TestCase):

    def test_greedy_cover_keeps_all_coverable_mass(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            M = rng.random((int(rng.integers(1, 30)), 40)) < 0.2
            hits = rng.integers(0, 3, 40).astype(float)
            chosen = greedy_cover(M, hits)
            self.assertEqual(len(chosen), len(set(chosen)))
            self.assertAlmostEqual(hits[M[chosen].any(0)].sum() if chosen else 0.0, hits[M.any(0)].sum())

    def test_shortest_per_pattern_is_a_shortest_minimal_cover(self):
        rng = np.random.default_rng(3)
        for _ in range(60):
            n = int(rng.integers(1, 10))
            w = rng.exponential(1.0, n)
            T = float(rng.uniform(0.1, 0.9) * w.sum())
            P = rng.random((5, n)) < 0.7
            got = {frozenset(c) for c in shortest_per_pattern(w, T, P)}
            for p in P:
                inside = [frozenset(np.flatnonzero(p)[list(c)])
                          for c in iter_min_covers(w[np.flatnonzero(p)], T)]
                if not inside:
                    continue
                k = min(len(c) for c in inside)
                mine = [c for c in got if c <= frozenset(np.flatnonzero(p))]
                self.assertTrue(any(len(c) == k and c in inside for c in mine))

    def test_simplify_keeps_precision_within_tolerance(self):
        rng = np.random.default_rng(4)
        for tol in (0.0, 0.05):
            for _ in range(40):
                full = rng.random((60, 8)) < 0.6
                mult = rng.integers(1, 5, 60).astype(float)
                hits = np.minimum(mult, rng.integers(0, 5, 60)).astype(float)
                lits = sorted(rng.choice(8, int(rng.integers(1, 6)), replace=False).tolist())
                m = full[:, lits].all(1)
                if mult[m].sum() == 0:
                    continue
                s = simplify(lits, full, mult, hits, tol)
                self.assertTrue(set(s) <= set(lits) and s)
                ms = full[:, s].all(1)
                self.assertGreaterEqual(hits[ms].sum() / mult[ms].sum() + 1e-12,
                                        hits[m].sum() / mult[m].sum() - tol)


class TestNegatedUnit(unittest.TestCase):

    def test_dual_threshold_rules_describe_not_firing(self):
        """NOT u holds iff some minimal set of FALSE literals weighs more than W - S."""
        rng = np.random.default_rng(0)
        for _ in range(80):
            n = int(rng.integers(1, 10))
            w = rng.exponential(1.0, n) * (rng.random(n) < 0.8)          # some pruned (0) weights
            if w.sum() == 0:
                continue
            S = float(rng.uniform(0.05, 1.0) * w.sum())
            T = w.sum() - S
            T += 1e-9 * max(1.0, abs(T))                                 # strict, as in unit_rules
            dual = [frozenset(c) for c in iter_min_covers(w, T)]
            for x in itertools.product([0, 1], repeat=n):
                fires = sum(w[i] for i in range(n) if x[i]) >= S
                false = frozenset(i for i in range(n) if not x[i])
                self.assertEqual(not fires, any(c <= false for c in dual), msg=f'w={w} S={S} x={x}')


class TestFormatting(unittest.TestCase):

    def test_fmt_counts(self):
        self.assertIsNone(fmt_counts({0, 1, 2, 3}, 0, 3))                # always true: omitted
        self.assertEqual(fmt_counts(set(), 0, 3), 'never (on observed counts)')
        self.assertEqual(fmt_counts({2, 3}, 0, 3), '>= 2')
        self.assertEqual(fmt_counts({0, 1}, 0, 3), '<= 1')
        self.assertEqual(fmt_counts({2}, 0, 3), '= 2')
        self.assertEqual(fmt_counts({1, 2}, 0, 3), 'in [1, 2]')
        self.assertEqual(fmt_counts({0, 2, 3, 5}, 0, 5), 'in {0, 2-3, 5}')
        self.assertEqual(fmt_counts({0, 4, 5}, 0, 5), 'in {0, >=4}')

    def test_patterns_inverse(self):
        rng = np.random.default_rng(1)
        X = rng.random((200, 7)) < 0.4
        rows, inv = patterns(X)
        np.testing.assert_array_equal(rows[inv], X)
        self.assertEqual(len({r.tobytes() for r in rows}), len(rows))


if __name__ == '__main__':
    unittest.main()
