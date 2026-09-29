"""Correctness of min_covers.py against brute force, the model's semantics, and the legacy code.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_min_covers.py -v

Every test is deterministic (fixed seeds). The oracle is exhaustive enumeration of
the 2^n subsets, so instances stay at n <= 14.
"""
import itertools
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from min_covers import (count_min_covers, extract_layer, iter_min_covers,  # noqa: E402
                        literal_true_intervals, min_covers, supported_min_covers,
                        unique_rows)


def brute_min_covers(w, S, max_len=None, eps=0.0):
    """Oracle: every subset of the weights > eps that reaches S and loses it when any item goes."""
    items = [i for i in range(len(w)) if w[i] > eps]
    out = set()
    for k in range(len(items) + 1):
        if max_len is not None and k > max_len:
            break
        for c in itertools.combinations(items, k):
            s = sum(w[i] for i in c)
            if s >= S and all(s - w[i] < S for i in c):
                out.add(frozenset(c))
    return out


def as_sets(covers):
    return [frozenset(c) for c in covers]


def random_instances(seed, count, n_max=12):
    """Weights from several regimes: uniform, heavy-tailed, integer with ties, mixed sign."""
    rng = np.random.default_rng(seed)
    for t in range(count):
        n = int(rng.integers(1, n_max + 1))
        kind = t % 4
        if kind == 0:
            w = rng.random(n)
        elif kind == 1:
            w = rng.exponential(1.0, n) ** 3                 # a few heavy, many tiny, like trained units
        elif kind == 2:
            w = rng.integers(0, 5, n).astype(float)          # ties and zeros
        else:
            w = rng.normal(0.3, 0.5, n)                      # some negative
        pos = w[w > 0].sum()
        S = float(rng.uniform(0.05, 1.1) * pos) if pos > 0 else float(rng.uniform(0.1, 1))
        if kind == 2:
            S = float(math.ceil(S)) or 1.0                   # integer threshold -> exact ties
        yield w, S


class TestEnumeration(unittest.TestCase):

    def test_matches_brute_force(self):
        for w, S in random_instances(0, 400):
            got = as_sets(iter_min_covers(w, S))
            self.assertEqual(set(got), brute_min_covers(w, S), msg=f'w={w} S={S}')

    def test_no_duplicates(self):
        for w, S in random_instances(1, 200):
            got = as_sets(iter_min_covers(w, S))
            self.assertEqual(len(got), len(set(got)))

    def test_each_output_is_a_minimal_cover(self):
        for w, S in random_instances(2, 200):
            for c in iter_min_covers(w, S):
                s = sum(w[i] for i in c)
                self.assertGreaterEqual(s, S)
                for i in c:
                    self.assertGreater(w[i], 0)
                    self.assertLess(s - w[i], S)

    def test_output_is_heaviest_first(self):
        for w, S in random_instances(3, 100):
            for c in iter_min_covers(w, S):
                ws = [w[i] for i in c]
                self.assertEqual(ws, sorted(ws, reverse=True))

    def test_dnf_is_the_threshold_function(self):
        """The disjunction of the covers fires on exactly the inputs where sum(w*x) >= S."""
        rng = np.random.default_rng(4)
        for _ in range(60):
            n = int(rng.integers(1, 11))
            w = rng.exponential(1.0, n)
            S = float(rng.uniform(0.1, 1.0) * w.sum())
            covers = as_sets(iter_min_covers(w, S))
            for x in itertools.product([0, 1], repeat=n):
                true = frozenset(i for i in range(n) if x[i])
                fires = sum(w[i] for i in true) >= S
                self.assertEqual(fires, any(c <= true for c in covers))

    def test_max_len_keeps_exactly_the_short_covers(self):
        rng = np.random.default_rng(5)
        for w, S in random_instances(5, 200):
            L = int(rng.integers(0, len(w) + 1))
            got = as_sets(iter_min_covers(w, S, max_len=L))
            self.assertEqual(set(got), {c for c in brute_min_covers(w, S) if len(c) <= L})

    def test_eps_is_a_sound_subset(self):
        """eps only removes covers, and every removed cover crosses S by less than eps."""
        for w, S in random_instances(6, 200):
            eps = float(np.quantile(np.abs(w), 0.4))
            full = brute_min_covers(w, S)
            kept = set(as_sets(iter_min_covers(w, S, eps=eps)))
            self.assertEqual(kept, {c for c in full if all(w[i] > eps for i in c)})
            for c in full - kept:
                self.assertLess(sum(w[i] for i in c) - S, eps + 1e-12)

    def test_edge_cases(self):
        self.assertEqual(min_covers([1.0, 2.0], 0.0), [()])            # always fires
        self.assertEqual(min_covers([1.0, 2.0], -3.0), [()])
        self.assertEqual(min_covers([-1.0, 0.0], 0.5), [])            # nothing positive
        self.assertEqual(min_covers([], 1.0), [])
        self.assertEqual(min_covers([0.2, 0.3], 1.0), [])             # never fires
        self.assertEqual(min_covers([0.2, 0.3], 0.5), [(1, 0)])       # exact equality counts
        self.assertEqual(set(as_sets(min_covers([1.0, 1.0, 1.0], 2.0))),
                         {frozenset(p) for p in itertools.combinations(range(3), 2)})
        self.assertEqual(set(as_sets(min_covers([3.0, 2.0, 1.0], 3.0))),
                         {frozenset([0]), frozenset([1, 2])})
        self.assertEqual(min_covers([1.0, 1.0, 1.0], 2.0, max_len=1), [])

    def test_max_out_truncates(self):
        w = np.ones(10)
        self.assertEqual(len(min_covers(w, 5.0, max_out=7)), 7)

    def test_large_n_runs_without_recursion(self):
        """Iterative search: a cover of 3000 items does not hit Python's recursion limit."""
        w = np.ones(3000)
        self.assertEqual(len(min_covers(w, 3000.0)), 1)


class TestCount(unittest.TestCase):

    def test_exact_on_integer_weights(self):
        for w, S in random_instances(7, 300):
            w = np.floor(np.abs(w) * 4)                    # integer weights, many ties and zeros
            S = float(max(1, math.ceil(S * 4)))
            got = count_min_covers(w, S, delta=1)
            ref = np.bincount([len(c) for c in brute_min_covers(w, S)], minlength=len(got))
            self.assertEqual(ref[len(got):].sum(), 0)
            np.testing.assert_array_equal(got, ref[:len(got)])

    def test_exact_with_max_len(self):
        rng = np.random.default_rng(8)
        for _ in range(100):
            n = int(rng.integers(1, 12))
            w = rng.integers(1, 6, n).astype(float)
            S = float(rng.integers(1, int(w.sum()) + 2))
            L = int(rng.integers(1, n + 1))
            got = count_min_covers(w, S, max_len=L, delta=1)
            ref = np.bincount([len(c) for c in brute_min_covers(w, S, max_len=L)], minlength=L + 1)
            np.testing.assert_array_equal(got, ref)

    def test_estimate_on_real_valued_weights(self):
        """Gridded counts are estimates; on a fine grid they stay within a few percent."""
        rng = np.random.default_rng(9)
        for _ in range(20):
            w = rng.random(14)
            S = float(0.5 * w.sum())
            est = count_min_covers(w, S, resolution=1 << 14).sum()
            ref = len(brute_min_covers(w, S))
            self.assertLess(abs(est - ref), 0.03 * ref + 2)

    def test_trivial_thresholds(self):
        np.testing.assert_array_equal(count_min_covers([1.0, 2.0], 0.0), [1.0])
        self.assertEqual(count_min_covers([0.2, 0.3], 1.0).sum(), 0)


class TestSupported(unittest.TestCase):

    def test_matches_brute_force_with_support(self):
        rng = np.random.default_rng(10)
        for w, S in random_instances(10, 200, n_max=10):
            n = len(w)
            X = rng.random((int(rng.integers(1, 40)), n)) < rng.uniform(0.2, 0.8)
            k = int(rng.integers(1, 4))
            L = None if rng.random() < 0.5 else int(rng.integers(1, n + 1))
            got = {frozenset(c): s for c, s in supported_min_covers(w, S, X, min_support=k, max_len=L)}
            ref = {}
            for c in brute_min_covers(w, S, max_len=L):
                s = int(X[:, sorted(c)].all(1).sum()) if c else len(X)
                if s >= k:
                    ref[c] = s
            self.assertEqual(got, ref, msg=f'w={w} S={S} k={k}')

    def test_precomputed_unique_rows(self):
        """Passing the layer's distinct rows once gives the same answer as passing X."""
        rng = np.random.default_rng(17)
        X = rng.random((300, 9)) < 0.5
        rows, mult = unique_rows(X)
        self.assertEqual(mult.sum(), len(X))
        self.assertEqual(len({r.tobytes() for r in rows}), len(rows))
        for _ in range(20):
            w = rng.exponential(1.0, 9)
            S = float(0.4 * w.sum())
            self.assertEqual(supported_min_covers(w, S, X, 2, eps=0.05),
                             supported_min_covers(w, S, rows, 2, eps=0.05, counts=mult))

    def test_sorted_by_support(self):
        rng = np.random.default_rng(11)
        w = rng.random(10)
        X = rng.random((200, 10)) < 0.6
        sup = [s for _, s in supported_min_covers(w, 0.4 * w.sum(), X)]
        self.assertEqual(sup, sorted(sup, reverse=True))


class TestExtractLayer(unittest.TestCase):

    def test_parallel_equals_serial(self):
        rng = np.random.default_rng(12)
        W = rng.exponential(1.0, (12, 14)) ** 2
        S = 0.4 * W.sum(1)
        a = extract_layer(W, S, n_jobs=1)
        b = extract_layer(W, S, n_jobs=3)
        for ra, rb in zip(a, b):
            self.assertEqual(ra['unit'], rb['unit'])
            self.assertEqual(set(as_sets(ra['covers'])), set(as_sets(rb['covers'])))

    def test_budget_picks_longest_length_that_fits(self):
        rng = np.random.default_rng(13)
        w = rng.integers(1, 6, 12).astype(float)
        S = 16.0                    # power of two: integer weights sit on the default grid S/2048
        ref = brute_min_covers(w, S)
        by_len = np.bincount([len(c) for c in ref])
        budget = int(by_len[:4].sum())                 # room for lengths <= 3, not for 4
        if by_len[4:].sum() == 0:
            self.skipTest('instance has no cover longer than 3')
        r = extract_layer(w[None], [S], budget=budget)[0]
        self.assertLessEqual(len(r['covers']), budget)
        self.assertEqual(set(as_sets(r['covers'])), {c for c in ref if len(c) <= r['max_len']})


class TestLiteralIntervals(unittest.TestCase):

    def test_matches_step_for_every_tau(self):
        try:
            import torch
            from models_proto.tell import step
        except ImportError as e:
            self.skipTest(f'torch unavailable: {e}')
        rng = np.random.default_rng(14)
        u = torch.linspace(0.0, 12.0, 24001, dtype=torch.float64)
        for _ in range(30):
            w, b = float(np.exp(rng.normal(0, 1))), float(rng.normal(0, 3))
            ivs = literal_true_intervals(w, b, 0.0, 12.0)
            inside = torch.zeros_like(u, dtype=torch.bool)
            for a, c in ivs:
                inside |= (u >= a) & (u <= c)
            for tau in (0, 5, 10):
                on = step(w * u + b, tau) >= 0.5
                bad = (on != inside)
                # disagreements only within float noise of an interval edge
                edges = torch.tensor([e for iv in ivs for e in iv] or [1e9], dtype=torch.float64)
                near = (u[:, None] - edges[None]).abs().min(1).values < 1e-6
                self.assertFalse((bad & ~near).any(), msg=f'w={w} b={b} tau={tau}')


class TestAgainstLegacy(unittest.TestCase):
    """The new search returns the same rules as latent_logic.find_logic_rules."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from latent_logic import find_logic_rules
        except Exception as e:                                   # torch_geometric missing etc.
            raise unittest.SkipTest(f'latent_logic not importable: {e}')
        cls.torch, cls.legacy = torch, staticmethod(find_logic_rules)

    def test_pure_weights(self):
        torch = self.torch
        for w, S in random_instances(15, 150, n_max=11):
            w = np.abs(w)
            old = self.legacy(torch.tensor(w), torch.zeros(len(w)), torch.tensor(S),
                              activations=None, max_rule_len=float('inf'), max_rules=float('inf'))
            new = as_sets(iter_min_covers(w, S, eps=1e-5))     # legacy drops w <= 1e-5
            self.assertEqual({frozenset(c) for c in old}, set(new))

    def test_with_support(self):
        torch = self.torch
        rng = np.random.default_rng(16)
        for w, S in random_instances(16, 150, n_max=10):
            w = np.abs(w)
            X = rng.random((int(rng.integers(5, 60)), len(w))) < 0.6
            k, L = int(rng.integers(1, 4)), int(rng.integers(1, len(w) + 1))
            old = self.legacy(torch.tensor(w), torch.zeros(len(w)), torch.tensor(S),
                              activations=torch.tensor(X), max_rule_len=L,
                              max_rules=float('inf'), min_support=k)
            new = supported_min_covers(w, S, X, min_support=k, max_len=L, eps=1e-5)
            self.assertEqual({frozenset(c) for c in old}, {frozenset(c) for c, _ in new})


if __name__ == '__main__':
    unittest.main()
