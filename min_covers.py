"""Minimal covers of a threshold: the exact DNF of one LogicalLayer unit.

A ``LogicalLayer`` unit fires when ``sum_i w_i * lit_i >= S`` with ``w >= 0`` and
``S = -b``. On Boolean literals that is a monotone function, and a monotone
function is exactly the disjunction of its *minimal true sets*: the sets ``C`` with
``sum(w[C]) >= S`` from which removing any element drops the sum below ``S``. This
module enumerates those sets.

Algorithm (``iter_min_covers``)
    Sort the weights in descending order and build every set by adding positions
    in increasing order, so the last position added is always the lightest one.
    With positive weights ``C`` is minimal iff ``sum(C) - min(C) < S``, so a set is
    emitted the moment it crosses ``S`` and is never extended. A candidate ``j`` is
    tried only if the best completion reachable from it still crosses ``S``; that
    bound shrinks with ``j``, so the scan breaks at the first failure. Every
    surviving node leads to an output (greedy completion crosses minimally), so
    there are no dead branches and the delay between two outputs is O(n).

    With ``max_len = L`` the bound becomes "the ``r`` heaviest remaining items",
    ``r`` being the slots left, which is still non-increasing in ``j`` - the
    no-dead-branch property survives the length cap.

Output size, not search, is the cost
    The number of minimal covers can be exponential (C(n, n/2) at worst), so
    ``count_min_covers`` computes it by length with a pseudo-polynomial DP before
    anything is enumerated, and ``extract_layer`` uses it to pick the longest
    length that fits a budget.

Small weights (``eps``)
    Dropping weights ``<= eps`` only removes covers: every set still returned is a
    minimal cover of the original problem, because minimality depends on the set
    alone. The covers that are lost all contain a weight ``<= eps`` and therefore
    cross ``S`` by a margin smaller than ``eps``.
"""
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _sorted_weights(w, eps):
    """Positions with ``w > eps``, heaviest first (stable, so ties keep index order)."""
    w = np.asarray(w, dtype=np.float64).ravel()
    keep = np.flatnonzero(w > eps)
    order = keep[np.argsort(-w[keep], kind='stable')]
    return order, w[order]


def iter_min_covers(w, S, max_len=None, eps=0.0):
    """Yield every minimal cover of ``S`` by ``w`` (with at most ``max_len`` items).

    Args:
        w:       [n] weights. Entries ``<= eps`` are ignored (see module docstring).
        S:       threshold. ``S <= 0`` yields only the empty cover.
        max_len: only covers with at most this many items.
        eps:     weights at or below this are dropped. 0 is exact.

    Yields:
        tuples of indices into ``w``, heaviest weight first.
    """
    S = float(S)
    if S <= 0.0:
        yield ()
        return
    order, ws = _sorted_weights(w, eps)
    n = len(ws)
    L = n if max_len is None else min(int(max_len), n)
    if n == 0 or L <= 0:
        return
    # P[j] = ws[:j].sum(); the best r more items from position j weigh P[j+r] - P[j]
    P = np.concatenate([[0.0], np.cumsum(ws)]).tolist()
    if P[L] < S:
        return
    ws, order = ws.tolist(), order.tolist()

    path, sums, j = [], [0.0], 0
    while True:
        cur = sums[-1]
        need = S - cur
        r = L - len(path)                       # slots left, always >= 1 here
        pushed = False
        while j < n:
            if P[min(j + r, n)] - P[j] < need:  # bound fails here and for every later j
                break
            if ws[j] >= need:                   # crosses S; ws[j] is the lightest -> minimal
                yield tuple(order[p] for p in path) + (order[j],)
                j += 1
            else:                               # r >= 2 here, otherwise the bound broke
                path.append(j)
                sums.append(cur + ws[j])
                j += 1
                pushed = True
                break
        if not pushed:
            if not path:
                return
            j = path.pop() + 1
            sums.pop()


def min_covers(w, S, max_len=None, eps=0.0, max_out=None):
    """``iter_min_covers`` as a list, stopping after ``max_out`` covers.

    ``max_out`` keeps the first covers in search order (heaviest literals first),
    which is not a meaningful sample - prefer bounding ``max_len`` with
    ``count_min_covers``.
    """
    out = []
    for c in iter_min_covers(w, S, max_len, eps):
        out.append(c)
        if max_out is not None and len(out) >= max_out:
            break
    return out


def count_min_covers(w, S, max_len=None, eps=0.0, delta=None, resolution=2048):
    """Number of minimal covers by size, without enumerating them.

    A minimal cover whose lightest item sits at sorted position ``j`` is ``{j}``
    plus a subset ``A`` of the heavier items with ``S - w_j <= sum(A) < S``. A
    subset-sum DP over the sorted items counts those ``A`` by size.

    Weights are put on a grid of step ``delta`` (default ``S / resolution``). The
    count is exact for the gridded weights, so exact whenever the weights are
    multiples of ``delta`` (``delta=1`` for integer weights) and an estimate
    otherwise. Cost: O(n * K * S/delta) with K the largest size counted.

    Returns:
        float array ``counts`` with ``counts[k]`` the number of covers of size k.
    """
    S = float(S)
    if S <= 0.0:
        return np.array([1.0])
    order, ws = _sorted_weights(w, eps)
    n = len(ws)
    K = n if max_len is None else min(int(max_len), n)
    counts = np.zeros(K + 1)
    if n == 0 or K == 0:
        return counts
    if delta is None:
        delta = S / resolution
    wq = np.rint(ws / delta).astype(np.int64)
    Sq = int(math.ceil(S / delta - 1e-9))
    f = np.zeros((K, Sq))       # f[k, s]: subsets of the items seen so far, k items, grid sum s < Sq
    f[0, 0] = 1.0
    for j in range(n):
        lo = max(0, Sq - int(wq[j]))
        counts[1:] += f[:, lo:].sum(1)
        if wq[j] < Sq and K > 1:
            f[1:, wq[j]:] += f[:-1, :Sq - wq[j]].copy()
    return counts


def unique_rows(X, counts=None):
    """Distinct rows of a bool matrix and how many rows each stands for.

    Rows are packed to bytes first, which makes this far faster than
    ``np.unique(axis=0)``. ``counts`` weights the input rows (default 1 each).
    """
    X = np.ascontiguousarray(X, dtype=bool)
    n_rows = len(X) if counts is None else int(np.sum(counts))
    if X.shape[1] == 0:
        return X[:1], np.array([n_rows] if len(X) else [], dtype=np.int64)
    packed = np.ascontiguousarray(np.packbits(X, axis=1))
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    mult = np.bincount(inv.ravel(), weights=counts, minlength=len(first))
    return X[first], np.rint(mult).astype(np.int64)


def supported_min_covers(w, S, X, min_support=1, max_len=None, eps=0.0, counts=None):
    """Minimal covers contained in at least ``min_support`` rows of ``X``.

    Because minimality depends on the set alone, a minimal cover contained in a
    row is exactly a minimal cover of the instance restricted to that row's true
    literals. So each distinct row is solved as a small independent problem and
    the results are merged: no search branch is spent on literal combinations
    that no example realises.

    Args:
        X:      [N, n] bool, the observed truth value of each literal.
        counts: if given, ``X`` already holds distinct rows and ``counts`` their
                multiplicities (``unique_rows``). Do that once per layer: the
                rows are shared by every unit of the layer.

    Returns:
        list of ``(cover, support)`` sorted by support (descending) then length.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    if counts is None:
        X, counts = unique_rows(X)
    cols = np.flatnonzero(w > eps)              # rows differing only on dropped literals merge
    pats, mult = unique_rows(np.asarray(X, dtype=bool)[:, cols], counts)
    # rows whose true literals cannot reach S hold no cover (loose tolerance: the search decides)
    reach = pats.astype(np.float64) @ w[cols] >= S - 1e-9 * max(1.0, abs(S))
    found = set()
    for p in pats[reach]:
        idx = cols[p]
        # a stable sort over ascending idx reproduces the global tie order
        for c in iter_min_covers(w[idx], S, max_len, eps):
            found.add(tuple(int(idx[i]) for i in c))
    pos = {int(c): k for k, c in enumerate(cols)}
    out = []
    for c in found:
        sup = int(mult[pats[:, [pos[i] for i in c]].all(1)].sum()) if c else int(mult.sum())
        if sup >= min_support:
            out.append((c, sup))
    out.sort(key=lambda cs: (-cs[1], len(cs[0]), cs[0]))
    return out


def _unit_job(args):
    unit, w, S, max_len, eps, budget = args
    t0 = time.perf_counter()
    counts = None
    L = max_len
    if budget is not None:
        counts = count_min_covers(w, S, max_len, eps)
        fits = np.flatnonzero(np.cumsum(counts) <= budget)
        L = int(fits[-1]) if len(fits) else 0
    covers = list(iter_min_covers(w, S, L, eps)) if (L is None or L > 0 or S <= 0) else []
    return {'unit': unit, 'covers': covers, 'max_len': L, 'counts': counts,
            'seconds': time.perf_counter() - t0}


def extract_layer(W, S, max_len=None, eps=0.0, budget=None, n_jobs=1):
    """Minimal covers of every unit of a layer.

    Args:
        W:       [out, in] non-negative weights (``LogicalLayer.weight``).
        S:       [out] thresholds (``-LogicalLayer.b``).
        budget:  if set, each unit is counted first and enumerated only up to the
                 longest length whose cumulative count fits the budget. The
                 result's ``max_len`` says where it stopped.
        n_jobs:  worker processes. Units are independent; they are submitted one
                 by one so a slow unit does not hold back a batch.

    Returns:
        list of dicts ``{unit, covers, max_len, counts, seconds}``, in unit order.
    """
    W = np.asarray(W, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64).ravel()
    jobs = [(u, W[u], float(S[u]), max_len, eps, budget) for u in range(W.shape[0])]
    if n_jobs == 1:
        return [_unit_job(j) for j in jobs]
    with ProcessPoolExecutor(n_jobs) as ex:
        return list(ex.map(_unit_job, jobs, chunksize=1))


def literal_true_intervals(w, b, lo, hi):
    """Where ``phi_in`` reads a literal as true: ``step(w*u + b) >= 0.5`` on [lo, hi].

    ``step`` is a partial Fourier series of a square wave. For every number of
    terms tau, its sign is that of ``sin``, so ``step(z) >= 0.5`` exactly when
    ``z mod 2*pi`` lies in ``[0, pi]``; the truth set does not depend on tau and
    needs no numerical scan. In ``u`` that is the union of the intervals
    ``[(2k*pi - b)/w, ((2k+1)*pi - b)/w]`` (``w > 0``), clipped to [lo, hi].
    """
    w, b, lo, hi = float(w), float(b), float(lo), float(hi)
    two_pi = 2 * math.pi
    k0 = math.floor((w * lo + b) / two_pi)
    k1 = math.floor((w * hi + b) / two_pi)
    out = []
    for k in range(k0, k1 + 1):
        a = max(lo, (two_pi * k - b) / w)
        c = min(hi, (two_pi * k + math.pi - b) / w)
        if a <= c:
            out.append((a, c))
    return out
