"""Paired comparison of two models on the same test set.

Usage:
    python mcnemar.py A/predictions.npz B/predictions.npz [--key center_crop]

Both files come from infer.py run on the same split. Reports each model's
accuracy, the exact McNemar test on the images where exactly one model is
right, and a bootstrap 95% CI for the accuracy difference (B - A).
"""
import argparse
import math
import numpy as np


def exact_mcnemar_p(b, c):
    """Two-sided exact binomial test of b vs c discordant pairs (p = 0.5)."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
                 for i in range(k + 1)]
    m = max(log_terms)
    tail = math.exp(m) * sum(math.exp(t - m) for t in log_terms)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--key', default='center_crop')
    ap.add_argument('--boot', type=int, default=10000)
    args = ap.parse_args()

    A, B = np.load(args.a), np.load(args.b)
    assert np.array_equal(A['labels'], B['labels']), 'different test sets/order - not comparable'
    y = A['labels']
    ra, rb = A[args.key] == y, B[args.key] == y

    b = int(np.sum(ra & ~rb))   # A right, B wrong
    c = int(np.sum(~ra & rb))   # A wrong, B right
    print('images: {}'.format(len(y)))
    print('acc A: {:.2%}   acc B: {:.2%}   diff (B-A): {:+.2%}'.format(ra.mean(), rb.mean(), rb.mean() - ra.mean()))
    print('only A right: {}   only B right: {}'.format(b, c))
    print('McNemar exact p = {:.4f}  ({})'.format(
        exact_mcnemar_p(b, c), 'significant at 0.05' if exact_mcnemar_p(b, c) < 0.05 else 'NOT significant'))

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(args.boot, len(y)))
    diffs = rb[idx].mean(1) - ra[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print('bootstrap 95% CI of diff: [{:+.2%}, {:+.2%}]'.format(lo, hi))


if __name__ == '__main__':
    main()
