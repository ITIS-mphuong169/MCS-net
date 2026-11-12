"""Ambiguity-aware label smoothing (SOFT_LABELS=artist|text).

Uniform label smoothing (LABEL_SMOOTH=e) moves e of the target mass away
from the labelled class and spreads it evenly over the other classes. Style
labels are not evenly ambiguous: an Impressionist canvas is far more likely
to be labelled Post-Impressionism than Ukiyo-e, and the same painter is
often filed under several neighbouring movements. These targets keep the
SAME amount of smoothing as uniform label smoothing (the labelled class
keeps 1 - e + e/C, exactly as torch's label_smoothing does) and only change
WHERE the remaining e * (C - 1) / C goes:

  artist  q(k) ~ how often the other training works of the same artist are
          labelled k (leave-one-out, k != y), plus SOFT_PSEUDO pseudo-counts
          spread uniformly so an artist with no other work of another style
          falls back to uniform smoothing. Needs the artist in the file
          name (WikiArt: '<artist>_<title>.jpg'; Pandora: parent folder).
          Training labels only; nothing extra is used at test time.
  text    q(k) ~ softmax(cos(t_y, t_k) / SOFT_TEMP) over k != y, from the CLIP
          text embeddings of the style names: a language prior, no data.

Both give one target distribution per training image: artist targets per
sample (StyleDataset returns them as a 4th element), text targets per class
(a C x C matrix looked up by label in train.py).
"""
import math
import os
from collections import Counter, defaultdict

import torch


def artist_of(dataset, rel):
    """Artist key of one sample from its path relative to the dataset root."""
    base = os.path.basename(rel)
    if dataset == 'pandora':  # <NN_Style>/<Artist>/<file>
        return os.path.basename(os.path.dirname(rel))
    return base.split('_')[0] if '_' in base else None  # wikiart: <artist>_<title>.jpg


def smoothing_mass(eps, num_classes):
    """(mass kept on the labelled class, mass spread over the others), the
    split torch's CrossEntropyLoss(label_smoothing=eps) uses."""
    return 1.0 - eps + eps / num_classes, eps * (num_classes - 1) / num_classes


def artist_targets(dataset, items, num_classes, eps, pseudo=1.0):
    """items: [(rel, label)] of ONE dataset's training split -> (N, C) float
    tensor of targets, plus a stats dict. Labels live in the unified label
    space; the histogram of an artist stays inside their dataset's classes."""
    artists = [artist_of(dataset, rel) for rel, _ in items]
    counts = defaultdict(Counter)
    for a, (_, y) in zip(artists, items):
        if a is not None:
            counts[a][y] += 1
    keep, spread = smoothing_mass(eps, num_classes)
    targets = torch.zeros(len(items), num_classes)
    informed = 0
    for i, (a, (_, y)) in enumerate(zip(artists, items)):
        q = torch.full((num_classes,), pseudo / (num_classes - 1))
        q[y] = 0.0
        if a is not None:
            others = counts[a].copy()
            others[y] -= 1  # leave this work out
            n_other = sum(n for k, n in others.items() if k != y)
            if n_other > 0:
                informed += 1
                for k, n in others.items():
                    if k != y:
                        q[k] += n
        q = q / q.sum()
        targets[i] = spread * q
        targets[i, y] += keep
    n_art = len(counts)
    stats = {'artists': n_art, 'unparsed': sum(a is None for a in artists),
             'works_per_artist': len(items) / max(n_art, 1), 'informed_fraction': informed / max(len(items), 1),
             'mean_entropy_bits': float(_entropy_bits(targets, keep)) }
    return targets, stats


def text_targets(prototypes, eps, temperature=0.05):
    """prototypes: (C, D) L2-normalised CLIP text embeddings -> (C, C) matrix,
    row y = target distribution for label y."""
    protos = torch.nn.functional.normalize(prototypes.float(), dim=-1)
    sim = protos @ protos.t()
    sim.fill_diagonal_(float('-inf'))  # the labelled class gets its mass separately
    q = torch.softmax(sim / temperature, dim=-1)
    keep, spread = smoothing_mass(eps, protos.size(0))
    return spread * q + keep * torch.eye(protos.size(0))


def _entropy_bits(targets, keep):
    """Mean entropy (bits) of the smoothing part of the targets: 0 = all of
    the spread mass on one class, log2(C - 1) = uniform (plain smoothing)."""
    spread = targets.clone()
    spread[torch.arange(len(targets)), targets.argmax(1)] -= keep
    spread = spread.clamp(min=0)
    spread = spread / spread.sum(1, keepdim=True).clamp(min=1e-12)
    ent = -(spread * torch.log2(spread.clamp(min=1e-12))).sum(1)
    return ent.mean()


def soft_cross_entropy(logits, targets):
    """Mean over the batch of -sum_k t_k log p_k. Same value as
    CrossEntropyLoss(label_smoothing=e) when targets are the uniform ones."""
    return -(targets * torch.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


def describe(mode, stats=None, num_classes=None):
    if mode == 'artist' and stats:
        return ('SOFT_LABELS=artist: {} artists ({:.1f} works each), {} files without an artist name, '
                '{:.1%} of training images get an artist-informed target, smoothing entropy {:.2f} bits '
                '(uniform would be {:.2f})').format(stats['artists'], stats['works_per_artist'], stats['unparsed'],
                                                     stats['informed_fraction'], stats['mean_entropy_bits'],
                                                     math.log2(max(num_classes - 1, 2)))
    return 'SOFT_LABELS={}'.format(mode)
