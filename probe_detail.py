"""Go / no-go test for a high-resolution "detail" view, and for learning
where to look (REINFORCE), before either is built.

MCS-Net's attention crop - and every crop / RL experiment we ran - cuts a
region out of the 224x224 input tensor and upsamples it (utils.batch_augment):
no new pixels. The WikiArt originals are much larger than 224, so whether
brushwork-level detail helps style recognition was never tested. This
script tests it without training a backbone.

Views of each image (all 224x224):
  global   Resize 256 + CenterCrop 224: the view every run is tested on
  detail   for each magnification m in PROBE_MAGS (default 2,3): the image
           resized to short side 256*m, its central (224*m)^2 square (the
           area the global view shows), cut into a 3x3 grid of crops, i.e.
           the same area at m times the linear resolution
  zoom     control: the global view itself blown up m times and cut the
           same way - MCS-Net's kind of crop, same field of view as the
           detail crop but no new pixels

(a) Is there information in the detail?
    frozen CLIP + logistic regression (protocol of probe.py):
        global   vs   global + zoom   vs   global + detail
    fine-tuned checkpoint, if PROBE_CKPT is given (logits averaged):
        global   vs   global + zoom   vs   global + detail
    'global + detail' minus 'global + zoom' is what the new pixels add;
    'global + zoom' minus 'global' is what zooming / averaging crops adds.

(b) Does it matter WHERE the detail crop is taken (what a policy could gain)?
    Prediction from the global view + ONE detail crop; test accuracy when
    that crop is: random (mean over the 9 positions), the centre, the most
    confident one, picked by a selector, or the oracle (any position right).
    The selector is a small MLP that sees only what a policy would see
    before cropping - the global view's patch tokens under each candidate
    region, the global feature, the position - trained on the val split to
    predict which position helps, scored on test. It is fully supervised
    (it is told the outcome of all 9 positions) while REINFORCE only gets
    bandit feedback, so its gain over a random position is an optimistic
    estimate of what RL could add.

Env:
  DATA_DIR         WikiArt root (originals, or the resize_dataset.py copy)
  PROBE_MODEL      frozen backbone (default vit_base_patch16_clip_224.openai,
                   the one probe.py scored 63.74 with)
  PROBE_CKPT       optional fine-tuned PLAIN_CLS checkpoint (last.ckpt); set
                   NET / PLAIN_CLS / TEXT_INIT / LABEL_ATTN as when it was trained
  PROBE_MAGS       magnifications (default 2,3)
  PROBE_OUT        output dir (default /kaggle/working/probe_detail)
  PROBE_BS         images per batch (default 16; each image is 19 or 37 crops)
  PROBE_TRAIN_LIMIT  training images used for the logistic regressions
                   (default 20000, a random subset: the split is shuffled;
                   0 = all). The comparison between views needs no more, and
                   the fine-tuned checkpoint's 'global' row is the sanity
                   check: it must reproduce infer.py's test accuracy.
  PROBE_FROZEN_ZOOM  also compute the zoom control for the frozen backbone
                   (default: only when no PROBE_CKPT is given - the
                   checkpoint gives that control without any training)
  PROBE_LIMIT      first N images of each split (smoke test)
  PROBE_EPOCHS     logistic-regression epochs (default 100, as probe.py)
  WORKERS          data-loader workers (default 4)
"""
import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm

import config
from probe import fit_probe, device
from data_multi import list_wikiart, seed_split

Image.MAX_IMAGE_PIXELS = 1000000000

MAGS = [int(m) for m in os.environ.get('PROBE_MAGS', '2,3').split(',') if m]
G = int(os.environ.get('PROBE_GRID', 3))   # G x G grid of crops per magnification (the training views use 3)
P = G * G
M = len(MAGS)
VD = 1 + M * P             # global + detail crops
V = VD + M * P             # ... + zoom (control) crops
MODEL = os.environ.get('PROBE_MODEL', 'vit_base_patch16_clip_224.openai')
CKPT = os.environ.get('PROBE_CKPT', '')
OUT = os.environ.get('PROBE_OUT', '/kaggle/working/probe_detail')
BS = int(os.environ.get('PROBE_BS', 16))
LIMIT = int(os.environ.get('PROBE_LIMIT', 0))
TRAIN_LIMIT = int(os.environ.get('PROBE_TRAIN_LIMIT', 20000))
FROZEN_ZOOM = os.environ.get('PROBE_FROZEN_ZOOM', '0' if os.environ.get('PROBE_CKPT') else '1') != '0'
EPOCHS = int(os.environ.get('PROBE_EPOCHS', 100))
PRETRAINED = os.environ.get('PROBE_PRETRAINED', '1') != '0'
WORKERS = int(os.environ.get('WORKERS', 4))
THRESH_DETAIL = 1.0        # points of accuracy 'global + detail' must add to be worth a branch
THRESH_SELECT = 1.0        # points a chosen position must add over a random one to be worth RL


def paired_diff(correct_a, correct_b, n_boot=2000, seed=0):
    """Accuracy of a minus accuracy of b, in points, with a paired bootstrap
    95% interval over the test images: (difference, low, high)."""
    d = correct_a.astype(np.float64) - correct_b.astype(np.float64)
    rng = np.random.RandomState(seed)
    boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return 100. * float(d.mean()), 100. * float(np.percentile(boots, 2.5)), 100. * float(np.percentile(boots, 97.5))


# --------------------------------------------------------------------------- views
def load(path):
    """RGB image decoded large enough for the biggest magnification, and the
    short side of the original file (how much real detail there is)."""
    need = 256 * max(MAGS)
    try:
        img = Image.open(path)
        short0 = min(img.size)
        img.draft('RGB', (need, need))  # JPEG: decode at reduced scale, staying >= need
        img = img.convert('RGB')
        if min(img.size) > 2 * need:
            img = TF.resize(img, 2 * need)
        return img, short0
    except Exception as e:
        print('[ERROR] unable to load {}: {}'.format(path, e))
        return Image.new('RGB', (224, 224)), 0


def grid_offsets(m):
    step = (224 * m - 224) / (G - 1)
    return [(round(r * step), round(c * step)) for r in range(G) for c in range(G)]


def grid_crops(img, m):
    return [img.crop((x, y, x + 224, y + 224)) for y, x in grid_offsets(m)]


def make_views(img, zoom):
    glob = TF.center_crop(TF.resize(img, 256), 224)
    views = [glob]
    for m in MAGS:  # detail: same area, from an m times larger resize of the original
        views += grid_crops(TF.center_crop(TF.resize(img, 256 * m), 224 * m), m)
    for m in (MAGS if zoom else []):  # zoom control: the global view blown up, no new pixels
        views += grid_crops(TF.resize(glob, 224 * m), m)
    return views


class DetailViews(Dataset):
    def __init__(self, root, items, zoom):
        self.root, self.items, self.zoom = root, items, zoom

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, y = self.items[i]
        img, short0 = load(os.path.join(self.root, rel))
        arr = np.stack([np.asarray(v, dtype=np.uint8) for v in make_views(img, self.zoom)])  # (VD or V, 224, 224, 3)
        return torch.from_numpy(arr).permute(0, 3, 1, 2), y, short0


# --------------------------------------------------------------------------- models
def region_bounds(m, grid):
    """Patch-index ranges, on the global view's patch grid, of the area each
    detail crop of magnification m shows."""
    out = []
    for y, x in grid_offsets(m):
        y0, x0 = y / m, x / m  # in global-view pixels
        rows = (int(np.floor(y0 * grid / 224 + 1e-6)), int(np.ceil((y0 + 224 / m) * grid / 224 - 1e-6)))
        cols = (int(np.floor(x0 * grid / 224 + 1e-6)), int(np.ceil((x0 + 224 / m) * grid / 224 - 1e-6)))
        out.append((rows, cols))
    return out


def pool_regions(fmap):
    """(B, C, H, W) patch tokens -> (B, M, 9, C): mean token under each candidate crop."""
    return torch.stack([torch.stack([fmap[:, :, a:b, c:d].mean((2, 3))
                                     for (a, b), (c, d) in region_bounds(m, fmap.size(2))], 1) for m in MAGS], 1)


class Frozen(nn.Module):
    """timm ViT, frozen: the pooled feature probe.py used, plus region tokens."""

    def __init__(self, name):
        super().__init__()
        self.net = timm.create_model(name, pretrained=PRETRAINED, num_classes=0)
        cfg = timm.data.resolve_data_config({}, model=self.net)
        self.register_buffer('mean', torch.tensor(cfg['mean']).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(cfg['std']).view(1, 3, 1, 1))

    def forward(self, x):
        tokens = self.net.forward_features((x.float() / 255. - self.mean) / self.std)
        assert tokens.dim() == 3, 'probe_detail needs a ViT (token) backbone'
        patches = tokens[:, self.net.num_prefix_tokens:]
        B, N, C = patches.shape
        H = int(round(N ** 0.5))
        return self.net.forward_head(tokens), pool_regions(patches.transpose(1, 2).reshape(B, C, H, H))


class Tuned(nn.Module):
    """A fine-tuned plain classifier (WSDAN_MCS, PLAIN_CLS=1): logits, pooled
    feature and region tokens. The model normalises for CLIP itself and
    expects ImageNet-normalised input, like the data loader gives it."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        p, pooled, fmap = self.model.plain_logits((x.float() / 255. - self.mean) / self.std)
        return p, pooled, pool_regions(fmap)


def build_tuned(path, class_names):
    from models import WSDAN_MCS
    model = WSDAN_MCS(num_classes=len(class_names), M=config.num_attentions, net=config.net, pretrained=False,
                      class_names=class_names)
    ck = torch.load(path, weights_only=False, map_location='cpu')
    missing = [k for k in model.state_dict() if k not in ck['state_dict']]
    assert not missing, 'PROBE_CKPT does not match NET/PLAIN_CLS/TEXT_INIT/LABEL_ATTN: missing {}'.format(missing[:5])
    model.load_state_dict(ck['state_dict'])
    assert model.plain, 'PROBE_CKPT must come from a PLAIN_CLS=1 run'
    print('fine-tuned model: {} from {} @ epoch {}'.format(config.net, path, ck.get('logs', {}).get('epoch')))
    return model


@torch.no_grad()
def run(module, x, n=320):
    outs = None
    for i in range(0, x.size(0), n):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            o = module(x[i:i + n])
        o = [t.float() for t in o]
        outs = [[t] for t in o] if outs is None else [a + [t] for a, t in zip(outs, o)]
    return [torch.cat(a) for a in outs]


def extract(ds, frozen, tuned, with_regions):
    """Features of one split. g: global feature (N, C); d / z: detail / zoom
    crop features (N, M, 9, C); reg: region tokens of the global view
    (N, M, 9, C); t*: the same from the fine-tuned model (logits for tg, td,
    tz; pooled feature tp; region tokens treg)."""
    loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=WORKERS, pin_memory=device.type == 'cuda')
    acc = {}

    def put(k, t):
        acc.setdefault(k, []).append(t.cpu().numpy().astype(np.float16))

    ys, shorts = [], []
    for X, y, short in tqdm(loader, unit=' batches'):
        B, Vx = X.shape[:2]  # Vx = VD (no zoom crops in this split) or V
        X = X.to(device, non_blocking=True)
        Vf = Vx if FROZEN_ZOOM else VD  # the frozen backbone skips the zoom crops unless asked
        cls, reg = run(frozen, X[:, :Vf].reshape(B * Vf, 3, 224, 224))
        cls = cls.view(B, Vf, -1)
        put('g', cls[:, 0])
        put('d', cls[:, 1:VD].reshape(B, M, P, -1))
        if Vf == V:
            put('z', cls[:, VD:].reshape(B, M, P, -1))
        if with_regions:
            put('reg', reg.view(B, Vf, M, P, -1)[:, 0])
            if tuned is not None:
                lg, pooled, treg = run(tuned, X.reshape(B * Vx, 3, 224, 224))
                lg = lg.view(B, Vx, -1)
                put('tg', lg[:, 0])
                put('td', lg[:, 1:VD].reshape(B, M, P, -1))
                if Vx == V:
                    put('tz', lg[:, VD:].reshape(B, M, P, -1))
                put('tp', pooled.view(B, Vx, -1)[:, 0])
                put('treg', treg.view(B, Vx, M, P, -1)[:, 0])
        ys.append(y.numpy())
        shorts.append(short.numpy())
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out['y'], out['short'] = np.concatenate(ys).astype(np.int64), np.concatenate(shorts).astype(np.int64)
    return out


# --------------------------------------------------------------------------- analysis
def f32(a):
    return np.asarray(a, dtype=np.float32)


def cat(*parts):
    return np.concatenate([f32(p) for p in parts], 1)


def predict(head, norm, X):
    mu, sd = norm
    with torch.no_grad():
        return head(torch.tensor((f32(X) - mu) / sd, device=device)).float().cpu().numpy()


def score(pred, y):
    return {'acc': 100. * float((pred == y).mean()), 'f1': 100. * float(f1_score(y, pred, average='macro'))}


def softmax(l):
    l = l - l.max(-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(-1, keepdims=True)


def fit_selector(Xv, tv, Xt, epochs=40):
    """MLP regressing, per (image, position), how much that position helps."""
    mu, sd = Xv.mean(0, keepdims=True), Xv.std(0, keepdims=True) + 1e-6
    Xv_t = torch.tensor((Xv - mu) / sd, device=device)
    tv_t = torch.tensor(tv, device=device)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(Xv.shape[1], 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(tv_t), device=device)
        for i in range(0, len(perm), 1024):
            idx = perm[i:i + 1024]
            loss = F.mse_loss(net(Xv_t[idx]).squeeze(1), tv_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor((Xt - mu) / sd, device=device)).squeeze(1).cpu().numpy()


def position_analysis(lv, yv, lt, yt, ctx_v, reg_v, ctx_t, reg_t):
    """lv / lt: (N, 9, K) logits of 'global + the detail crop at position p'
    on val / test. ctx: global feature (N, C); reg: region tokens (N, 9, C)."""
    def helps(l, y):  # centred log-probability of the true class per position
        lp = np.log(softmax(l.astype(np.float64)) + 1e-12)[np.arange(len(y))[:, None], np.arange(P)[None], y[:, None]]
        return np.clip(lp - lp.mean(1, keepdims=True), -5, 5).astype(np.float32)

    def selector_inputs(ctx, reg):
        n = len(ctx)
        onehot = np.tile(np.eye(P, dtype=np.float32)[None], (n, 1, 1))
        return np.concatenate([f32(reg), np.repeat(f32(ctx)[:, None], P, 1), onehot], 2).reshape(n * P, -1)

    n = len(yt)
    correct = lt.argmax(-1) == yt[:, None]  # (N, 9)
    pred_sel = fit_selector(selector_inputs(ctx_v, reg_v), helps(lv, yv).reshape(-1),
                            selector_inputs(ctx_t, reg_t)).reshape(n, P)
    sel = correct[np.arange(n), pred_sel.argmax(1)]
    conf = correct[np.arange(n), softmax(lt.astype(np.float64)).max(-1).argmax(1)]
    diff = sel.astype(np.float64) - correct.mean(1)  # per image: chosen position vs a random one
    rng = np.random.RandomState(0)
    boots = [diff[rng.randint(0, n, n)].mean() for _ in range(1000)]
    target_t = helps(lt, yt).reshape(-1)
    return {
        'random': 100. * float(correct.mean()),
        'center': 100. * float(correct[:, P // 2].mean()),
        'best_fixed': 100. * float(correct.mean(0).max()),
        'per_position': [100. * float(c) for c in correct.mean(0)],
        'max_confidence': 100. * float(conf.mean()),
        'selector': 100. * float(sel.mean()),
        'oracle': 100. * float(correct.any(1).mean()),
        'all_9': 100. * float((lt.mean(1).argmax(1) == yt).mean()),
        'selector_minus_random': 100. * float(diff.mean()),
        'selector_minus_random_ci95': [100. * float(np.percentile(boots, 2.5)), 100. * float(np.percentile(boots, 97.5))],
        'selector_corr': float(np.corrcoef(pred_sel.reshape(-1), target_t)[0, 1]) if target_t.std() > 0 else 0.0,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    samples = list_wikiart(config.DATA_DIR)
    class_names = sorted({s for _, s in samples})
    cidx = {c: i for i, c in enumerate(class_names)}
    items = [(rel, cidx[s]) for rel, s in samples]
    split = seed_split(len(items))  # the seed-1231 80/10/10 split of every run
    if TRAIN_LIMIT:
        split['train'] = split['train'][:TRAIN_LIMIT]  # the split is shuffled: a random subset
    if LIMIT:
        split = {k: v[:LIMIT] for k, v in split.items()}
    K = len(class_names)
    print({k: len(v) for k, v in split.items()}, K, 'classes; magnifications', MAGS,
          '; frozen zoom control', FROZEN_ZOOM, '; checkpoint', CKPT or None)

    # ---- features (cached per split) ----
    feats = {}
    todo = [s for s in ('train', 'val', 'test') if not os.path.exists(os.path.join(OUT, 'feats_{}.npz'.format(s)))]
    if todo:
        frozen = Frozen(MODEL).to(device).eval()
        tuned = Tuned(build_tuned(CKPT, class_names)).to(device).eval() if CKPT else None
        if torch.cuda.device_count() > 1:
            frozen = nn.DataParallel(frozen)
            tuned = nn.DataParallel(tuned) if tuned is not None else None
        for s in todo:
            t0 = time.time()
            # zoom crops only where something uses them: the frozen control, or the checkpoint on test
            zoom = FROZEN_ZOOM or (tuned is not None and s == 'test')
            f = extract(DetailViews(config.DATA_DIR, [items[i] for i in split[s]], zoom), frozen, tuned, s != 'train')
            np.savez(os.path.join(OUT, 'feats_{}.npz'.format(s)), **f)
            print('{}: {} images in {:.0f}s ({:.1f} img/s)'.format(s, len(f['y']), time.time() - t0,
                                                                  len(f['y']) / (time.time() - t0)), flush=True)
        del frozen, tuned
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    for s in ('train', 'val', 'test'):
        feats[s] = dict(np.load(os.path.join(OUT, 'feats_{}.npz'.format(s))))
    tr, va, te = feats['train'], feats['val'], feats['test']
    ytr, yva, yte = tr['y'], va['y'], te['y']
    has_tuned = 'tg' in te
    res = {'model': MODEL, 'mags': MAGS, 'n': {k: int(len(v['y'])) for k, v in feats.items()},
           'ckpt': CKPT if has_tuned else None, 'frozen': {}, 'tuned': {}, 'position': {}, 'resolution': {}}
    preds = {}

    def dmean(f, key, mi):
        return f32(f[key][:, mi]).mean(1)

    # ---- (a) frozen backbone, logistic regression ----
    def probe(name, build):
        head, norm, best = fit_probe(build(tr), ytr, build(va), yva, K, epochs=EPOCHS)
        preds['frozen/' + name] = predict(head, norm, build(te)).argmax(1)
        res['frozen'][name] = dict(score(preds['frozen/' + name], yte), val_acc=100. * best['val_acc'])
        print('  frozen  {:38s} acc {:.2f}  F1 {:.2f}'.format(name, res['frozen'][name]['acc'], res['frozen'][name]['f1']),
              flush=True)

    print('\n=== (a) frozen {} + logistic regression (test) ==='.format(MODEL))
    probe('global', lambda f: f32(f['g']))
    for mi, m in enumerate(MAGS):
        if 'z' in tr:
            probe('global + zoom x{} (no new pixels)'.format(m), lambda f, mi=mi: cat(f['g'], dmean(f, 'z', mi)))
        probe('global + detail x{}'.format(m), lambda f, mi=mi: cat(f['g'], dmean(f, 'd', mi)))
        probe('detail x{} only'.format(m), lambda f, mi=mi: dmean(f, 'd', mi))
    if M > 1:
        probe('global + detail all', lambda f: cat(f['g'], *[dmean(f, 'd', mi) for mi in range(M)]))

    # ---- (a) fine-tuned checkpoint, logits averaged ----
    if has_tuned:
        print('\n=== (a) fine-tuned checkpoint, logits averaged (test) ===')

        def tuned_row(name, logits):
            preds['tuned/' + name] = logits.argmax(1)
            res['tuned'][name] = score(preds['tuned/' + name], yte)
            print('  tuned   {:38s} acc {:.2f}  F1 {:.2f}'.format(name, res['tuned'][name]['acc'], res['tuned'][name]['f1']))

        tg = f32(te['tg'])
        tuned_row('global', tg)
        for mi, m in enumerate(MAGS):
            tuned_row('global + zoom x{} (no new pixels)'.format(m), (tg + dmean(te, 'tz', mi)) / 2)
            tuned_row('global + detail x{}'.format(m), (tg + dmean(te, 'td', mi)) / 2)
            tuned_row('detail x{} only'.format(m), dmean(te, 'td', mi))

    # ---- (b) one detail crop: does its position matter? ----
    print('\n=== (b) global + ONE detail crop: test accuracy by how the position is chosen ===')
    print('  {:14s} {:>7s} {:>7s} {:>9s} {:>9s} {:>7s} {:>7s}   {}'.format(
        '', 'random', 'centre', 'max-conf', 'selector', 'oracle', 'all 9', 'selector - random [95% CI]'))
    rng = np.random.RandomState(0)
    for mi, m in enumerate(MAGS):
        n_tr, n_va = len(ytr), len(yva)
        pos = rng.randint(0, P, size=(n_tr, 3))  # 3 random positions per training image
        Xtr = np.concatenate([cat(tr['g'], tr['d'][np.arange(n_tr), mi, pos[:, k]]) for k in range(3)])
        Xva = cat(va['g'], va['d'][np.arange(n_va), mi, rng.randint(0, P, size=n_va)])
        head, norm, _ = fit_probe(Xtr, np.tile(ytr, 3), Xva, yva, K, epochs=EPOCHS)
        del Xtr

        def all_positions(f):
            return np.stack([predict(head, norm, cat(f['g'], f['d'][:, mi, p])) for p in range(P)], 1)

        runs = {'frozen x{}'.format(m): (all_positions(va), all_positions(te), va['g'], va['reg'][:, mi],
                                         te['g'], te['reg'][:, mi])}
        if has_tuned:
            runs['tuned x{}'.format(m)] = ((f32(va['tg'])[:, None] + f32(va['td'][:, mi])) / 2,
                                           (f32(te['tg'])[:, None] + f32(te['td'][:, mi])) / 2,
                                           va['tp'], va['treg'][:, mi], te['tp'], te['treg'][:, mi])
        for name, (lv, lt, cv, rv, ct, rt) in runs.items():
            r = position_analysis(lv, yva, lt, yte, cv, rv, ct, rt)
            res['position'][name] = r
            print('  {:14s} {:7.2f} {:7.2f} {:9.2f} {:9.2f} {:7.2f} {:7.2f}   {:+.2f} [{:+.2f}, {:+.2f}]  (corr {:.2f})'.format(
                name, r['random'], r['center'], r['max_confidence'], r['selector'], r['oracle'], r['all_9'],
                r['selector_minus_random'], *r['selector_minus_random_ci95'], r['selector_corr']), flush=True)

    # ---- where the gain is: by original resolution, by class ----
    short = te['short']
    print('\noriginal short side of the test images: median {:.0f} px; '.format(np.median(short))
          + '; '.join('>= {} px: {:.1%}'.format(256 * m, float((short >= 256 * m).mean())) for m in MAGS))
    print('gain of "global + detail" over "global" (acc points), images with / without that much real resolution:')
    for src in (['frozen', 'tuned'] if has_tuned else ['frozen']):
        for m in MAGS:
            base, det = preds[src + '/global'] == yte, preds['{}/global + detail x{}'.format(src, m)] == yte
            real = short >= 256 * m
            g_real = 100. * float(det[real].mean() - base[real].mean()) if real.any() else None
            g_up = 100. * float(det[~real].mean() - base[~real].mean()) if (~real).any() else None
            res['resolution']['{} x{}'.format(src, m)] = {'real': g_real, 'upsampled': g_up, 'n_real': int(real.sum())}
            print('  {:7s} x{}: original large enough {} ({} images)   original smaller, upsampled {} ({} images)'.format(
                src, m, 'n/a' if g_real is None else '{:+.2f}'.format(g_real), int(real.sum()),
                'n/a' if g_up is None else '{:+.2f}'.format(g_up), int((~real).sum())))

    src = 'tuned' if has_tuned else 'frozen'
    best_m = max(MAGS, key=lambda m: res[src]['global + detail x{}'.format(m)]['acc'])
    r0 = recall_score(yte, preds[src + '/global'], labels=list(range(K)), average=None, zero_division=0)
    r1 = recall_score(yte, preds['{}/global + detail x{}'.format(src, best_m)], labels=list(range(K)), average=None,
                      zero_division=0)
    res['per_class_recall'] = {class_names[c]: [100. * float(r0[c]), 100. * float(r1[c])] for c in range(K)}
    print('\nper-class recall, {} model: global -> global + detail x{} (sorted by change)'.format(src, best_m))
    for c in np.argsort(-(r1 - r0)):
        print('  {:28s} {:6.1f} -> {:6.1f}  ({:+.1f})'.format(class_names[c], 100 * r0[c], 100 * r1[c], 100 * (r1[c] - r0[c])))

    # ---- decision ----
    def gains(d):
        out = {}
        for m in MAGS:
            det, zoom = d['global + detail x{}'.format(m)]['acc'], d.get('global + zoom x{} (no new pixels)'.format(m))
            out[m] = (det - d['global']['acc'], det - zoom['acc'] if zoom else None)
        return out

    print('\n=== decision ===')
    # The verdict is on the frozen backbone (the model the views will be
    # trained into is not available yet): GO needs, for some magnification,
    # 'global + detail' to beat BOTH 'global' and 'global + zoom' (the same
    # field of view with no new pixels) by THRESH_DETAIL points, and the 95%
    # paired-bootstrap interval of both differences to stay above zero.
    res['verdict'] = {}
    go_mags = []
    for src in (['frozen', 'tuned'] if has_tuned else ['frozen']):
        for m in MAGS:
            det = preds['{}/global + detail x{}'.format(src, m)] == yte
            vs_g = paired_diff(det, preds[src + '/global'] == yte)
            zname = '{}/global + zoom x{} (no new pixels)'.format(src, m)
            vs_z = paired_diff(det, preds[zname] == yte) if zname in preds else None
            res['verdict']['{} x{}'.format(src, m)] = {'vs_global': vs_g, 'vs_zoom': vs_z}
            print('(a) {:7s} x{}: global + detail vs global {:+.2f} [{:+.2f}, {:+.2f}];  vs global + zoom {}'.format(
                src, m, *vs_g, 'not computed' if vs_z is None else '{:+.2f} [{:+.2f}, {:+.2f}]'.format(*vs_z)))
            if src == 'frozen' and vs_z is not None and vs_g[0] >= THRESH_DETAIL and vs_g[1] > 0 \
                    and vs_z[0] >= THRESH_DETAIL and vs_z[1] > 0:
                go_mags.append(m)
    best_gain = max(v['vs_global'][0] for v in res['verdict'].values())
    sel_gain = max(r['selector_minus_random'] for r in res['position'].values())
    res['decision'] = {'detail_gain': best_gain, 'detail_go': bool(go_mags), 'go_mags': go_mags,
                       'selector_gain': sel_gain, 'rl_go': bool(sel_gain >= THRESH_SELECT)}
    print('\nVERDICT (a) detail view: {}'.format(
        'GO, magnification x{}'.format(max(go_mags, key=lambda m: res['verdict']['frozen x{}'.format(m)]['vs_zoom'][0]))
        if go_mags else 'NO-GO (no magnification adds {:.1f}+ points over both global and zoom with the interval above 0)'.format(THRESH_DETAIL)))
    print('VERDICT (b) learned / RL selection of the position: {} (best selector gain {:+.2f}, threshold {:.1f})'.format(
        'worth building' if sel_gain >= THRESH_SELECT else 'not worth it, use random or grid crops', sel_gain, THRESH_SELECT))
    with open(os.path.join(OUT, 'probe_detail_results.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print('saved', os.path.join(OUT, 'probe_detail_results.json'))


if __name__ == '__main__':
    main()
