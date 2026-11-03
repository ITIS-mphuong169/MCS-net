"""Linear-probe several frozen pretrained backbones on WikiArt.

Answers "is a stronger backbone worth it?" before any fine-tuning: every
image is decoded once, pushed through each backbone (frozen, no
augmentation, Resize 256 / CenterCrop 224 like infer.py), and the pooled
features are cached. A logistic-regression head is then trained on the
train features, its epoch and weight decay picked on val, and the test
accuracy / macro F1 reported per backbone. Same seed-1231 80/10/10 split
as data_loader.py, so the numbers are comparable to our fine-tuned runs.

Env:
  DATA_DIR        WikiArt root (config.DATA_DIR)
  PROBE_MODELS    comma list of timm names (default: the 3 below)
  PROBE_OUT       where features / results go (default /kaggle/working/probe)
  PROBE_BS        extraction batch size (default 64)
  PROBE_LIMIT     use only N images per split (smoke test)
  PROBE_PRETRAINED=0  random weights (offline smoke test only)
  WORKERS         data-loader workers (default 4)
"""
import os
import json
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import config

DEFAULT_MODELS = [
    'resnet101.tv_in1k',                      # what MCS-Net fine-tunes (reference)
    'convnext_base.fb_in22k_ft_in1k',         # modern conv net, ImageNet-22k
    'vit_base_patch16_clip_224.openai',       # CLIP image tower
]
MODELS = [m for m in os.environ.get('PROBE_MODELS', ','.join(DEFAULT_MODELS)).split(',') if m]
OUT = os.environ.get('PROBE_OUT', '/kaggle/working/probe')
BS = int(os.environ.get('PROBE_BS', 64))
LIMIT = int(os.environ.get('PROBE_LIMIT', 0))
PRETRAINED = os.environ.get('PROBE_PRETRAINED', '1') != '0'
WORKERS = int(os.environ.get('WORKERS', 4))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def safe_pil_loader(path):
    try:
        img = Image.open(path)
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024))
        return img.convert('RGB')
    except Exception as e:
        print(f'[ERROR] Unable to load image {path}: {e}')
        return Image.new('RGB', (224, 224))


def get_splits(data_dir, seed=1231, val_split=0.1, test_split=0.1):
    """Same shuffle as data_loader.get_data_loaders; tensors left in [0, 1]
    (each backbone normalises with its own mean/std on the GPU)."""
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    full = datasets.ImageFolder(data_dir, transform=tf, loader=safe_pil_loader)
    indices = list(range(len(full)))
    random.Random(seed).shuffle(indices)
    num_val, num_test = int(len(full) * val_split), int(len(full) * test_split)
    splits = {'val': indices[:num_val], 'test': indices[num_val:num_val + num_test],
              'train': indices[num_val + num_test:]}
    if LIMIT:
        splits = {k: v[:LIMIT] for k, v in splits.items()}
    return {k: Subset(full, v) for k, v in splits.items()}, full.classes


@torch.no_grad()
def extract(models, loader):
    feats = {name: [] for name in models}
    labels = []
    for X, y in tqdm(loader, unit=' batches'):
        X = X.to(device, non_blocking=True)
        for name, (net, mean, std) in models.items():
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                f = net((X - mean) / std)
            feats[name].append(f.float().cpu())
        labels.append(y)
    return {k: torch.cat(v).numpy() for k, v in feats.items()}, torch.cat(labels).numpy()


def fit_probe(Xtr, ytr, Xva, yva, num_classes, epochs=100, lr=1e-3, wds=(0.0, 1e-4, 1e-3)):
    """Multinomial logistic regression (Adam, full standardised features);
    the (weight decay, epoch) with best val accuracy wins."""
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd, device=device)
    Xva_t = torch.tensor((Xva - mu) / sd, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    best = {'val_acc': -1}
    for wd in wds:
        torch.manual_seed(0)
        head = torch.nn.Linear(Xtr.shape[1], num_classes).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
        n = len(ytr_t)
        for ep in range(1, epochs + 1):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, 1024):
                idx = perm[i:i + 1024]
                loss = F.cross_entropy(head(Xtr_t[idx]), ytr_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if ep % 5 == 0:
                with torch.no_grad():
                    va = (head(Xva_t).argmax(1).cpu().numpy() == yva).mean()
                if va > best['val_acc']:
                    best = {'val_acc': float(va), 'wd': wd, 'epoch': ep,
                            'state': {k: v.clone() for k, v in head.state_dict().items()}}
    head = torch.nn.Linear(Xtr.shape[1], num_classes).to(device)
    head.load_state_dict(best.pop('state'))
    return head, (mu, sd), best


def evaluate(head, norm, X, y, log_prior, taus=(0.0, 0.5, 1.0)):
    """Plain argmax, plus logit adjustment (logits - tau * log prior) which
    trades a little accuracy for the rare classes; tau is chosen on val."""
    mu, sd = norm
    with torch.no_grad():
        logits = head(torch.tensor((X - mu) / sd, device=device)).cpu()
    out = {}
    for tau in taus:
        pred = (logits - tau * log_prior).argmax(1).numpy()
        out[tau] = {'acc': accuracy_score(y, pred), 'f1_macro': f1_score(y, pred, average='macro')}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    splits, class_names = get_splits(config.DATA_DIR)
    print({k: len(v) for k, v in splits.items()}, len(class_names), 'classes')

    # ---- features (cached per model) ----
    todo = [m for m in MODELS if not os.path.exists(os.path.join(OUT, m + '.npz'))]
    if todo:
        models = {}
        for name in todo:
            net = timm.create_model(name, pretrained=PRETRAINED, num_classes=0).to(device).eval()
            cfg = timm.data.resolve_data_config({}, model=net)
            mean = torch.tensor(cfg['mean'], device=device).view(1, 3, 1, 1)
            std = torch.tensor(cfg['std'], device=device).view(1, 3, 1, 1)
            models[name] = (net, mean, std)
            print(f'{name}: {net.num_features} features, mean {cfg["mean"]}')
        for split, ds in splits.items():
            loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=WORKERS, pin_memory=True)
            t0 = time.time()
            feats, labels = extract(models, loader)
            print(f'{split}: {len(labels)} images in {time.time() - t0:.0f}s')
            for name in todo:
                path = os.path.join(OUT, name + '.npz')
                prev = dict(np.load(path)) if os.path.exists(path) else {}
                prev[split] = feats[name]
                prev[split + '_y'] = labels
                np.savez(path, **prev)
        del models
        torch.cuda.empty_cache()

    # ---- linear probes ----
    results = {}
    for name in MODELS:
        d = np.load(os.path.join(OUT, name + '.npz'))
        ytr = d['train_y']
        counts = np.bincount(ytr, minlength=len(class_names))
        log_prior = torch.log(torch.tensor(counts / counts.sum(), dtype=torch.float32))
        head, norm, best = fit_probe(d['train'], ytr, d['val'], d['val_y'], len(class_names))
        val = evaluate(head, norm, d['val'], d['val_y'], log_prior)
        tau = max(val, key=lambda t: val[t]['f1_macro'])          # tau picked on val F1
        test = evaluate(head, norm, d['test'], d['test_y'], log_prior)
        results[name] = {'dim': int(d['train'].shape[1]), 'wd': best['wd'], 'epoch': best['epoch'],
                         'val_acc': best['val_acc'], 'tau_for_f1': tau,
                         'test_plain': test[0.0], 'test_logit_adjusted': test[tau]}
        print(f"\n{name}  (dim {results[name]['dim']}, wd {best['wd']}, epoch {best['epoch']})")
        print(f"  val  acc {best['val_acc'] * 100:.2f}")
        print(f"  test acc {test[0.0]['acc'] * 100:.2f}  F1 {test[0.0]['f1_macro'] * 100:.2f}   (plain)")
        print(f"  test acc {test[tau]['acc'] * 100:.2f}  F1 {test[tau]['f1_macro'] * 100:.2f}   (logit adjusted, tau {tau})")

    print('\n=== summary (test, center crop) ===')
    print(f"{'backbone':40s} {'Acc':>7s} {'F1':>7s} | {'Acc':>7s} {'F1':>7s}  (plain | logit adj.)")
    for name, r in results.items():
        p, a = r['test_plain'], r['test_logit_adjusted']
        print(f"{name:40s} {p['acc'] * 100:7.2f} {p['f1_macro'] * 100:7.2f} | {a['acc'] * 100:7.2f} {a['f1_macro'] * 100:7.2f}")
    with open(os.path.join(OUT, 'probe_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
