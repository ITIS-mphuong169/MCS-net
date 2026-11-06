"""Make a fast-loading copy of an image-folder dataset (WikiArt).

Why: on Kaggle the data loader, not the GPU, bounds our epoch time - a
1-forward model (CLIP + linear) and a 3-forward model (CLIP + MCS-Net) both
took ~70 min per WikiArt epoch, and inference ran at ~30 img/s. The WikiArt
originals average ~400 KB and are decoded from scratch every epoch.

This script decodes every image once, resizes it so that its SHORT side is
at most --short pixels (never upscaling; 768 = 3x the 256-pixel test view,
enough for high-resolution detail crops), and stores the JPEGs,
uncompressed, in one zip. Relative paths and file names are kept, so
torchvision's ImageFolder lists the copy in exactly the same order as the
original and the seed-1231 split - hence the test set - is unchanged
(MANIFEST.txt inside the copy lets data_multi.py verify that at load time).

Next to the zip it writes <name>_meta.npz with, per image: the original
size, the stored size and the dHash dedup.py uses (so finding duplicates in
another dataset needs no second pass over WikiArt).

JPEG quality is picked by a pilot on a few hundred images: the highest of
92/90/88/85/82 whose projected total fits --budget-gb (Kaggle keeps at most
20 GB of notebook output); if none fits, the short side is reduced.

Usage (Kaggle CPU notebook, about an hour for WikiArt):
    python resize_dataset.py --src /kaggle/input/datasets/steubk/wikiart \
        --out /kaggle/working/wikiart768.zip
In a training session:
    python -m zipfile -e /kaggle/input/<prep notebook>/wikiart768.zip /kaggle/temp/
    DATA_DIR=/kaggle/temp/wikiart768
"""
import io
import os
import sys
import time
import random
import zipfile
import argparse
import numpy as np
from multiprocessing import Pool
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_multi import list_wikiart  # noqa: E402
from dedup import dhash, FAILED  # noqa: E402

QUALITIES = (92, 90, 88, 85, 82)
SHORTS = (768, 704, 640, 576, 512)
VERBATIM_MAX = 1500 * 1024  # an untouched JPEG is copied as is up to this size
MAX_ASPECT = 3              # long side capped at MAX_ASPECT x short (scroll paintings)


def load_resized(path, short):
    """RGB image with short side <= `short` (LANCZOS), plus what the file was."""
    img = Image.open(path)
    fmt, mode, size0 = img.format, img.mode, img.size
    if min(size0) >= 2 * short:
        img.draft('RGB', (short, short))  # JPEG only: decode at 1/2, 1/4, 1/8 scale, staying >= short
    img = img.convert('RGB')
    w, h = img.size
    scale = min(1.0, short / min(w, h))
    if max(w, h) * scale > MAX_ASPECT * short:
        scale = MAX_ASPECT * short / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img, size0, fmt, mode


def encode(img, quality):
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=quality)
    return buf.getvalue()


def pilot_one(args):
    path, short = args
    try:
        img, size0, fmt, mode = load_resized(path, short)
        if img.size == size0 and fmt == 'JPEG' and mode in ('RGB', 'L') and os.path.getsize(path) <= VERBATIM_MAX:
            n = os.path.getsize(path)
            return {q: n for q in QUALITIES}
        return {q: len(encode(img, q)) for q in QUALITIES}
    except Exception:
        return None


def convert_one(args):
    """-> (jpeg bytes, orig w, orig h, new w, new h, dhash, status) with
    status 0 = re-encoded, 1 = copied verbatim, 2 = unreadable (black
    224x224 placeholder, what data_loader.safe_pil_loader returns for it)."""
    path, short, quality = args
    h = dhash(path)
    try:
        img, size0, fmt, mode = load_resized(path, short)
        if img.size == size0 and fmt == 'JPEG' and mode in ('RGB', 'L') and os.path.getsize(path) <= VERBATIM_MAX:
            with open(path, 'rb') as f:
                return f.read(), size0[0], size0[1], img.size[0], img.size[1], h, 1
        return encode(img, quality), size0[0], size0[1], img.size[0], img.size[1], h, 0
    except Exception as e:
        print('[WARN] unreadable, stored as a black placeholder: {} ({})'.format(path, e), flush=True)
        return encode(Image.new('RGB', (224, 224)), 90), 0, 0, 224, 224, None, 2


def choose_setting(paths, short, budget_bytes, workers, n_pilot=300):
    rng = random.Random(0)
    sample = rng.sample(paths, min(n_pilot, len(paths)))
    for s in [x for x in SHORTS if x <= short] or [short]:
        with Pool(workers) as pool:
            sizes = [r for r in pool.map(pilot_one, [(p, s) for p in sample], chunksize=8) if r]
        proj = {q: float(np.mean([r[q] for r in sizes])) * len(paths) for q in QUALITIES}
        print('pilot, short side {}: projected total '.format(s)
              + ', '.join('q{} {:.1f} GB'.format(q, proj[q] / 1e9) for q in QUALITIES), flush=True)
        for q in QUALITIES:
            if proj[q] <= budget_bytes:
                return s, q, proj[q]
    q = QUALITIES[-1]
    print('[WARN] nothing fits the budget; using short side {} quality {}'.format(s, q))
    return s, q, proj[q]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='dataset root with one folder per class')
    ap.add_argument('--out', required=True, help='zip to write, e.g. /kaggle/working/wikiart768.zip')
    ap.add_argument('--short', type=int, default=768, help='max short side in pixels')
    ap.add_argument('--budget-gb', type=float, default=13.0, help='target size of the zip')
    ap.add_argument('--hard-limit-gb', type=float, default=17.0, help='abort if the zip grows past this')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--limit', type=int, default=0, help='only the first N images (smoke test)')
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.out))[0]
    samples = list_wikiart(args.src)  # ImageFolder order
    if args.limit:
        samples = samples[:args.limit]
    rels = [rel.replace(os.sep, '/') for rel, _ in samples]
    paths = [os.path.join(args.src, rel) for rel, _ in samples]
    print('{} images in {} classes at {}'.format(len(paths), len({s for _, s in samples}), args.src), flush=True)

    short, quality, proj = choose_setting(paths, args.short, args.budget_gb * 1e9, args.workers)
    print('=> short side <= {}, JPEG quality {}, about {:.1f} GB expected'.format(short, quality, proj / 1e9), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = len(paths)
    orig = np.zeros((n, 2), dtype=np.int32)
    new = np.zeros((n, 2), dtype=np.int32)
    hashes = np.full(n, FAILED, dtype=np.uint64)
    status = np.zeros(n, dtype=np.uint8)
    total, t0 = 0, time.time()
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_STORED, allowZip64=True) as zf, Pool(args.workers) as pool:
        tasks = ((p, short, quality) for p in paths)
        for i, (data, w0, h0, w1, h1, h, st) in enumerate(pool.imap(convert_one, tasks, chunksize=8)):
            zf.writestr('{}/{}'.format(name, rels[i]), data)
            orig[i], new[i], status[i] = (w0, h0), (w1, h1), st
            if h is not None:
                hashes[i] = h
            total += len(data)
            if total > args.hard_limit_gb * 1e9:
                raise SystemExit('zip passed {} GB after {} of {} images - rerun with a smaller --short'.format(
                    args.hard_limit_gb, i + 1, n))
            if (i + 1) % 2000 == 0 or i + 1 == n:
                el = time.time() - t0
                print('{:6d}/{}  {:.2f} GB  {:.0f} img/s  ETA {:.0f} min  (projected {:.1f} GB)'.format(
                    i + 1, n, total / 1e9, (i + 1) / el, (n - i - 1) / ((i + 1) / el) / 60, total / (i + 1) * n / 1e9),
                    flush=True)
        zf.writestr('{}/MANIFEST.txt'.format(name), '\n'.join(rels) + '\n')

    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), name + '_meta.npz')
    np.savez(meta_path, paths=np.array(rels), orig_size=orig, new_size=new, dhash=hashes, status=status,
             short=short, quality=quality)

    # the copy must list exactly like the original
    with zipfile.ZipFile(args.out) as zf:
        stored = [x[len(name) + 1:] for x in zf.namelist() if not x.endswith('MANIFEST.txt')]
    assert stored == rels, 'zip listing differs from the source listing'

    s0 = orig.min(1)[status != 2]
    print('\ndone in {:.0f} min: {} images, {:.2f} GB, {} re-encoded, {} copied verbatim, {} unreadable'.format(
        (time.time() - t0) / 60, n, total / 1e9, int((status == 0).sum()), int((status == 1).sum()),
        int((status == 2).sum())))
    print('original short side: median {:.0f} px; >= 512 px: {:.1%}; >= 768 px: {:.1%}; >= 1024 px: {:.1%}'.format(
        np.median(s0), (s0 >= 512).mean(), (s0 >= 768).mean(), (s0 >= 1024).mean()))
    print('written {} and {}'.format(args.out, meta_path))


if __name__ == '__main__':
    main()
