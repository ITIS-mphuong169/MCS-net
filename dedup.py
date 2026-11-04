"""Find Pandora18k images that are copies of WikiArt images.

Pandora18k was partly collected from WikiArt (its file names are WikiArt
slugs), so training on WikiArt + Pandora and testing on Pandora would
leak WikiArt training images into Pandora's test set. This script hashes
every image of both datasets (dHash, 64 bit, robust to re-encoding and
resizing) and lists the Pandora images within HAMMING bits of any WikiArt
image. data_multi.py drops them when EXCLUDE_LIST points at the output.

Usage (Kaggle, ~10-15 min for 100k images on 4 CPUs):
    DATA_DIR=/kaggle/input/datasets/steubk/wikiart DATA_DIR_PANDORA=/kaggle/working/pandora18k \
        python dedup.py --out /kaggle/working/pandora18k/pandora_dups.txt
"""
import os
import sys
import argparse
import numpy as np
from multiprocessing import Pool
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_multi import list_wikiart, list_pandora, DEFAULT_ROOTS  # noqa: E402


def dhash(path):
    """64-bit difference hash: 9x8 grayscale, each bit = left pixel > right."""
    try:
        img = Image.open(path)
        img.draft('L', (144, 128))  # fast JPEG downscale while decoding
        g = np.asarray(img.convert('L').resize((9, 8), Image.BILINEAR), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        return int(np.packbits(bits).view('>u8')[0])
    except Exception as e:
        print('[WARN] {}: {}'.format(path, e))
        return None


FAILED = 2 ** 64 - 1  # hash value standing for an unreadable image


def hash_all(root, rels, workers):
    paths = [os.path.join(root, r) for r in rels]
    with Pool(workers) as pool:
        hashes = pool.map(dhash, paths, chunksize=64)
    return np.array([FAILED if h is None else h for h in hashes], dtype=np.uint64)


POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def hamming_min(h, table_u8):
    """min Hamming distance from hash h to every row of table (n x 8 uint8)."""
    x = np.frombuffer(np.array([h], dtype=np.uint64).tobytes(), dtype=np.uint8)
    return POP[table_u8 ^ x].sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='pandora_dups.txt')
    ap.add_argument('--hamming', type=int, default=6, help='max bits differing (of 64) to call a duplicate')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    wroot = DEFAULT_ROOTS['wikiart']()
    wikiart = list_wikiart(wroot)
    proot, pandora = list_pandora(DEFAULT_ROOTS['pandora']())
    print('WikiArt {} images at {}\nPandora {} images at {}'.format(len(wikiart), wroot, len(pandora), proot))

    # 1) file-name match (WikiArt slugs): cheap first look
    wstems = {}
    for rel, style in wikiart:
        wstems.setdefault(os.path.splitext(os.path.basename(rel))[0].lower(), []).append((rel, style))
    name_hits = [(rel, style) for rel, style in pandora
                 if os.path.splitext(os.path.basename(rel))[0].lower() in wstems]
    print('Pandora images whose file name also exists in WikiArt: {}'.format(len(name_hits)))

    # 2) perceptual hash match
    print('hashing WikiArt ...', flush=True)
    wh = hash_all(wroot, [r for r, _ in wikiart], args.workers)
    print('hashing Pandora ...', flush=True)
    ph = hash_all(proot, [r for r, _ in pandora], args.workers)
    table = np.frombuffer(wh.tobytes(), dtype=np.uint8).reshape(-1, 8)

    dups = []
    label_pairs = {}
    for i, h in enumerate(ph):
        if int(h) == FAILED:
            continue
        dist = hamming_min(h, table)
        j = int(dist.argmin())
        if dist[j] <= args.hamming:
            rel, pstyle = pandora[i]
            wrel, wstyle = wikiart[j]
            dups.append((rel, pstyle, wrel, wstyle, int(dist[j])))
            label_pairs[(pstyle, wstyle)] = label_pairs.get((pstyle, wstyle), 0) + 1

    print('\nPandora images duplicated in WikiArt (Hamming <= {}): {} of {} ({:.1%})'.format(
        args.hamming, len(dups), len(pandora), len(dups) / max(len(pandora), 1)))
    per_style = {}
    for rel, pstyle, *_ in dups:
        per_style[pstyle] = per_style.get(pstyle, 0) + 1
    for s, n in sorted(per_style.items(), key=lambda kv: -kv[1]):
        print('  {:28s} {}'.format(s, n))
    print('\nlabel agreement of the duplicates (Pandora style -> WikiArt style):')
    for (ps, ws), n in sorted(label_pairs.items(), key=lambda kv: -kv[1])[:30]:
        print('  {:28s} -> {:28s} {}'.format(ps, ws, n))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('# Pandora18k images within Hamming {} of a WikiArt image (dedup.py); '
                'format: pandora/<relative path>\n'.format(args.hamming))
        for rel, pstyle, wrel, wstyle, dist in dups:
            f.write('pandora/{}\n'.format(rel))
    with open(os.path.splitext(args.out)[0] + '_pairs.tsv', 'w') as f:
        f.write('pandora_path\tpandora_style\twikiart_path\twikiart_style\thamming\n')
        for row in dups:
            f.write('\t'.join(str(x) for x in row) + '\n')
    print('\nwritten {} ({} lines)'.format(args.out, len(dups)))


if __name__ == '__main__':
    main()
