"""Multi-dataset art-style training (step 2: one model over several style
datasets with overlapping labels, cf. "An Erudite FGVC Model", CVPR 2023).

DATASETS=wikiart            single dataset, the default (same split as before)
DATASETS=pandora            Pandora18k alone
DATASETS=wikiart,pandora    joint training

Every dataset is split 80/10/10 with the same seed-1231 shuffle as
data_loader.py, independently of the others, so WikiArt's test set is the
one all earlier runs were measured on, and Pandora's split is the same
whether it is trained alone or jointly.

LABEL_SPACE=union     (default) one class per style name: WikiArt's and
                      Pandora's 'Impressionism' are the same class, so the
                      two datasets pool their images; styles only one
                      dataset has keep their own class (30 classes for
                      wikiart+pandora: 27 + Byzantine_Iconography,
                      Surrealism, Abstract_Art).
LABEL_SPACE=separate  Erudite-style: a class per (dataset, style), i.e. a
                      private head per dataset on a shared backbone (45).

Each sample carries its dataset index. The loss and the predictions are
restricted to the classes that exist in the sample's own dataset
(dataset_masks): a Pandora 'Cubism' image is never asked to choose between
WikiArt's Analytical and Synthetic Cubism, which it was never labelled
for. Accuracy is therefore always measured inside each dataset's own label
space, comparable to a model trained on that dataset alone.

EXCLUDE_LIST=file     relative image paths to drop (one per line, format
                      '<dataset>/<relative path>'), produced by dedup.py:
                      Pandora18k reuses WikiArt images, which would leak
                      WikiArt training images into Pandora's test set.
"""
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets

from data_loader import get_transforms, safe_pil_loader

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Pandora18k top-level folders -> WikiArt-style class names (style names that
# WikiArt also has must spell exactly like its folders, so 'union' merges them)
PANDORA_MAP = {
    '01_Byzantin_Iconography': 'Byzantine_Iconography',
    '02_Early_Renaissance': 'Early_Renaissance',
    '03_Northern_Renaissance': 'Northern_Renaissance',
    '04_High_Renaissance': 'High_Renaissance',
    '05_Baroque': 'Baroque',
    '06_Rococo': 'Rococo',
    '07_Romanticism': 'Romanticism',
    '08_Realism': 'Realism',
    '09_Impressionism': 'Impressionism',
    '10_Post_Impressionism': 'Post_Impressionism',
    '11_Expressionism': 'Expressionism',
    '12_Symbolism': 'Symbolism',
    '13_Fauvism': 'Fauvism',
    '14_Cubism': 'Cubism',
    '15_Surrealism': 'Surrealism',
    '16_AbstractArt': 'Abstract_Art',
    '17_NaiveArt': 'Naive_Art_Primitivism',
    '18_PopArt': 'Pop_Art',
}

DEFAULT_ROOTS = {
    'wikiart': lambda: __import__('config').DATA_DIR,
    'pandora': lambda: os.environ.get('DATA_DIR_PANDORA', '/kaggle/input/pandora18k'),
}


def find_root(path, marker):
    """`path` or the first directory below it that contains `marker`
    (Kaggle mounts an uploaded folder one or two levels deep)."""
    for cur, dirs, _ in os.walk(path):
        if marker in dirs:
            return cur
        if cur.count(os.sep) - path.count(os.sep) >= 3:
            dirs[:] = []
    raise FileNotFoundError('{} not found under {}'.format(marker, path))


def list_wikiart(root):
    """(relative path, style) in torchvision ImageFolder order - the order
    data_loader.py's split indices refer to."""
    folder = datasets.ImageFolder(root)
    return [(os.path.relpath(p, root), folder.classes[c]) for p, c in folder.samples]


def list_pandora(root):
    root = find_root(root, '01_Byzantin_Iconography')
    samples = []
    for top in sorted(os.listdir(root)):
        if top not in PANDORA_MAP:
            continue
        style = PANDORA_MAP[top]
        for cur, dirs, files in os.walk(os.path.join(root, top)):
            dirs.sort()
            for f in sorted(files):
                if f.lower().endswith(IMG_EXTS):
                    samples.append((os.path.relpath(os.path.join(cur, f), root), style))
    return root, samples


LISTERS = {
    'wikiart': lambda root: (root, list_wikiart(root)),
    'pandora': list_pandora,
}


def load_exclude(path):
    excl = set()
    if path and os.path.isfile(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    excl.add(line)
    return excl


def seed_split(n, seed=1231, val_split=0.1, test_split=0.1):
    """Same shuffle as data_loader.get_data_loaders."""
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    num_val, num_test = int(n * val_split), int(n * test_split)
    return {'val': indices[:num_val], 'test': indices[num_val:num_val + num_test],
            'train': indices[num_val + num_test:]}


class StyleDataset(Dataset):
    """Images of one source dataset. Returns (image, class index in the
    unified label space, dataset index)."""

    def __init__(self, root, items, transform, ds_index):
        self.root, self.items, self.transform, self.ds_index = root, items, transform, ds_index

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        img = safe_pil_loader(os.path.join(self.root, rel))
        return self.transform(img), label, self.ds_index


def dataset_names():
    return [d.strip() for d in os.environ.get('DATASETS', 'wikiart').split(',') if d.strip()]


def get_multi_loaders(batch_size=32, num_workers=4, seed=1231):
    """Returns (dataloaders, dataset_sizes, class_names, info) where info has
    'datasets' (names), 'masks' (num_datasets x num_classes bool tensor) and
    'per_dataset_sizes'. class_names are unified names; in 'separate' mode
    they are '<style>|<dataset>' (text_head strips the suffix)."""
    names = dataset_names()
    label_space = os.environ.get('LABEL_SPACE', 'union')
    assert label_space in ('union', 'separate'), label_space
    exclude = load_exclude(os.environ.get('EXCLUDE_LIST'))
    transforms_ = get_transforms()

    listed = []  # (name, root, [(rel, style)])
    for name in names:
        root, samples = LISTERS[name](DEFAULT_ROOTS[name]())
        before = len(samples)
        samples = [(rel, s) for rel, s in samples if '{}/{}'.format(name, rel) not in exclude]
        if before != len(samples):
            print('{}: dropped {} excluded images (EXCLUDE_LIST)'.format(name, before - len(samples)))
        listed.append((name, root, samples))

    def cname(style, name):
        return style if label_space == 'union' else '{}|{}'.format(style, name)

    class_names = sorted({cname(s, name) for name, _, samples in listed for _, s in samples})
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    masks = torch.zeros(len(names), len(class_names), dtype=torch.bool)

    splits = {'train': [], 'val': [], 'test': []}
    per_dataset_sizes = {}
    for di, (name, root, samples) in enumerate(listed):
        items = [(rel, class_to_idx[cname(s, name)]) for rel, s in samples]
        for _, c in items:
            masks[di, c] = True
        idx = seed_split(len(items), seed)
        per_dataset_sizes[name] = {k: len(v) for k, v in idx.items()}
        for split in splits:
            splits[split].append(StyleDataset(root, [items[i] for i in idx[split]], transforms_[split], di))
        print('{}: {} images, {} classes, train/val/test {}/{}/{}'.format(
            name, len(items), int(masks[di].sum()), *[len(idx[k]) for k in ('train', 'val', 'test')]))

    image_datasets = {k: ConcatDataset(v) for k, v in splits.items()}
    dataloaders = {
        k: DataLoader(ds, batch_size=batch_size, shuffle=(k == 'train'), num_workers=num_workers,
                      pin_memory=True, drop_last=(k == 'train'), persistent_workers=num_workers > 0)
        for k, ds in image_datasets.items()}
    dataset_sizes = {k: len(ds) for k, ds in image_datasets.items()}
    info = {'datasets': names, 'masks': masks, 'label_space': label_space, 'per_dataset_sizes': per_dataset_sizes}
    print('label space: {} ({} classes)'.format(label_space, len(class_names)))
    return dataloaders, dataset_sizes, class_names, info


def unpack(batch):
    """(X, y) or (X, y, d) -> X, y, d (d None for the single-dataset loader)."""
    if len(batch) == 2:
        return batch[0], batch[1], None
    return batch


def mask_logits(logits, d, masks):
    """Logits of classes absent from sample i's dataset -> very negative, so
    CE and argmax only range over that dataset's own label space."""
    if d is None or masks is None:
        return logits
    allowed = masks.to(logits.device)[d]
    return logits.masked_fill(~allowed, -1e4)
