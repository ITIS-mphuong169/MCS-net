"""Multi-view inputs: one global 224 view plus K high-resolution 'detail' crops.

Everything we tried so far that "looked closer" (MCS-Net's attention crop,
the RL crop selection) cut a region out of the 224x224 input tensor and
blew it up: no new pixels. The WikiArt originals have a short side of
about 1000-1400 px, so brushwork that vanishes at 224 is still there in the
file. These transforms hand the model that detail.

VIEWS=single   (default) the usual single 224 view, nothing changes
VIEWS=detail   global view + K crops from the image resized to a short side
               of 256 * DETAIL_MAG: the central (224 * MAG)^2 square, the
               area the global view shows, at MAG times the linear
               resolution, cut into 224 px crops
VIEWS=zoom     control: the same crops cut from the 224 global view
               blown up MAG times (same field of view, no new pixels)

DETAIL_MAG     magnification, default 2 (3 needs a short side of 768 px)
DETAIL_K       crops per training image, random positions + flip (default 2)
DETAIL_K_EVAL  crops per validation image (default 3): fixed positions, the
               first K of GRID_ORDER (centre, then corners, then edges)
DETAIL_K_TEST  crops per test image (default 9: the full 3x3 grid)

A sample becomes a (1 + K, 3, 224, 224) tensor; the model (WSDAN_MCS with
PLAIN_CLS=1) runs the shared backbone on every view and fuses the logits.
"""
import os
import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# order in which the cells of the 3x3 grid are used when fewer than 9 crops
# are asked for at evaluation: centre, the four corners, the four edges
GRID_ORDER = [4, 0, 2, 6, 8, 1, 3, 5, 7]


def grid_offsets(mag, grid=3):
    """(top, left) of the grid x grid crops of 224 px inside a (224 * mag)^2 square."""
    step = (224 * mag - 224) / (grid - 1)
    return [(round(r * step), round(c * step)) for r in range(grid) for c in range(grid)]


class DetailViews:
    """PIL image -> tensor (1 + k, 3, 224, 224): the global view (the ordinary
    transform of that split) followed by k detail crops."""

    def __init__(self, split, k, mag=2, source='hires', base=None, grid=3):
        assert split in ('train', 'val', 'test'), split
        assert source in ('hires', 'zoom'), source
        assert mag >= 2, 'a magnification below 2 shows nothing the global view does not'
        assert split == 'train' or k <= grid * grid, 'at most grid^2 fixed crops'
        self.split, self.k, self.mag, self.source, self.grid, self.base = split, k, mag, source, grid, base
        self.normalize = transforms.Normalize(MEAN, STD)

    def _square(self, img):
        """The (224 * mag)^2 square the detail crops are cut from."""
        if self.source == 'hires':
            return TF.center_crop(TF.resize(img, 256 * self.mag), 224 * self.mag)
        return TF.resize(TF.center_crop(TF.resize(img, 256), 224), 224 * self.mag)

    def _tensor(self, crop, flip=False):
        if flip:
            crop = TF.hflip(crop)
        return self.normalize(TF.to_tensor(crop))

    def __call__(self, img):
        views = [self.base(img)]
        if self.k > 0:
            square, side = self._square(img), 224 * self.mag
            if self.split == 'train':
                for _ in range(self.k):
                    top, left = random.randint(0, side - 224), random.randint(0, side - 224)
                    views.append(self._tensor(TF.crop(square, top, left, 224, 224), flip=random.random() < 0.5))
            else:
                offsets = grid_offsets(self.mag, self.grid)
                for cell in GRID_ORDER[:self.k]:
                    top, left = offsets[cell]
                    views.append(self._tensor(TF.crop(square, top, left, 224, 224)))
        return torch.stack(views)


def view_transforms(base):
    """base: {'train': t, 'val': t, 'test': t} of data_loader.get_transforms();
    returns the same dict, multi-view when VIEWS asks for it."""
    mode = os.environ.get('VIEWS', 'single')
    if mode == 'single':
        return base
    assert mode in ('detail', 'zoom'), 'VIEWS must be single, detail or zoom'
    source = 'hires' if mode == 'detail' else 'zoom'
    mag = int(os.environ.get('DETAIL_MAG', 2))
    ks = {'train': int(os.environ.get('DETAIL_K', 2)), 'val': int(os.environ.get('DETAIL_K_EVAL', 3)),
          'test': int(os.environ.get('DETAIL_K_TEST', 9))}
    print('VIEWS={}: global view + detail crops at x{} (train {}, val {}, test {} per image)'.format(
        mode, mag, ks['train'], ks['val'], ks['test']))
    return {split: DetailViews(split, ks[split], mag=mag, source=source, base=base[split]) for split in base}
