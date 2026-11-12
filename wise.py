"""WiSE-FT for the text-anchored CLIP classifier: weight-space ensembling of
the zero-shot model and the fine-tuned checkpoint (Wortsman et al., 2022).

    theta(alpha) = (1 - alpha) * theta_zero_shot + alpha * theta_fine_tuned

alpha = 0 is CLIP zero-shot (pretrained image tower, text prototypes as the
classifier), alpha = 1 the fine-tuned model. No training: every alpha is
scored on the validation split, the best one is then scored on the test
split (alpha = 1, the plain fine-tuned model, is always included, so the
choice can only help on val). The label-query attention head keeps its
fine-tuned weights (its gate is ~0, it has no zero-shot counterpart).

Environment (same recipe variables as training, set by server/clip.py):
    INFER_CKPT   fine-tuned checkpoint (last.ckpt)
    WISE_OUT     output folder: wise_results.json (the whole curve),
                 test_results.json + predictions.npz of the chosen alpha in
                 infer.py's format (so mcnemar.py can pair it with others)
    WISE_ALPHAS  comma list, default 0,0.1,...,1
    WISE_TEST    'all' scores every alpha on test too (the curve for a
                 figure); default: only the chosen alpha and alpha = 1
"""
import json
import os

import numpy as np
import torch

import config
import infer
from data_multi import get_multi_loaders
from data_loader import get_data_loaders
from models import WSDAN_MCS

device = infer.device


def interpolate(sd_zs, sd_ft, alpha):
    out = {}
    for k, v_ft in sd_ft.items():
        v_zs = sd_zs.get(k)
        if k.startswith('label_attn.') or v_zs is None or not v_ft.is_floating_point() or v_zs.shape != v_ft.shape:
            out[k] = v_ft
        else:
            out[k] = (1 - alpha) * v_zs.float() + alpha * v_ft.float()
    return out


def score(net, loader, class_names):
    res = infer.evaluate(loader, net, class_names)
    preds = res.pop('_preds')
    return res, preds


def main():
    ckpt = os.environ['INFER_CKPT']
    out_dir = os.environ.get('WISE_OUT', os.path.join(config.save_dir, 'wise'))
    os.makedirs(out_dir, exist_ok=True)
    alphas = [float(a) for a in os.environ.get('WISE_ALPHAS', '0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1').split(',')]
    assert 1.0 in alphas, 'alpha = 1 (the fine-tuned model) must be in WISE_ALPHAS'
    test_all = os.environ.get('WISE_TEST', '') == 'all'

    if infer.MULTI:
        loaders, sizes, class_names, info = get_multi_loaders(batch_size=config.batch_size)
        infer.DS_NAMES, infer.DS_MASKS = info['datasets'], info['masks']
    else:
        loaders, sizes, class_names = get_data_loaders(config.DATA_DIR, batch_size=config.batch_size)
    print('val images: {}, test images: {}'.format(sizes['val'], sizes['test']))

    # zero-shot weights: pretrained image tower + text prototypes (+ scale 100, gate 0)
    net = WSDAN_MCS(num_classes=len(class_names), M=config.num_attentions, net=config.net, pretrained=True,
                    class_names=class_names, num_datasets=len(infer.DS_NAMES) if infer.MULTI else 1)
    assert getattr(net, 'text_init', False) and getattr(net, 'plain', False), 'WiSE-FT needs PLAIN_CLS=1 TEXT_INIT=1'
    sd_zs = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
    checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')
    sd_ft = checkpoint['state_dict']
    missing = [k for k in sd_zs if k not in sd_ft]
    assert not missing, 'checkpoint does not match this model: missing {}'.format(missing[:5])
    epoch = checkpoint.get('logs', {}).get('epoch')
    n_mixed = sum(1 for k in sd_ft if k in sd_zs and not k.startswith('label_attn.') and sd_ft[k].is_floating_point())
    print('fine-tuned checkpoint {} @ epoch {}; {} tensors interpolated'.format(ckpt, epoch, n_mixed))

    infer.USE_TTA = infer.USE_TWO_VIEW = False
    net.to(device)
    curve = {}
    for alpha in alphas:
        net.load_state_dict(interpolate(sd_zs, sd_ft, alpha))
        print('\n=== alpha {:.2f}: validation'.format(alpha))
        val_res, _ = score(net, loaders['val'], class_names)
        curve[alpha] = {'val': val_res['center_crop']}
        print('alpha {:.2f}: val acc {:.2f}  F1 {:.2f}  top5 {:.2f}'.format(
            alpha, 100 * val_res['center_crop']['accuracy'], 100 * val_res['center_crop']['f1_macro'],
            100 * val_res['center_crop']['top5_accuracy']))
    best = max(alphas, key=lambda a: (curve[a]['val']['accuracy'], a))  # ties -> closer to fine-tuned
    print('\n>> alpha chosen on validation: {:.2f} (alpha 1.00 = fine-tuned: {:.2f} vs {:.2f})'.format(
        best, 100 * curve[1.0]['val']['accuracy'], 100 * curve[best]['val']['accuracy']))

    for alpha in (alphas if test_all else sorted({best, 1.0})):
        net.load_state_dict(interpolate(sd_zs, sd_ft, alpha))
        print('\n=== alpha {:.2f}: test'.format(alpha))
        test_res, preds = score(net, loaders['test'], class_names)
        curve[alpha]['test'] = test_res['center_crop']
        if alpha == best:
            test_res['epoch'] = epoch
            test_res['wise_alpha'] = alpha
            with open(os.path.join(out_dir, 'test_results.json'), 'w') as f:
                json.dump(test_res, f, indent=2, default=float)
            np.savez(os.path.join(out_dir, 'predictions.npz'), **preds)

    with open(os.path.join(out_dir, 'wise_results.json'), 'w') as f:
        json.dump({'checkpoint': ckpt, 'epoch': epoch, 'alpha': best,
                   'curve': {str(a): r for a, r in curve.items()}}, f, indent=2, default=float)
    print('\n{:>6s} {:>8s} {:>8s} {:>8s} {:>8s}'.format('alpha', 'val Acc', 'test Acc', 'test F1', 'top5'))
    for a in alphas:
        t = curve[a].get('test')
        print('{:6.2f} {:8.2f} {:>8s} {:>8s} {:>8s}{}'.format(
            a, 100 * curve[a]['val']['accuracy'],
            '{:.2f}'.format(100 * t['accuracy']) if t else '-', '{:.2f}'.format(100 * t['f1_macro']) if t else '-',
            '{:.2f}'.format(100 * t['top5_accuracy']) if t else '-', '  <- chosen' if a == best else ''))
    print('Saved to {}'.format(out_dir))


if __name__ == '__main__':
    main()
