"""Evaluate a trained checkpoint on the held-out test split.

Usage (Kaggle):
    INFER_CKPT=/path/to/model.ckpt python infer.py

INFER_CKPT defaults to config.save_dir + config.model_name (the best-val
checkpoint). Uses the same seed-1231 split as train.py, so the test images
were never seen during training or checkpoint selection. Test-time
augmentation (flip + 3 attention crops) follows the original author's code;
results are reported both with it and with a single center crop (the
protocol the paper describes).
"""
import os
import json
import time
import config
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from models import WSDAN_MCS
from utils import TopKAccuracyMetric, batch_augment
from data_loader import get_data_loaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# GPU settings
assert torch.cuda.is_available()
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

DATA_DIR = "/kaggle/input/datasets/steubk/wikiart"


def main():
    ckpt = os.environ.get('INFER_CKPT', config.save_dir + config.model_name)
    out_dir = os.environ.get('INFER_OUT', config.save_dir)
    os.makedirs(out_dir, exist_ok=True)

    dataloaders, dataset_sizes, class_names = get_data_loaders(DATA_DIR, batch_size=config.batch_size)
    test_loader = dataloaders['test']
    print('Test images: {}'.format(dataset_sizes['test']))

    net = WSDAN_MCS(num_classes=len(class_names), M=config.num_attentions, net=config.net, pretrained=False)
    checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')
    state_dict = checkpoint['state_dict']
    # WSDAN_MCS.load_state_dict silently skips mismatched keys, so check
    # here that the checkpoint really matches this branch's model.
    missing = [k for k in net.state_dict() if k not in state_dict]
    assert not missing, 'Checkpoint does not match this model (wrong branch?): missing {}'.format(missing[:5])
    net.load_state_dict(state_dict)
    ckpt_epoch = checkpoint.get('logs', {}).get('epoch')
    print('Network loaded from {} @ epoch {}'.format(ckpt, ckpt_epoch))

    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    results = evaluate(test_loader, net, class_names)
    results.update({'checkpoint': ckpt, 'epoch': ckpt_epoch, 'branch': config.tag})

    out_path = os.path.join(out_dir, 'test_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved to {}'.format(out_path))


def evaluate(data_loader, net, class_names):
    """One pass, two protocols: 'center_crop' (single center-cropped view,
    as the paper's sec. 3.3 describes) and 'tta' (the original author code's
    flip + 3 attention-crop averaging)."""
    preds = {'center_crop': [], 'tta': []}
    top5 = {k: TopKAccuracyMetric(topk=(1, 5)) for k in preds}
    all_labels = []

    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for X, y in tqdm(data_loader, unit=' batches'):
            X = X.to(device)
            y = y.to(device)
            X_m = torch.flip(X, [3])

            # model returns (p, p_effect, feature_matrix, attention_map, h)
            y_pred_raw, _, _, attention_map, _ = net(X)
            y_pred_raw_m, _, _, attention_map_m, _ = net(X_m)

            views, views_m = [y_pred_raw], [y_pred_raw_m]
            for theta, pad in ((0.3, 0.1), (0.2, 0.1), (0.1, 0.05)):
                views.append(net(batch_augment(X, attention_map, mode='crop', theta=theta, padding_ratio=pad))[0])
                views_m.append(net(batch_augment(X_m, attention_map_m, mode='crop', theta=theta, padding_ratio=pad))[0])
            y_pred_tta = (torch.stack(views).mean(0) + torch.stack(views_m).mean(0)) / 2.

            for name, y_pred in (('center_crop', y_pred_raw), ('tta', y_pred_tta)):
                top5[name](y_pred, y)
                preds[name].extend(y_pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print('\nTime {:.0f}s, {} test images'.format(time.time() - start_time, len(all_labels)))
    labels = list(range(len(class_names)))
    results = {'num_test_images': len(all_labels)}
    for name, p in preds.items():
        r = {
            'accuracy': accuracy_score(all_labels, p),
            'top5_accuracy': top5[name].corrects[1] / top5[name].num_samples,
            'precision_macro': precision_score(all_labels, p, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, p, average='macro', zero_division=0),
            'f1_macro': f1_score(all_labels, p, average='macro', zero_division=0),
            'precision_weighted': precision_score(all_labels, p, average='weighted', zero_division=0),
            'recall_weighted': recall_score(all_labels, p, average='weighted', zero_division=0),
            'f1_weighted': f1_score(all_labels, p, average='weighted', zero_division=0),
        }
        print('\n=== {} ==='.format(name))
        for k, v in r.items():
            print('{:20s} {:.4f}'.format(k, v))
        r['per_class'] = classification_report(all_labels, p, labels=labels, target_names=class_names,
                                               output_dict=True, zero_division=0)
        results[name] = r

    print('\n=== per-class report (center_crop) ===')
    print(classification_report(all_labels, preds['center_crop'], labels=labels,
                                target_names=class_names, digits=4, zero_division=0))
    return results


if __name__ == '__main__':
    main()
