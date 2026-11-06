import os
# Must be set before torch initializes its CUDA allocator. The AGM module
# (channel/spatial correlation attention) pushes GPU memory right up to
# the limit - confirmed OOM on Kaggle with "553.64 MiB reserved by
# PyTorch but unallocated" (fragmentation) while only ~98MB more was
# needed, so this directly addresses that per PyTorch's own suggestion.
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
import config
import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import random
from models import WSDAN_MCS
from models.mcs import StyleContrastiveLoss
from utils import AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment
import math
import numpy as np
import argparse
import wandb
from data_loader import get_data_loaders
from data_multi import get_multi_loaders, unpack, mask_logits
# import thop

# DATASETS=wikiart,pandora (see data_multi.py): several style datasets in one
# model. Each sample carries its dataset index; logits are masked to that
# dataset's own classes (DS_MASKS) in the loss and in the accuracy.
MULTI = 'DATASETS' in os.environ
DS_NAMES, DS_MASKS = None, None

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
# DEVICE=cpu: end-to-end smoke tests without a GPU (use with AMP=0)
device = torch.device(os.environ.get('DEVICE', 'cuda'))
torch.backends.cudnn.benchmark = True

# General loss functions - matches the paper's eq. 17 (L = Lcls + l1*Lcon +
# l2*Lcausal); no center_loss, the paper's loss doesn't have one
cross_entropy_loss = nn.CrossEntropyLoss()
# LABEL_SMOOTH=e: label smoothing on the training CE losses (validation and
# test keep plain CE, so val/loss stays comparable); 0 = off
LABEL_SMOOTH = float(os.environ.get('LABEL_SMOOTH', 0))
train_ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
contrastive_loss = StyleContrastiveLoss(temperature=config.tau)

# loss and metric
loss_container = AverageMeter(name='loss')
top1_container = AverageMeter(name='top1')
top5_container = AverageMeter(name='top5')
raw_metric = TopKAccuracyMetric(topk=(1, 5))
crop_metric = TopKAccuracyMetric(topk=(1, 5))
drop_metric = TopKAccuracyMetric(topk=(1, 5))

best_acc = 0.0
_wandb_batch_log_warned = False


# 固定随机种子
def seed_torch(seed=1231):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Ablation switches (all default to the normal full-paper run):
#   ABLATION_TAG    - separate save_dir and W&B run, so ablations never touch
#                     the main run's checkpoints or charts
#   TRAIN_FRACTION  - train on this fraction of the train split (val is full)
#   LAMBDA1         - override config.lambda1 (SCLM contrastive weight)
#   AUG_CE_WEIGHT   - weight of CE on the attention crop/drop images, the
#                     WS-DAN term dropped when the loss was matched to eq. 17
ABLATION_TAG = os.environ.get('ABLATION_TAG', '')
# SEED changes model init and data order only; the train/val/test split
# keeps its own fixed seed (data_loader.py), so every seed sees the same images
SEED = int(os.environ.get('SEED', 1231))
# Memory savers for small GPUs (both off by default):
#   AMP=1         - autocast for forward passes (losses stay float32)
#   AMP_DTYPE     - bf16 (default; Ampere+) or fp16 (Turing/T4: bf16 has no
#                   tensor-core support there; fp16 adds a GradScaler)
#   GRAD_ACCUM=N  - step the optimizer every N batches, so the effective batch
#                   is BATCH_SIZE * N (e.g. 16 x 2 = the paper's 32)
AMP = os.environ.get('AMP', '0') != '0'
AMP_DTYPE = torch.float16 if os.environ.get('AMP_DTYPE', 'bf16') == 'fp16' else torch.bfloat16
scaler = torch.amp.GradScaler('cuda', enabled=AMP and AMP_DTYPE == torch.float16)
GRAD_ACCUM = max(1, int(os.environ.get('GRAD_ACCUM', 1)))
# PLAIN_CLS=1 (model side, see WSDAN_MCS.forward_plain): backbone + linear
# classifier only. One forward pass per step, CE loss only: no attention
# crop/drop, no CCAM, no SCLM. The reference point for a new backbone.
PLAIN = os.environ.get('PLAIN_CLS', '0') != '0'
# Fine-tuning a pretrained transformer backbone (OPTIM=adamw):
#   OPTIM             sgd (paper: momentum 0.9) or adamw
#   BACKBONE_LR_MULT  lr multiplier for the backbone parameters (net.features);
#                     the heads keep config.learning_rate. A pretrained CLIP
#                     tower is destroyed by the head's lr, hence e.g. 0.01
#   SCHED             step (paper: x0.9 every 2 epochs) or cosine (per
#                     iteration, with WARMUP_EPOCHS of linear warm-up)
OPTIM = os.environ.get('OPTIM', 'sgd')
BACKBONE_LR_MULT = float(os.environ.get('BACKBONE_LR_MULT', 1.0))
SCHED = os.environ.get('SCHED', 'step')
WARMUP_EPOCHS = float(os.environ.get('WARMUP_EPOCHS', 1.0))
# RL_SELECT=1 (model side, see WSDAN_MCS.policy): REINFORCE on the crop-region
# policy. Reward = log p(true class | crop) - log p(true class | raw image),
# minus a moving-average baseline; RL_WEIGHT scales the policy loss and
# RL_ENTROPY adds an entropy bonus so the policy keeps exploring regions.
# The crop is also used at test time (infer.py 'two_view': raw + crop logits
# averaged), so the reward is measured on that fused prediction. The policy
# gets its own learning rate, RL_LR_MULT x the backbone lr (at the shared
# 1e-3 it stayed almost uniform over the 32 regions).
RL_WEIGHT = float(os.environ.get('RL_WEIGHT', 1.0))
RL_ENTROPY = float(os.environ.get('RL_ENTROPY', 0.01))
RL_LR_MULT = float(os.environ.get('RL_LR_MULT', 10))
rl_baseline = None
# WEIGHT_DECAY: SGD weight decay (paper Table 2: 1e-5)
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', 1e-5))
TRAIN_FRACTION = float(os.environ.get('TRAIN_FRACTION', 1.0))
# VIEWS=detail|zoom (data_views.py): the loss is CE of the fused logits plus
# VIEW_CE_WEIGHT x the mean CE of every view on its own, so each view has to
# be informative by itself (what test-time averaging relies on)
VIEW_CE_WEIGHT = float(os.environ.get('VIEW_CE_WEIGHT', 0.5))
AUG_CE_WEIGHT = float(os.environ.get('AUG_CE_WEIGHT', 0.0))
if 'LAMBDA1' in os.environ:
    config.lambda1 = float(os.environ['LAMBDA1'])
#   EPOCHS          - override config.epochs (e.g. 32 to fine-tune past 30)
#   FINETUNE_FROM   - start from this trained checkpoint with a fresh optimizer
#                     (lr FT_LR); with MLA_RANK > 0 its MHA weights are
#                     converted to latent attention by SVD (MHA2MLA)
if 'EPOCHS' in os.environ:
    config.epochs = int(os.environ['EPOCHS'])
FINETUNE_FROM = os.environ.get('FINETUNE_FROM', '')
FT_LR = float(os.environ.get('FT_LR', 5e-4))
if ABLATION_TAG:
    config.save_dir = config.save_dir.rstrip('/') + '_abl_{}/'.format(ABLATION_TAG)
    config.ckpt = config.save_dir + config.model_name


def main():
    global best_acc
    seed_torch(SEED)
    # Logging setting
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    # Fixed run id + resume="allow" so re-running this script in a later
    # Kaggle session (to continue from a checkpoint) appends to the same
    # W&B run instead of starting a new one.
    net_id = config.net.replace('.', '-')  # timm names carry a '.tag'; W&B ids can't
    wandb.init(
        project="mcs-net-wikiart",
        id="{}-{}-abl-{}".format(config.tag, net_id, ABLATION_TAG) if ABLATION_TAG
        else "{}-{}-v2".format(config.tag, net_id),
        resume="allow",
        config={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "net": config.net,
            "plain_cls": PLAIN,
            "text_init": os.environ.get("TEXT_INIT", "0") != "0",
            "text_freeze": os.environ.get("TEXT_FREEZE", "0") != "0",
            "optim": OPTIM,
            "backbone_lr_mult": BACKBONE_LR_MULT,
            "sched": SCHED,
            "warmup_epochs": WARMUP_EPOCHS if SCHED == 'cosine' else None,
            "amp_dtype": 'fp16' if AMP_DTYPE == torch.float16 else 'bf16',
            "num_attentions": config.num_attentions,
            "lambda1": config.lambda1,
            "lambda2": config.lambda2,
            "tau": config.tau,
            "ablation_tag": ABLATION_TAG,
            "seed": SEED,
            "rl_select": os.environ.get("RL_SELECT", "0") != "0",
            "rl_weight": RL_WEIGHT,
            "rl_lr_mult": RL_LR_MULT,
            "amp": AMP,
            "grad_accum": GRAD_ACCUM,
            "global_cls": os.environ.get("GLOBAL_CLS", "0") != "0",
            "part_residual": os.environ.get("PART_RESIDUAL", "0") != "0",
            "part_pos": os.environ.get("PART_POS", "0") != "0",
            "part_attn": config.PART_ATTN,
            "aug_rotate": float(os.environ.get("AUG_ROTATE", 0)),
            "feat_noise": float(os.environ.get("FEAT_NOISE", 0)),
            "fc_dropout": float(os.environ.get("FC_DROPOUT", 0)),
            "label_smooth": LABEL_SMOOTH,
            "weight_decay": WEIGHT_DECAY,
            "train_fraction": TRAIN_FRACTION,
            "aug_ce_weight": AUG_CE_WEIGHT,
            "mla_rank": int(os.environ.get('MLA_RANK', 0)),
            "finetune_from": FINETUNE_FROM,
            "ft_lr": FT_LR if FINETUNE_FROM else None,
            "datasets": os.environ.get('DATASETS', 'wikiart'),
            "label_space": os.environ.get('LABEL_SPACE', 'union') if MULTI else None,
            "exclude_list": os.path.basename(os.environ.get('EXCLUDE_LIST', '')) or None,
            "views": os.environ.get('VIEWS', 'single'),
            "detail_mag": int(os.environ.get('DETAIL_MAG', 2)),
            "detail_k": int(os.environ.get('DETAIL_K', 2)),
            "view_ce_weight": VIEW_CE_WEIGHT,
            "drop_path": float(os.environ.get('DROP_PATH', 0)),
            "label_attn": os.environ.get('LABEL_ATTN', '0') != '0',
            "label_attn_heads": int(os.environ.get('LABEL_ATTN_HEADS', 8)),
        },
    )

    # load dataset
    # train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)
    # train_loader, validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
    #                                            num_workers=config.workers, pin_memory=True, drop_last=True), \
    #                                 DataLoader(validate_dataset, batch_size=config.batch_size * 4, shuffle=False,
    #                                            num_workers=config.workers, pin_memory=True, drop_last=True)
    # num_classes = train_dataset.num_classes

    global DS_NAMES, DS_MASKS
    if MULTI:
        assert PLAIN, 'multi-dataset training is implemented for PLAIN_CLS=1 only'
        dataloaders, dataset_sizes, class_names, info = get_multi_loaders(batch_size=config.batch_size,
                                                                          num_workers=config.workers)
        DS_NAMES, DS_MASKS = info['datasets'], info['masks']
    else:
        dataloaders, dataset_sizes, class_names = get_data_loaders(config.DATA_DIR,
                                                                   batch_size=config.batch_size,
                                                                   num_workers=config.workers)
    train_loader = dataloaders['train']
    validate_loader = dataloaders['val']
    if TRAIN_FRACTION < 1.0:
        # the split is already seed-shuffled, so the first k images are a
        # random sample - identical across ablation runs
        train_subset = Subset(train_loader.dataset, range(int(len(train_loader.dataset) * TRAIN_FRACTION)))
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.workers, pin_memory=True, drop_last=True)
    print('Train images: {}, val images: {}'.format(len(train_loader.dataset), len(validate_loader.dataset)))
    num_classes = len(class_names)

    # Initialize model
    logs = {}
    start_epoch = 0
    net = WSDAN_MCS(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True,
                    class_names=class_names, num_datasets=len(DS_NAMES) if MULTI else 1)

    # Resume priority: RESUME_CKPT env var > last.ckpt (written every
    # epoch) > config.ckpt (best-only). model.ckpt is only rewritten when
    # val acc improves, so resuming from it can silently redo epochs.
    last_ckpt = os.path.join(config.save_dir, 'last.ckpt')
    if os.environ.get('RESUME_CKPT'):
        config.ckpt = os.environ['RESUME_CKPT']
    elif os.path.isfile(last_ckpt):
        config.ckpt = last_ckpt
    elif FINETUNE_FROM:
        # First session of a fine-tune: take the weights only. Optimizer,
        # scheduler and best-score start fresh (config.ckpt = None below).
        source = torch.load(FINETUNE_FROM, weights_only=False, map_location='cpu')
        state_dict = source['state_dict']
        if net.mla_rank > 0:
            from models.mcs import convert_mha_to_mla
            state_dict, energy = convert_mha_to_mla(state_dict, net.mla_rank)
            print('MHA -> MLA rank {}: SVD keeps {:.2%} of [W_k; W_v] energy'.format(net.mla_rank, energy))
        missing = [k for k in net.state_dict() if k not in state_dict]
        assert not missing, 'FINETUNE_FROM does not match this model: missing {}'.format(missing[:5])
        net.load_state_dict(state_dict)
        start_epoch = int(source['logs']['epoch'])
        config.ckpt = None
        config.learning_rate = FT_LR
        print('Fine-tuning from {} @ epoch {}, lr {}'.format(FINETUNE_FROM, start_epoch, FT_LR))
    if config.ckpt and os.path.isfile(config.ckpt):
        # Load ckpt and get state_dict
        # weights_only=False: PyTorch >=2.6 defaults to True, which
        # rejects the numpy arrays stored in our own 'logs' dict; safe
        # here since this checkpoint is one we wrote ourselves.
        checkpoint = torch.load(config.ckpt, weights_only=False)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])  # start from the beginning

        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(config.ckpt))
        print('Network loaded from {} @ {} epoch'.format(config.ckpt, start_epoch))

    logging.info('Network weights save to {}'.format(config.save_dir))

    # Re-seed per starting epoch: with one fixed seed, every resumed session
    # replayed the exact same shuffled + augmented images in its first epoch
    # (seen as an inflated train acc right after each resume). From-scratch
    # runs start at epoch 0, so their randomness is unchanged.
    if start_epoch > 0:
        seed_torch(SEED + start_epoch)

    # Use cuda
    net.to(device)
    if getattr(net, 'text_init', False) and start_epoch == 0:
        zero_shot_check(net, validate_loader)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    # # 计算FLOPs和参数量
    # input_sample = torch.randn(1, 3, 224, 224).to(device)  # 创建一个虚拟输入样本
    # flops, params = thop.profile(net, inputs=(input_sample,))
    # flops_G = flops / 1e9
    # params_M = params / 1e6
    # print(f"模型FLOPS: {flops_G:.2f}G")
    # print(f"模型参数量: {params_M:.2f}M")

    # Optimizer
    learning_rate = config.learning_rate
    # print('begin with ', learning_rate, ' learning rate')
    model = getattr(net, 'module', net)
    policy = getattr(model, 'policy', None)
    policy_ids = {id(p) for p in policy.parameters()} if policy is not None else set()
    backbone_ids = {id(p) for p in model.features.parameters()} if BACKBONE_LR_MULT != 1.0 else set()
    params = [{'params': [p for p in net.parameters() if id(p) not in policy_ids | backbone_ids]}]
    if backbone_ids:
        params.append({'params': list(model.features.parameters()), 'lr': learning_rate * BACKBONE_LR_MULT})
    if policy is not None:
        params.append({'params': list(policy.parameters()), 'lr': learning_rate * RL_LR_MULT})
    if OPTIM == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=WEIGHT_DECAY)
    if SCHED == 'cosine':
        # stepped once per optimizer step (train()), not per epoch
        total = config.epochs * len(train_loader) // GRAD_ACCUM
        warmup = int(WARMUP_EPOCHS * len(train_loader) // GRAD_ACCUM)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / warmup if s < warmup
                                                      else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    print('optimizer {} lr {:g} (backbone x{:g}), schedule {}, weight decay {:g}, plain {}'.format(
        OPTIM, learning_rate, BACKBONE_LR_MULT, SCHED, WEIGHT_DECAY, PLAIN))

    # Restore optimizer/scheduler state on resume (momentum buffers, LR
    # decay progress) - checkpoints from before this fix won't have these
    # keys, so this is a no-op for them (falls back to fresh state, same
    # as before).
    if config.ckpt and os.path.isfile(config.ckpt):
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        if 'scaler_state' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state'])

    # Model checkpoint
    callback_monitor = 'val_{}'.format(raw_metric.name)
    callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                               monitor=callback_monitor,
                               mode='max')
    if config.ckpt and os.path.isfile(config.ckpt) and 'best_score' in checkpoint:
        callback.set_best_score(checkpoint['best_score'])
        best_acc = checkpoint.get('best_acc', best_acc)
    elif callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()

    # logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
    #              format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
    # logging.info('')

    # Kaggle commit runs have a hard wall-clock limit and 100 epochs
    # won't fit in one. SESSION_EPOCH_LIMIT caps how many epochs this
    # process runs before exiting cleanly (exit code 0, valid output),
    # so the next commit can resume from the checkpoint instead of the
    # whole run getting cut off mid-epoch with output at risk.
    session_limit = int(os.environ.get('SESSION_EPOCH_LIMIT', config.epochs))
    end_epoch = min(config.epochs, start_epoch + session_limit)
    if end_epoch <= start_epoch:
        print('Nothing to do: start_epoch={} >= end_epoch={}'.format(start_epoch, end_epoch))
        return

    for epoch in range(start_epoch, end_epoch):
        callback.on_epoch_begin()
        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']
        # print('current lr =', optimizer.param_groups[0]['lr'])
        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))

        train(epoch=epoch,
              logs=logs,
              data_loader=train_loader,
              net=net,
              optimizer=optimizer,
              scheduler=scheduler,
              pbar=pbar)

        validate(logs=logs,
                 data_loader=validate_loader,
                 net=net,
                 pbar=pbar,
                 epoch=epoch)

        try:
            wandb.log({
                "lr": logs['lr'],
                "train/loss": logs['train_loss'],
                "train/raw_top1": logs['train_raw_topk_accuracy'][0],
                "train/raw_top5": logs['train_raw_topk_accuracy'][1],
                "train/crop_top1": logs['train_crop_topk_accuracy'][0],
                "train/crop_top5": logs['train_crop_topk_accuracy'][1],
                "train/drop_top1": logs['train_drop_topk_accuracy'][0],
                "train/drop_top5": logs['train_drop_topk_accuracy'][1],
                "val/loss": logs['val_loss'],
                "val/top1": logs['val_topk_accuracy'][0],
                "val/top5": logs['val_topk_accuracy'][1],
                "val/best_acc": best_acc,
                **({'train/attn_gate': float(getattr(net, 'module', net).label_attn.gate)}
                   if getattr(getattr(net, 'module', net), 'label_attn', None) is not None else {}),
                **({'train/view_w_global': float(torch.sigmoid(getattr(net, 'module', net).view_logit))}
                   if hasattr(getattr(net, 'module', net), 'view_logit') else {}),
                "epoch": logs['epoch'],
                **{'val/top1_{}'.format(k): v for k, v in logs.get('val_per_dataset', {}).items()},
            }, step=(epoch + 1) * len(train_loader))
            print('wandb.log OK for epoch {}'.format(logs['epoch']), flush=True)
        except Exception as e:
            print('wandb.log FAILED for epoch {}: {!r}'.format(logs['epoch'], e), flush=True)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val_loss'])
        elif SCHED != 'cosine':
            scheduler.step()
        callback.on_epoch_end(logs, net, optimizer=optimizer, scheduler=scheduler)

        # Always save the latest epoch so the next session resumes exactly
        # here (the best-only model.ckpt may lag several epochs behind).
        state_dict = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
        torch.save({
            'logs': logs,
            'state_dict': {k: v.cpu() for k, v in state_dict.items()},
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_score': callback.best_score,
            'best_acc': best_acc,
        }, last_ckpt)
        pbar.close()

    wandb.finish()


@torch.no_grad()
def zero_shot_check(net, data_loader, batches=32):
    """TEXT_INIT sanity check before any training: accuracy of the untrained
    text-initialised head on the first val batches. CLIP zero-shot on art
    styles should land well above chance (1/27 = 3.7%); near chance means
    the image and text towers are not in the same space."""
    net.eval()
    correct = total = 0
    for i, batch in enumerate(data_loader):
        if i >= batches:
            break
        X, y, d = unpack(batch)
        with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=AMP):
            p = (net(X.to(device)) if d is None else net(X.to(device), d.to(device)))[0]
        p = mask_logits(p.float(), None if d is None else d.to(p.device), DS_MASKS)
        correct += (p.argmax(1).cpu() == y).sum().item()
        total += len(y)
    print('zero-shot (text prototypes, no training): {:.2f}% on {} val images'.format(100. * correct / total, total))
    return correct / total


def adjust_learning(optimizer, epoch, iter):
    """Decay the learning rate based on schedule"""
    base_lr = config.learning_rate
    base_rate = 0.9
    base_duration = 2.0
    lr = base_lr * pow(base_rate, (epoch + iter) / base_duration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(**kwargs):
    # Retrieve training configuration
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    optimizer = kwargs['optimizer']
    scheduler = kwargs['scheduler']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()

    # begin training
    start_time = time.time()
    net.train()
    batch_len = len(data_loader)
    batch_info = ''
    for i, batch in enumerate(data_loader):
        # float_iter = float(i) / batch_len
        # adjust_learning(optimizer, epoch, float_iter)
        # now_lr = optimizer.param_groups[0]['lr']

        if i % GRAD_ACCUM == 0:
            optimizer.zero_grad()

        # obtain data for training
        X, y, d = unpack(batch)
        X = X.to(device)
        y = y.to(device)
        d = None if d is None else d.to(device)

        # raw image
        with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=AMP):
            y_pred_raw, y_pred_aux, feature_matrix, attention_map, h = net(X) if d is None else net(X, d)
        y_pred_raw, y_pred_aux, attention_map, h = y_pred_raw.float(), y_pred_aux.float(), attention_map.float(), h.float()
        y_pred_raw, y_pred_aux = mask_logits(y_pred_raw, d, DS_MASKS), mask_logits(y_pred_aux, d, DS_MASKS)
        # crop-policy outputs of this (raw-image) pass; the aug-image pass below overwrites them
        model = getattr(net, 'module', net)
        rl_logp, rl_entropy = getattr(model, 'rl_logp', None), getattr(model, 'rl_entropy', None)
        rl_reward = None

        if PLAIN:
            # plain classifier: CE on the raw image only (the crop/drop/aux
            # metrics below just mirror the raw prediction)
            batch_loss = train_ce(y_pred_raw, y)
            if y_pred_aux.dim() == 3:  # multi-view: also supervise every view on its own
                batch_loss = batch_loss + VIEW_CE_WEIGHT * train_ce(y_pred_aux.flatten(0, 1),
                                                                    y.repeat_interleave(y_pred_aux.size(1)))
            y_pred_aug, y_aug, y_pred_aux, y_aux = y_pred_raw, y, y_pred_raw, y
        else:
            ##################################
            # Attention Cropping
            ##################################
            with torch.no_grad():
                crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
                drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
            aug_images = torch.cat([crop_images, drop_images], dim=0)
            y_aug = torch.cat([y, y], dim=0)

            # crop images forward
            with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=AMP):
                y_pred_aug, y_pred_aux_aug, _, _, _ = net(aug_images)
            y_pred_aug, y_pred_aux_aug = y_pred_aug.float(), y_pred_aux_aug.float()

            y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
            y_aux = torch.cat([y, y_aug], dim=0)

            # loss - matches the paper's eq. 17 structure (L = Lcls + l1*Lcon +
            # l2*Lcausal). RCAL (the relation-counterfactual extension
            # previously added as a 4th term here) is dropped - confirmed on
            # Kaggle to hurt accuracy regardless of which other modules were
            # present, in a dedicated A/B comparison.
            Lcls = train_ce(y_pred_raw, y)
            Lcausal = train_ce(y_pred_aux, y_aux)
            Lcon = contrastive_loss(h, y)
            batch_loss = Lcls + config.lambda2 * Lcausal + config.lambda1 * Lcon
            if getattr(model, 'rl_select', False) and rl_logp is not None:
                global rl_baseline
                B = y.size(0)
                with torch.no_grad():
                    lp_fused = F.log_softmax((y_pred_raw + y_pred_aug[:B]) / 2, dim=1).gather(1, y[:, None]).squeeze(1)
                    lp_raw = F.log_softmax(y_pred_raw, dim=1).gather(1, y[:, None]).squeeze(1)
                    rl_reward = lp_fused - lp_raw
                    rl_baseline = rl_reward.mean() if rl_baseline is None else 0.9 * rl_baseline + 0.1 * rl_reward.mean()
                Lrl = -((rl_reward - rl_baseline) * rl_logp).mean() - RL_ENTROPY * rl_entropy.mean()
                batch_loss = batch_loss + RL_WEIGHT * Lrl
            if AUG_CE_WEIGHT:
                batch_loss = batch_loss + AUG_CE_WEIGHT * train_ce(y_pred_aug, y_aug)

        # backward (scaler is a no-op unless AMP_DTYPE=fp16)
        scaler.scale(batch_loss / GRAD_ACCUM).backward()
        if (i + 1) % GRAD_ACCUM == 0 or i + 1 == batch_len:
            scaler.step(optimizer)
            scaler.update()
            if SCHED == 'cosine':
                scheduler.step()

        # Defensive: release cached-but-unused CUDA memory periodically.
        # AGM+SCLM+ISAB+RCAL combined run right at this GPU's memory
        # limit (confirmed OOM a few batches in, not just on batch 1),
        # so guard against slow fragmentation growth over a long run.
        if i % 50 == 0:
            torch.cuda.empty_cache()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_crop_acc = crop_metric(y_pred_aug, y_aug)
            epoch_drop_acc = drop_metric(y_pred_aux, y_aux)

        # end of this batch
        batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Aug Acc ({:.2f}, {:.2f}), Aux Acc ({:.2f}, {:.2f})'.format(
            epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],
            epoch_crop_acc[0], epoch_crop_acc[1], epoch_drop_acc[0], epoch_drop_acc[1])

        pbar.update()
        pbar.set_postfix_str(batch_info)

        # live per-batch logging so loss is visible on W&B without
        # waiting for a full epoch (~70-90 min) to complete
        if i % 20 == 0:
            global_step = epoch * batch_len + i + 1
            try:
                wandb.log({
                    'batch/loss': batch_loss.item(),
                    'batch/raw_top1': epoch_raw_acc[0],
                    **({'rl/reward': rl_reward.mean().item(), 'rl/entropy': rl_entropy.mean().item()}
                       if rl_reward is not None else {}),
                    'epoch': epoch + 1,
                }, step=global_step)
            except Exception as e:
                global _wandb_batch_log_warned
                if not _wandb_batch_log_warned:
                    print('wandb.log (per-batch) FAILED: {!r}'.format(e), flush=True)
                    _wandb_batch_log_warned = True

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
    logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
    logs['train_info'] = batch_info
    end_time = time.time()

    # write food2k_log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))


def validate(**kwargs):
    # Retrieve training configuration
    global best_acc
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    drop_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    per_ds = {}  # dataset name -> [correct, total] (MULTI only)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # obtain data
            X, y, d = unpack(batch)
            X = X.to(device)
            y = y.to(device)
            d = None if d is None else d.to(device)

            ##################################
            # Raw Image
            ##################################
            with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=AMP):
                y_pred_raw, y_pred_aux, _, attention_map, _ = net(X) if d is None else net(X, d)
            y_pred_raw, y_pred_aux, attention_map = y_pred_raw.float(), y_pred_aux.float(), attention_map.float()
            y_pred_raw, y_pred_aux = mask_logits(y_pred_raw, d, DS_MASKS), mask_logits(y_pred_aux, d, DS_MASKS)
            if d is not None:
                hit = y_pred_raw.argmax(1) == y
                for di, name in enumerate(DS_NAMES):
                    sel = d == di
                    c = per_ds.setdefault(name, [0, 0])
                    c[0] += int(hit[sel].sum())
                    c[1] += int(sel.sum())

            if PLAIN:
                y_pred = y_pred_raw
                y_pred_aux = y_pred_raw  # (multi-view models return per-view logits here)
            else:
                crop_images3 = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
                with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=AMP):
                    y_pred_crop3, y_pred_aux_crop3, _, _, _ = net(crop_images3)
                y_pred_crop3, y_pred_aux_crop3 = y_pred_crop3.float(), y_pred_aux_crop3.float()

                ##################################
                # Final prediction
                ##################################
                y_pred = (y_pred_raw + y_pred_crop3) / 2.
                y_pred_aux = (y_pred_aux + y_pred_aux_crop3) / 2.

            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            epoch_loss = loss_container(batch_loss.item())

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)
            aux_acc = drop_metric(y_pred_aux, y)

    # end of validation
    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    if per_ds:
        logs['val_per_dataset'] = {k: 100. * c / max(n, 1) for k, (c, n) in per_ds.items()}
        print('val top1 per dataset: ' + ', '.join('{} {:.2f} ({} imgs)'.format(k, logs['val_per_dataset'][k], n)
                                                   for k, (c, n) in per_ds.items()), flush=True)
    end_time = time.time()

    # batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
    # pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    if epoch_acc[0] > best_acc:
        best_acc = epoch_acc[0]
        save_model(net, logs, 'model_bestacc.pth')

    if aux_acc[0] > best_acc:
        best_acc = aux_acc[0]
        save_model(net, logs, 'model_bestacc.pth')

    if epoch % 10 == 0:
        save_model(net, logs, 'model_epoch%d.pth' % epoch)

    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f}), Val Aux Acc ({:.2f}, {:.2f}), Best {:.2f}'.format(
        epoch_loss, epoch_acc[0], epoch_acc[1], aux_acc[0], aux_acc[1], best_acc)
    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))
    # print(batch_info)

    # write food2k_log for this epoch
    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    logging.info('')


def save_model(net, logs, ckpt_name):
    torch.save({'logs': logs, 'state_dict': net.state_dict()}, config.save_dir + 'model_bestacc.pth')


if __name__ == '__main__':
    main()
