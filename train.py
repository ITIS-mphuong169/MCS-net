import os
import config
import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from models import WSDAN_MCS
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment
import math
import numpy as np
import argparse
import wandb
from data_loader import get_data_loaders
# import thop

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
top1_container = AverageMeter(name='top1')
top5_container = AverageMeter(name='top5')
raw_metric = TopKAccuracyMetric(topk=(1, 5))
crop_metric = TopKAccuracyMetric(topk=(1, 5))
drop_metric = TopKAccuracyMetric(topk=(1, 5))

best_acc = 0.0


# 固定随机种子
def seed_torch(seed=1231):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch(1231)
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
    wandb.init(
        project="mcs-net-wikiart",
        id="{}-{}-v2".format(config.tag, config.net),
        resume="allow",
        config={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "net": config.net,
            "num_attentions": config.num_attentions,
            "beta": config.beta,
        },
    )

    # load dataset
    # train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)
    # train_loader, validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
    #                                            num_workers=config.workers, pin_memory=True, drop_last=True), \
    #                                 DataLoader(validate_dataset, batch_size=config.batch_size * 4, shuffle=False,
    #                                            num_workers=config.workers, pin_memory=True, drop_last=True)
    # num_classes = train_dataset.num_classes

    dataloaders, dataset_sizes, class_names = get_data_loaders("/kaggle/input/datasets/steubk/wikiart",
                                                               batch_size=32)
    train_loader = dataloaders['train']
    validate_loader = dataloaders['val']
    num_classes = 27

    # Initialize model
    logs = {}
    start_epoch = 0
    net = WSDAN_MCS(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)

    # feature_center: size of (#classes, #attention_maps * #channel_features)
    feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).cuda()

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

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].cuda()
            logging.info('feature_center loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    # Use cuda
    net.to(device)
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
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    # Model checkpoint
    callback_monitor = 'val_{}'.format(raw_metric.name)
    callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                               monitor=callback_monitor,
                               mode='max')
    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()

    # logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
    #              format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
    # logging.info('')

    for epoch in range(start_epoch, config.epochs):
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
              feature_center=feature_center,
              optimizer=optimizer,
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
            }, step=logs['epoch'])
            print('wandb.log OK for epoch {}'.format(logs['epoch']), flush=True)
        except Exception as e:
            print('wandb.log FAILED for epoch {}: {!r}'.format(logs['epoch'], e), flush=True)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val_loss'])
        else:
            scheduler.step()
        callback.on_epoch_end(logs, net, feature_center=feature_center)
        pbar.close()

    wandb.finish()


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
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
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
    for i, (X, y) in enumerate(data_loader):
        # float_iter = float(i) / batch_len
        # adjust_learning(optimizer, epoch, float_iter)
        # now_lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        # obtain data for training
        X = X.to(device)
        y = y.to(device)

        # raw image
        y_pred_raw, y_pred_aux, feature_matrix, attention_map = net(X)

        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
        aug_images = torch.cat([crop_images, drop_images], dim=0)
        y_aug = torch.cat([y, y], dim=0)

        # crop images forward
        y_pred_aug, y_pred_aux_aug, _, _ = net(aug_images)

        y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
        y_aux = torch.cat([y, y_aug], dim=0)

        # loss
        batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                     cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. + \
                     cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. + \
                     center_loss(feature_matrix, feature_center_batch)

        # backward
        batch_loss.backward()
        optimizer.step()

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
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            # obtain data
            X = X.to(device)
            y = y.to(device)

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, y_pred_aux, _, attention_map = net(X)

            crop_images3 = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop3, y_pred_aux_crop3, _, _ = net(crop_images3)

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
