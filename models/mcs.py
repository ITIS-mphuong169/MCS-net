import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.resnet as resnet

import random

__all__ = ['WSDAN_MCS']
EPSILON = 1e-6


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', attentions, features) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', fake_att, features) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + EPSILON)

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError(
            'Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


class MAB(nn.Module):
    """Multihead Attention Block (Lee et al., "Set Transformer", ICML 2019).
    Cross-attends Q (queries) against K (keys/values): MAB(Q, K)."""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K_, V_ = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K_.split(dim_split, 2), 0)
        V_ = torch.cat(V_.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / (self.dim_V ** 0.5), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O)
        return O


class ISAB(nn.Module):
    """Induced Set Attention Block (Set Transformer, Lee et al. 2019).
    Routes the input set through a small bank of learnable inducing points
    (num_inds) instead of attending every element to every other element,
    which also makes the block permutation-invariant over the input set -
    unlike plain self-attention, which has no notion of set vs. sequence."""

    def __init__(self, dim_in, dim_out, num_heads, num_inds):
        super(ISAB, self).__init__()
        self.inducing_points = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.inducing_points)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads)

    def forward(self, X):
        H = self.mab0(self.inducing_points.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class TransformerFeatures(nn.Module):
    """Wraps a timm features_only backbone (Swin/ConvNeXt/ViT) so it exposes
    a single forward(x) -> (B, C, H, W) feature map, matching the interface
    the resnet/inception branches already provide."""

    def __init__(self, backbone, num_features):
        super(TransformerFeatures, self).__init__()
        self.backbone = backbone
        self.num_features = num_features

    def forward(self, x):
        feat = self.backbone(x)[-1]
        # some timm backbones (e.g. Swin) return channels-last (B, H, W, C);
        # normalize to (B, C, H, W) like the conv backbones use
        if feat.dim() == 4 and feat.shape[1] != self.num_features and feat.shape[-1] == self.num_features:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat


class WSDAN_MCS(nn.Module):
    def __init__(self, num_classes, M=32, net='inception_mixed_6e', pretrained=False):
        super(WSDAN_MCS, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net

        # Network Initialization
        if 'inception' in net:
            from models.inception import inception_v3
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net)
        elif 'resnet' in net:
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif 'att' in net:
            print('==> Using MANet with resnet101 backbone')
            self.features = MANet()
            self.num_features = 2048
        elif 'swin' in net or 'convnext' in net or 'vit' in net:
            import timm
            backbone = timm.create_model(net, pretrained=pretrained, features_only=True)
            num_features = backbone.feature_info.channels()[-1]
            self.features = TransformerFeatures(backbone, num_features)
            self.num_features = num_features
            print('==> using timm backbone: %s, num_features=%d' % (net, num_features))
        else:
            raise ValueError('Unsupported net: %s' % net)

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)
        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')
        # Part-relationship modeling: BAP's M part-features form an
        # unordered SET (no inherent sequence order), not a sequence - so
        # use Set Transformer's ISAB (Lee et al. 2019) instead of plain
        # self-attention, which has no notion of permutation invariance.
        # Everything else here is unchanged from the self-attention version
        # (commit c24644c) this branch forked from, so the two are a fair
        # apples-to-apples comparison of just the attention mechanism.
        # Project down first (e.g. resnet101's 2048-dim parts) - running
        # the block at full num_features OOM'd a 15GB GPU (x2 for
        # real+counterfactual, x3 for the raw/crop/drop forward passes
        # per training step).
        part_dim = 256
        num_inducing_points = 16
        self.part_proj = nn.Linear(self.num_features, part_dim)
        self.part_transformer = ISAB(
            dim_in=part_dim, dim_out=part_dim,
            num_heads=8, num_inds=num_inducing_points,
        )
        # Classification Layer
        self.fc = nn.Linear(self.M * part_dim, self.num_classes, bias=False)

        logging.info(
            'WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes,
                                                                                               self.M))

    def _classify(self, feature_matrix, batch_size):
        parts = feature_matrix.view(batch_size, self.M, self.num_features)
        parts = self.part_proj(parts)
        parts = self.part_transformer(parts)
        # no *100 here: the original code scaled the tiny L2-normalized BAP
        # output before its single Linear layer, but part_transformer's
        # LayerNorm already renormalizes to unit scale, so the extra *100
        # just overshoots and blew up the loss to NaN in practice
        return self.fc(parts.reshape(batch_size, -1))

    def visualize(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]

        feature_matrix, _ = self.bap(feature_maps, attention_maps)
        p = self._classify(feature_matrix, batch_size)

        return p, attention_maps

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]

        feature_matrix, feature_matrix_hat = self.bap(feature_maps, attention_maps)

        # Classification
        p = self._classify(feature_matrix, batch_size)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        return p, p - self._classify(feature_matrix_hat, batch_size), feature_matrix, attention_map

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            print('%s: All params loaded' % type(self).__name__)
        else:
            print('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN_MCS, self).load_state_dict(model_dict)
