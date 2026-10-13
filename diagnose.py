"""Diagnose why the full-paper model learns slower than the simplified one.

Usage (Kaggle, GPU, ~5 min):
    DIAG_CKPT=/path/to/last.ckpt python diagnose.py

Measures, on a few training batches:
  1. AGM term scales: |F*| vs |lambda_c * F*P| vs |lambda_s * QF*|.
     If the channel term is orders of magnitude larger than F*, AGM's
     output is dominated by it instead of refining F*.
  2. Attention-map diversity: mean pairwise cosine similarity between
     the M attention maps of an image. Close to 1 = the M "parts" have
     collapsed onto the same region (BAP then sees one part M times).
  3. Gradient norm each loss term sends into layer4 and the attention
     conv: shows whether Lcon / Lcausal drown out Lcls.
"""
import os
import torch
import torch.nn.functional as F
import config
from models import WSDAN_MCS
from models.mcs import StyleContrastiveLoss
from data_loader import get_data_loaders

DATA_DIR = "/kaggle/input/datasets/steubk/wikiart"
NUM_BATCHES = int(os.environ.get('DIAG_BATCHES', 5))
device = torch.device('cuda')


def agm_terms(agm, F_star):
    """Same math as AttentionGenerationModule.forward, returning the
    three summands separately."""
    B, C, H, W = F_star.size()
    N = H * W
    phi_flat = torch.sigmoid(agm.phi(F_star)).view(B, C, N)
    varphi_flat = torch.sigmoid(agm.varphi(F_star)).view(B, C, N)
    F_flat = F_star.view(B, C, N).transpose(1, 2)
    channel_term = torch.bmm(torch.bmm(F_flat, phi_flat), varphi_flat.transpose(1, 2)) / N
    channel_term = channel_term.transpose(1, 2).view(B, C, H, W)
    alpha_flat = torch.tanh(agm.alpha(F_star)).view(B, C, N).transpose(1, 2)
    beta_flat = torch.tanh(agm.beta(F_star)).view(B, C, N).transpose(1, 2)
    Q = torch.bmm(alpha_flat, beta_flat.transpose(1, 2)) / C
    spatial_term = torch.bmm(Q, F_flat).transpose(1, 2).view(B, C, H, W)
    return agm.lambda_c * channel_term, agm.lambda_s * spatial_term


def attention_similarity(att):
    """Mean off-diagonal cosine similarity between the M maps, per image."""
    B, M = att.shape[:2]
    a = F.normalize(att.reshape(B, M, -1), dim=-1)
    sim = torch.bmm(a, a.transpose(1, 2))
    off = sim.sum(dim=(1, 2)) - sim.diagonal(dim1=1, dim2=2).sum(dim=1)
    return (off / (M * (M - 1))).mean().item()


def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)).item()


def main():
    ckpt = os.environ['DIAG_CKPT']
    net = WSDAN_MCS(num_classes=27, M=config.num_attentions, net=config.net, pretrained=False)
    checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    print('Loaded {} @ epoch {}\n'.format(ckpt, checkpoint['logs'].get('epoch')))

    loaders, _, _ = get_data_loaders(DATA_DIR, batch_size=config.batch_size, num_workers=2)
    ce = torch.nn.CrossEntropyLoss()
    con = StyleContrastiveLoss(temperature=config.tau)
    probe = {'layer4': list(net.features[-1].parameters()),
             'attention_conv': list(net.attentions.parameters())}

    stats = {k: [] for k in ['F_star', 'channel', 'spatial', 'att_sim', 'att_dead']}
    grads = {(l, p): [] for l in ['Lcls', 'l2*Lcausal', 'l1*Lcon'] for p in probe}

    for i, (X, y) in enumerate(loaders['train']):
        if i >= NUM_BATCHES:
            break
        X, y = X.to(device), y.to(device)

        # 1-2: forward statistics (eval mode, no grad)
        net.eval()
        with torch.no_grad():
            F_star = net.features(X)
            ch, sp = agm_terms(net.agm, F_star)
            att = net.attentions(net.agm(F_star))
            stats['F_star'].append(F_star.abs().mean().item())
            stats['channel'].append(ch.abs().mean().item())
            stats['spatial'].append(sp.abs().mean().item())
            stats['att_sim'].append(attention_similarity(att))
            stats['att_dead'].append((att.flatten(2).amax(-1) <= 1e-6).float().mean().item())

        # 3: per-loss gradient norms (train mode, raw images only)
        net.train()
        p, p_aux, _, _, h = net(X)
        losses = {'Lcls': ce(p, y),
                  'l2*Lcausal': config.lambda2 * ce(p_aux, y),
                  'l1*Lcon': config.lambda1 * con(h, y)}
        for lname, loss in losses.items():
            for pname, params in probe.items():
                grads[(lname, pname)].append(grad_norm(loss, params))
        net.zero_grad(set_to_none=True)

    avg = lambda v: sum(v) / len(v)
    print('=== 1. AGM term scale (mean |value|) ===')
    print('F* (backbone)          {:.4f}'.format(avg(stats['F_star'])))
    print('lambda_c * F*P         {:.4f}   ratio vs F*: {:.1f}x'.format(
        avg(stats['channel']), avg(stats['channel']) / avg(stats['F_star'])))
    print('lambda_s * QF*         {:.4f}   ratio vs F*: {:.1f}x'.format(
        avg(stats['spatial']), avg(stats['spatial']) / avg(stats['F_star'])))

    print('\n=== 2. Attention-map diversity ===')
    print('mean pairwise cosine sim between {} maps: {:.3f}  (1.0 = all identical)'.format(
        config.num_attentions, avg(stats['att_sim'])))
    print('fraction of all-zero (dead) maps:         {:.3f}'.format(avg(stats['att_dead'])))

    print('\n=== 3. Gradient norm per loss term ===')
    for pname in probe:
        print('[{}]'.format(pname))
        for lname in ['Lcls', 'l2*Lcausal', 'l1*Lcon']:
            print('  {:12s} {:.4f}'.format(lname, avg(grads[(lname, pname)])))


if __name__ == '__main__':
    main()
