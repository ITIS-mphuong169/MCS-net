#!/usr/bin/env python3
"""Chay CLIP ViT (toan canh + chi tiet) tren WikiArt - Windows, Linux, macOS, khong can bash.

    python server/clip.py check                          kiem tra GPU, goi Python, du lieu, W&B
    python server/clip.py probe                          phep thu ~25 phut (CLIP dong bang), in 2 dong VERDICT
    python server/clip.py train smoke                    chay thu ~10 phut, phai in "CHAY THU OK"
    python server/clip.py train both                     2 run lien tiep: single roi detail, moi run 100 epoch
    python server/clip.py train single | detail | zoom   chay rieng tung run
    python server/clip.py eval both                      cham test ca hai (epoch cuoi + epoch val tot nhat) + McNemar

Tuy chon chung: --data <thu muc WikiArt>  --epochs 100  --mag 2  --arch b16|l14  --gpu 0  --workers N
                --batch-size N --grad-accum N  --save-root <thu muc luu run>  --dry-run (chi in lenh, khong chay)
Bi dung giua chung (mat dien, tat may): chay lai DUNG lenh cu, tu train tiep tu epoch cuoi.

single : 1 view 224 (moc so sanh).  detail : toan canh 224 + 2 manh chi tiet cat tu anh goc (kiem tra: 9 manh),
zoom   : doi chung cua detail, manh cat tu chinh anh 224 phong to (khong co diem anh moi).
Tat ca dung cung cong thuc: CLIP ViT, head neo chu, attention theo nhan, AdamW, cosine, bfloat16, label smoothing 0.1.
"""
import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENTITY = 'phuongmt-b22cn641-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng'
# arch -> (timm name, run-tag prefix, results prefix)
ARCHS = {'b16': ('vit_base_patch16_clip_quickgelu_224.openai', 'srv_clip', 'clip'),
         'l14': ('vit_large_patch14_clip_quickgelu_224.openai', 'srv_clipL', 'clipL')}
MODES = ('single', 'detail', 'zoom')
RUN_DIR = 'wikiart_fullpaper_selfattn_abl_{}'   # train.py: save_root / this / (last.ckpt | model.ckpt)


def say(msg=''):
    print(msg, flush=True)


def find_data(path):
    """The folder that directly holds the 27 style folders (Kaggle zips often nest it one level deeper)."""
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_dir():
        return None
    for cur, dirs, _ in os.walk(p):
        if 'Impressionism' in dirs:
            return Path(cur)
        if Path(cur).relative_to(p).parts.__len__() >= 3:
            dirs[:] = []
    return p


def resolve_data(args):
    raw = args.data or os.environ.get('DATA_DIR')
    if not raw and (Path.home() / '.mcs_data_dir').is_file():
        raw = (Path.home() / '.mcs_data_dir').read_text().strip()
    d = find_data(raw)
    if d is None:
        sys.exit('!! Chua co duong dan du lieu WikiArt. Them:  --data D:\\duong\\dan\\wikiart  (thu muc co 27 thu muc phong cach)')
    n = sum(1 for x in d.iterdir() if x.is_dir())
    if n != 27:
        say('!! Canh bao: {} co {} thu muc con, phai la 27 (moi phong cach mot thu muc)'.format(d, n))
    return d


def gpu_info(gpu):
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(gpu)
        return {'name': p.name, 'mem': int(round(p.total_memory / 2 ** 30)), 'bf16': torch.cuda.is_bf16_supported()}
    except Exception:
        return None


def base_env(args):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HIP_VISIBLE_DEVICES=str(args.gpu), PYTHONUTF8='1',
               PYTHONIOENCODING='utf-8', HF_HUB_DISABLE_SYMLINKS_WARNING='1', WANDB_ENTITY=ENTITY)
    return env


def views_env(mode, args):
    if mode == 'single':
        return {'VIEWS': 'single'}
    view = {'VIEWS': 'detail' if mode in ('detail', 'smoke') else 'zoom', 'DETAIL_MAG': str(args.mag),
            'DETAIL_K': '2', 'DETAIL_K_EVAL': '2', 'DETAIL_K_TEST': '9'}
    return view


def batch_plan(args, mode, info):
    """Effective batch is always 32 images (views multiply the work per step)."""
    if args.batch_size or args.grad_accum:
        bs, acc = args.batch_size or 16, args.grad_accum or 2
        return bs, acc
    mem = info['mem'] if info else 12
    if args.arch == 'l14':
        return (8, 4) if mode == 'single' else (4, 8)
    limit = 12 if mode == 'single' else 24
    return (32, 1) if mem >= limit else (16, 2)


def run_cmd(cmd, env, args, label):
    say('>> {}'.format(label))
    if args.dry_run:
        shown = {k: v for k, v in env.items() if k in KEYS_SHOWN}
        say('   [dry-run] ' + ' '.join(cmd))
        for k in sorted(shown):
            say('   {}={}'.format(k, shown[k]))
        return 0
    return subprocess.call(cmd, cwd=str(REPO), env=env)


KEYS_SHOWN = {'CUDA_VISIBLE_DEVICES', 'DATA_DIR', 'SAVE_ROOT', 'BATCH_SIZE', 'GRAD_ACCUM', 'WORKERS', 'EPOCHS', 'SEED', 'NET',
              'PLAIN_CLS', 'TEXT_INIT', 'LABEL_ATTN', 'DATASETS', 'LABEL_SPACE', 'OPTIM', 'BACKBONE_LR_MULT', 'WEIGHT_DECAY',
              'SCHED', 'WARMUP_EPOCHS', 'AMP', 'AMP_DTYPE', 'LABEL_SMOOTH', 'DROP_PATH', 'ABLATION_TAG', 'VIEWS', 'DETAIL_MAG',
              'DETAIL_K', 'DETAIL_K_EVAL', 'DETAIL_K_TEST', 'TRAIN_FRACTION', 'WANDB_MODE', 'INFER_CKPT', 'INFER_OUT',
              'INFER_TTA', 'PROBE_OUT', 'PROBE_MAGS', 'PROBE_TRAIN_LIMIT', 'PROBE_BS'}


def cmd_check(args):
    say('He dieu hanh : {} {}'.format(platform.system(), platform.release()))
    say('Python       : {}  ({})'.format(platform.python_version(), sys.executable))
    for mod in ('torch', 'torchvision', 'timm', 'transformers', 'wandb', 'sklearn', 'tqdm', 'PIL', 'numpy'):
        try:
            m = __import__(mod)
            say('  {:13s} {}'.format(mod, getattr(m, '__version__', 'ok')))
        except Exception as e:
            say('  {:13s} !! THIEU ({}). Chay:  python -m pip install {}'.format(
                mod, type(e).__name__, {'sklearn': 'scikit-learn', 'PIL': 'pillow'}.get(mod, mod)))
    info = gpu_info(args.gpu)
    if info:
        say('GPU {}      : {}, {} GB, bfloat16: {}'.format(args.gpu, info['name'], info['mem'], info['bf16']))
    else:
        say('GPU          : !! Khong dung duoc (PyTorch ban CPU, hoac thieu driver). Cai:  '
            'python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124')
    try:
        d = resolve_data(args)
        say('Du lieu      : {}'.format(d))
    except SystemExit as e:
        say(str(e))
    netrc = [Path.home() / n for n in ('.netrc', '_netrc')]
    logged = bool(os.environ.get('WANDB_API_KEY')) or any(p.is_file() and 'wandb' in p.read_text(errors='ignore') for p in netrc)
    say('W&B          : {}'.format('da dang nhap' if logged else '!! chua dang nhap. Chay:  wandb login   (hoac dat WANDB_MODE=offline)'))
    say('Thu muc luu  : {}'.format(save_root(args)))


def save_root(args):
    return Path(args.save_root).expanduser() if args.save_root else Path.home() / 'runs'


def results_root(args):
    return Path(args.results).expanduser() if args.results else Path.home() / 'results'


def train_env(args, mode, data, info):
    netname, tagp, _ = ARCHS[args.arch]
    bs, acc = batch_plan(args, 'detail' if mode == 'smoke' else mode, info)
    bf16 = info['bf16'] if info else True
    env = base_env(args)
    env.update(DATA_DIR=str(data), SAVE_ROOT=save_root(args).as_posix() + '/', BATCH_SIZE=str(bs), GRAD_ACCUM=str(acc),
               WORKERS=str(args.workers), EPOCHS='1' if mode == 'smoke' else str(args.epochs), SEED=str(args.seed),
               NET=netname, PLAIN_CLS='1', TEXT_INIT='1', LABEL_ATTN=str(args.label_attn), DATASETS='wikiart',
               LABEL_SPACE='union', OPTIM='adamw', BACKBONE_LR_MULT='0.01', WEIGHT_DECAY='0.05', SCHED='cosine',
               WARMUP_EPOCHS='2', AMP='1', AMP_DTYPE='bf16' if bf16 else 'fp16', LABEL_SMOOTH='0.1', DROP_PATH='0.1',
               ABLATION_TAG='{}_{}'.format(tagp, mode))
    env.update(views_env(mode, args))
    if mode == 'smoke':
        env.update(TRAIN_FRACTION='0.02', WANDB_MODE='offline')
    return env, bs, acc


def cmd_train(args):
    data = resolve_data(args)
    info = gpu_info(args.gpu)
    if info is None and not args.dry_run:
        sys.exit('!! Khong thay GPU dung duoc. Chay:  python server/clip.py check')
    modes = ['single', 'detail'] if args.mode == 'both' else [args.mode]
    for mode in modes:
        env, bs, acc = train_env(args, mode, data, info)
        say('')
        say('=== {} | {} | GPU {} ({} GB) | batch {} x tich luy {} = {} anh | {} epoch | mag x{} | du lieu {}'.format(
            mode, args.arch, args.gpu, info['mem'] if info else '?', bs, acc, bs * acc, env['EPOCHS'], args.mag, data))
        rc = run_cmd([sys.executable, str(REPO / 'train.py')], env, args, 'train.py ({})'.format(mode))
        if rc != 0:
            sys.exit('!! train.py loi (ma {}). Doc dong loi o tren; chay lai dung lenh cu se tiep tuc tu epoch cuoi.'.format(rc))
        if mode == 'smoke':
            tagp = ARCHS[args.arch][1]
            if not args.dry_run:
                for d in glob.glob(str(save_root(args) / ('*' + tagp + '_smoke*'))):
                    shutil.rmtree(d, ignore_errors=True)
                say('>> CHAY THU OK')
    if args.mode == 'both':
        say('')
        say('>> Xong 2 run. Tiep theo:  python server/clip.py eval both --data {}'.format(data))


def cmd_probe(args):
    data = resolve_data(args)
    env = base_env(args)
    env.update(DATA_DIR=str(data), PROBE_OUT=str(results_root(args) / 'probe_detail'), PROBE_MAGS='2,3',
               PROBE_TRAIN_LIMIT='20000', WORKERS=str(args.workers), PROBE_BS='16')
    results_root(args).mkdir(parents=True, exist_ok=True)
    rc = run_cmd([sys.executable, str(REPO / 'probe_detail.py')], env, args, 'probe_detail.py (khoang 25 phut)')
    if rc != 0:
        sys.exit('!! probe_detail.py loi (ma {})'.format(rc))
    say('>> Ket qua: {}'.format(results_root(args) / 'probe_detail' / 'probe_detail_results.json'))
    say('>> Doc 2 dong VERDICT o cuoi: (a) nhanh chi tiet GO / NO-GO, (b) chon vi tri bang hoc (RL)')


def cmd_eval(args):
    data = resolve_data(args)
    netname, tagp, outp = ARCHS[args.arch]
    modes = ['single', 'detail'] if args.mode == 'both' else [args.mode]
    done = {}
    for mode in modes:
        run_dir = save_root(args) / RUN_DIR.format('{}_{}'.format(tagp, mode))
        for which, fname in (('last', 'last.ckpt'), ('best', 'model.ckpt')):
            src = run_dir / fname
            if not src.is_file():
                say('!! Khong thay {}'.format(src))
                continue
            if args.dry_run:
                ep = '?'
            else:
                import torch
                ep = int(torch.load(str(src), weights_only=False, map_location='cpu')['logs']['epoch'])
            copy = save_root(args) / 'eval_{}_{}_{}_ep{}.ckpt'.format(outp, mode, which, ep)
            out = results_root(args) / '{}_{}_{}_ep{}'.format(outp, mode, which, ep)
            if not args.dry_run:
                shutil.copy(str(src), str(copy))
            env = base_env(args)
            env.update(DATA_DIR=str(data), BATCH_SIZE=str(args.eval_batch), INFER_TTA='0', NET=netname, PLAIN_CLS='1',
                       TEXT_INIT='1', LABEL_ATTN=str(args.label_attn), DATASETS='wikiart', LABEL_SPACE='union',
                       WORKERS=str(args.workers), INFER_CKPT=str(copy), INFER_OUT=str(out))
            if mode == 'single':
                env.update(VIEWS='single')
            else:
                env.update(VIEWS='detail' if mode == 'detail' else 'zoom', DETAIL_MAG=str(args.mag), DETAIL_K_TEST='9')
            say('')
            say('=== cham test: {} / {} (epoch {})'.format(mode, which, ep))
            rc = run_cmd([sys.executable, str(REPO / 'infer.py')], env, args, 'infer.py ({} {})'.format(mode, which))
            if rc == 0:
                done[(mode, which)] = out / 'predictions.npz'
    if ('single', 'last') in done and ('detail', 'last') in done:
        say('')
        say('=== McNemar: detail (da gop toan canh + 9 manh) so voi single, epoch cuoi')
        run_cmd([sys.executable, str(REPO / 'mcnemar.py'), str(done[('single', 'last')]), str(done[('detail', 'last')])],
                base_env(args), args, 'mcnemar.py')
    say('')
    say('>> Dong "center_crop" la du doan da gop; "global_view" chi nhin toan canh; "detail_views" chi nhin 9 manh.')
    say('>> Ket qua: {}'.format(results_root(args)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['check', 'probe', 'train', 'eval'])
    ap.add_argument('mode', nargs='?', choices=['smoke', 'single', 'detail', 'zoom', 'both'],
                    help='train: smoke|single|detail|zoom|both   eval: single|detail|zoom|both')
    ap.add_argument('--data', help='thu muc WikiArt (co 27 thu muc phong cach)')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--mag', type=int, default=2, help='phong to cua manh chi tiet: 2 hoac 3 (theo VERDICT cua phep thu)')
    ap.add_argument('--arch', choices=list(ARCHS), default='b16', help='b16 = ViT-B/16 (mac dinh), l14 = ViT-L/14 (nang ~4 lan)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--workers', type=int, default=max(2, min(6, (os.cpu_count() or 4) - 2)),
                    help='so tien trinh doc anh; Windows moi tien trinh ton ~1GB RAM, giam xuong 4 neu may treo / het RAM')
    ap.add_argument('--batch-size', type=int, default=0)
    ap.add_argument('--grad-accum', type=int, default=0)
    ap.add_argument('--eval-batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--label-attn', type=int, default=1, choices=[0, 1])
    ap.add_argument('--save-root', help='noi luu run (mac dinh ~/runs)')
    ap.add_argument('--results', help='noi luu ket qua (mac dinh ~/results)')
    ap.add_argument('--dry-run', action='store_true', help='chi in ra se chay gi, khong chay')
    args = ap.parse_args()
    if args.command == 'train' and args.mode not in ('smoke', 'single', 'detail', 'zoom', 'both'):
        ap.error('train can mot MODE: smoke | single | detail | zoom | both')
    if args.command == 'eval' and args.mode not in ('single', 'detail', 'zoom', 'both'):
        ap.error('eval can mot MODE: single | detail | zoom | both')
    {'check': cmd_check, 'probe': cmd_probe, 'train': cmd_train, 'eval': cmd_eval}[args.command](args)


if __name__ == '__main__':
    main()
