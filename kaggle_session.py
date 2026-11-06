"""One Kaggle session for one account: data, runs, probe, summary.

    python kaggle_session.py tk1      # account 1
    python kaggle_session.py tk2      # account 2

The notebook needs nothing else (GPU T4, Internet on, input steubk/wikiart,
W&B key in the environment). In order, the session:

 1. Data. If the output of an earlier session of this notebook is attached
    as input, its wikiart768.zip / Pandora_18k.zip / pandora_dups.txt are
    used. Otherwise they are built into /kaggle/working/data (about an hour:
    resize_dataset.py, the Pandora18k download, dedup.py), which also makes
    them part of this session's output. Both sets are unpacked to
    /kaggle/temp, the fast local disk.
 2. The account's plan (PLANS below): training runs (train.py, then
    infer.py on last.ckpt) and, on account 1, the detail probe on the first
    run's checkpoint. A step whose result is already in the attached input
    is skipped. Before each step the time left in the session is checked;
    a failed step does not stop the following ones.
 3. A table of every run's test numbers, printed last and saved to
    /kaggle/working/results/summary.txt: the only thing to send back.

SEED=2 python kaggle_session.py tk1 runs the same plan with another seed.
"""
import os
import sys
import glob
import json
import time
import shutil
import zipfile
import subprocess
import urllib.request

ACCOUNT = sys.argv[1] if len(sys.argv) > 1 else ''
SEED = os.environ.get('SEED', '1')
INPUT = os.environ.get('KS_INPUT', '/kaggle/input')
WORK = os.environ.get('KS_WORK', '/kaggle/working')
TEMP = os.environ.get('KS_TEMP', '/kaggle/temp')
WIKIART_SRC = os.environ.get('KS_WIKIART_SRC', '/kaggle/input/datasets/steubk/wikiart')
PANDORA_URL = os.environ.get('KS_PANDORA_URL', 'http://imag.pub.ro/pandora/Download/Pandora_18k.zip')
LIMIT_H = float(os.environ.get('KS_LIMIT_H', 11.0))  # Kaggle stops a session at 12 h
REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
T0 = time.time()

# the recipe of every step-2 run (anything already set in the environment wins)
DEFAULTS = dict(NET='vit_base_patch16_clip_quickgelu_224.openai', EPOCHS='8', OPTIM='adamw',
                BACKBONE_LR_MULT='0.01', WEIGHT_DECAY='0.05', SCHED='cosine', WARMUP_EPOCHS='1',
                AMP='1', AMP_DTYPE='fp16', INFER_TTA='0', WORKERS='4', PLAIN_CLS='1', BATCH_SIZE='32',
                TEXT_INIT='1')


def plans(seed):
    """(kind, tag, datasets, label space, LABEL_ATTN) per step."""
    s = 's' + seed
    return {
        'tk1': [('run', 'clipq_text_' + s, 'wikiart', 'union', '0'),      # D   WikiArt, text-anchored head
                ('probe', 'clipq_text_' + s),                             #     detail / position probe on D
                ('run', 'clipq_lqa_' + s, 'wikiart', 'union', '1'),       # D-M + label-query attention
                ('run', 'pand_text_' + s, 'pandora', 'union', '0'),       # P   Pandora18k
                ('run', 'pand_lqa_' + s, 'pandora', 'union', '1')],       # P-M
        'tk2': [('run', 'joint_union_' + s, 'wikiart,pandora', 'union', '0'),      # U   both, unified labels
                ('run', 'joint_union_lqa_' + s, 'wikiart,pandora', 'union', '1'),  # U-M final model
                ('run', 'joint_sep_' + s, 'wikiart,pandora', 'separate', '0')],    # S   both, a head per dataset
    }


def hours():
    return (time.time() - T0) / 3600


def log(msg):
    print('\n[session {:5.2f} h] {}'.format(hours(), msg), flush=True)


def call(cmd, env=None):
    return subprocess.call(cmd, cwd=REPO, env=env if env is not None else os.environ.copy())


_FILES = None


def find(suffix):
    """First file under the notebook inputs whose path ends with `suffix`
    (the 81k WikiArt originals are not walked)."""
    global _FILES
    if _FILES is None:
        _FILES, src = [], os.path.abspath(WIKIART_SRC)
        for cur, dirs, files in os.walk(INPUT):
            if os.path.abspath(cur) == src:
                dirs[:] = []
                continue
            _FILES += [os.path.join(cur, f) for f in files]
        _FILES.sort()
    return next((p for p in _FILES if p.replace(os.sep, '/').endswith('/' + suffix)), None)


def download(url, dest, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dest + '.part', 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(dest + '.part', dest)
            return
        except Exception as e:
            print('download attempt {} failed: {}'.format(k + 1, e), flush=True)
            time.sleep(10)
    raise RuntimeError('could not download ' + url)


def prepare_data():
    """Sets DATA_DIR (and DATA_DIR_PANDORA / EXCLUDE_LIST); returns whether
    Pandora18k is available."""
    data = os.path.join(WORK, 'data')
    os.makedirs(data, exist_ok=True)
    os.makedirs(TEMP, exist_ok=True)

    wzip = find('wikiart768.zip') or os.path.join(data, 'wikiart768.zip')
    if os.path.isfile(wzip):
        log('WikiArt copy found: ' + wzip)
    else:
        if not os.path.isdir(WIKIART_SRC):
            raise SystemExit('No data: add the dataset steubk/wikiart to this notebook (Add Input), '
                             'or the output of a notebook that already built wikiart768.zip, then run again.')
        log('building the resized WikiArt copy (about an hour, only this once)')
        rc = call([PY, 'resize_dataset.py', '--src', WIKIART_SRC, '--out', wzip,
                   '--budget-gb', os.environ.get('KS_BUDGET_GB', '12')])
        if rc != 0 or not os.path.isfile(wzip):
            raise SystemExit('resize_dataset.py failed - nothing else can run')
    wdir = os.path.join(TEMP, 'wikiart768')
    if not os.path.isfile(os.path.join(wdir, 'MANIFEST.txt')):
        log('unpacking WikiArt to ' + TEMP)
        with zipfile.ZipFile(wzip) as z:
            z.extractall(TEMP)
    os.environ['DATA_DIR'] = wdir

    try:  # Pandora18k is not fatal: the WikiArt-only runs go on without it
        pzip = find('Pandora_18k.zip') or os.path.join(data, 'Pandora_18k.zip')
        if not os.path.isfile(pzip):
            log('downloading Pandora18k')
            download(PANDORA_URL, pzip)
        pdir = os.path.join(TEMP, 'pandora18k')
        if not os.path.isdir(pdir) or not os.listdir(pdir):
            log('unpacking Pandora18k')
            with zipfile.ZipFile(pzip) as z:
                z.extractall(pdir)
        dups = find('pandora_dups.txt') or os.path.join(data, 'pandora_dups.txt')
        if not os.path.isfile(dups):
            log('looking for Pandora18k images that are copies of WikiArt images')
            meta = find('wikiart768_meta.npz') or os.path.join(data, 'wikiart768_meta.npz')
            cmd = [PY, 'dedup.py', '--out', dups] + (['--wikiart-meta', meta] if os.path.isfile(meta) else [])
            if call(cmd, dict(os.environ, DATA_DIR_PANDORA=pdir)) != 0:
                raise RuntimeError('dedup.py failed')
        os.environ.update(DATA_DIR_PANDORA=pdir, EXCLUDE_LIST=dups)
        return True
    except Exception as e:
        log('Pandora18k NOT available ({}): the steps that need it will be skipped'.format(e))
        return False


def save_dir(tag):
    return os.path.join(WORK, 'FGVC', 'wikiart_fullpaper_selfattn_abl_' + tag)


def slim(path):
    """Keep only the weights of a finished run (1 GB with the optimizer
    state, a third without): infer.py and probe_detail.py need no more."""
    import torch
    ck = torch.load(path, weights_only=False, map_location='cpu')
    torch.save({'logs': ck.get('logs', {}), 'state_dict': ck['state_dict']}, path)


DURATION = {}  # tag -> hours of train + test


def do_run(tag, datasets, label_space, label_attn):
    out = os.path.join(WORK, 'results', tag)
    prev = find('results/{}/test_results.json'.format(tag))
    if prev:  # finished in an attached earlier session: keep its result files in this output too
        shutil.copytree(os.path.dirname(prev), out, dirs_exist_ok=True)
        return 'done in an earlier session'
    env = dict(os.environ, ABLATION_TAG=tag, DATASETS=datasets, LABEL_SPACE=label_space, LABEL_ATTN=label_attn)
    # a run cut off in an earlier session: its last.ckpt (written every epoch)
    # comes back through the attached output, and train.py resumes from it
    dst = os.path.join(save_dir(tag), 'last.ckpt')
    prev_ck = find('wikiart_fullpaper_selfattn_abl_{}/last.ckpt'.format(tag))
    if prev_ck and not os.path.isfile(dst):
        os.makedirs(save_dir(tag), exist_ok=True)
        shutil.copy(prev_ck, dst)
        log('{} resumes from {}'.format(tag, prev_ck))
    t = time.time()
    if call([PY, 'train.py'], env) != 0:
        return 'train.py FAILED'
    ck = os.path.join(save_dir(tag), 'last.ckpt')
    ienv = dict(env, INFER_CKPT=ck, INFER_OUT=out)
    if call([PY, 'infer.py'], ienv) != 0:
        log('infer.py failed, one more try on a single GPU')  # training used one GPU, infer.py uses all
        if call([PY, 'infer.py'], dict(ienv, CUDA_VISIBLE_DEVICES='0')) != 0:
            return 'infer.py FAILED (checkpoint kept: {})'.format(ck)
    DURATION[tag] = (time.time() - t) / 3600
    try:
        slim(ck)
        for extra in ('model.ckpt', 'model_bestacc.pth'):
            if os.path.isfile(os.path.join(save_dir(tag), extra)):
                os.remove(os.path.join(save_dir(tag), extra))
    except Exception as e:
        print('could not slim the checkpoint: {}'.format(e))
    return 'ok'


def do_probe(tag):
    dest = os.path.join(WORK, 'results', 'probe_detail_results.json')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    prev = find('results/probe_detail_results.json')
    if prev:
        shutil.copy(prev, dest)
        return 'done in an earlier session'
    ck = os.path.join(save_dir(tag), 'last.ckpt')
    if not os.path.isfile(ck):
        ck = find('wikiart_fullpaper_selfattn_abl_{}/last.ckpt'.format(tag)) or ''
    out = os.path.join(TEMP, 'probe_detail')
    env = dict(os.environ, PROBE_CKPT=ck, PROBE_MODEL=os.environ['NET'], PROBE_OUT=out, LABEL_ATTN='0')
    rc = call([PY, 'probe_detail.py'], env)
    if rc != 0:
        log('probe failed, one more try on a single GPU (the features already extracted are reused)')
        rc = call([PY, 'probe_detail.py'], dict(env, CUDA_VISIBLE_DEVICES='0'))
    if rc != 0:
        return 'probe_detail.py FAILED'
    shutil.copy(os.path.join(out, 'probe_detail_results.json'), dest)
    return 'ok' if ck else 'ok (no fine-tuned checkpoint: frozen backbone only)'


def summary(status):
    lines = ['', '===== KET QUA: {} seed {} ====='.format(ACCOUNT, SEED),
             '{:24s} {:9s} {:>6s} {:>7s} {:>7s} {:>6s}'.format('run', 'tap', 'anh', 'Acc', 'F1', 'gio')]
    for step, st in status:
        if step[0] != 'run':
            continue
        tag = step[1]
        path = os.path.join(WORK, 'results', tag, 'test_results.json')
        if not os.path.isfile(path):
            lines.append('{:24s} {}'.format(tag, st))
            continue
        with open(path) as f:
            r = json.load(f)
        rows = [(d, r[d]['center_crop']) for d in r.get('datasets', []) if d in r]
        if len(rows) != 1:
            rows.append(('tat ca', r['center_crop']))
        for i, (name, m) in enumerate(rows):
            n = m.get('per_class', {}).get('weighted avg', {}).get('support', 0)
            lines.append('{:24s} {:9s} {:6.0f} {:7.2f} {:7.2f} {:>6s}'.format(
                tag if i == 0 else '', name, n, 100 * m['accuracy'], 100 * m['f1_macro'],
                '{:.2f}'.format(DURATION[tag]) if i == 0 and tag in DURATION else ''))
    path = os.path.join(WORK, 'results', 'probe_detail_results.json')
    if os.path.isfile(path):
        with open(path) as f:
            p = json.load(f)
        lines += ['', '--- phep thu chi tiet (test Acc) ---']
        for src in ('frozen', 'tuned'):
            if p.get(src):
                lines.append('{:6s} '.format(src) + ' | '.join('{} {:.2f}'.format(k, v['acc']) for k, v in p[src].items()))
        for name, r in p.get('position', {}).items():
            lines.append('vi tri {:10s}: ngau nhien {:.2f}, giua {:.2f}, bo chon {:.2f}, oracle {:.2f}; '
                         'bo chon - ngau nhien {:+.2f} [{:+.2f}, {:+.2f}]'.format(
                             name, r['random'], r['center'], r['selector'], r['oracle'], r['selector_minus_random'],
                             *r['selector_minus_random_ci95']))
        d = p.get('decision', {})
        lines.append('ket luan: nhanh chi tiet {} ({:+.2f} diem), RL chon vi tri {} ({:+.2f} diem)'.format(
            'GO' if d.get('detail_go') else 'NO-GO', d.get('detail_gain', float('nan')),
            'dang xay' if d.get('rl_go') else 'khong dang', d.get('selector_gain', float('nan'))))
    lines += ['', '--- cac buoc ---'] + ['{:6s} {:24s} {}'.format(s[0], s[1], st) for s, st in status]
    lines.append('tong thoi gian {:.2f} h'.format(hours()))
    text = '\n'.join(lines)
    os.makedirs(os.path.join(WORK, 'results'), exist_ok=True)
    with open(os.path.join(WORK, 'results', 'summary.txt'), 'w') as f:
        f.write(text + '\n')
    print(text, flush=True)


def main():
    all_plans = plans(SEED)
    if ACCOUNT not in all_plans:
        raise SystemExit('usage: python kaggle_session.py tk1|tk2')
    for k, v in DEFAULTS.items():
        os.environ.setdefault(k, v)
    os.environ.update(SEED=SEED, SAVE_ROOT=os.path.join(WORK, 'FGVC') + '/')
    log('account {}, seed {}, {} steps'.format(ACCOUNT, SEED, len(all_plans[ACCOUNT])))

    has_pandora = prepare_data()
    status = []
    for step in all_plans[ACCOUNT]:
        left = LIMIT_H - hours()
        need = 1.3 if step[0] == 'probe' else 1.6 * max(DURATION.values(), default=0.0)
        if step[0] == 'run' and 'pandora' in step[2] and not has_pandora:
            st = 'skipped: Pandora18k not available'
        elif left < need:
            st = 'skipped: {:.1f} h left, about {:.1f} h needed - run the notebook again with this output as input'.format(
                left, need)
        else:
            log('{} {}'.format(step[0], ' '.join(step[1:])))
            try:
                st = do_probe(step[1]) if step[0] == 'probe' else do_run(*step[1:])
            except Exception as e:
                st = 'FAILED: {!r}'.format(e)
        log('{} {} -> {}'.format(step[0], step[1], st))
        status.append((step, st))
    summary(status)


if __name__ == '__main__':
    main()
