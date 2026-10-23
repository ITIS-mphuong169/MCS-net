import os

##################################################
# Training Config
##################################################
workers = int(os.environ.get('WORKERS', 4))  # number of Dataloader workers
epochs = 30  # number of epochs (reduced from the paper's 100 - free GPU
# quota can't realistically cover 100 epochs for 2 branches being compared;
# 30 is enough to see a clear convergence trend while staying achievable.
# Both exp/full-paper-baseline and exp/full-paper-self-attention use the
# same value so the comparison between them stays fair)
batch_size = int(os.environ.get('BATCH_SIZE', 16))  # batch size (reduced from paper's 32: AGM+SCLM combined
# push a 15GB GPU right to its limit at batch 32, with OOM recurring a few
# batches in even after the AGM memory fix + CUDA allocator tuning - kept
# at 16 for headroom even though self-attention replacing ISAB+RCAL here
# frees up some memory, since GPU quota is too scarce to risk re-testing)
learning_rate = 1e-3  # initial learning rate

##################################################
# Model Config
##################################################
image_size = (224, 224)  # size of training images
net = 'resnet101'  # inception_mixed_6e
num_attentions = 32  # number of attention maps
# loss weights matching the paper's eq. 17 (L = Lcls + lambda1*Lcon +
# lambda2*Lcausal), Table 2 values
lambda1 = 0.8  # weight of the SCLM contrastive loss (paper Table 2)
lambda2 = 0.6  # weight of the CCAM causal loss (paper Table 2)
tau = 0.07  # contrastive loss temperature (paper Table 2)

##################################################
# Dataset/Path Config
##################################################
tag = 'wikiart-fullpaper-selfattn'  # new tag: full paper modules (AGM+SCLM+
# CCAM) combined with self-attention for part-relationship modeling - the
# attention mechanism confirmed best in a separate fair comparison against
# ISAB and GAT. RCAL (tried here previously) is dropped - confirmed to hurt
# accuracy regardless of which other modules were present. Fresh tag/dir so
# this never resumes into or overwrites the rcal-v2 run's checkpoint/history

# DATA_DIR: WikiArt root containing the 27 class folders
DATA_DIR = os.environ.get('DATA_DIR', '/kaggle/input/datasets/steubk/wikiart')

# saving directory of .ckpt models
# SAVE_ROOT: where checkpoints/logs go (Kaggle: /kaggle/working/FGVC/)
SAVE_ROOT = os.environ.get('SAVE_ROOT', '/kaggle/working/FGVC/').rstrip('/') + '/'
save_dir = SAVE_ROOT + 'wikiart_fullpaper_selfattn/'
# PART_ATTN=0 builds the paper's own baseline (32 parts concatenated straight
# into the FC, no part-relationship module) from this same codebase, so the
# two models differ only in that module
PART_ATTN = os.environ.get('PART_ATTN', '1') != '0'
if not PART_ATTN:
    tag = 'wikiart-fullpaper-baseline'
    save_dir = SAVE_ROOT + 'wikiart_fullpaper_baseline/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
# points at the file ModelCheckpoint writes on val-accuracy improvement;
# train.py only loads it if os.path.isfile(ckpt), so this is a no-op on
# the very first run and auto-resumes on every run after that
ckpt = save_dir + model_name
visual_path = None
