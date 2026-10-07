##################################################
# Training Config
##################################################
workers = 4  # number of Dataloader workers
epochs = 100  # number of epochs
batch_size = 16  # batch size (reduced from paper's 32: AGM+SCLM combined
# push a 15GB GPU right to its limit at batch 32, with OOM recurring a few
# batches in even after the AGM memory fix + CUDA allocator tuning. Kept
# at 16 to match exp/full-paper-self-attention's batch_size exactly, so
# the two runs are comparable apples-to-apples)
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
tag = 'wikiart-fullpaper-baseline'  # new tag: the paper's own baseline -
# AGM+SCLM+CCAM, no part-relationship module added (32 parts just
# concatenated, as the paper itself does) - reference point to measure
# whether exp/full-paper-self-attention's added module is worth it.
# Fresh tag/dir so this never resumes into or overwrites any other run

# saving directory of .ckpt models
save_dir = '/kaggle/working/FGVC/wikiart_fullpaper_baseline/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
# points at the file ModelCheckpoint writes on val-accuracy improvement;
# train.py only loads it if os.path.isfile(ckpt), so this is a no-op on
# the very first run and auto-resumes on every run after that
ckpt = save_dir + model_name
visual_path = None
