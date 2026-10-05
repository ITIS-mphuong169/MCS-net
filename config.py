##################################################
# Training Config
##################################################
workers = 4  # number of Dataloader workers
epochs = 100  # number of epochs
batch_size = 16  # batch size (reduced from paper's 32: AGM+SCLM+ISAB+RCAL
# combined push a 15GB GPU right to its limit at batch 32, with OOM
# recurring a few batches in even after the AGM memory fix + CUDA
# allocator tuning - 16 gives real headroom instead of a razor's edge)
learning_rate = 1e-3  # initial learning rate

##################################################
# Model Config
##################################################
image_size = (224, 224)  # size of training images
net = 'resnet101'  # inception_mixed_6e
num_attentions = 32  # number of attention maps
# loss weights matching the paper's eq. 17 (L = Lcls + lambda1*Lcon +
# lambda2*Lcausal), Table 2 values; lambda_rel is RCAL's own addition
# (no paper equivalent), kept separate from lambda2 for independent tuning
lambda1 = 0.8  # weight of the SCLM contrastive loss (paper Table 2)
lambda2 = 0.6  # weight of the CCAM causal loss (paper Table 2)
lambda_rel = 0.5  # weight of the RCAL relation-counterfactual loss term
tau = 0.07  # contrastive loss temperature (paper Table 2)

##################################################
# Dataset/Path Config
##################################################
tag = 'wikiart-rcal-v2'  # new tag: loss reworked to match the paper's eq. 17
# (dropped center_loss/feature_center, fixed lambda1/lambda2 mismatch) -
# keep this as a fresh W&B run instead of appending onto the old
# wikiart-rcal run, whose early history used the broken loss and would
# otherwise mix misleadingly with the corrected data in the same chart

# saving directory of .ckpt models
save_dir = '/kaggle/working/FGVC/wikiart_rcal_v2/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
# points at the file ModelCheckpoint writes on val-accuracy improvement;
# train.py only loads it if os.path.isfile(ckpt), so this is a no-op on
# the very first run and auto-resumes on every run after that
ckpt = save_dir + model_name
visual_path = None
