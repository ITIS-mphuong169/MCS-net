##################################################
# Training Config
##################################################
workers = 4  # number of Dataloader workers
epochs = 100  # number of epochs
batch_size = 32  # batch size
learning_rate = 1e-3  # initial learning rate

##################################################
# Model Config
##################################################
image_size = (224, 224)  # size of training images
net = 'resnet101'  # inception_mixed_6e
num_attentions = 32  # number of attention maps
beta = 5e-2  # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'wikiart-cls-token'  # new tag/dir: CLS-token readout added on top of
# the self-attention run (commit c24644c) - classifier reads only the CLS
# token's output instead of flattening all 32 parts. Kept separate so this
# never resumes into or overwrites the self-attention/isab-fair/gat-fair runs

# saving directory of .ckpt models
save_dir = '/kaggle/working/FGVC/wikiart_cls_token/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
# points at the file ModelCheckpoint writes on val-accuracy improvement;
# train.py only loads it if os.path.isfile(ckpt), so this is a no-op on
# the very first run and auto-resumes on every run after that
ckpt = save_dir + model_name
visual_path = None
