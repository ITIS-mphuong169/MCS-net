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
lambda_rel = 0.5  # weight of the RCAL relation-counterfactual loss term
lambda1 = 0.8  # weight of the SCLM contrastive loss (paper Table 2)
tau = 0.07  # contrastive loss temperature (paper Table 2)

##################################################
# Dataset/Path Config
##################################################
tag = 'wikiart-rcal'  # distinct run id: RCAL changes the loss, keep it a separate experiment from plain ISAB

# saving directory of .ckpt models
save_dir = '/kaggle/working/FGVC/wikiart_rcal/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
# points at the file ModelCheckpoint writes on val-accuracy improvement;
# train.py only loads it if os.path.isfile(ckpt), so this is a no-op on
# the very first run and auto-resumes on every run after that
ckpt = save_dir + model_name
visual_path = None
