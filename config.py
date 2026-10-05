##################################################
# Training Config
##################################################
workers = 4  # number of Dataloader workers
epochs = 100  # number of epochs
batch_size = 32  # batch size (paper Table 2). Was reduced to 16 for the
# full-paper AGM+SCLM attempt (see exp/part-transformer), which pushed a
# 15GB GPU to its limit; without AGM, ISAB+RCAL alone matches the
# memory footprint of the already-stable plain-ISAB run (exp/part-
# transformer's early plain-ISAB commit), which ran fine at 32 - also
# lets this run compare directly on the Step axis with the other 3
# runs (all batch_size=32) without needing the epoch-axis workaround
learning_rate = 1e-3  # initial learning rate

##################################################
# Model Config
##################################################
image_size = (224, 224)  # size of training images
net = 'resnet101'  # inception_mixed_6e
num_attentions = 32  # number of attention maps
# loss weights: lambda2 matches the paper's CCAM causal term (Table 2);
# lambda_rel is RCAL's own addition (no paper equivalent). No lambda1/tau
# here - SCLM (contrastive loss) was dropped along with AGM, see
# exp/part-transformer for that full-paper attempt
lambda2 = 0.6  # weight of the CCAM-style causal loss (paper Table 2)
lambda_rel = 0.5  # weight of the RCAL relation-counterfactual loss term

##################################################
# Dataset/Path Config
##################################################
tag = 'wikiart-isab-rcal'  # new tag: simplified to ISAB+RCAL only (no
# AGM/SCLM/CCAM-shuffle - see exp/part-transformer for that attempt),
# fresh W&B run so it doesn't mix with either earlier attempt's history

# saving directory of .ckpt models
save_dir = '/kaggle/working/FGVC/wikiart_isab_rcal/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
# points at the file ModelCheckpoint writes on val-accuracy improvement;
# train.py only loads it if os.path.isfile(ckpt), so this is a no-op on
# the very first run and auto-resumes on every run after that
ckpt = save_dir + model_name
visual_path = None
