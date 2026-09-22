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
tag = 'wikiart'

# saving directory of .ckpt models
save_dir = '/kaggle/working/FGVC/wikiart/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
visual_path = None
# ckpt = save_dir + model_name
