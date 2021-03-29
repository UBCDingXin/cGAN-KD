print("\n===================================================================================================")

import os
import argparse
import shutil
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import autograd
from torchvision.utils import save_image
import csv
from tqdm import tqdm
import gc
import h5py

### import my stuffs ###
from opts import cnn_opts
from models import *
from utils import *
from train_cnn import train_cnn, test_cnn





#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = cnn_opts()
print(args)

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# CNN settings
## lr decay scheme
lr_decay_epochs = (args.lr_decay_epochs).split("_")
lr_decay_epochs = [int(epoch) for epoch in lr_decay_epochs]
## save freq
save_freq = (args.save_freq).split("_")
save_freq = [int(epoch) for epoch in save_freq]

#-------------------------------
# output folders
output_directory = os.path.join(args.root_path, 'output/NTrainPerLabel_{}'.format(args.max_num_img_per_label))
os.makedirs(output_directory, exist_ok=True)
save_models_folder = os.path.join(output_directory, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)



#######################################################################################
'''                                Data loader                                      '''
#######################################################################################
print('\n Loading real data...')
hf = h5py.File(os.path.join(args.data_path, 'RC-49_64x64_{}.h5'.format(args.max_num_img_per_label)), 'r')
images_odd = hf['images_odd'][:]
labels_odd = hf['labels_odd'][:]
types_odd = hf['types_odd'][:]
images_even = hf['images_even'][:]
labels_even = hf['labels_even'][:]
types_even = hf['types_even'][:]
indx_odd_train = hf['indx_odd_train'][:]
indx_odd_subtrain = hf['indx_odd_subtrain'][:]
indx_odd_subvalid = hf['indx_odd_subvalid'][:]
indx_odd_rest = hf['indx_odd_rest'][:]
hf.close()
assert labels_odd.max()<=args.max_label and labels_even.max()<=args.max_label
assert len(images_odd) == len(labels_odd) and len(images_odd) == len(types_odd)
assert len(images_even) == len(labels_even) and len(images_even) == len(types_even)

## normalize to [0,1]
labels_odd /= args.max_label
labels_even /= args.max_label

if args.validaiton_mode:
    images_train = images_odd[indx_odd_subtrain]
    labels_train = labels_odd[indx_odd_subtrain]
    images_test = images_odd[indx_odd_subvalid]
    labels_test = labels_odd[indx_odd_subvalid]
else:
    images_train = images_odd[indx_odd_train]
    labels_train = labels_odd[indx_odd_train]
    images_test = np.concatenate((images_odd[indx_odd_rest], images_even), axis=0)
    labels_test = np.concatenate((labels_odd[indx_odd_rest], labels_even), axis=0)

## load fake data if needed
if args.fake_dataset_name != 'None':
    fake_dataset_path = os.path.join(output_directory, 'fake_data/fake_RC49_NTrainPerLabel_{}_{}_seed_{}.h5'.format(args.max_num_img_per_label, args.fake_dataset_name, args.seed))
    print("\n Start loading fake data: {}...".format(fake_dataset_path))
    hf = h5py.File(fake_dataset_path, 'r')
    fake_images = hf['fake_images'][:]
    fake_labels = hf['fake_labels'][:]
    hf.close()
    print('\n Fake images: {}, min {}, max {}.'.format(fake_images.shape, fake_images.min(), fake_images.max()))
    print('\n Fake labels: {}, min {}, max {}.'.format(fake_labels.shape, fake_labels.min(), fake_labels.max()))
    assert np.max(fake_images)>1 and np.min(fake_images)>=0 and fake_labels.min()>=0
    if args.nfake<len(fake_images):
        indx_fake = np.random.choice(np.arange(len(fake_images)), size=int(args.nfake), replace=False)
        fake_images = fake_images[indx_fake]
        fake_labels = fake_labels[indx_fake]
    nfake = len(fake_images)
    assert nfake == args.nfake
    fake_labels /= args.max_label

    ## combine fake and real
    images_train = np.concatenate((images_train, fake_images), axis=0)
    labels_train = np.concatenate((labels_train, fake_labels))

    ## name for the training set
    fake_dataset_name_tmp = (args.fake_dataset_name).split('_')
    fake_dataset_name_tmp[-1] = str(nfake)
    fake_dataset_name_tmp = '_'.join(fake_dataset_name_tmp)
    train_dataset_name = 'real_nreal_{}_fake_{}'.format(len(images_train), fake_dataset_name_tmp)
else:
    nfake = 0
    train_dataset_name = 'real_nreal_{}_fake_None'.format(len(images_train))
## if

## data loader for the training set and test set
trainset = IMGs_dataset(images_train, labels_train, normalize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
testset = IMGs_dataset(images_test, labels_test, normalize=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)

## info of training set and test set
print("\n Training dataset name: {}.".format(train_dataset_name))
print("\n Training set: {}; {}x{}x{}x{}; Testing set: {}x{}x{}x{}.".format(train_dataset_name, images_train.shape[0], images_train.shape[1], images_train.shape[2], images_train.shape[3], images_test.shape[0], images_test.shape[1], images_test.shape[2], images_test.shape[3]))




#######################################################################################
'''                                  CNN Training                                   '''
#######################################################################################

### model initialization
net = cnn_initialization(args.cnn_name, img_size = args.img_size)
num_parameters = count_parameters(net)

### start training
filename_ckpt = os.path.join(save_models_folder, 'ckpt_baseline_{}_epoch_{}_seed_{}_data_{}_validation_{}.pth'.format(args.cnn_name, args.epochs, args.seed, train_dataset_name, args.validaiton_mode))
print('\n' + filename_ckpt)

# training
if not os.path.isfile(filename_ckpt):
    print("\n Start training the {} >>>".format(args.cnn_name))

    path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_baseline_{}_seed_{}_data_{}_validation_{}'.format(args.cnn_name, args.seed, train_dataset_name, args.validaiton_mode)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    net = train_cnn(net, args.cnn_name, trainloader, testloader, epochs=args.epochs, resume_epoch=args.resume_epoch, save_freq=save_freq, lr_base=args.lr_base, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, path_to_ckpt = path_to_ckpt_in_train, max_label=90)

    # store model
    torch.save({
        'net_state_dict': net.state_dict(),
    }, filename_ckpt)
    print("\n End training CNN.")
else:
    print("\n Loading pre-trained {}.".format(args.cnn_name))
    checkpoint = torch.load(filename_ckpt)
    net.load_state_dict(checkpoint['net_state_dict'])
#end if

# testing
test_mae = test_cnn(net, testloader, max_label=args.max_label, verbose=True)
print("\n Test MAE {}.".format(test_mae))



test_results_logging_fullpath = output_directory + '/Test_results_baseline_{}_seed_{}_data_{}_validation_{}.txt'.format(args.cnn_name, args.seed, train_dataset_name, args.validaiton_mode)
if not os.path.isfile(test_results_logging_fullpath):
    test_results_logging_file = open(test_results_logging_fullpath, "w")
    test_results_logging_file.close()
with open(test_results_logging_fullpath, 'a') as test_results_logging_file:
    test_results_logging_file.write("\n===================================================================================================")
    test_results_logging_file.write("\n Baseline {}; num paras: {}; seed: {} \n".format(args.cnn_name, num_parameters, args.seed))
    print(args, file=test_results_logging_file)
    test_results_logging_file.write("\n Test MAE {}.".format(test_mae))






print("\n===================================================================================================")
