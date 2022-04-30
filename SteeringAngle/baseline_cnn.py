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
plt.switch_backend('agg')
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

#-------------------------------
# output folders

if args.fake_data_path!="None" and args.nfake>0:
    fake_data_name = args.fake_data_path.split("/")[-1]
    output_directory = os.path.join(args.root_path, 'output/CNN/{}_useNfake_{}'.format(fake_data_name, args.nfake))
    cnn_info = '{}_lr_{}_decay_{}_finetune_{}'.format(args.cnn_name, args.lr_base, args.weight_decay, args.finetune)
else:
    output_directory = os.path.join(args.root_path, 'output/CNN/vanilla')
    cnn_info = '{}_lr_{}_decay_{}'.format(args.cnn_name, args.lr_base, args.weight_decay)
os.makedirs(output_directory, exist_ok=True)



#######################################################################################
'''                                Data loader                                      '''
#######################################################################################
print('\n Loading real data...')
hf = h5py.File(os.path.join(args.data_path, 'SteeringAngle_64x64_prop_0.8.h5'), 'r')
images_train = hf['images_train'][:]
labels_train = hf['labels_train'][:]
images_test = hf['images_test'][:]
labels_test = hf['labels_test'][:]
hf.close()

min_label = -88.13
max_label = 97.92
assert labels_train.min()>= min_label and labels_test.min()>=min_label
assert labels_train.max()<= max_label and labels_test.max()<=max_label


# some functions
def fn_norm_labels(labels):
    '''
    labels: unnormalized labels
    '''
    
    shift_value = np.abs(min_label)
    labels_after_shift = labels + shift_value
    max_label_after_shift = max_label + shift_value
    
    return labels_after_shift/max_label_after_shift


def fn_denorm_labels(labels):
    '''
    labels: normalized labels; numpy array
    '''
    shift_value = np.abs(min_label)
    max_label_after_shift = max_label + shift_value
    labels = labels * max_label_after_shift
    labels = labels - shift_value
    
    return labels


## normalize to [0,1]
labels_train = fn_norm_labels(labels_train)
labels_test = fn_norm_labels(labels_test)

# print(labels_train.min(), labels_train.max())
# print(labels_test.min(), labels_test.max())


## number of real images
nreal = len(labels_train)
assert len(labels_train) == len(images_train)

## load fake data if needed
if args.fake_data_path != 'None':
    print("\n Start loading fake data: {}...".format(args.fake_data_path))
    hf = h5py.File(args.fake_data_path, 'r')
    fake_images = hf['fake_images'][:]
    fake_labels = hf['fake_labels'][:]
    hf.close()
    print('\n Fake images: {}, min {}, max {}.'.format(fake_images.shape, fake_images.min(), fake_images.max()))
    print('\n Fake labels: {}, min {}, max {}.'.format(fake_labels.shape, fake_labels.min(), fake_labels.max()))
    assert np.max(fake_images)>1 and np.min(fake_images)>=0
    
    indx_fake = np.arange(len(fake_labels))
    np.random.shuffle(indx_fake)
    indx_fake = indx_fake[0:args.nfake]
    
    fake_images = fake_images[indx_fake]
    fake_labels = fake_labels[indx_fake]
    assert len(fake_images)==len(fake_labels)
    fake_labels = fn_norm_labels(fake_labels)
    fake_labels = np.clip(fake_labels, 0, 1)
    
    print("\n Range of normalized fake labels: ", fake_labels.min(), fake_labels.max())

    ## combine fake and real
    images_train = np.concatenate((images_train, fake_images), axis=0)
    labels_train = np.concatenate((labels_train, fake_labels))
## if

## data loader for the training set and test set
# trainset = IMGs_dataset(images_train, labels_train, normalize=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
testset = IMGs_dataset(images_test, labels_test, normalize=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)

## info of training set and test set
print("\n Training set: {}x{}x{}x{}; Testing set: {}x{}x{}x{}.".format(images_train.shape[0], images_train.shape[1], images_train.shape[2], images_train.shape[3], images_test.shape[0], images_test.shape[1], images_test.shape[2], images_test.shape[3]))




#######################################################################################
'''                                  CNN Training                                   '''
#######################################################################################

### model initialization

net = cnn_dict[args.cnn_name]()
num_parameters = count_parameters(net)

### start training
if args.finetune:
    filename_ckpt = os.path.join(output_directory, 'ckpt_{}_epoch_{}_finetune_True_last.pth'.format(args.cnn_name, args.epochs))
    ## load pre-trained model
    checkpoint = torch.load(args.init_model_path)
    net.load_state_dict(checkpoint['net_state_dict'])
else:
    filename_ckpt = os.path.join(output_directory, 'ckpt_{}_epoch_{}_last.pth'.format(args.cnn_name, args.epochs))
print('\n' + filename_ckpt)

# training
if not os.path.isfile(filename_ckpt):
    print("\n Start training the {} >>>".format(args.cnn_name))

    path_to_ckpt_in_train = output_directory + '/ckpts_in_train/{}'.format(cnn_info)    
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    train_cnn(net=net, net_name=args.cnn_name, train_images=images_train, train_labels=labels_train, testloader=testloader, epochs=args.epochs, resume_epoch=args.resume_epoch, save_freq=args.save_freq, batch_size=args.batch_size_train, lr_base=args.lr_base, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, path_to_ckpt = path_to_ckpt_in_train, fn_denorm_labels=fn_denorm_labels)

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
test_mae = test_cnn(net, testloader, fn_denorm_labels=fn_denorm_labels, verbose=True)
print("\n Test MAE {}.".format(test_mae))



test_results_logging_fullpath = output_directory + '/test_results_{}_MAE_{:.3f}.txt'.format(cnn_info, test_mae)
if not os.path.isfile(test_results_logging_fullpath):
    test_results_logging_file = open(test_results_logging_fullpath, "w")
    test_results_logging_file.close()
with open(test_results_logging_fullpath, 'a') as test_results_logging_file:
    test_results_logging_file.write("\n===================================================================================================")
    test_results_logging_file.write("\n {}; num paras: {}; seed: {} \n".format(cnn_info, num_parameters, args.seed))
    print(args, file=test_results_logging_file)
    test_results_logging_file.write("\n Test MAE {}.".format(test_mae))






print("\n===================================================================================================")
