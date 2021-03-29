print("\n ===================================================================================================")

import argparse
import os
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import autograd
from torchvision.utils import save_image
from tqdm import tqdm
import gc
from itertools import groupby
import multiprocessing
import h5py
import pickle
import copy

from opts import cnn_opts
from utils import *
from models import *
from train_cnn import train_cnn, test_cnn


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = cnn_opts()
print(args)

#--------------------------------
# system
# NCPU = multiprocessing.cpu_count()
NCPU = args.num_workers

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
save_models_folder = args.root_path + '/Output_CIFAR{}/saved_models'.format(args.num_classes)
os.makedirs(save_models_folder, exist_ok=True)




#######################################################################################
'''                                  Load Data                                      '''
#######################################################################################
## generate subset of real data
trainset_h5py_file = args.root_path + '/data/CIFAR{}_trainset_{}_seed_{}.h5'.format(args.num_classes, args.ntrain, args.seed)
hf = h5py.File(trainset_h5py_file, 'r')
images_all = hf['images_train'][:]
labels_all = hf['labels_train'][:]
subtrain_indx = hf['subtrain_indx'][:]
subvalid_indx = hf['subvalid_indx'][:]
hf.close()
assert np.max(images_all)>1 and np.min(labels_all)>=0

if args.validaiton_mode:
    images_train = images_all[subtrain_indx]
    labels_train = labels_all[subtrain_indx]
else:
    images_train = images_all
    labels_train = labels_all


## load fake data if needed
if args.fake_dataset_name != 'None':
    fake_dataset_path = os.path.join(args.root_path, 'data', 'CIFAR{}_ntrain_{}_{}_seed_{}.h5'.format(args.num_classes, args.ntrain, args.fake_dataset_name, args.seed))
    hf = h5py.File(fake_dataset_path, 'r')
    fake_images = hf['fake_images'][:]
    fake_labels = hf['fake_labels'][:]
    assert np.max(fake_images)>1 and np.min(fake_images)>=0
    hf.close()
    if args.nfake<len(fake_images):
        indx_fake = []
        for i in range(args.num_classes):
            indx_fake_i = np.where(fake_labels==i)[0]
            if i != (args.num_classes-1):
                nfake_i = args.nfake//args.num_classes
            else:
                nfake_i = args.nfake-(args.nfake//args.num_classes)*i
            indx_fake_i = np.random.choice(indx_fake_i, size=min(int(nfake_i), len(indx_fake_i)), replace=False)
            indx_fake.append(indx_fake_i)
        #end for i
        indx_fake = np.concatenate(indx_fake)
        # indx_fake = np.random.choice(np.arange(len(fake_images)), size=int(args.nfake), replace=False)
        fake_images = fake_images[indx_fake]
        fake_labels = fake_labels[indx_fake]
    nfake = len(fake_images)

    ## combine fake and real
    images_train = np.concatenate((images_train, fake_images), axis=0)
    labels_train = np.concatenate((labels_train, fake_labels))

    ## name for the training set
    fake_dataset_name_tmp = (args.fake_dataset_name).split('_')
    fake_dataset_name_tmp[-1] = str(nfake)
    fake_dataset_name_tmp = '_'.join(fake_dataset_name_tmp)
    train_dataset_name = 'real_nreal_{}_fake_{}'.format(args.ntrain, fake_dataset_name_tmp)
else:
    nfake = 0
    train_dataset_name = 'real_nreal_{}_fake_None'.format(args.ntrain)
## if

## compute the mean and std for normalization
train_means = []
train_stds = []
for i in range(3):
    images_i = images_train[:,i,:,:]
    images_i = images_i/255.0
    train_means.append(np.mean(images_i))
    train_stds.append(np.std(images_i))
## for i

## Test set
if not args.validaiton_mode:
    if args.num_classes == 10:
        cifar_testset = torchvision.datasets.CIFAR10(root = os.path.join(args.root_path, 'data'), train=False, download=True)
    elif args.num_classes == 100:
        cifar_testset = torchvision.datasets.CIFAR100(root = os.path.join(args.root_path, 'data'), train=False, download=True)
    images_test = cifar_testset.data
    images_test = np.transpose(images_test, (0, 3, 1, 2))
    labels_test = np.array(cifar_testset.targets)
else:
    images_test = images_all[subvalid_indx]
    labels_test = labels_all[subvalid_indx]

## info of training set and test set
print("\n Training set: {}; {}x{}x{}x{}; Testing set: {}x{}x{}x{}.".format(train_dataset_name, images_train.shape[0], images_train.shape[1], images_train.shape[2], images_train.shape[3], images_test.shape[0], images_test.shape[1], images_test.shape[2], images_test.shape[3]))



''' transformations '''
if args.transform:
    transform_cnn_train = transforms.Compose([
                transforms.RandomCrop((args.img_size, args.img_size), padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(train_means, train_stds),
                ])
else:
    transform_cnn_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(train_means, train_stds),
                ])

transform_cnn_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
            ])


trainset_cnn = IMGs_dataset(images_train, labels_train, transform=transform_cnn_train)
trainloader_cnn = torch.utils.data.DataLoader(trainset_cnn, batch_size=args.batch_size_train, shuffle=True, num_workers=NCPU)

testset_cnn = IMGs_dataset(images_test, labels_test, transform=transform_cnn_test)
testloader_cnn = torch.utils.data.DataLoader(testset_cnn, batch_size=args.batch_size_test, shuffle=False, num_workers=NCPU)




#######################################################################################
'''                                  Baseline CNN                                   '''
#######################################################################################

print("\n -----------------------------------------------------------------------------------------")
print("\n Start training cnn >>>")

#######################################
''' cnn training '''

# initialize cnn
net = cnn_initialization(args.cnn_name, num_classes=args.num_classes, img_size=args.img_size)
num_parameters = count_parameters(net)

# Filename
if args.validaiton_mode:
    filename_cnn_ckpt = save_models_folder + '/ckpt_baseline_{}_epoch_{}_transform_{}_seed_{}_data_{}_validation.pth'.format(args.cnn_name, args.epochs, args.transform, args.seed, train_dataset_name)
else:
    filename_cnn_ckpt = save_models_folder + '/ckpt_baseline_{}_epoch_{}_transform_{}_seed_{}_data_{}.pth'.format(args.cnn_name, args.epochs, args.transform, args.seed, train_dataset_name)
print('\n'+filename_cnn_ckpt)

# training
if not os.path.isfile(filename_cnn_ckpt):
    print("\n Start training the {} >>>".format(args.cnn_name))

    if args.validaiton_mode:
        path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_baseline_{}_seed_{}_data_{}_validation'.format(args.cnn_name, args.seed, train_dataset_name)
    else:
        path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_baseline_{}_seed_{}_data_{}'.format(args.cnn_name, args.seed, train_dataset_name)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    net = train_cnn(net, args.cnn_name, trainloader_cnn, testloader_cnn, epochs=args.epochs, resume_epoch=args.resume_epoch, save_freq=save_freq, lr_base=args.lr_base, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, seed = args.seed, extract_feature=False, path_to_ckpt = path_to_ckpt_in_train)

    # store model
    torch.save({
        'net_state_dict': net.state_dict(),
    }, filename_cnn_ckpt)
    print("\n End training CNN.")
else:
    print("\n Loading pre-trained {}.".format(args.cnn_name))
    checkpoint = torch.load(filename_cnn_ckpt)
    net.load_state_dict(checkpoint['net_state_dict'])
#end if

# testing
test_acc = test_cnn(net, testloader_cnn, extract_feature=False, verbose=True)
test_err = 100.0-test_acc
print("\n Test error rate {}.".format(test_err))


test_results_logging_fullpath = args.root_path + '/Output_CIFAR{}/Test_results_baseline_{}_seed_{}_data_{}_validation_{}.txt'.format(args.num_classes, args.cnn_name, args.seed, train_dataset_name, args.validaiton_mode)
if not os.path.isfile(test_results_logging_fullpath):
    test_results_logging_file = open(test_results_logging_fullpath, "w")
    test_results_logging_file.close()
with open(test_results_logging_fullpath, 'a') as test_results_logging_file:
    test_results_logging_file.write("\n===================================================================================================")
    test_results_logging_file.write("\n Baseline {}; num paras: {}; seed: {} \n".format(args.cnn_name, num_parameters, args.seed))
    print(args, file=test_results_logging_file)
    test_results_logging_file.write("\n Test error rate {}.".format(test_err))


print("\n===================================================================================================")
