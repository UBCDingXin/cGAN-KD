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

#-------------------------------
# some functions
def fn_norm_labels(labels):
    '''
    labels: unnormalized labels
    '''
    return labels/args.max_label

def fn_denorm_labels(labels):
    '''
    labels: normalized labels; numpy array
    '''
    if isinstance(labels, np.ndarray):
        return (labels*args.max_label).astype(int)
    elif torch.is_tensor(labels):
        return (labels*args.max_label).type(torch.int)
    else:
        return int(labels*args.max_label)





#######################################################################################
'''                                Data loader                                      '''
#######################################################################################
print('\n Loading real data...')
hf = h5py.File(os.path.join(args.data_path, 'UTKFace_64x64_prop_0.8.h5'), 'r')
images_train = hf['images_train'][:]
labels_train = hf['labels_train'][:]
images_test = hf['images_test'][:]
labels_test = hf['labels_test'][:]
hf.close()

## unique labels
unique_labels = np.sort(np.array(list(set(labels_train))))

## for each age, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original training set has {} images; For each age, take no more than {} images>>>".format(len(images_train), image_num_threshold))

sel_indx = []
for i in tqdm(range(len(unique_labels))):
    indx_i = np.where(labels_train == unique_labels[i])[0]
    if len(indx_i)>image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    sel_indx.append(indx_i)
sel_indx = np.concatenate(sel_indx, axis=0)
images_train = images_train[sel_indx]
labels_train = labels_train[sel_indx]
print("\r {} training images left.".format(len(images_train)))

## normalize to [0,1]
labels_train = fn_norm_labels(labels_train)
labels_test = fn_norm_labels(labels_test)

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
    
    # ## take no more than args.nfake_per_label imgs for each label
    # indx_fake = []
    # for i in range(len(unique_labels)):
    #     label_i = unique_labels[i]
    #     indx_i = np.where(fake_labels==label_i)[0]
    #     np.random.shuffle(indx_i)
    #     if args.nfake_per_label<len(indx_i):
    #         indx_i = indx_i[0:args.nfake_per_label]
    #     indx_fake.append(indx_i)
    # ###end for i
    # indx_fake = np.concatenate(indx_fake)
    
    indx_fake = np.arange(len(fake_labels))
    np.random.shuffle(indx_fake)
    indx_fake = indx_fake[:args.nfake]
    
    fake_images = fake_images[indx_fake]
    fake_labels = fake_labels[indx_fake]
    
    
    ### visualize data distribution
    unique_labels_unnorm = np.arange(1,int(args.max_label)+1)
    frequencies = []
    for i in range(len(unique_labels_unnorm)):
        indx_i = np.where(fake_labels==unique_labels_unnorm[i])[0]
        frequencies.append(len(indx_i))
    frequencies = np.array(frequencies).astype(int)
    width = 0.8
    x = np.arange(1,int(args.max_label)+1)
    # plot data in grouped manner of bar type
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.grid(color='lightgrey', linestyle='--', zorder=0)
    ax.bar(unique_labels_unnorm, frequencies, width, align='center', color='tab:green', zorder=3)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(args.root_path, "{}_UseNFake_{}_data_dist.pdf".format(fake_data_name, args.nfake)))
    plt.close()

    print('\n Frequence of ages: MIN={}, MEAN={}, MAX={}.'.format(np.min(frequencies),np.mean(frequencies),np.max(frequencies)))

    
    
    assert len(fake_images)==len(fake_labels)
    fake_labels = fn_norm_labels(fake_labels)

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
