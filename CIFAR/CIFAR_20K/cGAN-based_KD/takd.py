'''

Teacher Assistant Knowledge Distillation: TAKD

'''

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

from opts import takd_opts
from utils import *
from models import *
from train_cnn import train_cnn, test_cnn



#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = takd_opts()
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
images_train = hf['images_train'][:]
labels_train = hf['labels_train'][:]
hf.close()

## load fake data if needed
if args.fake_dataset_name != 'None':
    fake_dataset_path = os.path.join(args.root_path, 'data', 'CIFAR{}_ntrain_{}_{}_seed_{}.h5'.format(args.num_classes, args.ntrain, args.fake_dataset_name, args.seed))
    hf = h5py.File(fake_dataset_path, 'r')
    fake_images = hf['fake_images'][:]
    fake_labels = hf['fake_labels'][:]
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
if args.num_classes == 10:
    cifar_testset = torchvision.datasets.CIFAR10(root = os.path.join(args.root_path, 'data'), train=False, download=True)
elif args.num_classes == 100:
    cifar_testset = torchvision.datasets.CIFAR100(root = os.path.join(args.root_path, 'data'), train=False, download=True)

images_test = cifar_testset.data
images_test = np.transpose(images_test, (0, 3, 1, 2))
labels_test = np.array(cifar_testset.targets)

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
'''                   Train Teacher Asssistant and Student Net                      '''
#######################################################################################

''' load teacher net '''
net_teacher = cnn_initialization(args.teacher, num_classes=args.num_classes, img_size=args.img_size)
num_parameters_teacher = count_parameters(net_teacher)
checkpoint = torch.load(os.path.join(save_models_folder, args.teacher_ckpt_filename))
net_teacher.load_state_dict(checkpoint['net_state_dict'])
print("\n Loaded {} as teacher net: {} parameters..".format(args.teacher, num_parameters_teacher))

test_acc_teacher = test_cnn(net_teacher, testloader_cnn, extract_feature=False, verbose=True)
test_err_teacher = 100.0-test_acc_teacher
print("\n Teacher test error rate {}.".format(test_err_teacher))


''' init. TA net '''
net_TA = cnn_initialization(args.teacher_assistant, num_classes=args.num_classes, img_size=args.img_size)
num_parameters_TA = count_parameters(net_TA)

''' init. student net '''
net_student = cnn_initialization(args.student, num_classes=args.num_classes, img_size=args.img_size)
num_parameters_student = count_parameters(net_student)


print("\n -----------------------------------------------------------------------------------------")
print("\n Step 1: Start training {} as TA net with {} params >>>".format(args.teacher_assistant, num_parameters_TA))

# Filename
filename_cnn_ckpt = save_models_folder + '/ckpt_TA_{}_epoch_{}_transform_{}_lambda_{}_T_{}_seed_{}_with_teacher_{}_data_{}.pth'.format(args.teacher_assistant, args.epochs, args.transform, args.assistant_lambda_kd, args.assistant_T_kd, args.seed, args.teacher, train_dataset_name)
print('\n'+filename_cnn_ckpt)

# training
if not os.path.isfile(filename_cnn_ckpt):
    print("\n Start training the {} as TA >>>".format(args.teacher_assistant))

    path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_student_{}_transform_{}_lambda_{}_T_{}_seed_{}_with_teacher_{}_data_{}'.format(args.teacher_assistant, args.transform, args.assistant_lambda_kd, args.assistant_T_kd, args.seed, args.teacher, train_dataset_name)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    net_TA = train_cnn(net_TA, args.teacher_assistant, trainloader_cnn, testloader_cnn, epochs=args.epochs, resume_epoch=args.resume_epoch_1, save_freq=save_freq, lr_base=args.lr_base, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, seed = args.seed, extract_feature=False, path_to_ckpt = path_to_ckpt_in_train, net_teacher=net_teacher, lambda_kd=args.assistant_lambda_kd, T_kd=args.assistant_T_kd)

    # store model
    torch.save({
        'net_state_dict': net_TA.state_dict(),
    }, filename_cnn_ckpt)
    print("\n End training CNN.")
else:
    print("\n Loading pre-trained {}.".format(args.teacher_assistant))
    checkpoint = torch.load(filename_cnn_ckpt)
    net_TA.load_state_dict(checkpoint['net_state_dict'])
#end if

# testing
test_acc_TA = test_cnn(net_TA, testloader_cnn, extract_feature=False, verbose=True)
test_err_TA = 100.0-test_acc_TA
print("\n TA test error rate {}.".format(test_err_TA))






print("\n -----------------------------------------------------------------------------------------")
print("\n Step 2: Start training {} as student net with {} params >>>".format(args.student, num_parameters_student))

# Filename
filename_cnn_ckpt = save_models_folder + '/ckpt_student_{}_epoch_{}_transform_{}_lambda_{}_T_{}_seed_{}_with_teacher_{}_TA_{}_data_{}.pth'.format(args.student, args.epochs, args.transform, args.student_lambda_kd, args.student_T_kd, args.seed, args.teacher, args.teacher_assistant, train_dataset_name)
print('\n'+filename_cnn_ckpt)

# training
if not os.path.isfile(filename_cnn_ckpt):
    print("\n Start training the {} as student >>>".format(args.student))

    path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_student_{}_transform_{}_lambda_{}_T_{}_seed_{}_with_teacher_{}_TA_{}_data_{}'.format(args.student, args.transform, args.student_lambda_kd, args.student_T_kd, args.seed, args.teacher, args.teacher_assistant, train_dataset_name)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    net_student = train_cnn(net_student, args.student, trainloader_cnn, testloader_cnn, epochs=args.epochs, resume_epoch=args.resume_epoch_2, save_freq=save_freq, lr_base=args.lr_base, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, seed = args.seed, extract_feature=False, path_to_ckpt = path_to_ckpt_in_train, net_teacher=net_TA, lambda_kd=args.student_lambda_kd, T_kd=args.student_T_kd)

    # store model
    torch.save({
        'net_state_dict': net_student.state_dict(),
    }, filename_cnn_ckpt)
    print("\n End training CNN.")
else:
    print("\n Loading pre-trained {}.".format(args.student))
    checkpoint = torch.load(filename_cnn_ckpt)
    net_student.load_state_dict(checkpoint['net_state_dict'])
#end if

# testing
test_acc_student = test_cnn(net_student, testloader_cnn, extract_feature=False, verbose=True)
test_err_student = 100.0-test_acc_student
print("\n Student test error rate {}.".format(test_err_student))




''' Dump test results '''
test_results_logging_fullpath = args.root_path + '/Output_CIFAR{}/Test_results_TAKD_student_{}_teacher_{}_TA_{}_seed_{}_data_{}.txt'.format(args.num_classes, args.student, args.teacher, args.teacher_assistant, args.seed, train_dataset_name)
if not os.path.isfile(test_results_logging_fullpath):
    test_results_logging_file = open(test_results_logging_fullpath, "w")
    test_results_logging_file.close()
with open(test_results_logging_fullpath, 'a') as test_results_logging_file:
    test_results_logging_file.write("\n===================================================================================================")
    test_results_logging_file.write("\n Teacher: {} ({} params); TA: {} ({} params); Student: {} ({} params); seed: {} \n".format(args.teacher, num_parameters_teacher, args.teacher_assistant, num_parameters_TA, args.student, num_parameters_student, args.seed))
    print(args, file=test_results_logging_file)
    test_results_logging_file.write("\n Teacher test error rate {}.".format(test_err_teacher))
    test_results_logging_file.write("\n TA test error rate {}.".format(test_err_TA))
    test_results_logging_file.write("\n Student test error rate {}.".format(test_err_student))



print("\n ===================================================================================================")
