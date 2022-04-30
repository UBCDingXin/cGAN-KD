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
from models import model_dict
from train_cnn import train_cnn, test_cnn



#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = takd_opts()
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
# load teacher model
def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-1].split('_')
    if segments[1] != 'wrn':
        return segments[1]
    else:
        return segments[1] + '_' + segments[2] + '_' + segments[3]

def load_teacher(model_path, num_classes):
    model_name = get_teacher_name(model_path)
    print('==> loading teacher model: {}'.format(model_name))
    model = model_dict[model_name](num_classes=num_classes)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model, model_name

''' load teacher net '''
net_teacher, model_name = load_teacher(args.teacher_ckpt_path, num_classes=args.num_classes)
num_parameters_teacher = count_parameters(net_teacher)
args.teacher = model_name

#-------------------------------
# output folders
if (not args.use_fake_data) or args.nfake<=0:
    save_folder = os.path.join(args.root_path, 'output/vanilla')
else:
    fake_data_name = args.fake_data_path.split('/')[-1]
    save_folder = os.path.join(args.root_path, 'output/{}_useNfake_{}'.format(fake_data_name, args.nfake))
setting_name = 'S_{}_TA_{}_T_{}_L1_{}_T1_{}_L2_{}_T2_{}'.format(args.student, args.assistant, args.teacher, args.assistant_lambda_kd, args.assistant_T_kd, args.student_lambda_kd, args.student_T_kd)
save_folder = os.path.join(save_folder, setting_name)
os.makedirs(save_folder, exist_ok=True)

save_intrain_folder = os.path.join(save_folder, 'ckpts_in_train')
os.makedirs(save_intrain_folder, exist_ok=True)


#######################################################################################
'''                                  Load Data                                      '''
#######################################################################################
## get real data
train_set = torchvision.datasets.CIFAR100(root=args.data_path, download=True, train=True)
real_images = train_set.data
real_images = np.transpose(real_images, (0, 3, 1, 2))
real_labels = np.array(train_set.targets)

## load fake data if needed
if args.use_fake_data:
    ## load fake data
    with h5py.File(args.fake_data_path, "r") as f:
        train_images = f['fake_images'][:]
        train_labels = f['fake_labels'][:]
    train_labels = train_labels.astype(int)
    
    if args.nfake<len(train_labels):
        indx_fake = []
        for i in range(args.num_classes):
            indx_fake_i = np.where(train_labels==i)[0]
            if i != (args.num_classes-1):
                nfake_i = args.nfake//args.num_classes
            else:
                nfake_i = args.nfake-(args.nfake//args.num_classes)*i
            indx_fake_i = np.random.choice(indx_fake_i, size=min(int(nfake_i), len(indx_fake_i)), replace=False)
            indx_fake.append(indx_fake_i)
        #end for i
        indx_fake = np.concatenate(indx_fake)
        train_images = train_images[indx_fake]
        train_labels = train_labels[indx_fake]
    
    ## concatenate
    train_images = np.concatenate((real_images, train_images), axis=0)
    train_labels = np.concatenate((real_labels, train_labels), axis=0)    
    train_labels = train_labels.astype(int)  
else:
    train_images = real_images
    train_labels = real_labels

## compute normalizing constants
train_means = []
train_stds = []
for i in range(3):
    images_i = train_images[:,i,:,:]
    images_i = images_i/255.0
    train_means.append(np.mean(images_i))
    train_stds.append(np.std(images_i))
## for i

print("\n Final training set's dimensiona: ", train_images.shape)

## make training set
if args.transform:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_means, train_stds),
    ])
else:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_means, train_stds),
    ])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_means, train_stds),
])
train_set = IMGs_dataset(train_images, train_labels, transform=train_transform)
assert len(train_set)==len(train_images)
del train_images, train_labels; gc.collect()

test_set = torchvision.datasets.CIFAR100(root=args.data_path, download=True, train=False, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_test, shuffle=False)



#######################################################################################
'''                                  Implement TAKD                                 '''
#######################################################################################

test_acc_teacher = test_cnn(net_teacher, test_loader, verbose=False)
print("\n Teacher test accuracy {}.".format(test_acc_teacher))

''' init. assistant net '''
net_assistant = model_dict[args.assistant](num_classes=args.num_classes)
num_parameters_assistant = count_parameters(net_assistant)

''' init. student net '''
net_student = model_dict[args.student](num_classes=args.num_classes)
num_parameters_student = count_parameters(net_student)

## student model name, how to initialize student model, etc.
assistant_model_path = args.assistant + "_epoch_{}".format(args.epochs)
student_model_path = args.student + "_epoch_{}".format(args.epochs)
if args.finetune:
    print("\n Initialize assistant and student models by pre-trained ones")
    assistant_model_path = os.path.join(save_folder, assistant_model_path+'_finetune_last.pth')
    student_model_path = os.path.join(save_folder, student_model_path+'_finetune_last.pth')
    ## load pre-trained model
    checkpoint = torch.load(args.init_assistant_path)
    net_assistant.load_state_dict(checkpoint['model'])
    checkpoint = torch.load(args.init_student_path)
    net_student.load_state_dict(checkpoint['model'])
else:
    assistant_model_path = os.path.join(save_folder, assistant_model_path+'_last.pth')
    student_model_path = os.path.join(save_folder, student_model_path+'_last.pth')
print('\n ' + assistant_model_path) 
print('\r ' + student_model_path)    


print("\n -----------------------------------------------------------------------------------------")
print("\n Step 1: Start training {} as TA net with {} params >>>".format(args.assistant, num_parameters_assistant))

# training
if not os.path.isfile(assistant_model_path):
    print("\n Start training the {} as TA >>>".format(args.assistant))

    net_assistant = train_cnn(net_assistant, args.assistant, train_loader, test_loader, epochs=args.epochs, resume_epoch=args.resume_epoch_1, save_freq=save_freq, lr_base=args.lr_base1, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, seed = args.seed, path_to_ckpt = save_intrain_folder, net_teacher=net_teacher, lambda_kd=args.assistant_lambda_kd, T_kd=args.assistant_T_kd)

    # store model
    torch.save({
        'model': net_assistant.state_dict(),
    }, assistant_model_path)
    print("\n End training CNN.")
else:
    print("\n Loading pre-trained {}.".format(args.assistant))
    checkpoint = torch.load(assistant_model_path)
    net_assistant.load_state_dict(checkpoint['model'])
#end if

# testing
test_acc_assistant = test_cnn(net_assistant, test_loader, verbose=False)
test_err_assistant = 100.0-test_acc_assistant
print("\n Assistant test accuracy {}.".format(test_acc_assistant))
print("\r Assistant test error rate {}.".format(test_err_assistant))




print("\n -----------------------------------------------------------------------------------------")
print("\n Step 2: Start training {} as student net with {} params >>>".format(args.student, num_parameters_student))

# training
if not os.path.isfile(student_model_path):
    print("\n Start training the {} as student >>>".format(args.student))

    net_student = train_cnn(net_student, args.student, train_loader, test_loader, epochs=args.epochs, resume_epoch=args.resume_epoch_2, save_freq=save_freq, lr_base=args.lr_base2, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, seed = args.seed, path_to_ckpt = save_intrain_folder, net_teacher=net_assistant, lambda_kd=args.student_lambda_kd, T_kd=args.student_T_kd)

    # store model
    torch.save({
        'model': net_student.state_dict(),
    }, student_model_path)
    print("\n End training CNN.")
else:
    print("\n Loading pre-trained {}.".format(args.student))
    checkpoint = torch.load(student_model_path)
    net_student.load_state_dict(checkpoint['model'])
#end if

# testing
test_acc_student = test_cnn(net_student, test_loader, verbose=False)
test_err_student = 100.0-test_acc_student
print("\n Student test accuracy {}.".format(test_acc_student))
print("\r Student test error rate {}.".format(test_err_student))



''' Dump test results '''
test_results_logging_fullpath = save_folder + '/test_results.txt'
if not os.path.isfile(test_results_logging_fullpath):
    test_results_logging_file = open(test_results_logging_fullpath, "w")
    test_results_logging_file.close()
with open(test_results_logging_fullpath, 'a') as test_results_logging_file:
    test_results_logging_file.write("\n===================================================================================================")
    test_results_logging_file.write("\n Teacher: {} ({} params); Assistant: {} ({} params); Student: {} ({} params); seed: {} \n".format(args.teacher, num_parameters_teacher, args.assistant, num_parameters_assistant, args.student, num_parameters_student, args.seed))
    print(args, file=test_results_logging_file)
    test_results_logging_file.write("\n Teacher test accuracy {}.".format(test_acc_teacher))
    test_results_logging_file.write("\n Assistant test accuracy {}.".format(test_acc_assistant))
    test_results_logging_file.write("\n Student test accuracy {}.".format(test_acc_student))



print("\n ===================================================================================================")
