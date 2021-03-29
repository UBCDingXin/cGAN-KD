'''

Make the training set for GAN and CNN

'''

import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py
import random
from PIL import Image
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/xin/OneDrive/Working_directory/GAN-based_KD/CIFAR/cGAN-based_KD')
#parser.add_argument('--root_path', type=str, default='E:/OneDrive/Working_directory/GAN-based_KD/CIFAR/cGAN-based_KD')
parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')
parser.add_argument('--ntrain', type=int, default=20000)
parser.add_argument('--subvalid_prop', type=float, default=0.2)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--num_channels', type=int, default=3)
args = parser.parse_args()


os.chdir(args.root_path)


N_TRAIN = args.ntrain
N_SUBVALID_RATIO = args.subvalid_prop
N_SUBTRAIN = int(N_TRAIN * (1-N_SUBVALID_RATIO))
N_SUBVALID = N_TRAIN - N_SUBTRAIN
N_VALID = 50000 - N_TRAIN

IMG_SIZE = args.img_size
NC = args.num_channels
N_CLASS = args.num_classes


# random seed
SEED=args.seed
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if N_CLASS == 10:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
elif N_CLASS == 100:
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)

images_train_all = trainset.data
images_train_all = np.transpose(images_train_all, (0, 3, 1, 2))
labels_train_all = np.array(trainset.targets)

assert N_TRAIN+N_VALID==len(images_train_all)

if N_TRAIN<50000:
    indx_all = np.arange(len(images_train_all))
    np.random.shuffle(indx_all)
    indx_train = indx_all[0:N_TRAIN]
    indx_valid = indx_all[N_TRAIN:(N_TRAIN+N_VALID)]
    images_train = images_train_all[indx_train]
    labels_train = labels_train_all[indx_train]
    images_valid = images_train_all[indx_valid]
    labels_valid = labels_train_all[indx_valid]

    assert N_TRAIN==N_SUBTRAIN+N_SUBVALID
    train_indx = np.arange(N_TRAIN)
    subtrain_indx = train_indx[0:N_SUBTRAIN]
    subvalid_indx = train_indx[N_SUBTRAIN:N_TRAIN]
    # images_subtrain = images_train[subtrain_indx] #for hyper-parameter selection in DRE
    # labels_subtrain = labels_train[subtrain_indx]
    # images_subvalid = images_train[subvalid_indx] #for hyper-parameter selection in DRE
    # labels_subvalid = labels_train[subvalid_indx]

    h5py_file = args.root_path + '/data/CIFAR{}_trainset_{}_seed_{}.h5'.format(N_CLASS, N_TRAIN, SEED)
    with h5py.File(h5py_file, "w") as f:
        f.create_dataset('images_train', data = images_train, dtype = 'uint8')
        f.create_dataset('labels_train', data = labels_train)
        f.create_dataset('subtrain_indx', data = subtrain_indx)
        f.create_dataset('subvalid_indx', data = subvalid_indx)
        # f.create_dataset('images_valid', data = images_valid, dtype = 'uint8')
        # f.create_dataset('labels_valid', data = labels_valid)

     # make h5 file for BigGAN
    h5py_file = args.root_path + '/BigGAN/data/C{}_{}.hdf5'.format(N_CLASS, SEED)
    with h5py.File(h5py_file, "w") as f:
        f.create_dataset('imgs', data = images_train)
        f.create_dataset('labels', data = labels_train)


else:
    images_train = images_train_all.copy()
    labels_train = labels_train_all.copy()

    train_indx = np.arange(N_TRAIN)
    np.random.shuffle(train_indx)
    subtrain_indx = train_indx[0:N_SUBTRAIN]
    subvalid_indx = train_indx[N_SUBTRAIN:N_TRAIN]

    h5py_file = args.root_path  + '/data/CIFAR{}_trainset_{}_seed_{}.h5'.format(N_CLASS, N_TRAIN, SEED)
    with h5py.File(h5py_file, "w") as f:
        f.create_dataset('images_train', data = images_train, dtype = 'uint8')
        f.create_dataset('labels_train', data = labels_train)
        f.create_dataset('subtrain_indx', data = subtrain_indx)
        f.create_dataset('subvalid_indx', data = subvalid_indx)



##test h5 file
#hf = h5py.File(h5py_file, 'r')
#images_train = hf['images_train'][:]
#labels_train = hf['labels_train'][:]
#images_valid = hf['images_valid'][:]
#labels_valid = hf['labels_valid'][:]
#hf.close()
