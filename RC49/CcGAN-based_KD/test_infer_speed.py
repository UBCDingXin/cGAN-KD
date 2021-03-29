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
from models import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--cnn_name', type=str, default='ShuffleNet',
                    choices=['MobileNet', 'ShuffleNet',
                    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                    'VGG11', 'VGG13', 'VGG16', 'VGG19',
                    'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                    'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152',
                    'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161'],
                    help='The CNN used in the classification.')
parser.add_argument('--nsamp', type=int, default=100000, help='number of images')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

net = cnn_initialization(args.cnn_name, img_size = 64, parallel=False)
num_parameters = count_parameters(net, verbose=False)


## randomly generate args.nsamp images
images = np.random.randint(low=0, high=255, size=args.nsamp*3*64**2).reshape((args.nsamp, 3, 64, 64))
labels = np.random.uniform(low=0, high=1, size=args.nsamp)
#print(images.shape, labels.shape)
assert len(images)==len(labels)

trainset = IMGs_dataset(images, labels, normalize=True)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)


net = net.cuda()
net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    start_time = timeit.default_timer()
    for _, (images, _) in enumerate(dataloader):
        images = images.type(torch.float).cuda()
        outputs = net(images)
    total_time = timeit.default_timer()-start_time


print("\n {} has {} parameters...".format(args.cnn_name, num_parameters))
print('\r Total time: {}s; Inference FPS: {}'.format(total_time, args.nsamp/total_time))
