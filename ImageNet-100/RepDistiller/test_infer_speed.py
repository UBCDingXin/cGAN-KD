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
from models import model_dict


parser = argparse.ArgumentParser()
parser.add_argument('--cnn_name', type=str, default='',
                    help='The CNN used in the classification.')
parser.add_argument('--nsamp', type=int, default=10000, help='number of images')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--img_size', type=int, default=128)
args = parser.parse_args()


## torchstat does not support ResNet50
# from torchstat import stat
# net = model_dict[args.cnn_name](num_classes=100)
# stat(net, (3, 32, 32))


################################################################################
# Convenience function to count the number of parameters in a module
def count_parameters(module, verbose=True):
    num_parameters = sum([p.data.nelement() for p in module.parameters()])
    if verbose:
        print('Number of parameters: {}'.format(num_parameters))
    return num_parameters

# model
net = model_dict[args.cnn_name](num_classes=100)
num_parameters = count_parameters(net, verbose=False)


## randomly generate args.nsamp images
images = np.random.randint(low=0, high=255, size=args.nsamp*3*args.img_size**2).reshape((args.nsamp, 3, args.img_size, args.img_size))
print(images.shape)

class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, normalize=False):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
        self.normalize = normalize


    def __getitem__(self, index):

        image = self.images[index]

        if self.normalize:
            image = image/255.0
            image = (image-0.5)/0.5

        if self.labels is not None:
            label = self.labels[index]
            return (image, label)
        else:
            return image

    def __len__(self):
        return self.n_images

trainset = IMGs_dataset(images, None, normalize=True)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)


net = net.cuda()
net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    start_time = timeit.default_timer()
    for _, images in enumerate(dataloader):
        images = images.type(torch.float).cuda()
        outputs = net(images)
    total_time = timeit.default_timer()-start_time


print("\n {} has {} parameters...".format(args.cnn_name, num_parameters))
print('\r Total time: {}s; Inference FPS: {}'.format(total_time, args.nsamp/total_time))
