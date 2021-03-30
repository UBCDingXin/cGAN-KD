from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

from itertools import permutations

import h5py


class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, train=True, num_classes=200, real_data="", fake_data="None", nfake=1e30):
        super(IMGs_dataset, self).__init__()

        self.train = train
        self.num_classes = num_classes

        hf = h5py.File(real_data, 'r')
        self.images = hf['imgs'][:]
        self.labels = hf['labels'][:]
        hf.close()

        if fake_data != 'None':
            hf = h5py.File(fake_data, 'r')
            fake_images = hf['fake_images'][:]
            fake_labels = hf['fake_labels'][:]
            assert np.max(fake_images)>1 and np.min(fake_images)>=0
            hf.close()
            if nfake<len(fake_images):
                indx_fake = []
                for i in range(num_classes):
                    indx_fake_i = np.where(fake_labels==i)[0]
                    if i != (num_classes-1):
                        nfake_i = nfake//num_classes
                    else:
                        nfake_i = nfake-(nfake//num_classes)*i
                    indx_fake_i = np.random.choice(indx_fake_i, size=min(int(nfake_i), len(indx_fake_i)), replace=False)
                    indx_fake.append(indx_fake_i)
                #end for i
                indx_fake = np.concatenate(indx_fake)
                fake_images = fake_images[indx_fake]
                fake_labels = fake_labels[indx_fake]
            nfake = len(fake_images)

            ## combine fake and real
            self.images = np.concatenate((self.images, fake_images), axis=0)
            self.labels = np.concatenate((self.labels, fake_labels))
        ##end if

        ## compute the mean and std for normalization
        means = []
        stds = []
        assert (self.images).shape[1]==3
        for i in range(3):
            images_i = self.images[:,i,:,:]
            images_i = images_i/255.0
            means.append(np.mean(images_i))
            stds.append(np.std(images_i))
        ## for i

        self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC; after computing means and stds

        ## transforms
        self.transform = transforms.Compose([
            transforms.RandomCrop((64, 64), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
            ])

        ###

        if not self.train: ## NOTE: test mode
            hf = h5py.File(real_data, 'r')
            self.images = hf['imgs_val'][:]
            self.labels = hf['labels_val'][:]
            hf.close()
            self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC; after computing means and stds

            ## transforms
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
                ])

        self.n_images = len(self.images)

    def __getitem__(self, index):

        img, label = self.images[index], self.labels[index]
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:,::-1,:]

        img0 = np.rot90(img, 0).copy()
        img0 = Image.fromarray(img0)
        img0 = self.transform(img0)

        img1 = np.rot90(img, 1).copy()
        img1 = Image.fromarray(img1)
        img1 = self.transform(img1)

        img2 = np.rot90(img, 2).copy()
        img2 = Image.fromarray(img2)
        img2 = self.transform(img2)

        img3 = np.rot90(img, 3).copy()
        img3 = Image.fromarray(img3)
        img3 = self.transform(img3)

        img = torch.stack([img0,img1,img2,img3])

        return img, label

    def __len__(self):
        return self.n_images
