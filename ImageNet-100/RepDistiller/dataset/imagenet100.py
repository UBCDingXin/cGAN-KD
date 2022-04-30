from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import h5py
import torch
import gc


## for vanilla CNN training and KD other than CRD
class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, transform=None, is_instance=False):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        self.is_instance = is_instance
        
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
        self.transform = transform

    def __getitem__(self, index):

        ## for RGB only
        image = self.images[index]
        if self.transform is not None:
            image = np.transpose(image, (1, 2, 0)) #C * H * W ---->  H * W * C
            image = Image.fromarray(np.uint8(image), mode = 'RGB') #H * W * C
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[index]

            if self.is_instance:
                return image, label, index
            else:
                return image, label

        return image

    def __len__(self):
        return self.n_images


def get_imagenet100_dataloaders(data_folder, batch_size=128, num_workers=0, is_instance=False, use_fake_data=False, fake_data_folder=None, nfake=1e30):
    """
    imagenet 100
    """
    
    trainset_h5py_file = os.path.join(data_folder, 'ImageNet_128x128_100Class.h5')
    hf = h5py.File(trainset_h5py_file, 'r')
    real_images_train = hf['images_train'][:]
    real_labels_train = hf['labels_train'][:]
    real_images_test = hf['images_valid'][:]
    real_labels_test = hf['labels_valid'][:]
    hf.close()
                
    num_classes=100            

    if use_fake_data:
        ## load fake data
        with h5py.File(fake_data_folder, "r") as f:
            train_images = f['fake_images'][:]
            train_labels = f['fake_labels'][:]
        train_labels = train_labels.astype(int)
        
        if nfake<len(train_labels):
            indx_fake = []
            for i in range(num_classes):
                indx_fake_i = np.where(train_labels==i)[0]
                if i != (num_classes-1):
                    nfake_i = nfake//num_classes
                else:
                    nfake_i = nfake-(nfake//num_classes)*i
                indx_fake_i = np.random.choice(indx_fake_i, size=min(int(nfake_i), len(indx_fake_i)), replace=False)
                indx_fake.append(indx_fake_i)
            #end for i
            indx_fake = np.concatenate(indx_fake)
            train_images = train_images[indx_fake]
            train_labels = train_labels[indx_fake]
        
        ## concatenate
        train_images = np.concatenate((real_images_train, train_images), axis=0)
        train_labels = np.concatenate((real_labels_train, train_labels), axis=0)    
        train_labels = train_labels.astype(int)  
        
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
        train_transform = transforms.Compose([
            # transforms.RandomCrop(128, padding=4),
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
        ])
        train_set = IMGs_dataset(train_images, train_labels, transform=train_transform, is_instance=is_instance)
        assert len(train_set)==len(train_images)
        del train_images, train_labels; gc.collect()
    
    else:
        ## compute normalizing constants
        train_means = []
        train_stds = []
        for i in range(3):
            images_i = real_images_train[:,i,:,:]
            images_i = images_i/255.0
            train_means.append(np.mean(images_i))
            train_stds.append(np.std(images_i))
        ## for i     
        
        print("\n Final training set's dimensiona: ", real_images_train.shape)
        
        ## make training set
        train_transform = transforms.Compose([
            # transforms.RandomCrop(128, padding=4),
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
        ])
        train_set = IMGs_dataset(real_images_train, real_labels_train, transform=train_transform, is_instance=is_instance)
    
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = IMGs_dataset(real_images_test, real_labels_test, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=num_workers)

    if is_instance:
        n_data = len(train_set)
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader











## for CRD
class IMGs_dataset_CRD(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, k=4096, mode='exact', is_sample=True, percent=1.0):
        super(IMGs_dataset_CRD, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        
        if len(self.images) != len(self.labels):
            raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
        self.transform = transform
        
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        
        num_classes = 100
        num_samples = len(labels)
        
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[labels[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)
        
        
    def __getitem__(self, index):

        ## for RGB only
        image = self.images[index]
        if self.transform is not None:
            image = np.transpose(image, (1, 2, 0)) #C * H * W ---->  H * W * C
            image = Image.fromarray(np.uint8(image), mode = 'RGB') #H * W * C
            image = self.transform(image)
            
        label = self.labels[index]

        if not self.is_sample:
            # directly return
            return image, label, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[label], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[label]) else False
            neg_idx = np.random.choice(self.cls_negative[label], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return image, label, index, sample_idx

    def __len__(self):
        return self.n_images


def get_imagenet100_dataloaders_sample(data_folder, batch_size=128, num_workers=0, k=4096, mode='exact',
                                    is_sample=True, percent=1.0, 
                                    use_fake_data=False, fake_data_folder=None, nfake=1e30):
    """
    imagenet 100
    """
    
    trainset_h5py_file = os.path.join(data_folder, 'ImageNet_128x128_100Class.h5')
    hf = h5py.File(trainset_h5py_file, 'r')
    real_images_train = hf['images_train'][:]
    real_labels_train = hf['labels_train'][:]
    real_images_test = hf['images_valid'][:]
    real_labels_test = hf['labels_valid'][:]
    hf.close()
    
    num_classes=100
    
    if use_fake_data:
        ## load fake data
        with h5py.File(fake_data_folder, "r") as f:
            train_images = f['fake_images'][:]
            train_labels = f['fake_labels'][:]
        train_labels = train_labels.astype(int)
        
        if nfake<len(train_labels):
            indx_fake = []
            for i in range(num_classes):
                indx_fake_i = np.where(train_labels==i)[0]
                if i != (num_classes-1):
                    nfake_i = nfake//num_classes
                else:
                    nfake_i = nfake-(nfake//num_classes)*i
                indx_fake_i = np.random.choice(indx_fake_i, size=min(int(nfake_i), len(indx_fake_i)), replace=False)
                indx_fake.append(indx_fake_i)
            #end for i
            indx_fake = np.concatenate(indx_fake)
            train_images = train_images[indx_fake]
            train_labels = train_labels[indx_fake]
        
        ## concatenate
        train_images = np.concatenate((real_images_train, train_images), axis=0)
        train_labels = np.concatenate((real_labels_train, train_labels), axis=0)    
        train_labels = train_labels.astype(int)
        
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
        train_transform = transforms.Compose([
            # transforms.RandomCrop(128, padding=4),
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
        ])
        train_set = IMGs_dataset_CRD(train_images, train_labels, transform=train_transform, k=k, mode=mode, is_sample=is_sample, percent=percent)
        assert len(train_set)==len(train_images)
        del train_images, train_labels; gc.collect()
    else:
        ## compute normalizing constants
        train_means = []
        train_stds = []
        for i in range(3):
            images_i = real_images_train[:,i,:,:]
            images_i = images_i/255.0
            train_means.append(np.mean(images_i))
            train_stds.append(np.std(images_i))
        ## for i     
        
        print("\n Final training set's dimensiona: ", real_images_train.shape)
        
        ## make training set
        train_transform = transforms.Compose([
            # transforms.RandomCrop(128, padding=4),
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
        ])       
        train_set = IMGs_dataset_CRD(real_images_train, real_labels_train, transform=train_transform, k=k, mode=mode, is_sample=is_sample, percent=percent)
    
    n_data = len(train_set)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = IMGs_dataset(real_images_test, real_labels_test, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, n_data