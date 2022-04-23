import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import PIL
from PIL import Image
import h5py
import os



class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, transform=None):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
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

        # ## for grey scale only
        # image = self.images[index]
        # if self.transform is not None:
        #     image = Image.fromarray(np.uint8(image[0]), mode = 'L') #H * W * C
        #     image = self.transform(image)
        #     # image = np.array(image)
        #     # image = image[np.newaxis,:,:]
        #     # print(image.shape)

        if self.labels is not None:
            label = self.labels[index]

            return image, label

        return image

    def __len__(self):
        return self.n_images


def get_cifar(num_classes=100, real_data="", fake_data="None", nfake=1e30, batch_size=128, num_workers=0):

    ## load real data
    train_set = torchvision.datasets.CIFAR100(root=real_data, download=True, train=True)
    real_images = train_set.data
    real_images = np.transpose(real_images, (0, 3, 1, 2))
    real_labels = np.array(train_set.targets)

    ## load fake data
    if fake_data != 'None':
        ## load fake data
        with h5py.File(fake_data, "r") as f:
            images_train = f['fake_images'][:]
            labels_train = f['fake_labels'][:]
        labels_train = labels_train.astype(int)
        
        if nfake<len(labels_train):
            indx_fake = []
            for i in range(num_classes):
                indx_fake_i = np.where(labels_train==i)[0]
                if i != (num_classes-1):
                    nfake_i = nfake//num_classes
                else:
                    nfake_i = nfake-(nfake//num_classes)*i
                indx_fake_i = np.random.choice(indx_fake_i, size=min(int(nfake_i), len(indx_fake_i)), replace=False)
                indx_fake.append(indx_fake_i)
            #end for i
            indx_fake = np.concatenate(indx_fake)
            images_train = images_train[indx_fake]
            labels_train = labels_train[indx_fake]
        ## concatenate
        images_train = np.concatenate((real_images, images_train), axis=0)
        labels_train = np.concatenate((real_labels, labels_train), axis=0)    
        labels_train = labels_train.astype(int) 
    else:
        images_train = real_images
        labels_train = real_labels

    ## compute the mean and std for normalization
    train_means = []
    train_stds = []
    for i in range(3):
        images_i = images_train[:,i,:,:]
        images_i = images_i/255.0
        train_means.append(np.mean(images_i))
        train_stds.append(np.std(images_i))
    ## for i

    cifar_testset = torchvision.datasets.CIFAR100(root = real_data, train=False, download=True)
    images_test = cifar_testset.data
    images_test = np.transpose(images_test, (0, 3, 1, 2))
    labels_test = np.array(cifar_testset.targets)

    ## info of training set and test set
    print("\n Training set: {}x{}x{}x{}; Testing set: {}x{}x{}x{}.".format(images_train.shape[0], images_train.shape[1], images_train.shape[2], images_train.shape[3], images_test.shape[0], images_test.shape[1], images_test.shape[2], images_test.shape[3]))

    transform_cnn_train = transforms.Compose([
                transforms.RandomCrop((32, 32), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(train_means, train_stds),
                ])
    
    transform_cnn_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
            ])

    trainset_cnn = IMGs_dataset(images_train, labels_train, transform=transform_cnn_train)
    train_loader = torch.utils.data.DataLoader(trainset_cnn, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset_cnn = IMGs_dataset(images_test, labels_test, transform=transform_cnn_test)
    test_loader = torch.utils.data.DataLoader(testset_cnn, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

