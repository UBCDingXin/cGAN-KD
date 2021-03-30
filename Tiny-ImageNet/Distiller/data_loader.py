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
            # image = np.transpose(image, (1, 2, 0)) #C * H * W ---->  H * W * C
            image = Image.fromarray(np.uint8(image), mode = 'RGB') #H * W * C
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[index]

            return image, label

        return image

    def __len__(self):
        return self.n_images


def get_cifar(num_classes=200, real_data="", fake_data="None", nfake=1e30, batch_size=128, num_workers=4):

    hf = h5py.File(real_data, 'r')
    images_train = hf['imgs'][:]
    labels_train = hf['labels'][:]
    images_test = hf['imgs_val'][:]
    labels_test = hf['labels_val'][:]
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
        images_train = np.concatenate((images_train, fake_images), axis=0)
        labels_train = np.concatenate((labels_train, fake_labels))
    ##end if 

    ## compute the mean and std for normalization
    train_means = []
    train_stds = []
    for i in range(3):
        images_i = images_train[:,i,:,:]
        images_i = images_i/255.0
        train_means.append(np.mean(images_i))
        train_stds.append(np.std(images_i))
    ## for i

    images_train = images_train.transpose((0, 2, 3, 1))  # convert to HWC; after computing means and stds
    images_test = images_test.transpose((0, 2, 3, 1)) 

    ## info of training set and test set
    print("\n Training set: {}x{}x{}x{}; Testing set: {}x{}x{}x{}.".format(images_train.shape[0], images_train.shape[1], images_train.shape[2], images_train.shape[3], images_test.shape[0], images_test.shape[1], images_test.shape[2], images_test.shape[3]))

    transform_cnn_train = transforms.Compose([
                transforms.RandomCrop((64, 64), padding=4),
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

