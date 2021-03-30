import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import functional as F
import sys
import PIL
from PIL import Image

### import my stuffs ###
from models import *


# ################################################################################
# Progress Bar
class SimpleProgressBar():
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100:
            print('')


################################################################################
# torch dataset from numpy array
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


################################################################################
def PlotLoss(loss, filename):
    x_axis = np.arange(start = 1, stop = len(loss)+1)
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_axis, np.array(loss))
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.legend()
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3)
    #plt.title('Training Loss')
    plt.savefig(filename)


################################################################################
# Convenience function to count the number of parameters in a module
def count_parameters(module, verbose=True):
    num_parameters = sum([p.data.nelement() for p in module.parameters()])
    if verbose:
        print('Number of parameters: {}'.format(num_parameters))
    return num_parameters


# ################################################################################
# Init. CNNs
def cnn_initialization(cnn_name, num_classes=200, img_size=64, parallel=True):
    if cnn_name == "MobileNet":
        net = mobilenet_v2(num_classes=num_classes)
    elif cnn_name == "ShuffleNet":
        net = shufflenet_v2_x1_0(num_classes=num_classes)
    # elif cnn_name.split('-')[0] == "efficientnet":
    #     net = EfficientNet.from_name(model_name=cnn_name, num_classes=num_classes, image_size=img_size)
    elif cnn_name == "efficientnet-b0":
        net = EfficientNetB0(num_classes=num_classes)
    elif cnn_name == "ResNet8":
        net = ResNet8_custom(num_classes=num_classes)
    elif cnn_name == "ResNet20":
        net = ResNet20_custom(num_classes=num_classes)
    elif cnn_name == "ResNet110":
        net = ResNet110_custom(num_classes=num_classes)
    elif cnn_name == "ResNet18":
        net = ResNet18(num_classes=num_classes)
    elif cnn_name == "ResNet34":
        net = ResNet34(num_classes=num_classes)
    elif cnn_name == "ResNet50":
        net = ResNet50(num_classes=num_classes)
    elif cnn_name == "ResNet101":
        net = ResNet101(num_classes=num_classes)
    elif cnn_name == "ResNet152":
        net = ResNet152(num_classes=num_classes)
    elif cnn_name[0:3] == "VGG":
        net = VGG(cnn_name, num_classes=num_classes)
    elif cnn_name == "PreActResNet18":
        net = PreActResNet18(num_classes=num_classes)
    elif cnn_name == "PreActResNet34":
        net = PreActResNet34(num_classes=num_classes)
    elif cnn_name == "PreActResNet50":
        net = PreActResNet50(num_classes=num_classes)
    elif cnn_name == "PreActResNet101":
        net = PreActResNet101(num_classes=num_classes)
    elif cnn_name == "DenseNet121":
        net = DenseNet121(num_classes=num_classes)
    elif cnn_name == "DenseNet169":
        net = DenseNet169(num_classes=num_classes)
    elif cnn_name == "DenseNet201":
        net = DenseNet201(num_classes=num_classes)
    elif cnn_name == "DenseNet161":
        net = DenseNet161(num_classes=num_classes)

    if parallel:
        net = nn.DataParallel(net)
    net = net.cuda()

    return net


def cnn_extract_initialization(cnn_name, num_classes=10, img_size=32, parallel=True):
    if cnn_name == "ResNet34":
        net = ResNet34_extract(num_classes=num_classes)
    elif cnn_name == "ResNet50":
        net = ResNet50_extract(num_classes=num_classes)
    elif cnn_name == "ResNet101":
        net = ResNet101_extract(num_classes=num_classes)
    elif cnn_name == "ResNet152":
        net = ResNet152_extract(num_classes=num_classes)
    elif cnn_name == "DenseNet121":
        net = DenseNet121_extract(num_classes=num_classes)
    elif cnn_name == "DenseNet169":
        net = DenseNet169_extract(num_classes=num_classes)
    elif cnn_name == "DenseNet201":
        net = DenseNet201_extract(num_classes=num_classes)
    elif cnn_name == "DenseNet161":
        net = DenseNet161_extract(num_classes=num_classes)

    if parallel:
        net = nn.DataParallel(net)
    net = net.cuda()

    return net
