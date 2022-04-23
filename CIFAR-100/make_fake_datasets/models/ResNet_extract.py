'''
ResNet-based model to map an image from pixel space to a features space.
Need to be pretrained on the dataset.

codes are based on
@article{
zhang2018mixup,
title={mixup: Beyond Empirical Risk Minimization},
author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
journal={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=r1Ddp1-Rb},
}
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

IMG_SIZE=32
NC=3
resize=(32,32)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_extract(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, nc=NC, img_height=IMG_SIZE, img_width=IMG_SIZE):
        super(ResNet_extract, self).__init__()
        self.in_planes = 64

        self.main = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False),  # h=h
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(block, 64, num_blocks[0], stride=1),  # h=h
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2),
            nn.AvgPool2d(kernel_size=4)
        )
        self.classifier_1 = nn.Sequential(
                nn.Linear(512*block.expansion, img_height*img_width*nc),
                )
        self.classifier_2 = nn.Sequential(
                nn.BatchNorm1d(img_height*img_width*nc),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(img_height*img_width*nc, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(256, num_classes),
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = nn.functional.interpolate(x,size=resize,mode='bilinear',align_corners=True)
        features = self.main(x)
        features = features.view(features.size(0), -1)
        features = self.classifier_1(features)
        out = self.classifier_2(features)
        return out, features


def ResNet18_extract(num_classes=10):
    return ResNet_extract(BasicBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34_extract(num_classes=10):
    return ResNet_extract(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50_extract(num_classes=10):
    return ResNet_extract(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101_extract(num_classes=10):
    return ResNet_extract(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152_extract(num_classes=10):
    return ResNet_extract(Bottleneck, [3,8,36,3], num_classes=num_classes)


if __name__ == "__main__":
    net = ResNet34_extract(num_classes=10).cuda()
    x = torch.randn(16,3,32,32).cuda()
    out, features = net(x)
    print(out.size())
    print(features.size())

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    print(get_parameter_number(net))
