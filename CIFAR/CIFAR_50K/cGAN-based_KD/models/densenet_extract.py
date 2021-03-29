'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

NC=3
resize=(32,32)
IMG_SIZE = 32



class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet_extract(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, nc=NC, img_height=IMG_SIZE, img_width=IMG_SIZE):
        super(DenseNet_extract, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(NC, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        # self.linear = nn.Linear(num_planes, num_classes)

        self.classifier_1 = nn.Sequential(
                nn.Linear(num_planes, img_height*img_width*nc),
                nn.BatchNorm1d(img_height*img_width*nc),
                nn.ReLU(),
                )
        self.classifier_2 = nn.Sequential(
                nn.Linear(img_height*img_width*nc, num_classes)
        )

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = nn.functional.interpolate(x,size=resize,mode='bilinear',align_corners=True)
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        features = self.classifier_1(out)
        out = self.classifier_2(features)
        return out, features


def DenseNet121_extract(num_classes=10):
   return DenseNet_extract(Bottleneck, [6,12,24,16], growth_rate=32, num_classes=num_classes)

def DenseNet169_extract(num_classes=10):
   return DenseNet_extract(Bottleneck, [6,12,32,32], growth_rate=32, num_classes=num_classes)

def DenseNet201_extract(num_classes=10):
   return DenseNet_extract(Bottleneck, [6,12,48,32], growth_rate=32, num_classes=num_classes)

def DenseNet161_extract(num_classes=10):
   return DenseNet_extract(Bottleneck, [6,12,36,24], growth_rate=48, num_classes=num_classes)


def test_densenet():
    net = DenseNet121_extract(num_classes=10).cuda()
    net = nn.DataParallel(net)
    x = torch.randn(100,3,32,32).cuda()
    o,feat = net(x)
    print(o.shape)
    print(feat.shape)


if __name__ == "__main__":
    test_densenet()
