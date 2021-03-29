'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

NC=3
# resize=(32,32)
IMG_SIZE = 64



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


class DenseNet_embed(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, dim_embed=128):
        super(DenseNet_embed, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(NC, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.maxpool2d = nn.MaxPool2d(2, stride=2)

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

        self.x2h_res = nn.Sequential(
            nn.Linear(num_planes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, dim_embed),
            nn.BatchNorm1d(dim_embed),
            nn.ReLU(),
        )

        self.h2y = nn.Sequential(
            nn.Linear(dim_embed, 1),
            nn.ReLU()
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
        out = self.maxpool2d(out)
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        features = self.x2h_res(out)
        out = self.h2y(features)
        return out, features


def DenseNet121_embed(dim_embed=128):
   return DenseNet_embed(Bottleneck, [6,12,24,16], growth_rate=32, dim_embed=dim_embed)

def DenseNet169_embed(dim_embed=128):
   return DenseNet_embed(Bottleneck, [6,12,32,32], growth_rate=32, dim_embed=dim_embed)

def DenseNet201_embed(dim_embed=128):
   return DenseNet_embed(Bottleneck, [6,12,48,32], growth_rate=32, dim_embed=dim_embed)

def DenseNet161_embed(dim_embed=128):
   return DenseNet_embed(Bottleneck, [6,12,36,24], growth_rate=48, dim_embed=dim_embed)



#------------------------------------------------------------------------------
# map labels to the embedding space
class model_y2h(nn.Module):
    def __init__(self, dim_embed=128):
        super(model_y2h, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, dim_embed),
            # nn.BatchNorm1d(dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            # nn.BatchNorm1d(dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            # nn.BatchNorm1d(dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            # nn.BatchNorm1d(dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            nn.ReLU()
        )

    def forward(self, y):
        y = y.view(-1, 1) + 1e-8
        return self.main(y)




def test_densenet():
    net = DenseNet121_embed(dim_embed=128).cuda()
    net = nn.DataParallel(net)
    x = torch.randn(16,3,64,64).cuda()
    o,feat = net(x)
    print(o.shape)
    print(feat.shape)

    y = torch.randn(16, 1).cuda()
    y2h = model_y2h(128).cuda()
    y2h = nn.DataParallel(y2h)
    h = y2h(y)
    print(h.shape)


if __name__ == "__main__":
    test_densenet()
