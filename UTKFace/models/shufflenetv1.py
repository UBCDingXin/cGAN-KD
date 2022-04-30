'''ShuffleNet in PyTorch.
See the paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" for more details.

To fit 128x128 images, I modified the first conv layer and add an extra max_pool2d after it (Following Table 5 of "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design")

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.stride = stride

        mid_planes = int(out_planes/4)
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        preact = torch.cat([out, res], 1) if self.stride == 2 else out+res
        out = F.relu(preact)
        # out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out+res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, cfg):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        # self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False) #original
        self.conv1 = nn.Conv2d(3, 24, kernel_size=4, padding=1, stride=2, bias=False) #h=h//2
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        
        self.linear = nn.Sequential(
            nn.Linear(out_planes[2], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
                
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1),
            nn.ReLU(),
        )
        

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes,
                                     stride=stride,
                                     groups=groups,
                                     is_last=(i == num_blocks - 1)))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ShuffleV1(**kwargs):
    cfg = {
        'out_planes': [240, 480, 960],
        'num_blocks': [4, 8, 4],
        'groups': 3
    }
    return ShuffleNet(cfg, **kwargs)


if __name__ == '__main__':

    x = torch.randn(2, 3, 64, 64)
    net = ShuffleV1()
    import time
    a = time.time()
    out = net(x, is_feat=True, preact=True)
    b = time.time()
    print(b - a)
    print(out.shape)
