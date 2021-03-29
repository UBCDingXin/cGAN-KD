'''

https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py

'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out



class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GenBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.snconv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class generator(nn.Module):
    """Generator."""

    def __init__(self, nz, nc, gen_ch=128):
        super(generator, self).__init__()

        self.nz = nz
        self.gen_ch = gen_ch
        self.snlinear0 = snlinear(in_features=nz, out_features=gen_ch*8*4*4)
        self.block1 = GenBlock(gen_ch*8, gen_ch*4)
        self.block2 = GenBlock(gen_ch*4, gen_ch*2)
        self.self_attn = Self_Attn(gen_ch*2)
        self.block3 = GenBlock(gen_ch*2, gen_ch)
        self.bn = nn.BatchNorm2d(gen_ch, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=gen_ch, out_channels=nc, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z):
        # n x nz
        act0 = self.snlinear0(z)            # n x gen_ch*8*4*4
        act0 = act0.view(-1, self.gen_ch*8, 4, 4) # n x gen_ch*8 x 4 x 4
        act1 = self.block1(act0)    # n x gen_ch*4 x 8 x 8
        act2 = self.block2(act1)    # n x gen_ch*2 x 16 x 16
        act2 = self.self_attn(act2)         # n x gen_ch*2 x 16 x 16
        act3 = self.block3(act2)    # n x gen_ch x 32 x 32
        act3 = self.bn(act3)                # n x gen_ch  x 32 x 32
        act3 = self.relu(act3)              # n x gen_ch  x 32 x 32
        act4 = self.snconv2d1(act3)         # n x 3 x 32 x 32
        act4 = self.tanh(act4)              # n x 3 x 32 x 32
        return act4


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, nc, disc_ch=64):
        super(discriminator, self).__init__()
        self.disc_ch = disc_ch
        self.opt_block1 = DiscOptBlock(nc, disc_ch)
        self.self_attn = Self_Attn(disc_ch)
        self.block1 = DiscBlock(disc_ch, disc_ch*2)
        self.block2 = DiscBlock(disc_ch*2, disc_ch*4)
        self.block3 = DiscBlock(disc_ch*4, disc_ch*8)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear = snlinear(in_features=disc_ch*8, out_features=1)

        # Weight init
        self.apply(init_weights)

    def forward(self, x):
        # n x 3 x 32 x 32
        h0 = self.opt_block1(x) # n x disc_ch   x 16 x 16
        h0 = self.self_attn(h0) # n x disc_ch x 16 x 16
        h1 = self.block1(h0)    # n x disc_ch*2 x 8 x 8
        h2 = self.block2(h1)    # n x disc_ch*4 x 4 x 4
        h3 = self.block3(h2, downsample=False)    # n x disc_ch*8 x  4 x  4
        h3 = self.relu(h3)              # n x disc_ch*8 x 4 x 4
        h3 = torch.sum(h3, dim=[2,3])   # n x disc_ch*8
        output = torch.squeeze(self.snlinear(h3)) # n
        return output



if __name__=="__main__":
    #test
    n = 4
    nz = 128
    netG = generator(nz=nz, gen_ch=32, nc=3).cuda()
    netD = discriminator(nc=3, disc_ch=32).cuda()
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

    z = torch.randn(n, nz).cuda()
    x = netG(z)
    o = netD(x)
    print(x.size())
    print(o.size())
