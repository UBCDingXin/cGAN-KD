import torch
from torch import nn



class encoder_extract(nn.Module):
    def __init__(self, dim_bottleneck=64*64*3, ch=64):
        super(encoder_extract, self).__init__()
        self.ch = ch
        self.dim_bottleneck = dim_bottleneck

        self.conv = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=4, stride=2, padding=1), #h=h/2; 32
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch),
            nn.ReLU(),

            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2, padding=1), #h=h/2; 16
            nn.BatchNorm2d(ch*2),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*2, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*2),
            nn.ReLU(),

            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2, padding=1), #h=h/2; 8
            nn.BatchNorm2d(ch*4),
            nn.ReLU(),
            nn.Conv2d(ch*4, ch*4, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*4),
            nn.ReLU(),

            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2, padding=1), #h=h/2; 4
            nn.BatchNorm2d(ch*8),
            nn.ReLU(),
            nn.Conv2d(ch*8, ch*12, kernel_size=3, stride=1, padding=1), #h=h; 4x4x64x12=12288=64x64x3
            # nn.BatchNorm2d(ch*12),
            nn.ReLU(),
        )

    def forward(self, x):
        feature = self.conv(x)
        feature = feature.view(-1, self.ch*12*4*4)
        return feature



class decoder_extract(nn.Module):
    def __init__(self, dim_bottleneck=64*64*3, ch=64):
        super(decoder_extract, self).__init__()
        self.ch = ch
        self.dim_bottleneck = dim_bottleneck

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ch*12, ch*8, kernel_size=4, stride=2, padding=1), #h=2h; 8
            nn.BatchNorm2d(ch*8),
            nn.ReLU(True),
            nn.Conv2d(ch*8, ch*8, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*8),
            nn.ReLU(),

            nn.ConvTranspose2d(ch*8, ch*4, kernel_size=4, stride=2, padding=1), #h=2h; 16
            nn.BatchNorm2d(ch*4),
            nn.ReLU(True),
            nn.Conv2d(ch*4, ch*4, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*4),
            nn.ReLU(),

            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=4, stride=2, padding=1), #h=2h; 32
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),
            nn.Conv2d(ch*2, ch*2, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*2),
            nn.ReLU(),

            nn.ConvTranspose2d(ch*2, ch, kernel_size=4, stride=2, padding=1), #h=2h; 64
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1), #h=h
            nn.Tanh()
        )

        self.predict = nn.Sequential(
            nn.Linear(self.dim_bottleneck, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 1),
            nn.ReLU(),
        )

    def forward(self, feature):
        pred = self.predict(feature)
        feature = feature.view(-1, self.ch*12, 4, 4)
        out = self.deconv(feature)
        return out, pred


if __name__=="__main__":
    #test

    net_encoder = encoder_extract(dim_bottleneck=64*64*3, ch=64).cuda()
    net_decoder = decoder_extract(dim_bottleneck=64*64*3, ch=64).cuda()
    net_encoder = nn.DataParallel(net_encoder)
    net_decoder = nn.DataParallel(net_decoder)

    x = torch.randn(10, 3, 64,64).cuda()
    f = net_encoder(x)
    xh, yh = net_decoder(f)
    print(f.size())
    print(xh.size())
    print(yh.size())
