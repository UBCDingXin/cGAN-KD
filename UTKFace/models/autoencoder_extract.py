import torch
from torch import nn



class encoder_extract(nn.Module):
    def __init__(self, dim_bottleneck=64*64*3, ch=32):
        super(encoder_extract, self).__init__()
        self.ch = ch
        self.dim_bottleneck = dim_bottleneck

        self.conv = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=4, stride=2, padding=1), #h=h/2; 32
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch),
            nn.ReLU(True),

            nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1), #h=h/2; 16
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch*2, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),

            nn.Conv2d(ch*2, ch*2, kernel_size=4, stride=2, padding=1), #h=h/2; 8
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),
            nn.Conv2d(ch*2, ch*4, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*4),
            nn.ReLU(True),

            nn.Conv2d(ch*4, ch*4, kernel_size=4, stride=2, padding=1), #h=h/2; 4
            nn.BatchNorm2d(ch*4),
            nn.ReLU(True),
            nn.Conv2d(ch*4, ch*4, kernel_size=3, stride=1, padding=1), #h=h; 4
            nn.BatchNorm2d(ch*4),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(
            nn.Linear(ch*4*4*4, dim_bottleneck),
            nn.ReLU()
        )


    def forward(self, x):
        feature = self.conv(x)
        feature = feature.view(-1, self.ch*4*4*4)
        feature = self.fc(feature)
        return feature



class decoder_extract(nn.Module):
    def __init__(self, dim_bottleneck=64*64*3, ch=32):
        super(decoder_extract, self).__init__()
        self.ch = ch
        self.dim_bottleneck = dim_bottleneck

        self.fc = nn.Sequential(
            nn.Linear(dim_bottleneck, ch*4*4*4),
            nn.BatchNorm1d(ch*4*4*4),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ch*4, ch*4, kernel_size=4, stride=2, padding=1), #h=2h; 8
            nn.BatchNorm2d(ch*4),
            nn.ReLU(True),
            nn.Conv2d(ch*4, ch*2, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch*2, ch*2, kernel_size=4, stride=2, padding=1), #h=2h; 16
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),
            nn.Conv2d(ch*2, ch, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1), #h=2h; 32
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1), #h=2h; 64
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch),
            nn.ReLU(True),

            nn.Conv2d(ch, 3, kernel_size=1, stride=1, padding=0), #h=h
            nn.Tanh()
        )

    def forward(self, feature):
        feature = self.fc(feature)
        feature = feature.view(-1, self.ch*4, 4, 4)
        out = self.deconv(feature)
        return out



class decoder_predict(nn.Module):
    def __init__(self, dim_bottleneck=64*64*3):
        super(decoder_predict, self).__init__()
        self.dim_bottleneck = dim_bottleneck

        self.predict = nn.Sequential(
            nn.Linear(self.dim_bottleneck, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 1),
            nn.ReLU(),
        )
    def forward(self, feature):
        return self.predict(feature)



if __name__=="__main__":
    #test

    net_encoder = encoder_extract(dim_bottleneck=64*64*3, ch=64).cuda()
    net_decoder = decoder_extract(dim_bottleneck=64*64*3, ch=64).cuda()
    net_predict = decoder_predict(dim_bottleneck=64*64*3).cuda()
    net_encoder = nn.DataParallel(net_encoder)
    net_decoder = nn.DataParallel(net_decoder)
    net_predict = nn.DataParallel(net_predict)

    x = torch.randn(10, 3, 64,64).cuda()
    f = net_encoder(x)
    xh = net_decoder(f)
    yh = net_predict(f)
    print(f.size())
    print(xh.size())
    print(yh.size())

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(net_encoder))
    print(get_parameter_number(net_decoder))
    print(get_parameter_number(net_predict))


