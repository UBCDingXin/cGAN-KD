'''

Conditional Density Ration Estimation via Multilayer Perceptron

Multilayer Perceptron : trained to model density ratio in a feature space

Its input is the output of a pretrained Deep CNN, say ResNet-34

'''

import torch
import torch.nn as nn

IMG_SIZE=32
NC=3
N_CLASS = 100

cfg = {"MLP3": [512,256,128],
       "MLP5": [2048,1024,512,256,128],
       "MLP7": [2048,2048,1024,1024,512,256,128],
       "MLP9": [2048,2048,1024,1024,512,512,256,256,128],}


class cDR_MLP(nn.Module):
    def __init__(self, MLP_name, p_dropout=0.5, init_in_dim = IMG_SIZE**2*NC, num_classes = N_CLASS):
        super(cDR_MLP, self).__init__()
        self.init_in_dim = init_in_dim
        self.p_dropout=p_dropout
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, num_classes)

        layers = self._make_layers(cfg[MLP_name])
        layers += [nn.Linear(cfg[MLP_name][-1], 1)]
        layers += [nn.ReLU()]

        self.main = nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_dim = self.init_in_dim #initial input dimension
        for x in cfg:
            if in_dim == self.init_in_dim:
                layers += [nn.Linear(in_dim+self.num_classes, x),
                           nn.GroupNorm(8, x),
                           nn.ReLU(inplace=True),
                           nn.Dropout(self.p_dropout) # do we really need dropout?
                           ]
            else:
                layers += [nn.Linear(in_dim, x),
                           nn.GroupNorm(8, x),
                           nn.ReLU(inplace=True),
                           nn.Dropout(self.p_dropout) # do we really need dropout?
                           ]
            in_dim = x
        return layers

    def forward(self, x, labels):
        x = torch.cat((self.label_emb(labels), x), -1)
        out = self.main(x)
        return out


if __name__ == "__main__":
    net = cDR_MLP('MLP5').cuda()
    x = torch.randn((5,IMG_SIZE**2*NC)).cuda()
    labels = torch.LongTensor(5).random_(0, N_CLASS).cuda()
    out = net(x, labels)
    print(out.size())

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(net))
