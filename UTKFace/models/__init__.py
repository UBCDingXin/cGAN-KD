from .autoencoder_extract import *
from .cDR_MLP import cDR_MLP
from .SNGAN import SNGAN_Generator, SNGAN_Discriminator
from .SAGAN import SAGAN_Generator, SAGAN_Discriminator
from .shufflenetv1 import ShuffleV1
from .shufflenetv2 import ShuffleV2
from .mobilenet import mobilenet_v2
from .efficientnet import EfficientNetB0
from .vgg import vgg8, vgg11, vgg13, vgg16, vgg19
from .resnet import resnet8x4, resnet20, resnet32x4, resnet56, resnet110
from .resnetv2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet161
from .ResNet_embed import ResNet18_embed, ResNet34_embed, ResNet50_embed, model_y2h
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2

cnn_dict = {
    'resnet20': resnet20,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'vgg8': vgg8,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'MobileNetV2': mobilenet_v2,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'efficientnetb0': EfficientNetB0,
    'densenet121': DenseNet121,
    'densenet169': DenseNet169,
    'densenet201': DenseNet201,
    'densenet161': DenseNet161,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
}
