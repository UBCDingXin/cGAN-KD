import argparse

'''

Options for Some Baseline CNN Training

'''
def cnn_opts():
    parser = argparse.ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--fake_dataset_name', type=str, default='None',
                        help="If it equals to 'None', then use real data only")
    parser.add_argument('--cnn_name', type=str, default='MobileNet',
                        choices=['MobileNet', 'ShuffleNet',
                        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                        'VGG11', 'VGG13', 'VGG16', 'VGG19',
                        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                        'ResNet8', 'ResNet20', 'ResNet110',
                        'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152',
                        'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161'],
                        help='The CNN used in the classification.')
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Datast Settings '''
    parser.add_argument('--ntrain', type=int, default=30000, help='number of images for training')
    parser.add_argument('--nfake', type=float, default=1e30, help='number of fake images for training')
    parser.add_argument('--num_classes', type=int, default=10, metavar='N',choices=[10, 100]) #CIFAR10 or CIFAR100
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=32, metavar='N')

    ''' CNN Settings '''
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--save_freq', type=str, default='150_250')
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=256)
    parser.add_argument('--lr_base', type=float, default=0.1, help='base learning rate of CNNs')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='150_250', help='decay lr at which epoch; separate by _')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--transform', action='store_true', default=False, help='conventional data augmentation for CNN training')
    parser.add_argument('--validaiton_mode', action='store_true', default=False, help='Select filtering threshold on the validation set')

    args = parser.parse_args()

    return args



'''

Options for GAN, DRE, and synthetic data generation

'''
def gen_synth_data_opts():
    parser = argparse.ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Datast Settings '''
    parser.add_argument('--ntrain', type=int, default=30000, help='number of images for training')
    parser.add_argument('--num_classes', type=int, default=10, metavar='N',choices=[10, 100]) #CIFAR10 or CIFAR100
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=32, metavar='N')

    ''' GAN Settings '''
    parser.add_argument('--gan_name', type=str, default='BigGAN',
                        choices=['BigGAN', 'cGAN', 'GAN'])
    parser.add_argument('--gan_epochs', type=int, default=600)
    parser.add_argument('--gan_d_niters', type=int, default=1,
                        help='train d ? iterations when g is trained once')
    parser.add_argument('--gan_loss', type=str, default='vanilla',choices=['vanilla','hinge','wasserstein'])
    parser.add_argument('--gan_lr_g', type=float, default=1e-4,
                        help='learning rate for generator')
    parser.add_argument('--gan_lr_d', type=float, default=1e-4,
                        help='learning rate for discriminator')
    parser.add_argument('--gan_dim_g', type=int, default=128,
                        help='Latent dimension of GAN')
    parser.add_argument('--gan_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training GAN')
    parser.add_argument('--gan_resume_epoch', type=int, default=0)
    parser.add_argument('--gan_transform', action='store_true', default=False)

    ''' DRE Settings '''
    ## Pre-trained CNN for feature extraction
    parser.add_argument('--dre_precnn_net', type=str, default='ResNet50',
                        choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                                 'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161'],
                        help='Pre-trained CNN for DRE in Feature Space; Candidate: ResNetXX')
    parser.add_argument('--dre_precnn_epochs', type=int, default=350)
    parser.add_argument('--dre_precnn_resume_epoch', type=int, default=0, metavar='N')
    parser.add_argument('--dre_precnn_lr_base', type=float, default=0.1, help='base learning rate of CNNs')
    parser.add_argument('--dre_precnn_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--dre_precnn_lr_decay_epochs', type=str, default='150_250', help='decay lr at which epoch; separate by _')
    parser.add_argument('--dre_precnn_batch_size_train', type=int, default=128, metavar='N')
    parser.add_argument('--dre_precnn_weight_decay', type=float, default=5e-4)
    parser.add_argument('--dre_precnn_transform', action='store_true', default=False)
    ## DR model in the feature space
    parser.add_argument('--dre_net', type=str, default='MLP7',
                        choices=['MLP3', 'MLP5', 'MLP7', 'MLP9'], help='DR Model in the feature space') # DRE in Feature Space
    parser.add_argument('--dre_epochs', type=int, default=400)
    parser.add_argument('--dre_lr_base', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dre_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--dre_lr_decay_epochs', type=str, default='100_200', help='decay lr at which epoch; separate by _')
    parser.add_argument('--dre_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training DRE')
    parser.add_argument('--dre_lambda', type=float, default=1e-3, help='penalty in DRE')
    parser.add_argument('--dre_resume_epoch', type=int, default=0)

    ''' Sampling Settings '''
    parser.add_argument('--subsampling', action='store_true', default=False)
    parser.add_argument('--samp_batch_size', type=int, default=1000) #also used for computing density ratios after the dre training
    parser.add_argument('--samp_burnin_size', type=int, default=50000)
    parser.add_argument('--samp_nfake_per_class', type=int, default=10000) #number of fake images per class for data augmentation
    parser.add_argument('--samp_filter_precnn_net', type=str, default='DenseNet121',
                        choices=['ResNet101', 'ResNet152', 'ResNet110', 'PreActResNet101', 'PreActResNet152',
                        'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161'],
                        help='Pre-trained CNN for DRE in Feature Space;')
    parser.add_argument('--samp_filter_precnn_net_ckpt_filename', type=str, default='')
    parser.add_argument('--samp_filter_ce_percentile_threshold', type=float, default=1.0,
                        help='The percentile threshold of cross entropy to filter out bad synthetic images by a pre-trained net')
    parser.add_argument('--unfiltered_fake_dataset_filename', type=str, default='')
    parser.add_argument('--adjust_label', action='store_true', default=False)

    args = parser.parse_args()

    return args



'''

Options for Baseline Knowledge Distillation: BLKD

'''
def blkd_opts():
    parser = argparse.ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--fake_dataset_name', type=str, default='None',
                        help="If it equals to 'None', then use real data only")
    parser.add_argument('--teacher', type=str, default='ResNet110',
                        choices=['MobileNet', 'ShuffleNet',
                        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                        'VGG11', 'VGG13', 'VGG16', 'VGG19',
                        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                        'ResNet8', 'ResNet20', 'ResNet110',
                        'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152',
                        'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161'],
                        help='Teacher Net.')
    parser.add_argument('--teacher_ckpt_filename', type=str, default='')
    parser.add_argument('--student', type=str, default='ResNet8',
                        choices=['MobileNet', 'ShuffleNet',
                        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                        'VGG11', 'VGG13', 'VGG16', 'VGG19',
                        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                        'ResNet8', 'ResNet20', 'ResNet110',
                        'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152',
                        'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161'],
                        help='Teacher Assistant Net.')
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Datast Settings '''
    parser.add_argument('--ntrain', type=int, default=30000, help='number of images for training')
    parser.add_argument('--nfake', type=float, default=1e30, help='number of fake images for training')
    parser.add_argument('--num_classes', type=int, default=10, metavar='N',choices=[10, 100]) #CIFAR10 or CIFAR100
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=32, metavar='N')

    ''' CNN Settings '''
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--save_freq', type=str, default='150_250')
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=256)
    parser.add_argument('--lr_base', type=float, default=0.1, help='base learning rate of CNNs')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='150_250', help='decay lr at which epoch; separate by _')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--transform', action='store_true', default=False, help='conventional data augmentation for CNN training')

    ''' Knowledge Distillation Settings '''
    parser.add_argument('--lambda_kd', type=float, default=0.05)
    parser.add_argument('--T_kd', type=int, default=1)

    args = parser.parse_args()

    return args




'''

Options for Teacher Assistant Knowledge Distillation: TAKD

Mirzadeh, Seyed Iman, et al. "Improved knowledge distillation via teacher assistant." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 04. 2020.

'''
def takd_opts():
    parser = argparse.ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--fake_dataset_name', type=str, default='None',
                        help="If it equals to 'None', then use real data only")
    parser.add_argument('--teacher', type=str, default='DenseNet121',
                        choices=['MobileNet', 'ShuffleNet',
                        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                        'VGG11', 'VGG13', 'VGG16', 'VGG19',
                        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                        'ResNet8', 'ResNet20', 'ResNet110',
                        'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152',
                        'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161'],
                        help='Teacher Net.')
    parser.add_argument('--teacher_ckpt_filename', type=str, default='')
    parser.add_argument('--teacher_assistant', type=str, default='ResNet34',
                        choices=['MobileNet', 'ShuffleNet',
                        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                        'VGG11', 'VGG13', 'VGG16', 'VGG19',
                        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                        'ResNet8', 'ResNet20', 'ResNet110',
                        'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152',
                        'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161'],
                        help='Teacher Assistant Net.')
    parser.add_argument('--student', type=str, default='MobileNet',
                        choices=['MobileNet', 'ShuffleNet',
                        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                        'VGG11', 'VGG13', 'VGG16', 'VGG19',
                        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                        'ResNet8', 'ResNet20', 'ResNet110',
                        'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152',
                        'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161'],
                        help='Teacher Assistant Net.')
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)


    ''' Datast Settings '''
    parser.add_argument('--ntrain', type=int, default=30000, help='number of images for training')
    parser.add_argument('--nfake', type=float, default=1e30, help='number of fake images for training')
    parser.add_argument('--num_classes', type=int, default=10, metavar='N',choices=[10, 100]) #CIFAR10 or CIFAR100
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=32, metavar='N')

    ''' CNN Settings '''
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--resume_epoch_1', type=int, default=0) #for TA
    parser.add_argument('--resume_epoch_2', type=int, default=0) #for student
    parser.add_argument('--save_freq', type=str, default='150_250')
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=256)
    parser.add_argument('--lr_base', type=float, default=0.1, help='base learning rate of CNNs')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='150_250', help='decay lr at which epoch; separate by _')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--transform', action='store_true', default=False, help='conventional data augmentation for CNN training')


    ''' Knowledge Distillation Settings '''
    parser.add_argument('--assistant_lambda_kd', type=float, default=0.5)
    parser.add_argument('--assistant_T_kd', type=int, default=10)

    parser.add_argument('--student_lambda_kd', type=float, default=0.5)
    parser.add_argument('--student_T_kd', type=int, default=10)

    args = parser.parse_args()

    return args
