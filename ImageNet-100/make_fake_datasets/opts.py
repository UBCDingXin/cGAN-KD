import argparse


def gen_synth_data_opts():
    parser = argparse.ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--eval_ckpt_path', type=str, default='')
    parser.add_argument('--gan_ckpt_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Datast Settings '''
    parser.add_argument('--ntrain', type=int, default=50000, help='number of images for training')
    parser.add_argument('--num_classes', type=int, default=100, metavar='N',choices=[10, 100]) #CIFAR10 or CIFAR100
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=128, metavar='N')
    parser.add_argument('--visualize_fake_images', action='store_true', default=False)

    ''' GAN Settings '''
    parser.add_argument('--gan_net', type=str, default='BigGANdeep')
    parser.add_argument('--gan_dim_g', type=int, default=128)

    ''' DRE Settings '''
    ## Pre-trained CNN for feature extraction
    parser.add_argument('--dre_precnn_net', type=str, default='ResNet34',
                        help='Pre-trained CNN for DRE in Feature Space; Candidate: ResNetXX')
    parser.add_argument('--dre_precnn_epochs', type=int, default=350)
    parser.add_argument('--dre_precnn_resume_epoch', type=int, default=0, metavar='N')
    parser.add_argument('--dre_precnn_lr_base', type=float, default=0.1, help='base learning rate of CNNs')
    parser.add_argument('--dre_precnn_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--dre_precnn_lr_decay_epochs', type=str, default='150_250', help='decay lr at which epoch; separate by _')
    parser.add_argument('--dre_precnn_batch_size_train', type=int, default=128, metavar='N')
    parser.add_argument('--dre_precnn_weight_decay', type=float, default=1e-4)
    parser.add_argument('--dre_precnn_transform', action='store_true', default=False)
    parser.add_argument('--dre_precnn_lambda', type=float, default=0.0, help='penalty strength for the reconstruction loss')
    
    ## DR model in the feature space
    parser.add_argument('--dre_net', type=str, default='MLP5',
                        choices=['MLP3', 'MLP5', 'MLP7', 'MLP9'], help='DR Model in the feature space') # DRE in Feature Space
    parser.add_argument('--dre_epochs', type=int, default=200)
    parser.add_argument('--dre_lr_base', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dre_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--dre_lr_decay_epochs', type=str, default='100_250', help='decay lr at which epoch; separate by _')
    parser.add_argument('--dre_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training DRE')
    parser.add_argument('--dre_lambda', type=float, default=1e-2, help='penalty in DRE')
    parser.add_argument('--dre_resume_epoch', type=int, default=0)
    parser.add_argument('--dre_transform', action='store_true', default=False)

    ''' Sampling Settings '''
    parser.add_argument('--subsampling', action='store_true', default=False)
    parser.add_argument('--samp_batch_size', type=int, default=1000) #also used for computing density ratios after the dre training
    parser.add_argument('--samp_burnin_size', type=int, default=5000)
    parser.add_argument('--samp_nfake_per_class', type=int, default=1000) #number of fake images per class for evaluation

    ''' Filtering and label adjust Settings '''
    parser.add_argument('--filter', action='store_true', default=False)
    parser.add_argument('--adjust', action='store_true', default=False)
    parser.add_argument('--samp_filter_precnn_net', type=str, default='DenseNet121',
                        help='Pre-trained CNN for filtering;')
    parser.add_argument('--samp_filter_precnn_net_ckpt_path', type=str, default='')
    parser.add_argument('--samp_filter_ce_percentile_threshold', type=float, default=1.0,
                        help='The percentile threshold of cross entropy to filter out bad synthetic images by a pre-trained net')
    parser.add_argument('--samp_filter_batch_size', type=int, default=100)
    parser.add_argument('--samp_filter_burnin_size', type=int, default=1000)

    parser.add_argument('--visualize_filtered_images', action='store_true', default=False)

    ''' Evaluation '''
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--eval_FID_batch_size', type=int, default=200)

    args = parser.parse_args()

    return args



