import argparse



'''

Options for Some Baseline CNN Training

'''
def cnn_opts():
    parser = argparse.ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--fake_dataset_name', type=str, default='None',
                        help="If it equals to 'None', then use real data only")
    parser.add_argument('--cnn_name', type=str, default='ShuffleNet',
                        choices=['MobileNet', 'ShuffleNet',
                        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                        'VGG11', 'VGG13', 'VGG16', 'VGG19',
                        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                        'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152',
                        'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161'],
                        help='The CNN used in the classification.')
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Datast Settings '''
    parser.add_argument('--nfake', type=float, default=1e30, help='number of fake images for training')
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=64, metavar='N')
    parser.add_argument('--min_label', type=float, default=1.0)
    parser.add_argument('--max_label', type=float, default=60.0)
    parser.add_argument('--max_num_img_per_label', type=int, default=1e30, metavar='N')
    parser.add_argument('--max_num_img_per_label_after_replica', type=int, default=200, metavar='N')

    ''' CNN Settings '''
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--save_freq', type=str, default='50_100_150_200_250_300_350')
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--lr_base', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='150_250', help='decay lr at which epoch; separate by _')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # parser.add_argument('--transform', action='store_true', default=False, help='conventional data augmentation for CNN training')
    parser.add_argument('--validaiton_mode', action='store_true', default=False, help='Select filtering threshold on the validation set')

    args = parser.parse_args()

    return args





'''

Options for GAN, DRE, and synthetic data generation

'''

def gen_synth_data_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Dataset '''
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=64, metavar='N')
    parser.add_argument('--min_label', type=float, default=1.0)
    parser.add_argument('--max_label', type=float, default=60.0)
    parser.add_argument('--max_num_img_per_label', type=int, default=9999, metavar='N')
    parser.add_argument('--max_num_img_per_label_after_replica', type=int, default=200, metavar='N')

    ''' GAN settings '''
    # label embedding setting
    parser.add_argument('--gan_dim_embed', type=int, default=128) #dimension of the embedding space

    parser.add_argument('--gan_embed_x2y_net_name', type=str, default='ResNet34')
    parser.add_argument('--gan_embed_x2y_epoch', type=int, default=200) #epoch of cnn training for label embedding
    parser.add_argument('--gan_embed_x2y_resume_epoch', type=int, default=0) #epoch of cnn training for label embedding
    parser.add_argument('--gan_embed_x2y_batch_size', type=int, default=256, metavar='N')
    parser.add_argument('--gan_embed_x2y_lr_base', type=float, default=1e-2, help='base learning rate of CNNs')
    parser.add_argument('--gan_embed_x2y_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--gan_embed_x2y_lr_decay_epochs', type=str, default='60_120', help='decay lr at which epoch; separate by _')

    parser.add_argument('--gan_embed_y2h_epoch', type=int, default=500)
    parser.add_argument('--gan_embed_y2h_batch_size', type=int, default=256, metavar='N')
    parser.add_argument('--gan_embed_y2h_lr_base', type=float, default=1e-2, help='base learning rate of CNNs')
    parser.add_argument('--gan_embed_y2h_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--gan_embed_y2h_lr_decay_epochs', type=str, default='150_250_350', help='decay lr at which epoch; separate by _')

    # gan setting
    parser.add_argument('--gan_name', type=str, default='SNGAN')
    parser.add_argument('--gan_loss_type', type=str, default='hinge')
    parser.add_argument('--gan_niters', type=int, default=40000, help='number of iterations')
    parser.add_argument('--gan_resume_niters', type=int, default=0)
    parser.add_argument('--gan_save_niters_freq', type=int, default=5000, help='frequency of saving checkpoints')
    parser.add_argument('--gan_d_niters', type=int, default=1, help='update D multiple times while update G once')
    parser.add_argument('--gan_lr_g', type=float, default=1e-4, help='learning rate for generator')
    parser.add_argument('--gan_lr_d', type=float, default=1e-4, help='learning rate for discriminator')
    parser.add_argument('--gan_dim_g', type=int, default=256, help='Latent dimension of GAN')
    parser.add_argument('--gan_batch_size_disc', type=int, default=512)
    parser.add_argument('--gan_batch_size_gene', type=int, default=512)
    parser.add_argument('--gan_gene_ch', type=int, default=64)
    parser.add_argument('--gan_disc_ch', type=int, default=64)

    # ccgan setting
    parser.add_argument('--gan_kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--gan_threshold_type', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--gan_kappa', type=float, default=-1)
    parser.add_argument('--gan_nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')

    # DiffAugment setting
    parser.add_argument('--gan_DiffAugment', action='store_true', default=False)
    parser.add_argument('--gan_DiffAugment_policy', type=str, default='color,translation,cutout')


    ''' DRE Settings '''
    ## Pre-trained AE for feature extraction
    parser.add_argument('--dre_presae_epochs', type=int, default=200)
    parser.add_argument('--dre_presae_resume_epoch', type=int, default=0, metavar='N')
    parser.add_argument('--dre_presae_lr_base', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--dre_presae_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--dre_presae_lr_decay_freq', type=int, default=50)
    parser.add_argument('--dre_presae_batch_size_train', type=int, default=128, metavar='N')
    parser.add_argument('--dre_presae_weight_decay', type=float, default=1e-4)
    parser.add_argument('--dre_presae_lambda_sparsity', type=float, default=1e-3, help='Control the sparsity of the sparse AE.')
    ## DR model in the feature space
    parser.add_argument('--dre_net', type=str, default='MLP5',
                        choices=['MLP3', 'MLP5', 'MLP7', 'MLP9'], help='DR Model in the feature space') # DRE in Feature Space
    parser.add_argument('--dre_epochs', type=int, default=350)
    parser.add_argument('--dre_lr_base', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dre_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--dre_lr_decay_epochs', type=str, default='100_200', help='decay lr at which epoch; separate by _')
    parser.add_argument('--dre_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training DRE')
    parser.add_argument('--dre_lambda', type=float, default=1e-3, help='penalty in DRE')
    parser.add_argument('--dre_resume_epoch', type=int, default=0)

    parser.add_argument('--dre_threshold_type', type=str, default='hard', choices=['soft', 'hard'])
    parser.add_argument('--dre_kappa', type=float, default=1e-20)
    parser.add_argument('--dre_nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')


    ''' Sampling Settings '''
    parser.add_argument('--subsampling', action='store_true', default=False,
                        help='cDRE-F-SP based subsampling')
    parser.add_argument('--samp_batch_size', type=int, default=1000) #also used for computing density ratios after the dre training
    parser.add_argument('--samp_burnin_size', type=int, default=5000)
    parser.add_argument('--samp_num_fake_labels', type=int, default=-1,
                        help='If equals to -1, then generate fake images for the test labels; otherwise, generate samp_num_fake_labels in [0,1]')
    parser.add_argument('--samp_nfake_per_label', type=int, default=100)
    parser.add_argument('--samp_filter_precnn_net', type=str, default='VGG11',
                        help='Pre-trained CNN for filtering and label adjustment;')
    parser.add_argument('--samp_filter_precnn_net_ckpt_filename', type=str, default='')
    parser.add_argument('--samp_filter_mae_percentile_threshold', type=float, default=1.0,
                        help='The percentile threshold of MAE to filter out bad synthetic images by a pre-trained net')
    parser.add_argument('--unfiltered_fake_dataset_filename', type=str, default='')

    args = parser.parse_args()

    return args
