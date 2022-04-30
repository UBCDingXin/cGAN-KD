import argparse

'''

Options for Teacher Assistant Knowledge Distillation: TAKD

Mirzadeh, Seyed Iman, et al. "Improved knowledge distillation via teacher assistant." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 04. 2020.

'''
def takd_opts():
    parser = argparse.ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=2021, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)


    ''' Datast Settings '''
    parser.add_argument('--num_classes', type=int, default=100, metavar='N',choices=[10, 100]) #CIFAR10 or CIFAR100
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=32, metavar='N')
    
    parser.add_argument('--use_fake_data', action='store_true', default=False)
    parser.add_argument('--fake_data_path', type=str, default='')
    parser.add_argument('--nfake', type=int, default=100000)
    

    ''' CNN Settings '''
    parser.add_argument('--student', type=str, default='MobileNet', help='Student Net.')
    parser.add_argument('--assistant', type=str, default='MobileNet', help='Teacher Assistant Net.')
    parser.add_argument('--teacher_ckpt_path', type=str, default='')
    
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--init_assistant_path', type=str, default='')
    parser.add_argument('--init_student_path', type=str, default='')
    
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--resume_epoch_1', type=int, default=0) #for TA
    parser.add_argument('--resume_epoch_2', type=int, default=0) #for student
    parser.add_argument('--save_freq', type=str, default='60_120_180_240')
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr_base1', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--lr_base2', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='150_180_210', help='decay lr at which epoch; separate by _')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--transform', action='store_true', default=False, help='conventional data augmentation for CNN training')


    ''' Knowledge Distillation Settings '''
    parser.add_argument('--assistant_lambda_kd', type=float, default=0.5)
    parser.add_argument('--assistant_T_kd', type=int, default=10)

    parser.add_argument('--student_lambda_kd', type=float, default=0.5)
    parser.add_argument('--student_T_kd', type=int, default=10)

    args = parser.parse_args()

    return args
