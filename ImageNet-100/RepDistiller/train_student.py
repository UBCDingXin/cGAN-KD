"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.imagenet100 import get_imagenet100_dataloaders, get_imagenet100_dataloaders_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')

    ## use cGAN-generated fake data
    parser.add_argument('--use_fake_data', action='store_true', default=False)
    parser.add_argument('--fake_data_path', type=str, default='')
    parser.add_argument('--nfake', type=int, default=100000)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--init_student_path', type=str, default='')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')
    parser.add_argument('--resume_epoch', type=int, default=0)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if (opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']) and not opt.finetune:
        opt.learning_rate = 0.01


    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    if (not opt.use_fake_data) or opt.nfake<=0:
        opt.save_folder = os.path.join(opt.root_path, 'output/student_models/vanilla')
    else:
        fake_data_name = opt.fake_data_path.split('/')[-1]
        opt.save_folder = os.path.join(opt.root_path, 'output/student_models/{}_useNfake_{}'.format(fake_data_name, opt.nfake))
    os.makedirs(opt.save_folder, exist_ok=True)

    opt.model_name = 'S_{}_T_{}_{}_r_{}_a_{}_b_{}_{}'.format(opt.model_s, opt.model_t, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.save_intrain_folder = os.path.join(opt.save_folder, 'ckpts_in_train', opt.model_name)
    os.makedirs(opt.save_intrain_folder, exist_ok=True)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-1].split('_')
    if segments[1] != 'wrn':
        return segments[1]
    else:
        return segments[1] + '_' + segments[2] + '_' + segments[3]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    best_acc = 0

    opt = parse_option()

    # torch.manual_seed(2021)
    # torch.cuda.manual_seed(2021)
    # torch.backends.cudnn.deterministic = True

    # dataloader
    if opt.distill in ['crd']:
        train_loader, val_loader, n_data = get_imagenet100_dataloaders_sample(opt.data_path, batch_size=opt.batch_size,
                                                                            num_workers=opt.num_workers,
                                                                            k=opt.nce_k,
                                                                            mode=opt.mode,
                                                                            use_fake_data=opt.use_fake_data, fake_data_folder=opt.fake_data_path, nfake=opt.nfake)
    else:
        train_loader, val_loader, n_data = get_imagenet100_dataloaders(opt.data_path, batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True,
                                                                    use_fake_data=opt.use_fake_data, fake_data_folder=opt.fake_data_path, nfake=opt.nfake)
    n_cls = 100

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    ## student model name, how to initialize student model, etc.
    student_model_filename = 'S_{}_T_{}_{}_r_{}_a_{}_b_{}_epoch_{}'.format(opt.model_s, opt.model_t, opt.distill,opt.gamma, opt.alpha, opt.beta, opt.epochs)
    if opt.finetune:
        ckpt_cnn_filename = os.path.join(opt.save_folder, student_model_filename+'_finetune_True_last.pth')
        ## load pre-trained model
        checkpoint = torch.load(opt.init_student_path)
        model_s.load_state_dict(checkpoint['model'])
    else:
        ckpt_cnn_filename = os.path.join(opt.save_folder,student_model_filename+'_last.pth')
    print('\n ' + ckpt_cnn_filename)


    data = torch.randn(2, 3, 128, 128)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True


    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)


    if not os.path.isfile(ckpt_cnn_filename):

        print("\n Start training the {} >>>".format(opt.model_s))

        ## resume training
        if opt.resume_epoch>0:
            save_file = opt.save_intrain_folder + "/ckpt_{}_epoch_{}.pth".format(opt.model_s, opt.resume_epoch)
            checkpoint = torch.load(save_file)
            model_s.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            module_list.load_state_dict(checkpoint['module_list'])
            trainable_list.load_state_dict(checkpoint['trainable_list'])
            criterion_list.load_state_dict(checkpoint['criterion_list'])

            # module_list = checkpoint['module_list']
            # criterion_list = checkpoint['criterion_list']
            # # trainable_list = checkpoint['trainable_list']
            # ckpt_test_accuracy = checkpoint['accuracy']
            # ckpt_epoch = checkpoint['epoch']

            # print('\n Resume training: epoch {}, test_acc {}...'.format(ckpt_epoch, ckpt_test_accuracy))

            if torch.cuda.is_available():
                module_list.cuda()
                criterion_list.cuda()
        #end if


        for epoch in range(opt.resume_epoch, opt.epochs):

            adjust_learning_rate(epoch, opt, optimizer)
            print("==> training...")

            time1 = time.time()
            train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

            # regular saving
            if (epoch+1) % opt.save_freq == 0:
                print('==> Saving...')
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'module_list': module_list.state_dict(),
                    'criterion_list': criterion_list.state_dict(),
                    'trainable_list': trainable_list.state_dict(),
                    'accuracy': test_acc,
                }
                save_file = os.path.join(opt.save_intrain_folder, 'ckpt_{}_epoch_{}.pth'.format(opt.model_s, epoch+1))
                torch.save(state, save_file)
        ##end for epoch
        # store model
        torch.save({
            'opt':opt,
            'model': model_s.state_dict(),
        }, ckpt_cnn_filename)
        print("\n End training CNN.")

    else:
        print("\n Loading pre-trained {}.".format(opt.model_s))
        checkpoint = torch.load(ckpt_cnn_filename)
        model_s.load_state_dict(checkpoint['model'])

    test_acc, test_acc_top5, _ = validate(val_loader, model_s, criterion_cls, opt)
    print("\n {}, test_acc:{:.3f}, test_acc_top5:{:.3f}.".format(opt.model_s, test_acc, test_acc_top5))

    eval_results_fullpath = opt.save_folder + "/test_result_" + opt.model_name + ".txt"
    if not os.path.isfile(eval_results_fullpath):
        eval_results_logging_file = open(eval_results_fullpath, "w")
        eval_results_logging_file.close()
    with open(eval_results_fullpath, 'a') as eval_results_logging_file:
        eval_results_logging_file.write("\n===================================================================================================")
        eval_results_logging_file.write("\n Test results for {} \n".format(opt.model_name))
        print(opt, file=eval_results_logging_file)
        eval_results_logging_file.write("\n Test accuracy: Top1 {:.3f}, Top5 {:.3f}.".format(test_acc, test_acc_top5))
        eval_results_logging_file.write("\n Test error rate: Top1 {:.3f}, Top5 {:.3f}.".format(100-test_acc, 100-test_acc_top5))




if __name__ == '__main__':
    print("\n ===================================================================================================")
    main()
    print("\n ===================================================================================================")
