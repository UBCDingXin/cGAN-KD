from __future__ import print_function

import os
import argparse
import socket
import time

# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate


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
    parser.add_argument('--init_model_path', type=str, default='')
    
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--resume_epoch', type=int, default=0)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--model', type=str, default='resnet110')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if (opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']) and not opt.finetune:
        opt.learning_rate = 0.01

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    if (not opt.use_fake_data) or opt.nfake<=0:
        opt.save_folder = os.path.join(opt.root_path, 'output/teacher_models/vanilla')
        opt.model_name = '{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)
    else:
        fake_data_name = opt.fake_data_path.split('/')[-1]
        opt.save_folder = os.path.join(opt.root_path, 'output/teacher_models/{}_useNfake_{}'.format(fake_data_name, opt.nfake))
        opt.model_name = '{}_lr_{}_decay_{}_finetune_{}_trial_{}'.format(opt.model, opt.learning_rate,
                                                            opt.weight_decay, opt.finetune, opt.trial)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def main():
    # best_acc = 0

    opt = parse_option()

    # dataloader
    train_loader, val_loader = get_cifar100_dataloaders(opt.data_path, batch_size=opt.batch_size, num_workers=opt.num_workers, use_fake_data=opt.use_fake_data, fake_data_folder=opt.fake_data_path, nfake=opt.nfake)
    n_cls = 100

    # model
    model = model_dict[opt.model](num_classes=n_cls)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    if opt.finetune:
        ckpt_cnn_filename = os.path.join(opt.save_folder, 'ckpt_{}_epoch_{}_finetune_True_last.pth'.format(opt.model, opt.epochs))
        ## load pre-trained model
        checkpoint = torch.load(opt.init_model_path)
        model.load_state_dict(checkpoint['model'])
    else:
        ckpt_cnn_filename = os.path.join(opt.save_folder, 'ckpt_{}_epoch_{}_last.pth'.format(opt.model, opt.epochs))
    print('\n ' + ckpt_cnn_filename)

    if not os.path.isfile(ckpt_cnn_filename):
        print("\n Start training the {} >>>".format(opt.model))
        
        opt.save_intrain_folder = os.path.join(opt.save_folder, 'ckpts_in_train', opt.model_name)
        os.makedirs(opt.save_intrain_folder, exist_ok=True)
        
        if opt.resume_epoch>0:
            save_file = opt.save_intrain_folder + "/ckpt_{}_epoch_{}.pth".format(opt.model, opt.resume_epoch)
            checkpoint = torch.load(save_file)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        #end if

        for epoch in range(opt.resume_epoch, opt.epochs):
            adjust_learning_rate(epoch, opt, optimizer)
            print("==> training...")

            time1 = time.time()
            train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
            test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)
            time2 = time.time()
            print('\r epoch {}/{}: train_acc:{:.3f}, test_acc:{:.3f}, total time {:.2f}'.format(epoch+1, opt.epochs, train_acc, test_acc, time2 - time1))
        
            # regular saving
            if (epoch+1) % opt.save_freq == 0:
                print('==> Saving...')
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'accuracy': test_acc,
                }
                save_file = os.path.join(opt.save_intrain_folder, 'ckpt_{}_epoch_{}.pth'.format(opt.model, epoch+1))
                torch.save(state, save_file)
        
        ##end for epoch
        # store model
        torch.save({
            'opt':opt,
            'model': model.state_dict(),
        }, ckpt_cnn_filename)
        print("\n End training CNN.")
    
    else:
        print("\n Loading pre-trained {}.".format(opt.model))
        checkpoint = torch.load(ckpt_cnn_filename)
        model.load_state_dict(checkpoint['model'])
        
        
    test_acc, test_acc_top5, _ = validate(val_loader, model, criterion, opt)
    print("\n {}, test_acc:{:.3f}, test_acc_top5:{:.3f}.".format(opt.model, test_acc, test_acc_top5))

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
    main()
