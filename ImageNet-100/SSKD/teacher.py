print("\n ===================================================================================================")

import os
import os.path as osp
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from utils import AverageMeter, accuracy
from models import model_dict

from teacher_data_loader import get_imagenet100



torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train teacher network.')

parser.add_argument('--root_path', default="", type=str)
parser.add_argument('--real_data', default="", type=str)
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--fake_data', default="None", type=str)
parser.add_argument('--nfake', default=1e30, type=float)
parser.add_argument('--finetune', action='store_true', default=False)
parser.add_argument('--init_model_path', type=str, default='')

parser.add_argument('--arch', type=str)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_epochs', type=str, default='30_60_90', help='decay lr at which epoch; separate by _')
parser.add_argument('--lr_decay_factor', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--save_interval', type=int, default=20)

args = parser.parse_args()

#######################################
''' set seed '''
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

''' save folders '''
if args.fake_data=="None" or args.nfake<=0:
    save_folder = os.path.join(args.root_path, 'output/teachers/vanilla')
else:
    fake_data_name = args.fake_data.split('/')[-1]
    save_folder = os.path.join(args.root_path, 'output/teachers/{}_useNfake_{}'.format(fake_data_name, args.nfake))
os.makedirs(save_folder, exist_ok=True)

setting_name = "{}_lr_{}_decay_{}".format(args.arch, args.lr, args.weight_decay)

save_intrain_folder = os.path.join(save_folder, 'ckpts_in_train', setting_name)
os.makedirs(save_intrain_folder, exist_ok=True)


''' dataloaders '''
train_loader, val_loader = get_imagenet100(num_classes=args.num_classes, real_data=args.real_data, fake_data=args.fake_data, nfake=args.nfake, batch_size=args.batch_size, num_workers=args.num_workers)



#######################################
''' learning rate decay '''
lr_decay_epochs = (args.lr_decay_epochs).split("_")
lr_decay_epochs = [int(epoch) for epoch in lr_decay_epochs]

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate """
    lr = args.lr

    num_decays = len(lr_decay_epochs)
    for decay_i in range(num_decays):
        if epoch >= lr_decay_epochs[decay_i]:
            lr = lr * args.lr_decay_factor
        #end if epoch
    #end for decay_i
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

''' init. models '''
model = model_dict[args.arch](num_classes=args.num_classes).cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


''' whether finetune '''
if args.finetune:
    ckpt_model_filename = os.path.join(save_folder, 'ckpt_{}_epoch_{}_finetune_last.pth'.format(args.arch, args.epochs))
    ## load pre-trained model
    checkpoint = torch.load(args.init_model_path)
    model.load_state_dict(checkpoint['model'])
else:
    ckpt_model_filename = os.path.join(save_folder, 'ckpt_{}_epoch_{}_last.pth'.format(args.arch, args.epochs))
print('\n ' + ckpt_model_filename)

''' start training '''

if not os.path.isfile(ckpt_model_filename):
    print("\n Start training {} >>>".format(args.arch))
    
    if args.resume_epoch>0:
        save_file = save_intrain_folder + "/ckpt_{}_epoch_{}.pth".format(args.arch, args.resume_epoch)
        checkpoint = torch.load(save_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    #end if
    
    total_train_time = 0
    for epoch in range(args.resume_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        ##########################
        ## Training Stage
        
        model.train()
        loss_record = AverageMeter()
        acc_record = AverageMeter()
    
        start = time.time()
        for x, target in train_loader:

            optimizer.zero_grad()
            x = x.cuda()
            target = target.type(torch.long).cuda()

            output = model(x)
            loss = F.cross_entropy(output, target)

            loss.backward()
            optimizer.step()

            batch_acc = accuracy(output, target, topk=(1,))[0]
            loss_record.update(loss.item(), x.size(0))
            acc_record.update(batch_acc.item(), x.size(0))

        run_time = time.time() - start
        total_train_time+=run_time
    
        info = '\n train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}'.format(epoch+1, args.epochs, run_time, loss_record.avg, acc_record.avg)
        print(info)
    
        ##########################
        ## Testing Stage
        model.eval()
        acc_record = AverageMeter()
        loss_record = AverageMeter()
        start = time.time()
        for x, target in val_loader:

            x = x.cuda()
            target = target.type(torch.long).cuda()
            with torch.no_grad():
                output = model(x)
                loss = F.cross_entropy(output, target)

            batch_acc = accuracy(output, target, topk=(1,))[0]
            loss_record.update(loss.item(), x.size(0))
            acc_record.update(batch_acc.item(), x.size(0))

        run_time = time.time() - start

        info = '\r test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}'.format(epoch+1, args.epochs, run_time, loss_record.avg, acc_record.avg)
        print(info)
        
        print('\r Total training time: {:.2f} seconds.'.format(total_train_time))
        
        ##########################
        ## save checkpoint
        if (epoch+1) % args.save_interval==0 :
            save_file = save_intrain_folder + "/ckpt_{}_epoch_{}.pth".format(args.arch, epoch+1)
            torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
            }, save_file)
    ## end for epoch
    print("\n End training CNN")
    
    ## save model
    torch.save({'model': model.state_dict()}, ckpt_model_filename)
else:
    print("\n Loading pre-trained {}.".format(args.arch))
    checkpoint = torch.load(ckpt_model_filename)
    model.load_state_dict(checkpoint['model'])


##########################
model = model.cuda()
model.eval()
acc_record = AverageMeter()
loss_record = AverageMeter()

for x, target in val_loader:

    x = x.cuda()
    target = target.type(torch.long).cuda()
    with torch.no_grad():
        output = model(x)
        loss = F.cross_entropy(output, target)

    batch_acc = accuracy(output, target, topk=(1,))[0]
    loss_record.update(loss.item(), x.size(0))
    acc_record.update(batch_acc.item(), x.size(0))


print('\n Test accuracy of {}: {:.2f}'.format(args.arch, acc_record.avg))


print("\n ===================================================================================================")
