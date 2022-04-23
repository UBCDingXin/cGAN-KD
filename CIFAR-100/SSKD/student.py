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
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from utils import AverageMeter, accuracy
from wrapper import wrapper

from models import model_dict

from student_dataset import IMGs_dataset

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train SSKD student network.')

parser.add_argument('--root_path', default="", type=str)
parser.add_argument('--real_data', default="", type=str)
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--fake_data', default="None", type=str)
parser.add_argument('--nfake', default=1e30, type=float)
parser.add_argument('--finetune', action='store_true', default=False)
parser.add_argument('--init_student_path', type=str, default='')

parser.add_argument('--s_arch', type=str) # student architecture
parser.add_argument('--t_path', type=str) # teacher checkpoint path

parser.add_argument('--t_epochs', type=int, default=60, help="for training ssp_head")

parser.add_argument('--epochs', type=int, default=240)
parser.add_argument('--resume_epoch', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--lr_decay_epochs', type=str, default='150_180_210', help='decay lr at which epoch; separate by _')
parser.add_argument('--lr_decay_factor', type=float, default=0.1)
parser.add_argument('--t_lr', type=float, default=0.05)
parser.add_argument('--t_lr_decay_epochs', type=str, default='30_45', help='decay lr at which epoch; separate by _')
parser.add_argument('--t_lr_decay_factor', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)

parser.add_argument('--save_interval', type=int, default=40)
parser.add_argument('--ce_weight', type=float, default=0.1) # cross-entropy
parser.add_argument('--kd_weight', type=float, default=0.9) # knowledge distillation
parser.add_argument('--tf_weight', type=float, default=2.7) # transformation
parser.add_argument('--ss_weight', type=float, default=10.0) # self-supervision

parser.add_argument('--kd_T', type=float, default=4.0) # temperature in KD
parser.add_argument('--tf_T', type=float, default=4.0) # temperature in LT
parser.add_argument('--ss_T', type=float, default=0.5) # temperature in SS

parser.add_argument('--ratio_tf', type=float, default=1.0) # keep how many wrong predictions of LT
parser.add_argument('--ratio_ss', type=float, default=0.75) # keep how many wrong predictions of SS

args = parser.parse_args()


#######################################
''' set seed '''
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

''' save folders '''
def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-1].split('_')
    if segments[1] != 'wrn':
        return segments[1]
    else:
        return segments[1] + '_' + segments[2] + '_' + segments[3]

args.t_arch = get_teacher_name(args.t_path) ## get teacher's name

setting_name = 'S_{}_T_{}_lr_{}_decay_{}'.format(args.s_arch, args.t_arch, args.lr, args.weight_decay)

if args.fake_data=="None" or args.nfake<=0:
    save_folder = os.path.join(args.root_path, 'output/students/vanilla')
else:
    fake_data_name = args.fake_data.split('/')[-1]
    save_folder = os.path.join(args.root_path, 'output/students/{}_useNfake_{}'.format(fake_data_name, args.nfake))

save_folder_ssp = os.path.join(args.root_path, 'output/students/vanilla', 'ssp_heads')
os.makedirs(save_folder_ssp, exist_ok=True)

save_intrain_folder = os.path.join(save_folder, 'ckpts_in_train', setting_name)
os.makedirs(save_intrain_folder, exist_ok=True)


''' dataloaders '''
trainset = IMGs_dataset(train=True, num_classes=args.num_classes, real_data=args.real_data, fake_data=args.fake_data, nfake=args.nfake)
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

valset = IMGs_dataset(train=False, num_classes=args.num_classes, real_data=args.real_data, fake_data=args.fake_data, nfake=args.nfake)
val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

trainset_ssp = IMGs_dataset(train=True, num_classes=args.num_classes, real_data=args.real_data, fake_data="None")
train_loader_ssp = DataLoader(trainset_ssp, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

# train_loader_ssp = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)



#####################################################
''' Load teacher model and train ssp head '''
t_lr_decay_epochs = (args.t_lr_decay_epochs).split("_")
t_lr_decay_epochs = [int(epoch) for epoch in t_lr_decay_epochs]

def adjust_learning_rate1(optimizer, epoch):
    """decrease the learning rate """
    lr = args.t_lr

    num_decays = len(t_lr_decay_epochs)
    for decay_i in range(num_decays):
        if epoch >= t_lr_decay_epochs[decay_i]:
            lr = lr * args.t_lr_decay_factor
        #end if epoch
    #end for decay_i
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

## load teacher
t_model = model_dict[args.t_arch](num_classes=args.num_classes).cuda()
state_dict = torch.load(args.t_path)['model']
t_model.load_state_dict(state_dict)
t_model = wrapper(module=t_model).cuda()

t_optimizer = optim.SGD([{'params':t_model.backbone.parameters(), 'lr':0.0},
                        {'params':t_model.proj_head.parameters(), 'lr':args.t_lr}],
                        momentum=args.momentum, weight_decay=args.weight_decay)
t_model.eval()

## pretrained teacher model's test accuracy
t_acc_record = AverageMeter()
t_loss_record = AverageMeter()
start = time.time()
for x, target in val_loader:

    x = x[:,0,:,:,:].cuda()
    target = target.long().cuda()
    with torch.no_grad():
        output, _, feat = t_model(x)
        loss = F.cross_entropy(output, target)

    batch_acc = accuracy(output, target, topk=(1,))[0]
    t_acc_record.update(batch_acc.item(), x.size(0))
    t_loss_record.update(loss.item(), x.size(0))

run_time = time.time() - start
info = '\n teacher cls_acc:{:.2f}'.format(t_acc_record.avg)
print(info)

# train ssp_head
ssp_head_path = save_folder_ssp+'/ckpt_{}_ssp_head_epoch_{}.pth'.format(args.t_arch, args.t_epochs)
if not os.path.isfile(ssp_head_path):
    print("\n Start training SSP head >>>")
    for epoch in range(args.t_epochs):

        adjust_learning_rate1(t_optimizer, epoch)

        t_model.eval()
        loss_record = AverageMeter()
        acc_record = AverageMeter()

        start = time.time()
        for x, _ in train_loader_ssp:

            t_optimizer.zero_grad()

            x = x.cuda()
            c,h,w = x.size()[-3:]
            x = x.view(-1, c, h, w)

            _, rep, feat = t_model(x, bb_grad=False)
            batch = int(x.size(0) / 4)
            nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
            aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

            nor_rep = rep[nor_index]
            aug_rep = rep[aug_index]
            nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
            simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
            target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            loss = F.cross_entropy(simi, target)

            loss.backward()
            t_optimizer.step()

            batch_acc = accuracy(simi, target, topk=(1,))[0]
            loss_record.update(loss.item(), 3*batch)
            acc_record.update(batch_acc.item(), 3*batch)

        run_time = time.time() - start
        info = '\n teacher_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}'.format(
            epoch+1, args.t_epochs, run_time, loss_record.avg, acc_record.avg)
        print(info)

        t_model.eval()
        acc_record = AverageMeter()
        loss_record = AverageMeter()
        start = time.time()
        for x, _ in val_loader:

            x = x.cuda()
            c,h,w = x.size()[-3:]
            x = x.view(-1, c, h, w)

            with torch.no_grad():
                _, rep, feat = t_model(x)
            batch = int(x.size(0) / 4)
            nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
            aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

            nor_rep = rep[nor_index]
            aug_rep = rep[aug_index]
            nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
            simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
            target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            loss = F.cross_entropy(simi, target)

            batch_acc = accuracy(simi, target, topk=(1,))[0]
            acc_record.update(batch_acc.item(),3*batch)
            loss_record.update(loss.item(), 3*batch)

        run_time = time.time() - start

        info = '\r ssp_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}'.format(
                epoch+1, args.t_epochs, run_time, loss_record.avg, acc_record.avg)
        print(info)
    ## end for epoch
    ## save teacher ckpt
    torch.save(t_model.state_dict(), ssp_head_path)
else:
    print("\n Loading pre-trained SSP head>>>")
    checkpoint = torch.load(ssp_head_path)
    t_model.load_state_dict(checkpoint)


#####################################################
'''        Train Student model via SSKD           '''
lr_decay_epochs = (args.lr_decay_epochs).split("_")
lr_decay_epochs = [int(epoch) for epoch in lr_decay_epochs]

def adjust_learning_rate2(optimizer, epoch):
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


## init. student net
s_model = model_dict[args.s_arch](num_classes=args.num_classes)
s_model = wrapper(module=s_model)
s_model_path = "ckpt_" + setting_name + "_epoch_{}".format(args.epochs)
if args.finetune:
    print("\n Initialize student model by pre-trained one")
    s_model_path = os.path.join(save_folder, s_model_path+'_finetune_last.pth')
    ## load pre-trained model
    checkpoint = torch.load(args.init_student_path)
    s_model.load_state_dict(checkpoint['model'])
else:
    s_model_path = os.path.join(save_folder, s_model_path+'_last.pth')
s_model = s_model.cuda()
print('\r ' + s_model_path)

## optimizer
optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


## training
print("\n -----------------------------------------------------------------------------------------")
# training
if not os.path.isfile(s_model_path):
    print("\n Start training {} as student net >>>".format(args.s_arch))

    ## resume training
    if args.resume_epoch>0:
        save_file = os.path.join(save_intrain_folder, "ckpt_{}_epoch_{}.pth".format(args.s_arch, args.resume_epoch))
        checkpoint = torch.load(save_file)
        s_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    total_train_time = 0
    for epoch in range(args.resume_epoch, args.epochs):

        adjust_learning_rate2(optimizer, epoch)

        ##########################
        # train
        s_model.train()
        loss1_record = AverageMeter()
        loss2_record = AverageMeter()
        loss3_record = AverageMeter()
        loss4_record = AverageMeter()
        cls_acc_record = AverageMeter()
        ssp_acc_record = AverageMeter()

        start = time.time()
        for x, target in train_loader:

            optimizer.zero_grad()

            c,h,w = x.size()[-3:]
            x = x.view(-1,c,h,w).cuda()
            target = target.long().cuda()

            batch = int(x.size(0) / 4)
            nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
            aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

            output, s_feat, _ = s_model(x, bb_grad=True)
            log_nor_output = F.log_softmax(output[nor_index] / args.kd_T, dim=1)
            log_aug_output = F.log_softmax(output[aug_index] / args.tf_T, dim=1)
            with torch.no_grad():
                knowledge, t_feat, _ = t_model(x)
                nor_knowledge = F.softmax(knowledge[nor_index] / args.kd_T, dim=1)
                aug_knowledge = F.softmax(knowledge[aug_index] / args.tf_T, dim=1)

            # error level ranking
            aug_target = target.unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            rank = torch.argsort(aug_knowledge, dim=1, descending=True)
            rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
            index = torch.argsort(rank)
            tmp = torch.nonzero(rank, as_tuple=True)[0]
            wrong_num = tmp.numel()
            correct_num = 3*batch - wrong_num
            wrong_keep = int(wrong_num * args.ratio_tf)
            index = index[:correct_num+wrong_keep]
            distill_index_tf = torch.sort(index)[0]

            s_nor_feat = s_feat[nor_index]
            s_aug_feat = s_feat[aug_index]
            s_nor_feat = s_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            s_aug_feat = s_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
            s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, dim=1)

            t_nor_feat = t_feat[nor_index]
            t_aug_feat = t_feat[aug_index]
            t_nor_feat = t_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            t_aug_feat = t_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
            t_simi = F.cosine_similarity(t_aug_feat, t_nor_feat, dim=1)

            t_simi = t_simi.detach()
            aug_target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            rank = torch.argsort(t_simi, dim=1, descending=True)
            rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
            index = torch.argsort(rank)
            tmp = torch.nonzero(rank, as_tuple=True)[0]
            wrong_num = tmp.numel()
            correct_num = 3*batch - wrong_num
            wrong_keep = int(wrong_num * args.ratio_ss)
            index = index[:correct_num+wrong_keep]
            distill_index_ss = torch.sort(index)[0]

            log_simi = F.log_softmax(s_simi / args.ss_T, dim=1)
            simi_knowledge = F.softmax(t_simi / args.ss_T, dim=1)

            loss1 = F.cross_entropy(output[nor_index], target)
            loss2 = F.kl_div(log_nor_output, nor_knowledge, reduction='batchmean') * args.kd_T * args.kd_T
            loss3 = F.kl_div(log_aug_output[distill_index_tf], aug_knowledge[distill_index_tf], \
                            reduction='batchmean') * args.tf_T * args.tf_T
            loss4 = F.kl_div(log_simi[distill_index_ss], simi_knowledge[distill_index_ss], \
                            reduction='batchmean') * args.ss_T * args.ss_T

            loss = args.ce_weight * loss1 + args.kd_weight * loss2 + args.tf_weight * loss3 + args.ss_weight * loss4

            loss.backward()
            optimizer.step()

            cls_batch_acc = accuracy(output[nor_index], target, topk=(1,))[0]
            ssp_batch_acc = accuracy(s_simi, aug_target, topk=(1,))[0]
            loss1_record.update(loss1.item(), batch)
            loss2_record.update(loss2.item(), batch)
            loss3_record.update(loss3.item(), len(distill_index_tf))
            loss4_record.update(loss4.item(), len(distill_index_ss))
            cls_acc_record.update(cls_batch_acc.item(), batch)
            ssp_acc_record.update(ssp_batch_acc.item(), 3*batch)

        run_time = time.time() - start
        total_train_time += run_time
        info = '\n student_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ce_loss:{:.3f}\t kd_loss:{:.3f}\t cls_acc:{:.2f}'.format(epoch+1, args.epochs, run_time, loss1_record.avg, loss2_record.avg, cls_acc_record.avg)
        print(info)

        ##########################
        # cls val
        s_model.eval()
        acc_record = AverageMeter()
        loss_record = AverageMeter()
        start = time.time()
        for x, target in val_loader:

            x = x[:,0,:,:,:].cuda()
            target = target.long().cuda()
            with torch.no_grad():
                output, _, feat = s_model(x)
                loss = F.cross_entropy(output, target)

            batch_acc = accuracy(output, target, topk=(1,))[0]
            acc_record.update(batch_acc.item(), x.size(0))
            loss_record.update(loss.item(), x.size(0))

        run_time = time.time() - start

        info = '\r student_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_acc:{:.2f}'.format(epoch+1, args.epochs, run_time, acc_record.avg)
        print(info)

        ##########################
        ## save checkpoint
        if (epoch+1) % args.save_interval==0 :
            save_file = save_intrain_folder + "/ckpt_{}_epoch_{}.pth".format(args.s_arch, epoch+1)
            torch.save({
                    'model': s_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
            }, save_file)
    ## end for epoch
    print("\n End training CNN")

    ## save model
    torch.save({'model': s_model.state_dict()}, s_model_path)
else:
    print("\n Loading pre-trained {}.".format(args.s_arch))
    checkpoint = torch.load(s_model_path)
    s_model.load_state_dict(checkpoint['model'])
## end if

##########################
s_model = s_model.cuda()
s_model.eval()
acc_record = AverageMeter()
loss_record = AverageMeter()
start = time.time()
for x, target in val_loader:

    x = x[:,0,:,:,:].cuda()
    target = target.long().cuda()
    with torch.no_grad():
        output, _, feat = s_model(x)
        loss = F.cross_entropy(output, target)

    batch_acc = accuracy(output, target, topk=(1,))[0]
    acc_record.update(batch_acc.item(), x.size(0))
    loss_record.update(loss.item(), x.size(0))

print('\n Test accuracy of {}: {:.2f}'.format(args.s_arch, acc_record.avg))

''' Dump test results '''
test_results_logging_fullpath = save_folder + '/test_results_S_{}_T_{}.txt'.format(args.s_arch, args.t_arch)
if not os.path.isfile(test_results_logging_fullpath):
    test_results_logging_file = open(test_results_logging_fullpath, "w")
    test_results_logging_file.close()
with open(test_results_logging_fullpath, 'a') as test_results_logging_file:
    test_results_logging_file.write("\n===================================================================================================")
    test_results_logging_file.write("\n Teacher: {}; Student: {}; seed: {} \n".format(args.t_arch, args.s_arch, args.seed))
    print(args, file=test_results_logging_file)
    test_results_logging_file.write("\n Teacher test accuracy {}.".format(t_acc_record.avg))
    test_results_logging_file.write("\n Student test accuracy {}.".format(acc_record.avg))



print("\n ===================================================================================================")
