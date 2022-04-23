import pdb
import time
import argparse
import numpy as np
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='CNN')

parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')

parser.add_argument('--mode', type=str, default='')
parser.add_argument('--dataset', '-d', default='cifar100',
                    )
parser.add_argument('--model', '-a', default='resnet18',
                    )
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=240,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning rate')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay ratio')
parser.add_argument('--lr_adjust_step', default=[150, 180, 210], type=int, nargs='+',
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')

parser.add_argument('--teacher', type=str, default='',
                    help='teacher model')
parser.add_argument('--teacher-weight', type=str, default='',
                    help='teacher model weight path')
parser.add_argument('--kd-loss-weight', type=float, default=1.0,
                    help='review kd loss weight')
parser.add_argument('--kd-warm-up', type=float, default=20.0,
                    help='feature konwledge distillation loss weight warm up epochs')

parser.add_argument('--use-kl', action='store_true', default=False,
                    help='use kl kd loss')
parser.add_argument('--kl-loss-weight', type=float, default=1.0,
                    help='kl konwledge distillation loss weight')
parser.add_argument('-T', type=float, default=4.0,
                    help='knowledge distillation loss temperature')
parser.add_argument('--ce-loss-weight', type=float, default=1.0,
                    help='cross entropy loss weight')


args = parser.parse_args()
assert torch.cuda.is_available()

cudnn.deterministic = True
cudnn.benchmark = False
if args.seed == 0:
    args.seed = np.random.randint(1000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)


from util.misc import *
from util.kd import DistillKL

from model.resnet import ResNet18, ResNet50
from model.resnet_cifar import build_resnet_backbone, build_resnetx4_backbone
from model.resnetv2_cifar import ResNet50
from model.vgg import vgg_dict
from model.mobilenetv2 import mobile_half
from model.shufflenetv1 import ShuffleV1
from model.shufflenetv2 import ShuffleV2
from model.wide_resnet_cifar import wrn
from model.wide_resnet import WideResNet
from model.reviewkd import build_review_kd, hcl


output_directory = os.path.join(args.root_path, "output")
os.makedirs(output_directory, exist_ok=True)

log_directory = os.path.join(output_directory, "{}/log".format(args.mode))
os.makedirs(log_directory, exist_ok=True)
ckpt_directory = os.path.join(output_directory, "{}/models".format(args.mode))
os.makedirs(ckpt_directory, exist_ok=True)

if args.mode=="vanilla":
    test_id = "{}_epoch_{}_last".format(args.model, args.epochs)
elif args.mode=="distill":
    test_id = "S_{}_T_{}_epoch_{}_last".format(args.model, args.teacher, args.epochs)
else:
    raise Exception("Wrong mode!!!")

log_filename = os.path.join(log_directory, "log_{}.txt".format(test_id))
logger = Logger(args=args, filename=log_filename)
print(args)

with open(log_filename, 'a') as f:
    f.write("\n===================================================================================================\n")

# Image Preprocessing
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

# dataset
# if args.dataset == 'cifar10':
#     num_classes = 10
#     train_dataset = datasets.CIFAR10(root='data/',train=True,transform=train_transform,download=True)
#     test_dataset = datasets.CIFAR10(root='data/',train=False,transform=test_transform,download=True)
# elif args.dataset == 'cifar100':
#     num_classes = 100
#     train_dataset = datasets.CIFAR100(root='data/',train=True,transform=train_transform,download=True)
#     test_dataset = datasets.CIFAR100(root='data/',train=False,transform=test_transform,download=True)
# else:
#     assert False

num_classes = 100
train_dataset = datasets.CIFAR100(root=args.data_path,train=True,transform=train_transform,download=True)
test_dataset = datasets.CIFAR100(root=args.data_path,train=False,transform=test_transform,download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                           shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.batch_size,
                                          shuffle=False,pin_memory=True)

# teacher model
if 'x4' in args.teacher:
    teacher = build_resnetx4_backbone(depth = int(args.teacher[6:-2]), num_classes=num_classes)
elif 'resnet' in args.teacher:
    teacher = build_resnet_backbone(depth = int(args.teacher[6:]), num_classes=num_classes)
elif 'ResNet50' in args.teacher:
    teacher = ResNet50(num_classes=num_classes)
elif 'vgg' in args.teacher:
    teacher = vgg_dict[args.teacher](num_classes=num_classes)
elif 'mobile' in args.teacher:
    teacher = mobile_half(num_classes=num_classes)
elif 'wrn' in args.teacher:
    teacher = wrn(depth = int(args.teacher[4:6]), widen_factor = int(args.teacher[-1:]), num_classes=num_classes)
elif args.teacher == '':
    teacher = None
else:
    assert False
if teacher is not None:
    load_teacher_weight(teacher, args.teacher_weight, args.teacher)
    teacher = teacher.cuda()
# model
if teacher is not None:
    cnn = build_review_kd(args.model, num_classes=num_classes, teacher = args.teacher)
elif 'x4' in args.model:
    cnn = build_resnetx4_backbone(depth = int(args.model[6:-2]), num_classes=num_classes)
elif 'resnet' in args.model:
    cnn = build_resnet_backbone(depth = int(args.model[6:]), num_classes=num_classes)
elif 'ResNet50' in args.model:
    cnn = ResNet50(num_classes=num_classes)
elif 'vgg' in args.model:
    cnn = vgg_dict[args.model](num_classes=num_classes)
elif 'Mobile' in args.model:
    cnn = mobile_half(num_classes=num_classes)
elif 'ShuffleV1' in args.model:
    cnn = ShuffleV1(num_classes=num_classes)
elif 'ShuffleV2' in args.model:
    cnn = ShuffleV2(num_classes=num_classes)
elif 'wrn' in args.model:
    cnn = wrn(depth = int(args.model[4:6]), widen_factor = int(args.model[-1:]), num_classes=num_classes)
elif args.model == 'wideresnet':
    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)
else:
    assert False

if 'Shuffle' in args.model or 'Mobile' in args.model:
    # args.lr = 0.02 ## ReviewKD
    args.lr = 0.01 ## to be consistent with RepDistiller

trainable_parameters = nn.ModuleList()
trainable_parameters.append(cnn)

criterion = nn.CrossEntropyLoss().cuda()
kl_criterion = DistillKL(args.T)
wd = args.wd
lr = args.lr
cnn_optimizer = torch.optim.SGD(trainable_parameters.parameters(), lr=args.lr,
                                momentum=0.9, nesterov=True, weight_decay=wd)

# test
def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            pred = cnn(images)
        if teacher is not None:
            fs, pred = pred

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc



ckpt_cnn_filename = os.path.join(ckpt_directory, 'ckpt_{}.pth'.format(test_id))
print('\n ' + ckpt_cnn_filename)

if not os.path.isfile(ckpt_cnn_filename):
    # train
    best_acc = 0.0
    st_time = time.time()
    for epoch in range(args.epochs):
        loss_avg = {}
        correct = 0.
        total = 0.
        cnt_ft = {}
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()

            cnn.zero_grad()
            losses = {}
            if teacher is not None:
                s_features, pred = cnn(images)
                t_features, t_pred = teacher(images, is_feat = True, preact=True)
                t_features = t_features[1:]
                feature_kd_loss = hcl(s_features, t_features)
                losses['review_kd_loss'] = feature_kd_loss * min(1, epoch/args.kd_warm_up) * args.kd_loss_weight
                if args.use_kl:
                    losses['kl_kd_loss'] = kl_criterion(pred, t_pred) * args.kl_loss_weight
            else:
                pred = cnn(images)
            
            xentropy_loss = criterion(pred, labels)

            losses['cls_loss'] = xentropy_loss * args.ce_loss_weight
            loss = sum(losses.values())
            loss.backward()
            cnn_optimizer.step()

            for key in losses:
                if not key in loss_avg:
                    loss_avg[key] = AverageMeter()
                else:
                    loss_avg[key].update(losses[key])

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

        test_acc = test(test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
        lr = lr_schedule(lr, epoch, cnn_optimizer, args)

        loss_avg = {k: loss_avg[k].val for k in loss_avg}
        row = { 'epoch': str(epoch), 
                'train_acc': '%.2f'%(accuracy*100), 
                'test_acc': '%.2f'%(test_acc*100), 
                'best_acc': '%.2f'%(best_acc*100), 
                'lr': '%.5f'%(lr),
                'loss': '%.5f'%(sum(loss_avg.values())),
                }
        loss_avg = {k: '%.5f'%loss_avg[k] for k in loss_avg}
        row.update(loss_avg)
        row.update({
                'time': format_time(time.time()-st_time),
                'eta': format_time((time.time()-st_time)/(epoch+1)*(args.epochs-epoch-1)),
                })
        print(row)
        logger.writerow(row)
    ##end for epoch
    torch.save({
            'model': cnn.state_dict(),
        }, ckpt_cnn_filename)
    logger.close()
else:
    print("\n Loading pretrained model...")
    cnn.load_state_dict(torch.load(ckpt_cnn_filename)['model'])
    cnn = cnn.cuda()

val_acc = test(test_loader)*100.0
print(val_acc)
with open(log_filename, 'a') as f:
    f.write("\n===================================================================================================")
    
    
eval_results_fullpath = ckpt_directory + "/test_result_" + test_id + ".txt"
if not os.path.isfile(eval_results_fullpath):
    eval_results_logging_file = open(eval_results_fullpath, "w")
    eval_results_logging_file.close()
with open(eval_results_fullpath, 'a') as eval_results_logging_file:
    eval_results_logging_file.write("\n===================================================================================================")
    eval_results_logging_file.write("\n Test results for {} \n".format(test_id))
    print(args, file=eval_results_logging_file)
    eval_results_logging_file.write("\n Test accuracy: Top1 {:.3f}.".format(val_acc))
    eval_results_logging_file.write("\n Test error rate: Top1 {:.3f}.".format(100-val_acc))