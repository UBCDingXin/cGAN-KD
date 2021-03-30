''' For CNN training and testing. '''


import os
import timeit
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F



''' function for cnn training '''
def train_cnn(net, net_name, trainloader, testloader, epochs, resume_epoch=0, save_freq=[50,100,150,200,250,300], lr_base=0.1, lr_decay_factor=0.1, lr_decay_epochs=[150, 250], weight_decay=1e-4, seed = 2020, extract_feature=False, path_to_ckpt = None, net_teacher=None, lambda_kd=0.5, T_kd=5):

    ''' learning rate decay '''
    def adjust_learning_rate(optimizer, epoch):
        """decrease the learning rate """
        lr = lr_base

        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
            #end if epoch
        #end for decay_i
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = lr_base, momentum= 0.9, weight_decay=weight_decay)

    if path_to_ckpt is not None and resume_epoch>0:
        save_file = path_to_ckpt + "/{}_checkpoint_epoch_{}.pth".format(net_name, resume_epoch)
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    ## if teache net exists
    if net_teacher is not None:
        net_teacher = net_teacher.cuda()
        net_teacher.eval()

    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate(optimizer, epoch)
        for batch_idx, (batch_train_images, batch_train_labels) in enumerate(trainloader):

            batch_train_images = batch_train_images.type(torch.float).cuda()
            batch_train_labels = batch_train_labels.type(torch.long).cuda()

            if len(batch_train_images)==trainloader.batch_size: #skip the last iteration in each epoch
                #Forward pass
                if extract_feature:
                    outputs,_ = net(batch_train_images)
                else:
                    outputs = net(batch_train_images)

                ## standard CE loss
                loss = criterion(outputs, batch_train_labels)

                if net_teacher is not None:
                    teacher_outputs = net_teacher(batch_train_images)
                    # Knowledge Distillation Loss
                    loss_KD = nn.KLDivLoss(reduction='mean')(F.log_softmax(outputs / T_kd, dim=1), F.softmax(teacher_outputs / T_kd, dim=1))
                    loss = (1 - lambda_kd) * loss + lambda_kd * T_kd * T_kd * loss_KD

                #backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().item()
        #end for batch_idx
        test_acc = test_cnn(net, testloader, extract_feature, verbose=False)

        print('%s: [epoch %d/%d] train_loss:%.3f, test_acc:%.3f Time: %.4f' % (net_name, epoch+1, epochs, train_loss/(batch_idx+1), test_acc, timeit.default_timer()-start_time))

        # save checkpoint
        if path_to_ckpt is not None and ((epoch+1) in save_freq or (epoch+1) == epochs) :
            save_file = path_to_ckpt + "/{}_checkpoint_epoch_{}.pth".format(net_name, epoch+1)
            torch.save({
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for epoch

    return net



def test_cnn(net, testloader, extract_feature=False, verbose=False):

    net = net.cuda()
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(testloader):
            images = images.type(torch.float).cuda()
            labels = labels.type(torch.long).cuda()
            if extract_feature:
                outputs,_ = net(images)
            else:
                outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        if verbose:
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100.0 * correct / total))
    return 100.0 * correct / total
