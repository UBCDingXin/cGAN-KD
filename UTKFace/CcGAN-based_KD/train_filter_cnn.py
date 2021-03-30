''' For CNN training and testing. '''


import os
import timeit
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

def hflip_images(batch_images):
    uniform_threshold = np.random.uniform(0,1,len(batch_images))
    indx_gt = np.where(uniform_threshold>0.5)[0]
    batch_images[indx_gt] = torch.flip(batch_images[indx_gt], dims=[3])
    return batch_images

''' function for cnn training '''
def train_filter_cnn(net, net_name, trainloader, testloader, epochs, resume_epoch=0, save_freq=[50, 100, 150, 200, 250, 300], lr_base=0.01, lr_decay_factor=0.1, lr_decay_epochs=[150, 250], weight_decay=1e-4, seed = 2020, path_to_ckpt = None):

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
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = lr_base, momentum= 0.9, weight_decay=weight_decay)

    if path_to_ckpt is not None and resume_epoch>0:
        save_file = path_to_ckpt + "/{}_checkpoint_epoch_{}.pth".format(net_name, resume_epoch)
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if


    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate(optimizer, epoch)
        for batch_idx, (batch_train_images, batch_train_labels) in enumerate(trainloader):
    
            ## random horizontal flipping
            batch_train_images = hflip_images(batch_train_images)

            batch_train_images = batch_train_images.type(torch.float).cuda()
            batch_train_labels = batch_train_labels.type(torch.float).view(-1,1).cuda()

            #Forward pass
            outputs = net(batch_train_images)

            ## standard CE loss
            loss = criterion(outputs, batch_train_labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        #end for batch_idx
        test_mae, _ = test_filter_cnn(net, testloader, verbose=False)

        print('%s: [epoch %d/%d] train_loss:%.3f, test_mae:%.3f Time: %.4f' % (net_name, epoch+1, epochs, train_loss/(batch_idx+1), test_mae, timeit.default_timer()-start_time))

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



def test_filter_cnn(net, testloader, verbose=True):

    net = net.cuda()
    net.eval()  
    with torch.no_grad():
        total = 0
        loss_all = 0
        loss_all_list = []
        for batch_idx, (images, labels) in enumerate(testloader):
            images = images.type(torch.float).cuda()
            labels = labels.type(torch.float).view(-1)
            labels_pred = net(images)
            labels_pred = labels_pred.cpu().view(-1)
            loss = torch.abs(labels_pred-labels)
            loss_all += loss.sum().item()
            total += labels.size(0)
            loss_all_list.append(loss)
        loss_all_list = (torch.cat(loss_all_list)).numpy()
        if verbose:
            print('MAE of the model on the test images: {} (normalized scale).'.format(loss_all / total))
        assert len(loss_all_list) == total
    return loss_all / total, loss_all_list
