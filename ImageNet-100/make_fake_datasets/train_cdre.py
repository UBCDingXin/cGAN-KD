'''

Functions for Training Class-conditional Density-ratio model

'''

import torch
import torch.nn as nn
import numpy as np
import os
import timeit

from utils import SimpleProgressBar
from opts import gen_synth_data_opts

''' Settings '''
args = gen_synth_data_opts()



# training function
def train_cdre(trainloader, dre_net, dre_precnn_net, netG, path_to_ckpt=None):

    # some parameters in the opts
    dim_gan = args.gan_dim_g
    dre_net_name = args.dre_net
    dre_epochs = args.dre_epochs
    dre_lr_base = args.dre_lr_base
    dre_lr_decay_factor = args.dre_lr_decay_factor
    dre_lr_decay_epochs = (args.dre_lr_decay_epochs).split("_")
    dre_lr_decay_epochs = [int(epoch) for epoch in dre_lr_decay_epochs]
    dre_lambda = args.dre_lambda
    dre_resume_epoch = args.dre_resume_epoch


    ''' learning rate decay '''
    def adjust_learning_rate(optimizer, epoch):
        lr = dre_lr_base
        num_decays = len(dre_lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= dre_lr_decay_epochs[decay_i]:
                lr = lr * dre_lr_decay_factor
            #end if epoch
        #end for decay_i
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #end def adjust lr


    # nets
    dre_precnn_net = dre_precnn_net.cuda()
    netG = netG.cuda()
    dre_precnn_net.eval()
    netG.eval()

    dre_net = dre_net.cuda()


    # define optimizer
    optimizer = torch.optim.Adam(dre_net.parameters(), lr = dre_lr_base, betas=(0.5, 0.999), weight_decay=1e-4)


    if path_to_ckpt is not None and dre_resume_epoch>0:
        print("Loading ckpt to resume training dre_net >>>")
        ckpt_fullpath = path_to_ckpt + "/cDRE_{}_checkpoint_epoch_{}.pth".format(dre_net_name, dre_resume_epoch)
        checkpoint = torch.load(ckpt_fullpath)
        dre_net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #load d_loss and g_loss
        logfile_fullpath = path_to_ckpt + "/cDRE_{}_train_loss_epoch_{}.npz".format(dre_net_name, dre_resume_epoch)
        if os.path.isfile(logfile_fullpath):
            avg_train_loss = list(np.load(logfile_fullpath))
        else:
            avg_train_loss = []
    else:
        avg_train_loss = []

    start_time = timeit.default_timer()
    for epoch in range(dre_resume_epoch, dre_epochs):

        adjust_learning_rate(optimizer, epoch)

        train_loss = 0

        for batch_idx, (batch_real_images, batch_real_labels) in enumerate(trainloader):
            dre_net.train()

            batch_size_curr = batch_real_images.shape[0]

            batch_real_images = batch_real_images.type(torch.float).cuda()
            num_unique_classes = len(list(set(batch_real_labels.numpy())))
            batch_real_labels = batch_real_labels.type(torch.long).cuda()

            with torch.no_grad():
                z = torch.randn(batch_size_curr, dim_gan, dtype=torch.float).cuda()
                batch_fake_images = netG(z, batch_real_labels)
                batch_fake_images = batch_fake_images.detach()
                _, batch_features_real = dre_precnn_net(batch_real_images)
                batch_features_real = batch_features_real.detach()
                _, batch_features_fake = dre_precnn_net(batch_fake_images)
                batch_features_fake = batch_features_fake.detach()

            # density ratios for real and fake images
            DR_real = dre_net(batch_features_real, batch_real_labels)
            DR_fake = dre_net(batch_features_fake, batch_real_labels)

            #Softplus loss
            softplus_fn = torch.nn.Softplus(beta=1,threshold=20)
            sigmoid_fn = torch.nn.Sigmoid()
            SP_div = torch.mean(sigmoid_fn(DR_fake) * DR_fake) - torch.mean(softplus_fn(DR_fake)) - torch.mean(sigmoid_fn(DR_real))
            #penalty term: prevent assigning zero to all fake image
            # penalty = dre_lambda * (torch.mean(DR_fake)/num_unique_classes - 1)**2
            # loss = SP_div/num_unique_classes + penalty
            penalty = dre_lambda * (torch.mean(DR_fake) - 1)**2
            loss = SP_div + penalty

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        # end for batch_idx

        print("cDRE+{}+lambda{}: [epoch {}/{}] [train loss {}] [Time {}]".format(dre_net_name, dre_lambda, epoch+1, dre_epochs, train_loss/(batch_idx+1), timeit.default_timer()-start_time))

        avg_train_loss.append(train_loss/(batch_idx+1))

        # save checkpoint
        if path_to_ckpt is not None and ((epoch+1) % 50 == 0 or (epoch+1)==dre_epochs):
            ckpt_fullpath = path_to_ckpt + "/cDRE_{}_checkpoint_epoch_{}.pth".format(dre_net_name, epoch+1)
            torch.save({
                    'epoch': epoch,
                    'net_state_dict': dre_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_fullpath)

            # save loss
            logfile_fullpath = path_to_ckpt + "/cDRE_{}_train_loss_epoch_{}.npz".format(dre_net_name, epoch+1)
            np.savez(logfile_fullpath, np.array(avg_train_loss))

    #end for epoch
    netG = netG.cpu() #back to memory
    return dre_net, avg_train_loss
#end for def
