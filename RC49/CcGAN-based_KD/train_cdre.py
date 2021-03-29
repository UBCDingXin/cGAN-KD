'''

Functions for Training Class-conditional Density-ratio model

'''

import torch
import torch.nn as nn
import numpy as np
import os
import timeit

from utils import *
from opts import gen_synth_data_opts

''' Settings '''
args = gen_synth_data_opts()

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
dre_batch_size = args.dre_batch_size

threshold_type = args.dre_threshold_type
nonzero_soft_weight_threshold = args.dre_nonzero_soft_weight_threshold


# training function
def train_cdre(kappa, train_images, train_labels, test_labels, dre_net, dre_precnn_net, netG, net_y2h, PreNetFilter=None, filter_mae_cutoff_point=1e30, path_to_ckpt=None):
    '''
    Note that train_images are not normalized to [-1,1]
    train_labels are normalized to [0,1]
    filter_mae_cutoff_point must have the same scale as normalized labels
    '''

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
    net_y2h = net_y2h.cuda()
    dre_precnn_net.eval()
    netG.eval()
    net_y2h.eval()

    if PreNetFilter is not None:
        PreNetFilter = PreNetFilter.cuda()
        PreNetFilter.eval()

    dre_net = dre_net.cuda()

    # define optimizer
    # optimizer = torch.optim.Adam(dre_net.parameters(), lr=dre_lr_base, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer = torch.optim.Adam(dre_net.parameters(), lr=dre_lr_base, betas=(0.5, 0.999))

    if path_to_ckpt is not None and dre_resume_epoch > 0:
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

    #################
    unique_train_labels = np.sort(np.array(list(set(train_labels))))
    if test_labels is not None:
        unique_test_labels = np.sort(np.array(list(set(test_labels))))

    start_time = timeit.default_timer()
    for epoch in range(dre_resume_epoch, dre_epochs):

        adjust_learning_rate(optimizer, epoch)

        train_loss = 0

        for batch_idx in range(len(train_images)//dre_batch_size):

            dre_net.train()

            ## Target labels
            if test_labels is not None:
                batch_fake_labels = np.random.choice(unique_test_labels, size=dre_batch_size, replace=True)
            else:
                batch_fake_labels = np.random.uniform(low=0.0, high=1.0, size=dre_batch_size)

            ## Find index of real images with labels in the vicinity of batch_fake_labels
            batch_real_indx = np.zeros(dre_batch_size, dtype=int) #index of images in the datata; the labels of these images are in the vicinity
            for j in range(dre_batch_size):
                ## index for real images
                if threshold_type == "hard":
                    indx_real_in_vicinity = np.where(np.abs(train_labels-batch_fake_labels[j])<= kappa)[0]
                else:
                    # reverse the weight function for SVDL
                    indx_real_in_vicinity = np.where((train_labels-batch_fake_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

                if test_labels is not None:
                    ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
                    while len(indx_real_in_vicinity)<1:
                        batch_fake_labels[j] = np.random.uniform(low=0.0, high=1.0, size=1)
                        ## index for real images
                        if threshold_type == "hard":
                            indx_real_in_vicinity = np.where(np.abs(train_labels-batch_fake_labels[j])<= kappa)[0]
                        else:
                            # reverse the weight function for SVDL
                            indx_real_in_vicinity = np.where((train_labels-batch_fake_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]
                    #end while len(indx_real_in_vicinity)<1

                assert len(indx_real_in_vicinity)>=1

                batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]
            #end for j

            ## draw the real image batch from the training set
            batch_real_images = train_images[batch_real_indx]
            assert batch_real_images.max()>1
            batch_real_labels = train_labels[batch_real_indx]
            batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).cuda()
            batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).cuda()

            ## normalize real images
            batch_real_images = (batch_real_images/255.0-0.5)/0.5
            batch_real_images = torch.from_numpy(batch_real_images).type(torch.float).cuda()
            assert len(batch_real_images) == dre_batch_size
            batch_real_images = batch_real_images.type(torch.float).cuda()
            assert batch_real_images.max().item()<=1

            ## extract features
            with torch.no_grad():
                if PreNetFilter is not None:
                    ## get fake images with assigined labels not far from their predicted labels
                    z = torch.randn(dre_batch_size, dim_gan, dtype=torch.float).cuda()
                    batch_fake_images = netG(z, net_y2h(batch_fake_labels))
                    batch_fake_labels_pred = PreNetFilter(batch_fake_images)
                    batch_mae_loss = torch.abs(batch_fake_labels.view(-1)-batch_fake_labels_pred.view(-1))
                    indx_mae_larger = batch_mae_loss>(filter_mae_cutoff_point*1.2) ##some fake images have assigned labels which are far from the predicted labels
                    niter_tmp = 0
                    while indx_mae_larger.sum().item()>0 and niter_tmp<=10:
                        # print('\n epoch:{}/{}, batch_idx:{}/{}: {} fake images have assigned labels far from gt with threshold {}'.format(epoch+1, dre_epochs, batch_idx, len(train_images)//dre_batch_size, indx_mae_larger.sum().item(), filter_mae_cutoff_point))
                        batch_size_tmp = indx_mae_larger.sum().item()
                        batch_fake_labels_tmp = batch_fake_labels[indx_mae_larger]
                        assert len(batch_fake_labels_tmp)==batch_size_tmp
                        z = torch.randn(batch_size_tmp, dim_gan, dtype=torch.float).cuda()
                        batch_fake_images_tmp = netG(z, net_y2h(batch_fake_labels_tmp))
                        batch_fake_labels_pred_tmp = PreNetFilter(batch_fake_images_tmp)
                        batch_mae_loss_tmp = torch.abs(batch_fake_labels_tmp.view(-1)-batch_fake_labels_pred_tmp.view(-1))
                        ## update
                        batch_fake_images[indx_mae_larger] = batch_fake_images_tmp
                        batch_mae_loss[indx_mae_larger] = batch_mae_loss_tmp
                        indx_mae_larger = batch_mae_loss>filter_mae_cutoff_point
                        niter_tmp+=1
                    ###end while
                    
                    indx_mae_smaller = batch_mae_loss<=filter_mae_cutoff_point
                    batch_fake_images = batch_fake_images[indx_mae_smaller]
                    batch_fake_labels = batch_fake_labels[indx_mae_smaller]
                    batch_real_images = batch_real_images[indx_mae_smaller]
                    batch_real_labels = batch_real_labels[indx_mae_smaller]
                    # print('\n epoch:{}/{}, batch_idx:{}/{}: {} fake images have assigned labels within the threshold {}'.format(epoch+1, dre_epochs, batch_idx, len(train_images)//dre_batch_size, indx_mae_smaller.sum().item(), filter_mae_cutoff_point))
                else:
                    z = torch.randn(dre_batch_size, dim_gan, dtype=torch.float).cuda()
                    batch_fake_images = netG(z, net_y2h(batch_fake_labels))
                batch_fake_images = batch_fake_images.detach()
                batch_features_real = dre_precnn_net(batch_real_images)
                batch_features_real = batch_features_real.detach()
                batch_features_fake = dre_precnn_net(batch_fake_images)
                batch_features_fake = batch_features_fake.detach()

            # density ratios for real and fake images
            DR_real = dre_net(batch_features_real, net_y2h(batch_fake_labels))
            DR_fake = dre_net(batch_features_fake, net_y2h(batch_fake_labels))

            ## weight vector
            if threshold_type == "soft":
                real_weights = torch.exp(-kappa*(batch_real_labels-batch_fake_labels)**2).cuda()
            else:
                real_weights = torch.ones(len(DR_real), dtype=torch.float).cuda()
            #end if threshold type

            #Softplus loss: vicinal trick for fake images only
            softplus_fn = torch.nn.Softplus(beta=1,threshold=20)
            sigmoid_fn = torch.nn.Sigmoid()
            SP_div_fake = torch.mean(sigmoid_fn(DR_fake) * DR_fake) - torch.mean(softplus_fn(DR_fake))
            SP_div_real = torch.mean(real_weights.view(-1) * (sigmoid_fn(DR_real)).view(-1))
            #penalty term: prevent assigning zero to all fake image
            penalty = dre_lambda * (torch.mean(DR_fake) - 1)**2
            loss = SP_div_fake - SP_div_real + penalty

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        ### end for batch_idx

        print("cDRE+{}+lambda{}: [epoch {}/{}] [train loss {}] [Time {}]".format(dre_net_name, dre_lambda, epoch+1, dre_epochs, train_loss/(batch_idx+1), timeit.default_timer()-start_time))

        avg_train_loss.append(train_loss/(batch_idx+1))


        ## debug
        ### density ratios in train mode
        batch_labels_debug = np.random.choice(unique_train_labels, size=dre_batch_size, replace=True)
        batch_real_indx = np.zeros(dre_batch_size, dtype=int)
        for j in range(dre_batch_size):
            ## index for real images
            indx_real_in_vicinity = np.where(np.abs(train_labels-batch_labels_debug[j])<= 1e-20)[0]
            assert len(indx_real_in_vicinity)>=1
            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]
        #end for j
        batch_real_images = train_images[batch_real_indx]
        assert batch_real_images.max()>1
        batch_real_images = (batch_real_images/255.0-0.5)/0.5
        batch_real_images = torch.from_numpy(batch_real_images).type(torch.float).cuda()
        assert batch_real_images.max().item()<=1

        batch_labels_debug = torch.from_numpy(batch_labels_debug).type(torch.float).cuda()
        z = torch.randn(dre_batch_size, dim_gan, dtype=torch.float).cuda()
        batch_fake_images = netG(z, net_y2h(batch_labels_debug))
        batch_fake_images = batch_fake_images.detach()

        with torch.no_grad():
            batch_features_real = dre_precnn_net(batch_real_images)
            batch_features_real = batch_features_real.detach()
            batch_features_fake = dre_precnn_net(batch_fake_images)
            batch_features_fake = batch_features_fake.detach()

        dre_net.train()
        DR_real_1 = dre_net(batch_features_real, net_y2h(batch_labels_debug))
        DR_fake_1 = dre_net(batch_features_fake, net_y2h(batch_labels_debug))
        dre_net.eval()
        DR_real_2 = dre_net(batch_features_real, net_y2h(batch_labels_debug))
        DR_fake_2 = dre_net(batch_features_fake, net_y2h(batch_labels_debug))

        print("\r Train mode: real {}, fake {}.".format(DR_real_1.mean().item(), DR_fake_1.mean().item()))
        print("\r Eval mode: real {}, fake {}.".format(DR_real_2.mean().item(), DR_fake_2.mean().item()))


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
    return dre_net, avg_train_loss
#end for def
