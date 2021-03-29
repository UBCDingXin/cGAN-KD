''' For cGAN training '''

import torch
import torch.nn as nn
from torch import autograd
from torchvision.utils import save_image
import numpy as np
import os
import timeit

from utils import SimpleProgressBar
from opts import gen_synth_data_opts


''' Settings '''
args = gen_synth_data_opts()



## function for computing gradient penalty for wgan
def calc_gradient_penalty_cwgan(netD, real_data, fake_data, labels, wgan_lambda=10):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size(0), args.num_channels*args.img_height*args.img_height)
    alpha = alpha.view(real_data.size(0), args.num_channels, args.img_height, args.img_height)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()

    interpolates = autograd.Variable(interpolates, requires_grad=True)
    # labels = autograd.Variable(labels_data, requires_grad=True)

    disc_interpolates = netD(interpolates, labels)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() ,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * wgan_lambda
    return gradient_penalty


def train_cgan(gan_name, trainloader, netG, netD, save_GANimages_folder, path_to_ckpt = None):

    # some parameters in the opts
    resume_epoch = args.gan_resume_epoch
    loss_type = args.gan_loss
    gan_epochs = args.gan_epochs
    dim_gan = args.gan_dim_g
    num_classes = args.num_classes
    lr_g = args.gan_lr_g
    lr_d = args.gan_lr_d
    critic_iters = args.gan_d_niters

    # define optimizer
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.0, 0.999)) #SAGAN setting
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.0, 0.999))

    netG = netG.cuda()
    netD = netD.cuda()

    if path_to_ckpt is not None and resume_epoch>0:
        print("\r Resume training >>>")
        save_file = path_to_ckpt + "/{}_{}_checkpoint_epoch_{}.pth".format(gan_name, loss_type, resume_epoch)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        gen_iterations = checkpoint['gen_iterations']
    else:
        gen_iterations = 0
    #end if


    # fixed z and label for outputing fake images during training
    n_row=10
    z_fixed = torch.randn(n_row**2, dim_gan, dtype=torch.float).cuda()
    class_fixed = list(np.linspace(0, num_classes-1, 10).astype(int))
    labels_fixed = []
    for i in range(n_row):
        labels_fixed.extend(list(class_fixed[i]*np.ones(n_row)))
    labels_fixed = np.array(labels_fixed)
    labels_fixed = torch.from_numpy(labels_fixed).type(torch.long).cuda()


    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, gan_epochs):

        data_iter = iter(trainloader)
        batch_idx = 0

        while (batch_idx < len(trainloader)):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # netD.train()

            for batch_idx_D in range(critic_iters):

                if batch_idx == len(trainloader):
                    break

                (batch_train_images, batch_train_labels) = data_iter.next()
                batch_idx += 1

                batch_size_curr = batch_train_images.shape[0]
                batch_train_images = batch_train_images.type(torch.float).cuda()
                batch_train_labels = batch_train_labels.type(torch.long).cuda()

                # Adversarial ground truths. for vanilla loss only
                gt_real = torch.ones(batch_size_curr, 1).cuda()
                gt_fake = torch.zeros(batch_size_curr,1).cuda()

                d_out_real = netD(batch_train_images, batch_train_labels)
                z = torch.randn(batch_size_curr, dim_gan, dtype=torch.float).cuda()
                gen_imgs = netG(z, batch_train_labels)
                d_out_fake = netD(gen_imgs.detach(), batch_train_labels)
                if loss_type == "vanilla":
                    d_out_real = torch.nn.Sigmoid()(d_out_real)
                    d_out_fake = torch.nn.Sigmoid()(d_out_fake)
                    d_loss_real = nn.BCELoss()(d_out_real, gt_real)
                    d_loss_fake = nn.BCELoss()(d_out_fake, gt_fake)
                    d_loss = d_loss_real + d_loss_fake
                elif loss_type == "hinge":
                    d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                    d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
                    d_loss = d_loss_real + d_loss_fake
                elif loss_type == "wasserstein":
                    gradient_penalty = calc_gradient_penalty_cwgan(netD, batch_train_images.data, gen_imgs.data, batch_train_labels)
                    d_loss_real = d_out_real.mean()
                    d_loss_fake = d_out_fake.mean()
                    d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    Wasserstein_D = d_loss_real.cpu().item() - d_loss_fake.cpu().item()

                # Backward
                optimizerD.zero_grad()
                d_loss.backward()
                optimizerD.step()

            #end for batch_idx_D

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation

            # netG.train()

            z = torch.randn(batch_size_curr, dim_gan, dtype=torch.float).cuda()
            gen_imgs = netG(z, batch_train_labels)
            g_out_fake = netD(gen_imgs, batch_train_labels)

            if loss_type == "vanilla":
                g_out_fake = torch.nn.Sigmoid()(g_out_fake)
                g_loss = nn.BCELoss()(g_out_fake, gt_real)
            elif loss_type in ["hinge", "wasserstein"]:
                g_loss = - g_out_fake.mean()

            # Backward
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            gen_iterations += 1

            if gen_iterations % 100 == 0:
                # netG.eval()
                with torch.no_grad():
                    gen_imgs = netG(z_fixed, labels_fixed)
                    gen_imgs = gen_imgs.detach()
                os.makedirs(save_GANimages_folder, exist_ok=True)
                save_image(gen_imgs.data, save_GANimages_folder +'/%d.png' % gen_iterations, nrow=n_row, normalize=True)

            if gen_iterations%20 == 0:
                if loss_type in ["vanilla", "hinge"]:
                    print ("%s+%s: [Iter %d] [Epoch %d/%d] [D loss: %.4f] [G loss: %.4f] [D out real:%.4f] [D out fake:%.4f] [Time: %.4f]" % (gan_name, loss_type, gen_iterations, epoch + 1, gan_epochs, d_loss.item(), g_loss.item(), d_out_real.mean().item(), d_out_fake.mean().item(), timeit.default_timer()-start_time))
                elif loss_type == "wasserstein":
                    print ("%s+%s: [Iter %d] [Epoch %d/%d] [D loss: %.4f] [G loss: %.4f] [D out real:%.4f] [D out fake:%.4f] [W Dist: %.4f] [Time: %.4f]" % (gan_name, loss_type, gen_iterations, epoch + 1, gan_epochs, d_loss.item(), g_loss.item(), d_out_real.mean().item(), d_out_fake.mean().item(), Wasserstein_D, timeit.default_timer()-start_time))


        if path_to_ckpt is not None and ((epoch+1) % 500 == 0 or (epoch+1) == gan_epochs):
            save_file = path_to_ckpt + "/{}_{}_checkpoint_epoch_{}.pth".format(gan_name, loss_type, epoch+1)
            torch.save({
                    'gen_iterations': gen_iterations,
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)

    #end for epoch
    return netG, netD


def SampcGAN_given_label(netG, given_label, nfake=10000, batch_size = 500, to_numpy=True):

    # some parameters in opts
    dim_gan = args.gan_dim_g
    num_channels = args.num_channels
    img_height = args.img_height
    img_width = args.img_width

    if batch_size>nfake:
        batch_size = nfake

    raw_fake_images = []
    raw_fake_labels = []

    netG=netG.cuda()
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < nfake:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).cuda()
            labels = (given_label*torch.ones(batch_size)).type(torch.long).cuda()
            batch_fake_images = netG(z, labels)
            raw_fake_images.append(batch_fake_images.cpu())
            raw_fake_labels.append(labels.cpu().view(-1))
            tmp += batch_size
    raw_fake_images = torch.cat(raw_fake_images, dim=0)
    raw_fake_labels = torch.cat(raw_fake_labels)

    if to_numpy:
        raw_fake_images = raw_fake_images.numpy()
        raw_fake_labels = raw_fake_labels.numpy()

    return raw_fake_images[0:nfake], raw_fake_labels[0:nfake]




# def SampcGAN_given_label(netG, given_label, nfake=10000, batch_size = 500):
#
#     # some parameters in opts
#     dim_gan = args.gan_dim_g
#     num_channels = args.num_channels
#     img_height = args.img_height
#     img_width = args.img_width
#
#     if batch_size>nfake:
#         batch_size = nfake
#     raw_fake_images = np.zeros((nfake+batch_size, num_channels, img_height, img_width))
#     raw_fake_labels = np.zeros(nfake+batch_size)
#     netG=netG.cuda()
#     netG.eval()
#     with torch.no_grad():
#         tmp = 0
#         while tmp < nfake:
#             z = torch.randn(batch_size, dim_gan, dtype=torch.float).cuda()
#             labels = torch.from_numpy(given_label*np.ones(batch_size)).type(torch.long).cuda()
#             raw_fake_labels[tmp:(tmp+batch_size)] = labels.cpu().numpy()
#             batch_fake_images = netG(z, labels)
#             raw_fake_images[tmp:(tmp+batch_size)] = batch_fake_images.detach().cpu().numpy()
#             tmp += batch_size
#     #remove extra entries
#     raw_fake_images = raw_fake_images[0:nfake]
#     raw_fake_labels = raw_fake_labels[0:nfake]
#
#     return raw_fake_images, raw_fake_labels
