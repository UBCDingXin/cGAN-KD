''' For GAN training '''

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


## function for computing gradient penalty
def calc_gradient_penalty_wgan(netD, real_data, fake_data, wgan_lambda=10):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size(0), args.num_channels*args.img_height*args.img_height)
    alpha = alpha.view(real_data.size(0), args.num_channels, args.img_height, args.img_height)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() ,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * wgan_lambda
    return gradient_penalty



def train_gan(gan_name, trainloader, which_class, netG, netD, save_GANimages_folder, path_to_ckpt = None):

    # some parameters in the opts
    resume_epoch = args.gan_resume_epoch
    loss_type = args.gan_loss
    gan_epochs = args.gan_epochs
    dim_gan = args.gan_dim_g
    lr_g = args.gan_lr_g
    lr_d = args.gan_lr_d
    critic_iters = args.gan_d_niters

    # define optimizer
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.0, 0.999)) #SAGAN setting
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.0, 0.999))

    netG = netG.cuda()
    netD = netD.cuda()


    if path_to_ckpt is not None and resume_epoch>0:
        save_file = path_to_ckpt + "/{}_{}_checkpoint_epoch_{}_class_{}.pth".format(gan_name, loss_type, resume_epoch, which_class)
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

    n_row=10
    z_fixed = torch.randn(n_row**2, dim_gan, dtype=torch.float).cuda()

    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, gan_epochs):

        data_iter = iter(trainloader)
        batch_idx = 0

        while (batch_idx < len(trainloader)):
            ############################
            # (1) Update D network
            ###########################
            # for p in netD.parameters():  # reset requires_grad
            #     p.requires_grad = True  # they are set to False below in netG update

            netD.train()

            for batch_idx_D in range(critic_iters):

                if batch_idx == len(trainloader):
                    break

                (batch_train_images, batch_train_labels) = data_iter.next()
                batch_idx += 1

                batch_size_curr = batch_train_images.shape[0]
                batch_train_images = batch_train_images.type(torch.float).cuda()
                assert batch_train_labels[0].item() == which_class

                # Adversarial ground truths. for vanilla loss only
                gt_real = torch.ones(batch_size_curr, 1).cuda()
                gt_fake = torch.zeros(batch_size_curr,1).cuda()

                d_out_real = netD(batch_train_images)
                z = torch.randn(batch_size_curr, dim_gan, dtype=torch.float).cuda()
                gen_imgs = netG(z)
                d_out_fake = netD(gen_imgs.detach())
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
                    gradient_penalty = calc_gradient_penalty_cwgan(netD, batch_train_images.data, gen_imgs.data)
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
            # for p in netD.parameters():
            #     p.requires_grad = False  # to avoid computation

            netG.train()

            z = torch.randn(batch_size_curr, dim_gan, dtype=torch.float).cuda()
            gen_imgs = netG(z)
            g_out_fake = netD(gen_imgs)

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

            if gen_iterations%20 == 0:
                if loss_type in ["vanilla", "hinge"]:
                    print ("%s+%s: [Class %d] [Iter %d] [Epoch %d/%d] [D loss: %.4f] [G loss: %.4f] [D out real:%.4f] [D out fake:%.4f] [Time: %.4f]" % (gan_name, loss_type, which_class, gen_iterations, epoch + 1, gan_epochs, d_loss.item(), g_loss.item(), d_out_real.mean().item(), d_out_fake.mean().item(), timeit.default_timer()-start_time))
                elif loss_type == "wasserstein":
                    print ("%s+%s: [Class %d] [Iter %d] [Epoch %d/%d] [D loss: %.4f] [G loss: %.4f] [D out real:%.4f] [D out fake:%.4f] [W Dist: %.4f] [Time: %.4f]" % (gan_name, loss_type, which_class, gen_iterations, epoch + 1, gan_epochs, d_loss.item(), g_loss.item(), d_out_real.mean().item(), d_out_fake.mean().item(), Wasserstein_D, timeit.default_timer()-start_time))

            if gen_iterations % 100 == 0:
                netG.eval()
                with torch.no_grad():
                    gen_imgs = netG(z_fixed)
                    gen_imgs = gen_imgs.detach()

                save_image(gen_imgs.data, save_GANimages_folder + '/class_{}_{}.png'.format(which_class, gen_iterations), nrow=n_row, normalize=True)


        if path_to_ckpt is not None and (epoch+1) % 200 == 0:
            save_file = path_to_ckpt + "/{}_{}_checkpoint_epoch_{}_class_{}.pth".format(gan_name, loss_type, epoch+1, which_class)
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


# sampling from the trained GAN
def SampGAN(netG, nfake = 10000, batch_size = 500, to_numpy=True):
    nfake=int(nfake); batch_size = int(batch_size);

    if batch_size > nfake:
        batch_size = nfake

    # some parameters in opts
    dim_gan = args.gan_dim_g
    num_channels = args.num_channels
    img_height = args.img_height
    img_width = args.img_width

    raw_fake_images = []

    netG=netG.cuda()
    netG.eval()
    with torch.no_grad():
        pb = SimpleProgressBar()
        tmp = 0
        while tmp < nfake:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).cuda()
            batch_fake_images = netG(z)
            raw_fake_images.append(batch_fake_images.cpu())
            tmp += batch_size
            pb.update(min(float(tmp)/nfake, 1)*100)
    #remove extra entries
    raw_fake_images = torch.cat(raw_fake_images, dim=0)
    if to_numpy:
        raw_fake_images = raw_fake_images.numpy()

    return raw_fake_images[0:nfake]
