
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit

from utils import SimpleProgressBar
from opts import gen_synth_data_opts

''' Settings '''
args = gen_synth_data_opts()

# some parameters in the opts
epochs = args.dre_presae_epochs
base_lr = args.dre_presae_lr_base
lr_decay_epochs = args.dre_presae_lr_decay_freq
lr_decay_factor = args.dre_presae_lr_decay_factor
lambda_sparsity = args.dre_presae_lambda_sparsity
resume_epoch = args.dre_presae_resume_epoch

# horizontal flip
def hflip_images(batch_images):
    uniform_threshold = np.random.uniform(0,1,len(batch_images))
    indx_gt = np.where(uniform_threshold>0.5)[0]
    batch_images[indx_gt] = torch.flip(batch_images[indx_gt], dims=[3])
    return batch_images

# decay learning rate every args.dre_lr_decay_epochs epochs
def adjust_learning_rate(epoch, epochs, optimizer, base_lr, lr_decay_epochs, lr_decay_factor):
    lr = base_lr #1e-4

    for i in range(epochs//lr_decay_epochs):
        if epoch >= (i+1)*lr_decay_epochs:
            lr *= lr_decay_factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_sparseAE(trainloader, net_encoder, net_decoder, save_sae_images_folder, path_to_ckpt=None):

    # nets
    net_encoder = net_encoder.cuda()
    net_decoder = net_decoder.cuda()

    # define optimizer
    params = list(net_encoder.parameters()) + list(net_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr = base_lr, betas=(0.5, 0.999), weight_decay=1e-4)

    # criterion
    criterion = nn.MSELoss()

    if path_to_ckpt is not None and resume_epoch>0:
        print("Loading ckpt to resume training sparseAE >>>")
        ckpt_fullpath = path_to_ckpt + "/sparseAE_checkpoint_intrain/sparseAE_checkpoint_epoch_{}_lambda_{}.pth".format(resume_epoch, lambda_sparsity)
        checkpoint = torch.load(ckpt_fullpath)
        net_encoder.load_state_dict(checkpoint['net_encoder_state_dict'])
        net_decoder.load_state_dict(checkpoint['net_decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        gen_iterations = checkpoint['gen_iterations']
    else:
        gen_iterations = 0

    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):

        adjust_learning_rate(epoch, epochs, optimizer, base_lr, lr_decay_epochs, lr_decay_factor)

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0

        for batch_idx, (batch_real_images, batch_real_labels) in enumerate(trainloader):

            net_encoder.train()
            net_decoder.train()

            batch_size_curr = batch_real_images.shape[0]

            ## random horizontal flipping
            batch_real_images = hflip_images(batch_real_images)

            batch_real_images = batch_real_images.type(torch.float).cuda()
            batch_real_labels = batch_real_labels.type(torch.float).cuda()

            batch_features = net_encoder(batch_real_images)
            batch_recons_images, batch_pred_labels = net_decoder(batch_features)

            '''
            based on https://debuggercafe.com/sparse-autoencoders-using-l1-regularization-with-pytorch/
            '''
            loss1 = criterion(batch_recons_images, batch_real_images) + lambda_sparsity * batch_features.mean() 
            loss2 = criterion(batch_pred_labels.view(-1), batch_real_labels.view(-1))
            loss = loss1 + loss2

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            train_loss1 += loss1.cpu().item()
            train_loss2 += loss2.cpu().item()

            gen_iterations += 1

            if gen_iterations % 100 == 0:
                n_row=min(10, int(np.sqrt(batch_size_curr)))
                with torch.no_grad():
                    batch_recons_images, _ = net_decoder(net_encoder(batch_real_images[0:n_row**2]))
                    batch_recons_images = batch_recons_images.detach().cpu()
                save_image(batch_recons_images.data, save_sae_images_folder + '/{}.png'.format(gen_iterations), nrow=n_row, normalize=True)

            if gen_iterations % 20 == 0:
                print("SparseAE+lambda{}: [step {}] [epoch {}/{}] [train loss {:.4f}={:.4f}+{:.4f}] [Time {:.4f}]".format(lambda_sparsity, gen_iterations, epoch+1, epochs, train_loss/(batch_idx+1), train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), timeit.default_timer()-start_time) )
        # end for batch_idx

        if path_to_ckpt is not None and (epoch+1) % 50 == 0:
            save_file = path_to_ckpt + "/sparseAE_checkpoint_intrain/sparseAE_checkpoint_epoch_{}_lambda_{}.pth".format(epoch+1, lambda_sparsity)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'gen_iterations': gen_iterations,
                    'net_encoder_state_dict': net_encoder.state_dict(),
                    'net_decoder_state_dict': net_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for epoch

    return net_encoder, net_decoder
