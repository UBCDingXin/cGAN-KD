
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
lambda_regression = args.dre_presae_lambda_regression
resume_epoch = args.dre_presae_resume_epoch
weigth_decay = args.dre_presae_weight_decay
batch_size = args.dre_presae_batch_size_train


## horizontal flipping
def hflip_images(batch_images):
    ''' for numpy arrays '''
    uniform_threshold = np.random.uniform(0,1,len(batch_images))
    indx_gt = np.where(uniform_threshold>0.5)[0]
    batch_images[indx_gt] = np.flip(batch_images[indx_gt], axis=3)
    return batch_images

## normalize images
def normalize_images(batch_images):
    batch_images = batch_images/255.0
    batch_images = (batch_images - 0.5)/0.5
    return batch_images


# decay learning rate every args.dre_lr_decay_epochs epochs
def adjust_learning_rate(epoch, epochs, optimizer, base_lr, lr_decay_epochs, lr_decay_factor):
    lr = base_lr #1e-4

    for i in range(epochs//lr_decay_epochs):
        if epoch >= (i+1)*lr_decay_epochs:
            lr *= lr_decay_factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_sparseAE(unique_labels, train_images, train_labels, net_encoder, net_decoder, net_predict, save_sae_images_folder, path_to_ckpt=None):
    '''
    train_images: unnormalized
    train_labels: normalized to [0,1]
    '''
    assert train_images.max()>1.0 and train_images.max()<=255.0
    assert train_labels.max()<=1.0 and train_labels.min()>=0

    # nets
    net_encoder = net_encoder.cuda()
    net_decoder = net_decoder.cuda()
    net_predict = net_predict.cuda()

    # define optimizer
    params = list(net_encoder.parameters()) + list(net_decoder.parameters()) + list(net_predict.parameters())
    optimizer = torch.optim.SGD(params, lr = base_lr, momentum= 0.9, weight_decay=weigth_decay)
    # optimizer = torch.optim.Adam(params, lr = base_lr, betas=(0, 0.999), weight_decay=weigth_decay)

    # criterion
    criterion = nn.MSELoss()

    if path_to_ckpt is not None and resume_epoch>0:
        print("Loading ckpt to resume training sparseAE >>>")
        ckpt_fullpath = path_to_ckpt + "/PreSAEForDRE_checkpoint_intrain/PreSAEForDRE_checkpoint_epoch_{}_sparsity_{:.3f}_regre_{:.3f}.pth".format(resume_epoch, lambda_sparsity, lambda_regression)
        checkpoint = torch.load(ckpt_fullpath)
        net_encoder.load_state_dict(checkpoint['net_encoder_state_dict'])
        net_decoder.load_state_dict(checkpoint['net_decoder_state_dict'])
        net_predict.load_state_dict(checkpoint['net_predict_state_dict'])
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

        for batch_idx in range(len(train_labels)//batch_size):

            net_encoder.train()
            net_decoder.train()
            net_predict.train()

            #################################################
            ''' generate target labels '''
            batch_target_labels = np.random.choice(unique_labels, size=batch_size, replace=True)
            batch_unique_labels, batch_unique_label_counts = np.unique(batch_target_labels, return_counts=True)

            batch_real_indx = []
            for j in range(len(batch_unique_labels)):
                indx_j = np.where(train_labels==batch_unique_labels[j])[0]
                indx_j = np.random.choice(indx_j, size=batch_unique_label_counts[j])
                batch_real_indx.append(indx_j)
            batch_real_indx = np.concatenate(batch_real_indx)
            batch_real_indx = batch_real_indx.reshape(-1)


            #################################################
            ## get some real images for training
            batch_real_images = train_images[batch_real_indx]
            batch_real_images = hflip_images(batch_real_images) ## randomly flip real images
            batch_real_images = normalize_images(batch_real_images) ## normalize real images
            batch_real_images = torch.from_numpy(batch_real_images).type(torch.float).cuda()
            assert batch_real_images.max().item()<=1.0
            batch_real_labels = train_labels[batch_real_indx]
            batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).cuda()


            #################################################
            ## forward pass
            batch_features = net_encoder(batch_real_images)
            batch_recons_images = net_decoder(batch_features)
            batch_pred_labels = net_predict(batch_features)

            '''
            based on https://debuggercafe.com/sparse-autoencoders-using-l1-regularization-with-pytorch/
            '''
            loss1 = criterion(batch_recons_images, batch_real_images) + lambda_sparsity * batch_features.mean() 
            loss2 = criterion(batch_pred_labels.view(-1), batch_real_labels.view(-1))
            loss = loss1 + loss2*lambda_regression

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            train_loss1 += loss1.cpu().item()
            train_loss2 += loss2.cpu().item()

            gen_iterations += 1

            if gen_iterations % 100 == 0:
                net_encoder.eval()
                net_decoder.eval()
                n_row=min(10, int(np.sqrt(batch_size)))
                with torch.no_grad():
                    batch_recons_images = net_decoder(net_encoder(batch_real_images[0:n_row**2]))
                    batch_recons_images = batch_recons_images.detach().cpu()
                save_image(batch_recons_images.data, save_sae_images_folder + '/{}.png'.format(gen_iterations), nrow=n_row, normalize=True)

            if gen_iterations % 20 == 0:
                print("\r SparseAE+sparsity{:.3f}+regre{:.3f}: [step {}] [epoch {}/{}] [train loss {:.4f}={:.4f}+{:.4f}] [Time {:.4f}]".format(lambda_sparsity, lambda_regression, gen_iterations, epoch+1, epochs, train_loss/(batch_idx+1), train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), timeit.default_timer()-start_time) )
        # end for batch_idx

        if path_to_ckpt is not None and (epoch+1) % 50 == 0:
            save_file = path_to_ckpt + "/PreSAEForDRE_checkpoint_intrain/PreSAEForDRE_checkpoint_epoch_{}_sparsity_{:.3f}_regre_{:.3f}.pth".format(epoch+1, lambda_sparsity, lambda_regression)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'gen_iterations': gen_iterations,
                    'net_encoder_state_dict': net_encoder.state_dict(),
                    'net_decoder_state_dict': net_decoder.state_dict(),
                    'net_predict_state_dict': net_predict.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for epoch

    net_encoder = net_encoder.cpu()
    net_decoder = net_decoder.cpu()
    net_predict = net_predict.cpu()

    return net_encoder, net_decoder, net_predict
