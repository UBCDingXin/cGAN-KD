print("\n===================================================================================================")

import argparse
import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
import h5py
import os
import random
from tqdm import tqdm, trange
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import timeit
from PIL import Image

### import my stuffs ###
from opts import gen_synth_data_opts
from utils import *
from models import *
from train_ccgan import train_ccgan, SampCcGAN_given_labels
from train_net_for_label_embed import train_net_embed, train_net_y2h
from train_sparseAE import train_sparseAE
from train_cdre import train_cdre
from eval_metrics import cal_FID, cal_labelscore


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)

if args.subsampling:
    subsampling_method = "cDRE-F-SP+RS_{}_{}".format(args.dre_threshold_type, args.dre_kappa)
else:
    subsampling_method = "None"

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)


#-------------------------------
# output folders
eval_models_folder = os.path.join(args.root_path, 'output/eval_models')
assert os.path.exists(eval_models_folder)

output_directory = os.path.join(args.root_path, 'output/NTrainPerLabel_{}'.format(args.max_num_img_per_label))
os.makedirs(output_directory, exist_ok=True)
save_models_folder = os.path.join(output_directory, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = os.path.join(output_directory, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)
save_traincurves_folder = os.path.join(output_directory, 'training_curves')
os.makedirs(save_traincurves_folder, exist_ok=True)
fake_data_folder = os.path.join(output_directory, 'fake_data')
os.makedirs(fake_data_folder, exist_ok=True)


#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
hf = h5py.File(os.path.join(args.data_path, 'RC-49_64x64_{}.h5'.format(args.max_num_img_per_label)), 'r')
images_odd = hf['images_odd'][:]
labels_odd = hf['labels_odd'][:]
types_odd = hf['types_odd'][:]
images_even = hf['images_even'][:]
labels_even = hf['labels_even'][:]
types_even = hf['types_even'][:]
indx_odd_train = hf['indx_odd_train'][:]
hf.close()
assert labels_odd.max()<=args.max_label and labels_even.max()<=args.max_label
assert len(images_odd) == len(labels_odd) and len(images_odd) == len(types_odd)
assert len(images_even) == len(labels_even) and len(images_even) == len(types_even)
## concatenate
images_train = images_odd[indx_odd_train]
labels_train_raw = labels_odd[indx_odd_train]
types_train = types_odd[indx_odd_train]
images_test = np.concatenate((images_odd, images_even), axis=0) ##all images are used for the computing Intra-FID, etc.
labels_test_raw = np.concatenate((labels_odd, labels_even), axis=0)
types_test = np.concatenate((types_odd, types_even), axis=0)
del images_odd, images_even, labels_odd, labels_even, types_odd, types_even; gc.collect()


## some example real images
nrow = 10
ncol=10
unique_labels_show = np.array(sorted(list(set(labels_train_raw))))
indx_show = np.arange(0, len(unique_labels_show), len(unique_labels_show)//9)
unique_labels_show = unique_labels_show[indx_show]
nrow = len(unique_labels_show)
sel_labels_indx = []
for i in range(nrow):
    curr_label = unique_labels_show[i]
    indx_curr_label = np.where(labels_train_raw==curr_label)[0]
    np.random.shuffle(indx_curr_label)
    indx_curr_label = indx_curr_label[0:ncol]
    sel_labels_indx.extend(list(indx_curr_label))
sel_labels_indx = np.array(sel_labels_indx)
images_show = images_train[sel_labels_indx]
# print(images_show.mean())
images_show = (images_show/255.0-0.5)/0.5
images_show = torch.from_numpy(images_show)
save_image(images_show.data, save_images_folder +'/real_images_grid_{}x{}.png'.format(nrow, ncol), nrow=ncol, normalize=True)


## normalize to [0,1]
print("\n Range of unnormalized train labels: ({},{})".format(np.min(labels_train_raw), np.max(labels_train_raw)))
print("\n Range of unnormalized test labels: ({},{})".format(np.min(labels_test_raw), np.max(labels_test_raw)))
labels_train = labels_train_raw / args.max_label
labels_test = labels_test_raw / args.max_label

# unique normalized training labels
unique_labels_train_norm = np.sort(np.array(list(set(labels_train))))
unique_labels_test_norm = np.sort(np.array(list(set(labels_test))))


## set sigma and kappa/nu in CcGAN
if args.gan_kernel_sigma<0:
    std_label = np.std(labels_train)
    args.gan_kernel_sigma = 1.06*std_label*(len(labels_train))**(-1/5)

    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\n The std of {} labels is {} so the kernel sigma is {}".format(len(labels_train), std_label, args.gan_kernel_sigma))
##end if

if args.gan_kappa<0:
    n_unique = len(unique_labels_train_norm)

    diff_list = []
    for i in range(1,n_unique):
        diff_list.append(unique_labels_train_norm[i] - unique_labels_train_norm[i-1])
    kappa_base = np.abs(args.gan_kappa)*np.max(np.array(diff_list))

    if args.gan_threshold_type=="hard":
        args.gan_kappa = kappa_base
    else:
        args.gan_kappa = 1/kappa_base**2
## end if


#######################################################################################
'''               Pre-trained CNN and GAN for label embedding                       '''
#######################################################################################
net_embed_x2y_filename_ckpt = save_models_folder + '/ckpt_embed_{}_epoch_{}_seed_{}.pth'.format(args.gan_embed_x2y_net_name, args.gan_embed_x2y_epoch, args.seed)
print(net_embed_x2y_filename_ckpt)
net_embed_y2h_filename_ckpt = save_models_folder + '/ckpt_embed_y2h_epoch_{}_seed_{}.pth'.format(args.gan_embed_y2h_epoch, args.seed)
print(net_embed_y2h_filename_ckpt)

trainset_embed_x2y = IMGs_dataset(images_train, labels_train, normalize=True)
trainloader_embed_x2y = torch.utils.data.DataLoader(trainset_embed_x2y, batch_size=args.gan_embed_x2y_batch_size, shuffle=True, num_workers=args.num_workers)
testset_embed_x2y = IMGs_dataset(images_test, labels_test, normalize=True)
testloader_embed_x2y = torch.utils.data.DataLoader(testset_embed_x2y, batch_size=128, shuffle=False, num_workers=args.num_workers)

# if args.gan_embed_x2y_net_name == "DenseNet121":
#     net_embed_x2y = DenseNet121_embed(dim_embed=args.gan_dim_embed)
# elif args.gan_embed_x2y_net_name == "DenseNet169":
#     net_embed_x2y = DenseNet169_embed(dim_embed=args.gan_dim_embed)
# elif args.gan_embed_x2y_net_name == "DenseNet169":
#     net_embed_x2y = DenseNet201_embed(dim_embed=args.gan_dim_embed)
# elif args.gan_embed_x2y_net_name == "DenseNet169":
#     net_embed_x2y = DenseNet161_embed(dim_embed=args.gan_dim_embed)
# else:
#     raise Exception("Wrong embedding net name!")

if args.gan_embed_x2y_net_name == "ResNet18":
    net_embed_x2y = ResNet18_embed(dim_embed=args.gan_dim_embed)
elif args.gan_embed_x2y_net_name == "ResNet34":
    net_embed_x2y = ResNet34_embed(dim_embed=args.gan_dim_embed)
elif args.gan_embed_x2y_net_name == "ResNet50":
    net_embed_x2y = ResNet50_embed(dim_embed=args.gan_dim_embed)
else:
    raise Exception("Wrong embedding net name!")

net_embed_x2y = net_embed_x2y.cuda()
net_embed_x2y = nn.DataParallel(net_embed_x2y)

net_embed_y2h = model_y2h(dim_embed=args.gan_dim_embed)
net_embed_y2h = net_embed_y2h.cuda()
net_embed_y2h = nn.DataParallel(net_embed_y2h)


## (1). Train net_embed first: x2h+h2y
if not os.path.isfile(net_embed_x2y_filename_ckpt):
    print("\n Start training CNN for label embedding >>>")

    # lr decay epochs
    net_embed_x2y_lr_decay_epochs = (args.gan_embed_x2y_lr_decay_epochs).split("_")
    net_embed_x2y_lr_decay_epochs = [int(epoch) for epoch in net_embed_x2y_lr_decay_epochs]

    # ckpts in training
    ckpts_in_train_net_embed_x2y = os.path.join(save_models_folder, 'ckpts_in_train_embed_x2y_{}'.format(args.gan_embed_x2y_net_name))
    os.makedirs(ckpts_in_train_net_embed_x2y, exist_ok=True)

    # training function
    net_embed_x2y = train_net_embed(net=net_embed_x2y, net_name=args.gan_embed_x2y_net_name, trainloader=trainloader_embed_x2y, testloader=testloader_embed_x2y, epochs=args.gan_embed_x2y_epoch, resume_epoch = args.gan_embed_x2y_resume_epoch, lr_base=args.gan_embed_x2y_lr_base, lr_decay_factor=args.gan_embed_x2y_lr_decay_factor, lr_decay_epochs=net_embed_x2y_lr_decay_epochs, weight_decay=1e-4, path_to_ckpt = ckpts_in_train_net_embed_x2y, max_label=args.max_label)

    # save model
    torch.save({
    'net_state_dict': net_embed_x2y.state_dict(),
    }, net_embed_x2y_filename_ckpt)
else:
    print("\n net_embed ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_embed_x2y_filename_ckpt)
    net_embed_x2y.load_state_dict(checkpoint['net_state_dict'])
#end not os.path.isfile

## (2). Train y2h
#train a net which maps a label back to the embedding space
if not os.path.isfile(net_embed_y2h_filename_ckpt):
    print("\n Start training net_embed_y2h >>>")

    # lr decay epochs
    net_embed_y2h_lr_decay_epochs = (args.gan_embed_y2h_lr_decay_epochs).split("_")
    net_embed_y2h_lr_decay_epochs = [int(epoch) for epoch in net_embed_y2h_lr_decay_epochs]

    # training function
    net_embed_y2h = train_net_y2h(unique_labels_norm=unique_labels_train_norm, net_y2h=net_embed_y2h, net_embed=net_embed_x2y, epochs=args.gan_embed_y2h_epoch, lr_base=args.gan_embed_y2h_lr_base, lr_decay_factor=args.gan_embed_y2h_lr_decay_factor, lr_decay_epochs=net_embed_y2h_lr_decay_epochs, weight_decay=1e-4, batch_size=args.gan_embed_y2h_batch_size)

    # save model
    torch.save({
    'net_state_dict': net_embed_y2h.state_dict(),
    }, net_embed_y2h_filename_ckpt)
else:
    print("\n net_embed_y2h ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_embed_y2h_filename_ckpt)
    net_embed_y2h.load_state_dict(checkpoint['net_state_dict'])
#end not os.path.isfile

##some simple test after the embedding nets training
indx_tmp = np.arange(len(unique_labels_train_norm))
np.random.shuffle(indx_tmp)
indx_tmp = indx_tmp[:10]
labels_tmp = unique_labels_train_norm[indx_tmp].reshape(-1,1)
labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).cuda()
epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1,1).type(torch.float).cuda()
labels_tmp = torch.clamp(labels_tmp+epsilons_tmp, 0.0, 1.0)
net_embed_x2y.eval()
net_embed_h2y = net_embed_x2y.module.h2y
net_embed_y2h.eval()
with torch.no_grad():
    labels_rec_tmp = net_embed_h2y(net_embed_y2h(labels_tmp)).cpu().numpy().reshape(-1,1)
results = np.concatenate((labels_tmp.cpu().numpy(), labels_rec_tmp), axis=1)
print("\n labels vs reconstructed labels")
print(results)







#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
print("\n Start CcGAN training: {}, Sigma is {}, Kappa is {}".format(args.gan_threshold_type, args.gan_kernel_sigma, args.gan_kappa))

path_to_ckpt_ccgan = os.path.join(save_models_folder, 'ckpt_CcGAN_loss_{}_niters_{}_seed_{}_{}_{}_{}.pth'.format(args.gan_loss_type, args.gan_niters, args.seed, args.gan_threshold_type, args.gan_kernel_sigma, args.gan_kappa))
print(path_to_ckpt_ccgan)

start = timeit.default_timer()
if not os.path.isfile(path_to_ckpt_ccgan):
    ## images generated during training
    images_in_train_ccgan = os.path.join(save_images_folder, 'images_in_train_ccgan')
    os.makedirs(images_in_train_ccgan, exist_ok=True)

    # ckpts in training
    ckpts_in_train_ccgan = os.path.join(save_models_folder, 'ckpts_in_train_ccgan')
    os.makedirs(ckpts_in_train_ccgan, exist_ok=True)

    # init models
    netG = CcGAN_Generator(z_dim=args.gan_dim_g, gene_ch=args.gan_gene_ch, dim_embed=args.gan_dim_embed).cuda()
    netD = CcGAN_Discriminator(disc_ch=args.gan_disc_ch, dim_embed=args.gan_dim_embed).cuda()
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

    # training function
    netG, netD = train_ccgan(kernel_sigma=args.gan_kernel_sigma, kappa=args.gan_kappa, train_images=images_train, train_labels=labels_train, netG=netG, netD=netD, net_y2h = net_embed_y2h, save_images_folder = images_in_train_ccgan, path_to_ckpt = ckpts_in_train_ccgan, clip_label=False)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, path_to_ckpt_ccgan)

else:
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(path_to_ckpt_ccgan)
    netG = CcGAN_Generator(z_dim=args.gan_dim_g, gene_ch=args.gan_gene_ch, dim_embed=args.gan_dim_embed).cuda()
    netG = nn.DataParallel(netG)
    netG.load_state_dict(checkpoint['netG_state_dict'])
## end if
stop = timeit.default_timer()
print("CcGAN training finished; Time elapses: {}s".format(stop - start))

## functions for sampling
def fn_sampleGAN_given_labels(labels, batch_size, to_numpy=True, verbose=True):
    fake_images, fake_labels = SampCcGAN_given_labels(netG=netG, net_y2h=net_embed_y2h, labels=labels, batch_size = batch_size, to_numpy=to_numpy, verbose=verbose)
    return fake_images, fake_labels






#######################################################################################
'''                                    cDRE training                                 '''
#######################################################################################

if args.subsampling:
    ##############################################
    ''' Pre-trained sparse AE for feature extraction '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Pre-trained sparse AE for feature extraction")

    filename_presae_ckpt = save_models_folder + '/ckpt_PreSAEForDRE_epoch_{}_lambda_{}_seed_{}.pth'.format(args.dre_presae_epochs, args.dre_presae_lambda_sparsity, args.seed)
    print('\n' + filename_presae_ckpt)

    # training
    if not os.path.isfile(filename_presae_ckpt):

        save_sae_images_InTrain_folder = save_images_folder + '/SAE_lambda_{}_InTrain_{}'.format(args.dre_presae_lambda_sparsity, args.seed)
        os.makedirs(save_sae_images_InTrain_folder, exist_ok=True)

        # dataloader
        trainset = IMGs_dataset(images_train, labels_train, normalize=True)
        trainloader_sparseAE = torch.utils.data.DataLoader(trainset, batch_size=args.dre_presae_batch_size_train, shuffle=True, num_workers=args.num_workers)

        # initialize net
        dre_presae_encoder_net = encoder_extract(ch=64, dim_bottleneck=args.img_size*args.img_size*args.num_channels).cuda()
        dre_presae_decoder_net = decoder_extract(ch=64, dim_bottleneck=args.img_size*args.img_size*args.num_channels).cuda()
        dre_presae_encoder_net = nn.DataParallel(dre_presae_encoder_net)
        dre_presae_decoder_net = nn.DataParallel(dre_presae_decoder_net)

        print("\n Start training sparseAE model for feature extraction in the DRE >>>")
        dre_presae_encoder_net, dre_presae_decoder_net = train_sparseAE(trainloader=trainloader_sparseAE, net_encoder=dre_presae_encoder_net, net_decoder=dre_presae_decoder_net, save_sae_images_folder=save_sae_images_InTrain_folder, path_to_ckpt=save_models_folder)
        # store model
        torch.save({
            'encoder_net_state_dict': dre_presae_encoder_net.state_dict(),
            # 'decoder_net_state_dict': dre_presae_decoder_net.state_dict(),
        }, filename_presae_ckpt)
        print("\n End training CNN.")
    else:
        print("\n Loading pre-trained sparseAE for feature extraction in DRE.")
        dre_presae_encoder_net = encoder_extract(ch=64, dim_bottleneck=args.img_size*args.img_size*args.num_channels).cuda()
        dre_presae_encoder_net = nn.DataParallel(dre_presae_encoder_net)
        checkpoint = torch.load(filename_presae_ckpt)
        dre_presae_encoder_net.load_state_dict(checkpoint['encoder_net_state_dict'])
    #end if


    ##############################################
    ''' DRE Training '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n cDRE training")

    if args.dre_kappa<0:
        n_unique = len(unique_labels_train_norm)

        diff_list = []
        for i in range(1,n_unique):
            diff_list.append(unique_labels_train_norm[i] - unique_labels_train_norm[i-1])
        kappa_base = np.abs(args.dre_kappa)*np.max(np.array(diff_list))

        if args.dre_threshold_type=="hard":
            args.dre_kappa = kappa_base
        else:
            args.dre_kappa = 1/kappa_base**2
    #end if

    ## dre filename
    drefile_fullpath = save_models_folder + '/ckpt_cDRE-F-SP_{}_epochs_{}_lambda_{}_type_{}_kappa_{}_CcGAN_{}_{}_{}_{}_seed_{}.pth'.format(args.dre_net, args.dre_epochs, args.dre_lambda, args.dre_threshold_type, args.dre_kappa, args.gan_niters, args.gan_threshold_type, args.gan_kernel_sigma, args.gan_kappa, args.seed)
    print('\n' + drefile_fullpath)

    path_to_ckpt_in_train = save_models_folder + '/ckpt_cDRE-F-SP_{}_lambda_{}_type_{}_kappa_{}_CcGAN_{}_{}_{}_{}_seed_{}'.format(args.dre_net, args.dre_lambda, args.dre_threshold_type, args.dre_kappa, args.gan_niters, args.gan_threshold_type, args.gan_kernel_sigma, args.gan_kappa, args.seed)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    dre_loss_file_fullpath = save_traincurves_folder + '/train_loss_cDRE-F-SP_{}_epochs_{}_lambda_{}_type_{}_kappa_{}_CcGAN_{}_{}_{}_{}_seed_{}.png'.format(args.dre_net, args.dre_epochs, args.dre_lambda, args.dre_threshold_type, args.dre_kappa, args.gan_niters, args.gan_threshold_type, args.gan_kernel_sigma, args.gan_kappa, args.seed)

    ### init net
    dre_net = cDR_MLP(args.dre_net, p_dropout=0.5, init_in_dim=args.num_channels*args.img_size*args.img_size, dim_embed=args.gan_dim_embed).cuda()
    dre_net = nn.DataParallel(dre_net)

    #if DR model exists, then load the pretrained model; otherwise, start training the model.
    if not os.path.isfile(drefile_fullpath):
        print("\n Begin Training conditional DR in Feature Space: >>>")

        if args.dre_no_vicinal:
            cdre_target_labels = labels_train
        else:
            cdre_target_labels = labels_test_raw/args.max_label

        dre_net, avg_train_loss = train_cdre(kappa=args.dre_kappa, train_images=images_train, train_labels=labels_train, test_labels=cdre_target_labels, dre_net=dre_net, dre_precnn_net=dre_presae_encoder_net, netG=netG, net_y2h=net_embed_y2h, path_to_ckpt=path_to_ckpt_in_train)

        # save model
        torch.save({
        'net_state_dict': dre_net.state_dict(),
        }, drefile_fullpath)
        PlotLoss(avg_train_loss, dre_loss_file_fullpath)

    else:
        # if already trained, load pre-trained DR model
        checkpoint_dre_net = torch.load(drefile_fullpath)
        dre_net.load_state_dict(checkpoint_dre_net['net_state_dict'])
    ##end if not

    # Compute density ratio: function for computing a bunch of images in a numpy array
    def comp_cond_density_ratio(imgs, labels, batch_size=args.samp_batch_size):
        #imgs: a torch tensor
        n_imgs = len(imgs)
        if batch_size>n_imgs:
            batch_size = n_imgs

        ##make sure the last iteration has enough samples
        imgs = torch.cat((imgs, imgs[0:batch_size]), dim=0)
        labels = torch.cat((labels, labels[0:batch_size]), dim=0)

        density_ratios = []
        dre_net.eval()
        dre_presae_encoder_net.eval()
        net_embed_y2h.eval()
        # print("\n Begin computing density ratio for images >>")
        with torch.no_grad():
            n_imgs_got = 0
            while n_imgs_got < n_imgs:
                batch_images = imgs[n_imgs_got:(n_imgs_got+batch_size)]
                batch_labels = labels[n_imgs_got:(n_imgs_got+batch_size)]
                batch_images = batch_images.type(torch.float).cuda()
                batch_labels = batch_labels.type(torch.float).cuda()
                batch_features = dre_presae_encoder_net(batch_images)
                batch_ratios = dre_net(batch_features, net_embed_y2h(batch_labels))
                density_ratios.append(batch_ratios.cpu().detach())
                n_imgs_got += batch_size
            ### while n_imgs_got
        density_ratios = torch.cat(density_ratios)
        density_ratios = density_ratios[0:n_imgs].numpy()
        return density_ratios


    # Enhanced sampler based on the trained DR model
    # Rejection Sampling:"Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
    def fn_enhancedSampler_given_label(nfake, given_label, batch_size=args.samp_batch_size, n_burnin=args.samp_burnin_size):
        ## Burn-in Stage
        burnin_labels = given_label * torch.ones(nfake)
        burnin_imgs, _ = fn_sampleGAN_given_labels(burnin_labels, batch_size, to_numpy=False, verbose=False)
        burnin_densityratios = comp_cond_density_ratio(burnin_imgs, burnin_labels)

        # print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
        M_bar = np.max(burnin_densityratios)
        del burnin_imgs, burnin_densityratios; gc.collect()
        ## Rejection sampling
        enhanced_imgs = []
        num_imgs = 0
        while num_imgs < nfake:
            batch_labels = given_label * torch.ones(batch_size)
            batch_imgs, _ = fn_sampleGAN_given_labels(batch_labels, batch_size, to_numpy=False, verbose=False)
            batch_ratios = comp_cond_density_ratio(batch_imgs, batch_labels)
            M_bar = np.max([M_bar, np.max(batch_ratios)])
            #threshold
            batch_p = batch_ratios/M_bar
            batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
            indx_accept = np.where(batch_psi<=batch_p)[0]
            if len(indx_accept)>0:
                enhanced_imgs.append(batch_imgs[indx_accept])
            num_imgs+=len(indx_accept)
            del batch_imgs, batch_ratios; gc.collect()
        # pbar.close()
        enhanced_imgs = np.concatenate(enhanced_imgs, axis=0)
        enhanced_imgs = enhanced_imgs[0:nfake]
        return enhanced_imgs, given_label*np.ones(nfake)








#######################################################################################
'''                                  Evaluation                                     '''
#######################################################################################

if args.eval:
    print("\n Start evaluation ...")

    # for intra-FID
    PreNetFID = encoder_eval(dim_bottleneck=512).cuda()
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = os.path.join(eval_models_folder, 'ckpt_AE_epoch_200_seed_2020_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # for Diversity: entropy of predicted races within each eval center
    PreNetDiversity = ResNet34_class_eval(num_classes=49, ngpu = torch.cuda.device_count()).cuda() #49 chair types
    Filename_PreCNNForEvalGANs_Diversity = os.path.join(eval_models_folder, 'ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre_eval(ngpu = torch.cuda.device_count()).cuda()
    Filename_PreCNNForEvalGANs_LS = os.path.join(eval_models_folder, 'ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # generate nfake images
    print("\n Start sampling {} fake images per label from CcGAN >>>".format(args.eval_nfake_per_label))
    eval_labels = np.sort(np.array(list(set(labels_test_raw)))) #not normalized
    unique_eval_labels = list(set(eval_labels))
    print("\n There are {} unique eval labels.".format(len(unique_eval_labels)))
    eval_labels_norm = eval_labels/args.max_label #normalized

    for i in range(len(eval_labels)):
        curr_label = eval_labels_norm[i]
        if i == 0:
            fake_labels_assigned = np.ones(args.eval_nfake_per_label)*curr_label
        else:
            fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(args.eval_nfake_per_label)*curr_label))
    fake_images, _ = fn_sampleGAN_given_labels(labels=fake_labels_assigned, batch_size=args.eval_batch_size)
    assert len(fake_images) == args.eval_nfake_per_label*len(eval_labels)
    assert len(fake_labels_assigned) == args.eval_nfake_per_label*len(eval_labels)

    # dump fake images for evaluation: NIQE
    if args.eval_dump_fake_for_NIQE:

        dump_fake_images_folder = os.path.join(args.root_path, "dump_fake_data/fake_images_CcGAN_{}_subsampling_{}_nsamp_{}".format(args.gan_threshold_type, subsampling_method, len(fake_images)))
        os.makedirs(dump_fake_images_folder, exist_ok=True)

        for i in tqdm(range(len(fake_images))):
            label_i = fake_labels_assigned[i]*args.max_label
            filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
            image_i = fake_images[i]
            image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
            image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
            image_i_pil.save(filename_i)
        #end for i
    ##end if dump for NIQE

    print("End sampling! We got {} fake images.".format(len(fake_images)))


    # Compute Intra-FID, Diversity, and Label Score
    ## normalize real images
    images_test = (images_test/255.0-0.5)/0.5

    nfake_all = len(fake_images)
    nreal_all = len(images_test)

    center_start = np.min(labels_test_raw)+args.eval_FID_radius
    center_stop = np.max(labels_test_raw)-args.eval_FID_radius

    if args.eval_FID_num_centers<=0 and args.eval_FID_radius==0: #completely overlap
        centers_loc = eval_labels.copy() #not normalized
    elif args.eval_FID_num_centers>0:
        centers_loc = np.linspace(center_start, center_stop, args.eval_FID_num_centers) #not normalized
    else:
        raise Exception('center location error!')

    FID_over_centers = np.zeros(len(centers_loc))
    entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
    labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
    num_realimgs_over_centers = np.zeros(len(centers_loc))
    for i in range(len(centers_loc)):
        center = centers_loc[i]
        interval_start = (center - args.eval_FID_radius)#/args.max_label
        interval_stop = (center + args.eval_FID_radius)#/args.max_label
        indx_real = np.where((labels_test_raw>=interval_start)*(labels_test_raw<=interval_stop)==True)[0]
        np.random.shuffle(indx_real)
        real_images_curr = images_test[indx_real]
        num_realimgs_over_centers[i] = len(real_images_curr)
        indx_fake = np.where((fake_labels_assigned>=(interval_start/args.max_label))*(fake_labels_assigned<=(interval_stop/args.max_label))==True)[0]
        np.random.shuffle(indx_fake)
        fake_images_curr = fake_images[indx_fake]
        fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
        # FID
        FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size = 500, resize = None)
        # Entropy of predicted class labels
        predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=500, num_workers=args.num_workers)
        entropies_over_centers[i] = compute_entropy(predicted_class_labels)
        # Label score
        labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = 500, resize = None, num_workers=args.num_workers)

        print("\n [{}/{}] Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(i+1, len(centers_loc), center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))
    # end for i
    # average over all centers
    print("\n Subsampling {} SFID: {}({}); min/max: {}/{}.".format(subsampling_method, np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
    print("\n Subsampling {} LS over centers: {}({}); min/max: {}/{}.".format(subsampling_method, np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
    print("\n Subsampling {} entropy over centers: {}({}); min/max: {}/{}.".format(subsampling_method, np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

    # dump FID versus number of samples (for each center) to npy
    dump_fid_ls_entropy_over_centers_filename = args.root_path + "/CcGAN_{}_subsampling_{}_fid_ls_entropy_over_centers".format(args.gan_threshold_type, subsampling_method)
    np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)

    # Overall LS: abs(y_assigned - y_predicted)
    ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = 200, resize = None)
    print("\n Subsampling {}: overall LS of {} fake images: {}({}).".format(subsampling_method, nfake_all, ls_mean_overall, ls_std_overall))

##end if args.eval







#######################################################################################
'''                                   Sampling                                      '''
#######################################################################################

if not args.eval: #if not evaluate, then sampling

    #--------------------------------------------------------------------------------------
    ''' Synthetic Data Generation '''

    print('\n Start sampling ...')

    fake_data_h5file_fullpath = os.path.join(fake_data_folder, args.unfiltered_fake_dataset_filename)
    if os.path.isfile(fake_data_h5file_fullpath) and args.samp_filter_mae_percentile_threshold < 1:
        print("\n Loading exiting unfiltered fake data >>>")
        hf = h5py.File(fake_data_h5file_fullpath, 'r')
        fake_images = hf['fake_images'][:]
        fake_labels = hf['fake_labels'][:] #unnormalized
        hf.close()
    else:
        if args.samp_num_fake_labels>0:
            target_labels_norm = np.random.uniform(low=0.0, high=1.0, size=args.samp_num_fake_labels)
        else:
            target_labels_norm = np.sort(np.array(list(set(labels_test_raw))))
            target_labels_norm = target_labels_norm/args.max_label
            assert target_labels_norm.min()>=0 and target_labels_norm.max()<=1

        if args.subsampling:
            print("\n Generating {} fake images for each of {} distinct labels with subsampling: {}.".format(args.samp_nfake_per_label, len(target_labels_norm), subsampling_method))
            fake_images = []
            fake_labels = []
            for i in trange(len(target_labels_norm)):
                fake_labels_i = target_labels_norm[i]*np.ones(args.samp_nfake_per_label)
                fake_images_i, _ = fn_enhancedSampler_given_label(args.samp_nfake_per_label, target_labels_norm[i], batch_size=args.samp_batch_size, n_burnin=args.samp_burnin_size)
                ### denormlize: [-1,1]--->[0,255]
                fake_images_i = (fake_images_i*0.5+0.5)*255.0
                fake_images_i = fake_images_i.astype(np.uint8)
                ### denormalize labels
                fake_labels_i = (fake_labels_i*args.max_label).astype(np.float)
                ### append
                fake_images.append(fake_images_i)
                fake_labels.append(fake_labels_i)
            ##end for i
            fake_images = np.concatenate(fake_images, axis=0)
            fake_labels = np.concatenate(fake_labels, axis=0)
        else:
            print("\n Generating {} fake images for each of {} distinct labels without subsampling.".format(args.samp_nfake_per_label, len(target_labels_norm)))
            fake_images = []
            fake_labels = []
            for i in trange(len(target_labels_norm)):
                fake_labels_i = target_labels_norm[i]*np.ones(args.samp_nfake_per_label)
                fake_images_i, _ = fn_sampleGAN_given_labels(labels=fake_labels_i, batch_size=args.samp_batch_size, to_numpy=True, verbose=False)
                ### denormlize: [-1,1]--->[0,255]
                fake_images_i = (fake_images_i*0.5+0.5)*255.0
                fake_images_i = fake_images_i.astype(np.uint8)
                ### denormalize labels
                fake_labels_i = (fake_labels_i*args.max_label).astype(np.float)
                ### append
                fake_images.append(fake_images_i)
                fake_labels.append(fake_labels_i)
            ##end for i
            fake_images = np.concatenate(fake_images, axis=0)
            fake_labels = np.concatenate(fake_labels, axis=0)
            # fake_images, _ = fn_sampleGAN_given_labels(labels=fake_labels, batch_size=args.samp_batch_size, to_numpy=True, verbose=True)
        assert len(fake_images) == args.samp_nfake_per_label*len(target_labels_norm)
        assert len(fake_labels) == args.samp_nfake_per_label*len(target_labels_norm)
        assert np.max(fake_images)>1

        ##end if
        # ### denormlize: [-1,1]--->[0,255]
        # fake_images = (fake_images*0.5+0.5)*255.0
        # fake_images = fake_images.astype(np.uint8)
        # ### denormalize labels
        # fake_labels = (target_labels_norm*args.max_label).astype(np.float)
    ##end if os


    #--------------------------------------------------------------------------------------
    ''' Filtered and Adjusted by a pre-trained CNN '''
    if args.samp_filter_mae_percentile_threshold < 1 or args.adjust_label:
        print("\n -----------------------------------------------------------------------------------------")
        print("\n Start Filtering Synthetic Data >>>")

        ## dataset
        assert fake_labels.max()>1
        dataset_filtering = IMGs_dataset(fake_images, fake_labels, normalize=True)
        dataloader_filtering = torch.utils.data.DataLoader(dataset_filtering, batch_size=200, shuffle=False, num_workers=args.num_workers)

        ## load pre-trained cnn
        filter_precnn_net = cnn_initialization(cnn_name=args.samp_filter_precnn_net, img_size=args.img_size)
        checkpoint = torch.load(os.path.join(save_models_folder, args.samp_filter_precnn_net_ckpt_filename))
        filter_precnn_net.load_state_dict(checkpoint['net_state_dict'])

        ## evaluate on fake data
        fake_mae_loss = []
        fake_labels_pred = []
        filter_precnn_net.eval()
        pbar = tqdm(total=len(fake_images))
        with torch.no_grad():
            total = 0
            loss_all = 0
            for batch_idx, (images, labels) in enumerate(dataloader_filtering):
                images = images.type(torch.float).cuda()
                labels = labels.type(torch.float) #unnormalized label
                labels_pred = filter_precnn_net(images)
                labels_pred = (labels_pred.cpu())*args.max_label #denormalize
                labels = labels.view(-1)
                labels_pred = labels_pred.view(-1)
                loss = torch.abs(labels_pred-labels)
                loss_all += loss.sum().item()
                total += labels.size(0)
                fake_labels_pred.append(labels_pred.numpy())
                fake_mae_loss.append(loss.numpy())
                pbar.update(len(images))
            print('\n Test MAE of {} on the {} fake images: {}.'.format(args.samp_filter_precnn_net, len(fake_images), loss_all / total))
        fake_mae_loss = np.concatenate(fake_mae_loss, axis=0)
        fake_labels_pred = np.concatenate(fake_labels_pred, axis=0)

        mae_cutoff_point = np.quantile(fake_mae_loss, q=args.samp_filter_mae_percentile_threshold)
        # indx_sel = np.where(fake_mae_loss<mae_cutoff_point)[0]
        # fake_images = fake_images[indx_sel]
        # fake_labels = fake_labels_pred[indx_sel] #adjust the labels of fake data by using the pre-trained big CNN
        if args.samp_filter_mae_percentile_threshold < 1:
            indx_sel = np.where(fake_mae_loss<mae_cutoff_point)[0]
        else:
            indx_sel = np.arange(len(fake_images))
        fake_images = fake_images[indx_sel]
        if args.adjust_label:
            fake_labels = fake_labels_pred[indx_sel] #adjust the labels of fake data by using the pre-trained big CNN
        else:
            fake_labels = fake_labels[indx_sel]

        ## histogram of MAEs
        fig = plt.figure()
        ax = plt.subplot(111)
        n, bins, patches = plt.hist(fake_mae_loss, 100, density=False, facecolor='g', alpha=0.75)
        plt.axvline(x=mae_cutoff_point, c='grey')
        plt.xlabel('MAE')
        plt.ylabel('Frequency')
        plt.title('Histogram of MAE')
        plt.grid(True)
        #plt.show()
        plt.savefig(os.path.join(fake_data_folder, 'histogram_of_fake_data_MAE_with_threshold_{}.png'.format(args.samp_filter_mae_percentile_threshold)))


    #--------------------------------------------------------------------------------------
    ''' Dump synthetic data to h5 file '''

    nfake = len(fake_images)

    fake_dataset_name = '{}_{}_niters_{}_subsampling_{}_FilterMAEPct_{}_AdjustLabel_{}_nfake_{}_seed_{}'.format(args.gan_name, args.gan_loss_type, args.gan_niters, subsampling_method, args.samp_filter_mae_percentile_threshold, args.adjust_label, nfake, args.seed)

    fake_h5file_fullpath = os.path.join(fake_data_folder, 'fake_RC49_NTrainPerLabel_{}_{}.h5'.format(args.max_num_img_per_label, fake_dataset_name))

    if os.path.isfile(fake_h5file_fullpath):
        os.remove(fake_h5file_fullpath)

    ## dump fake iamges into h5 file
    with h5py.File(fake_h5file_fullpath, "w") as f:
        f.create_dataset('fake_images', data = fake_images, dtype='uint8')
        f.create_dataset('fake_labels', data = fake_labels, dtype='float')








print("\n===================================================================================================")
