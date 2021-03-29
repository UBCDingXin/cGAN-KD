'''

Train GAN, Pre-CNN, and DRE. Generate synthetic data.

'''

print("\n ===================================================================================================")

#----------------------------------------
import argparse
import os
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from torch import autograd
from torchvision.utils import save_image
from tqdm import tqdm, trange
import gc
from itertools import groupby
import multiprocessing
import h5py
import pickle
import copy

#----------------------------------------
from opts import gen_synth_data_opts
from utils import *
from models import *
from train_cnn import train_cnn, test_cnn
from train_gan import *
from train_cgan import *
from train_dre import train_dre
from train_cdre import train_cdre


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)

#--------------------------------
# system
# NCPU = multiprocessing.cpu_count()
NCPU = args.num_workers

#-------------------------------
# GAN and DRE
dre_precnn_lr_decay_epochs  = (args.dre_precnn_lr_decay_epochs).split("_")
dre_precnn_lr_decay_epochs = [int(epoch) for epoch in dre_precnn_lr_decay_epochs]

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# output folders
save_models_folder = args.root_path + '/Output_CIFAR{}/saved_models'.format(args.num_classes)
os.makedirs(save_models_folder, exist_ok=True)

save_images_folder = args.root_path + '/Output_CIFAR{}/saved_images'.format(args.num_classes)
os.makedirs(save_images_folder, exist_ok=True)

save_traincurves_folder = args.root_path + '/Output_CIFAR{}/Training_loss_fig'.format(args.num_classes)
os.makedirs(save_traincurves_folder, exist_ok=True)



#######################################################################################
'''                                  Load Data                                      '''
#######################################################################################
## generate subset
trainset_h5py_file = args.root_path + '/data/CIFAR{}_trainset_{}_seed_{}.h5'.format(args.num_classes, args.ntrain, args.seed)
hf = h5py.File(trainset_h5py_file, 'r')
images_train = hf['images_train'][:]
labels_train = hf['labels_train'][:]
hf.close()

if args.num_classes == 10:
    cifar_testset = torchvision.datasets.CIFAR10(root = os.path.join(args.root_path, 'data'), train=False, download=True)
elif args.num_classes == 100:
    cifar_testset = torchvision.datasets.CIFAR100(root = os.path.join(args.root_path, 'data'), train=False, download=True)

# compute the mean and std for normalization
assert images_train.shape[1]==3
train_means = []
train_stds = []
for i in range(3):
    images_i = images_train[:,i,:,:]
    images_i = images_i/255.0
    train_means.append(np.mean(images_i))
    train_stds.append(np.std(images_i))
## for i

images_test = cifar_testset.data
images_test = np.transpose(images_test, (0, 3, 1, 2))
labels_test = np.array(cifar_testset.targets)

print("\n Training set shape: {}x{}x{}x{}; Testing set shape: {}x{}x{}x{}.".format(images_train.shape[0], images_train.shape[1], images_train.shape[2], images_train.shape[3], images_test.shape[0], images_test.shape[1], images_test.shape[2], images_test.shape[3]))


''' transformations '''
if args.dre_precnn_transform:
    transform_precnn_train = transforms.Compose([
                transforms.RandomCrop((args.img_size, args.img_size), padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(train_means, train_stds),
                ])
else:
    transform_precnn_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(train_means, train_stds),
                ])

if args.gan_transform:
    transform_gan_dre = transforms.Compose([
                # transforms.RandomCrop((args.img_size, args.img_size), padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                ])
else:
    transform_gan_dre = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                ])

# test set for cnn
transform_precnn_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
            ])
testset_precnn = IMGs_dataset(images_test, labels_test, transform=transform_precnn_test)
testloader_precnn = torch.utils.data.DataLoader(testset_precnn, batch_size=100, shuffle=False, num_workers=NCPU)




#######################################################################################
'''                             GAN and DRE Training                                '''
#######################################################################################
#--------------------------------------------------------------------------------------
''' Pre-trained CNN for feature extraction '''
if args.subsampling:
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Pre-trained CNN for feature extraction")
    # data loader
    trainset_dre_precnn = IMGs_dataset(images_train, labels_train, transform=transform_precnn_train)
    trainloader_dre_precnn = torch.utils.data.DataLoader(trainset_dre_precnn, batch_size=args.dre_precnn_batch_size_train, shuffle=True, num_workers=NCPU)
    # Filename
    filename_precnn_ckpt = save_models_folder + '/ckpt_PreCNNForDRE_{}_epoch_{}_transform_{}_ntrain_{}_seed_{}.pth'.format(args.dre_precnn_net, args.dre_precnn_epochs, args.dre_precnn_transform, args.ntrain, args.seed)
    print('\n' + filename_precnn_ckpt)

    path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_PreCNNForDRE_{}_ntrain_{}_seed_{}'.format(args.dre_precnn_net, args.ntrain, args.seed)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    # initialize cnn
    dre_precnn_net = cnn_extract_initialization(args.dre_precnn_net, num_classes=args.num_classes, img_size=args.img_size)
    num_parameters = count_parameters(dre_precnn_net)
    # training
    if not os.path.isfile(filename_precnn_ckpt):
        print("\n Start training CNN for feature extraction in the DRE >>>")
        dre_precnn_net = train_cnn(dre_precnn_net, 'PreCNNForDRE_{}'.format(args.dre_precnn_net), trainloader_dre_precnn, testloader_precnn, epochs=args.dre_precnn_epochs, resume_epoch=args.dre_precnn_resume_epoch, lr_base=args.dre_precnn_lr_base, lr_decay_factor=args.dre_precnn_lr_decay_factor, lr_decay_epochs=dre_precnn_lr_decay_epochs, weight_decay=args.dre_precnn_weight_decay, seed = args.seed, extract_feature=True, path_to_ckpt = path_to_ckpt_in_train)

        # store model
        torch.save({
            'net_state_dict': dre_precnn_net.state_dict(),
        }, filename_precnn_ckpt)
        print("\n End training CNN.")
    else:
        print("\n Loading pre-trained CNN for feature extraction in DRE.")
        checkpoint = torch.load(filename_precnn_ckpt)
        dre_precnn_net.load_state_dict(checkpoint['net_state_dict'])
    #end if

    # testing
    _ = test_cnn(dre_precnn_net, testloader_precnn, extract_feature=True, verbose=True)



#--------------------------------------------------------------------------------------
''' Data Loader for GAN and DRE training '''
if args.gan_name in ['cGAN', 'BigGAN']: #conditional GAN and conditional DRE
    trainset_gan_dre = IMGs_dataset(images_train, labels_train, transform=transform_gan_dre)
    ## data loader for gan
    trainloader_gan = torch.utils.data.DataLoader(trainset_gan_dre, batch_size=args.gan_batch_size, shuffle=True, num_workers=NCPU)
    ## data loader for dre
    trainloader_dre = torch.utils.data.DataLoader(trainset_gan_dre, batch_size=args.dre_batch_size, shuffle=True, num_workers=NCPU)
else:
    trainloader_gan_list = []
    trainloader_dre_list = []
    for i in range(args.num_classes):
        indx_train_i = np.where(labels_train==i)[0]
        trainset_gan_dre_i = IMGs_dataset(images_train[indx_train_i], labels_train[indx_train_i], transform=transform_gan_dre)
        ## data loader for gan
        trainloader_gan_i = torch.utils.data.DataLoader(trainset_gan_dre_i, batch_size=args.gan_batch_size, shuffle=True, num_workers=NCPU)
        trainloader_gan_list.append(trainloader_gan_i)
        ## data loader for dre
        trainloader_dre_i = torch.utils.data.DataLoader(trainset_gan_dre_i, batch_size=args.dre_batch_size, shuffle=True, num_workers=NCPU)
        trainloader_dre_list.append(trainloader_dre_i)
    #end for i
# end if gan_name


#--------------------------------------------------------------------------------------
''' GAN training '''

path_to_imgs_in_train = save_images_folder + '/{}_{}_ntrain_{}_transform_{}_seed_{}_InTrain'.format(args.gan_name, args.gan_loss, args.ntrain, args.gan_transform, args.seed)
os.makedirs(path_to_imgs_in_train, exist_ok=True)

if args.gan_name == 'BigGAN': #BigGAN
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Start training BigGAN >>>")

    ganfile_fullpath = save_models_folder + '/ckpt_BigGAN_cifar{}_ntrain_{}_seed_{}/G_ema.pth'.format(args.num_classes, args.ntrain, args.seed)

    ckpt_g = torch.load(ganfile_fullpath)
    netG = BigGAN_Generator(resolution=args.img_size, G_attn='0', n_classes=args.num_classes, G_shared=False).cuda()
    netG.load_state_dict(ckpt_g)
    netG = nn.DataParallel(netG)

    def fn_sampleGAN_given_label(nfake, given_label, batch_size, to_numpy=True):
        raw_fake_images = []
        raw_fake_labels = []
        netG.eval()
        with torch.no_grad():
            tmp = 0
            while tmp < nfake:
                z = torch.randn(batch_size, args.gan_dim_g, dtype=torch.float).cuda()
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

elif args.gan_name == 'cGAN':
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Start training cGAN >>>")

    ganfile_fullpath = save_models_folder + '/ckpt_{}_{}_epochs_{}_ntrain_{}_transform_{}_seed_{}.pth'.format(args.gan_name, args.gan_loss, args.gan_epochs, args.ntrain, args.gan_transform, args.seed)
    print('\n' + ganfile_fullpath)

    path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_{}_{}_ntrain_{}_transform_{}_seed_{}'.format(args.gan_name, args.gan_loss, args.ntrain, args.gan_transform, args.seed)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)


    if not os.path.isfile(ganfile_fullpath):
        start = timeit.default_timer()
        print("\n Begin Training GAN:")
        #model initialization
        netG = cond_generator(nz=args.gan_dim_g, num_classes=args.num_classes, nc=args.num_channels).cuda()
        netD = cond_discriminator(nc=args.num_channels, num_classes=args.num_classes).cuda()
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

        # Start training
        netG, netD = train_cgan(args.gan_name, trainloader_gan, netG, netD, save_GANimages_folder = path_to_imgs_in_train, path_to_ckpt = path_to_ckpt_in_train)

        # store model
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
        }, ganfile_fullpath)

        stop = timeit.default_timer()
        print("GAN training finished! Time elapses: {}s".format(stop - start))
    else:
        print("\n Load pre-trained GAN:")
        checkpoint = torch.load(ganfile_fullpath)
        netG = cond_generator(nz=args.gan_dim_g, num_classes=args.num_classes, nc=args.num_channels).cuda()
        netG = nn.DataParallel(netG)
        netG.load_state_dict(checkpoint['netG_state_dict'])
    # end if

    # function for sampling from a trained cGAN
    def fn_sampleGAN_given_label(nfake, given_label, batch_size, to_numpy=True):
        images, labels = SampcGAN_given_label(netG, given_label, nfake = nfake, batch_size = batch_size, to_numpy=to_numpy)
        return images, labels

else: ##unconditional GAN
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Start training {} GANs >>>".format(args.num_classes))

    netG_list = []
    start = timeit.default_timer()
    for i in range(args.num_classes): #fit one DCGAN for one class
        ganfile_i_fullpath = save_models_folder + '/ckpt_{}_{}_epochs_{}_ntrain_{}_class_{}_transform_{}_seed_{}.pth'.format(args.gan_name, args.gan_loss, args.gan_epochs, args.ntrain, i, args.gan_transform, args.seed)
        print('\n' + ganfile_i_fullpath)

        path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_{}_{}_ntrain_{}_transform_{}_seed_{}'.format(args.gan_name, args.gan_loss, args.ntrain, args.gan_transform, args.seed)
        os.makedirs(path_to_ckpt_in_train, exist_ok=True)

        if not os.path.isfile(ganfile_i_fullpath):
            print("\n Begin Training Unconditional GAN for class " + str(i) + ":")
            #dataloader
            trainloader_gan_i = trainloader_gan_list[i]

            #model initialization
            netG = generator(nz=args.gan_dim_g, nc=args.num_channels).cuda()
            netD = discriminator(nc=args.num_channels).cuda()
            netG = nn.DataParallel(netG)
            netD = nn.DataParallel(netD)

            # Start training
            netG, netD = train_gan(args.gan_name, trainloader_gan_i, i, netG, netD, save_GANimages_folder = path_to_imgs_in_train, path_to_ckpt = path_to_ckpt_in_train)
            # store model
            torch.save({
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
            }, ganfile_i_fullpath)

            stop = timeit.default_timer()
            print("GAN training for class %d finished! Time elapses: %4f" % (i, stop - start))
        else:
            print("\n Load pre-trained GAN for class " + str(i) + ":")
            checkpoint = torch.load(ganfile_i_fullpath)
            netG = generator(nz=args.gan_dim_g, nc=args.num_channels).cuda()
            netG = nn.DataParallel(netG)
            netG.load_state_dict(checkpoint['netG_state_dict'])
        netG_list.append(netG.cpu())
    #end for i

    # function for sampling from a trained cGAN
    def fn_sampleGAN_given_label(nfake, given_label, batch_size, to_numpy=True):
        netG = netG_list[given_label]; netG = netG.cuda()
        labels = np.ones(nfake)*given_label
        images = SampGAN(netG, nfake = nfake, batch_size = batch_size, to_numpy=to_numpy)
        return images, labels

# end if gan_name


#--------------------------------------------------------------------------------------
''' DRE training '''

if args.subsampling:
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Start training DRE model >>>")

    ##################################################
    # conditional DRE
    if args.gan_name in ['cGAN', 'BigGAN']:
        start = timeit.default_timer()
        ## dre filename
        drefile_fullpath = save_models_folder + '/ckpt_cDRE-F-SP_{}_epochs_{}_lambda_{}_{}_{}_epochs_{}_transform_{}_ntrain_{}_seed_{}.pth'.format(args.dre_net, args.dre_epochs, args.dre_lambda, args.gan_name, args.gan_loss, args.gan_epochs, args.gan_transform, args.ntrain, args.seed)
        print('\n' + drefile_fullpath)

        path_to_ckpt_in_train = save_models_folder + '/ckpt_cDRE-F-SP_{}_lambda_{}_{}_{}_epochs_{}_transform_{}_ntrain_{}_seed_{}'.format(args.dre_net, args.dre_lambda, args.gan_name, args.gan_loss, args.gan_epochs, args.gan_transform, args.ntrain, args.seed)
        os.makedirs(path_to_ckpt_in_train, exist_ok=True)

        dre_loss_file_fullpath = save_traincurves_folder + '/cDRE-F-SP_{}_epochs_{}_lambda_{}_{}_{}_epochs_{}_transform_{}_ntrain_{}_seed_{}.png'.format(args.dre_net, args.dre_epochs, args.dre_lambda, args.gan_name, args.gan_loss, args.gan_epochs, args.gan_transform, args.ntrain, args.seed)

        ## initialize conditional density ratio net
        dre_net = cDR_MLP(args.dre_net, p_dropout=0.5, init_in_dim = args.num_channels*args.img_size*args.img_size, num_classes = args.num_classes).cuda()
        dre_net = nn.DataParallel(dre_net)
        #if DR model exists, then load the pretrained model; otherwise, start training the model.
        if not os.path.isfile(drefile_fullpath):
            print("\n Begin Training DR in Feature Space: >>>")
            dre_net, avg_train_loss = train_cdre(trainloader_dre, dre_net, dre_precnn_net, netG, path_to_ckpt=path_to_ckpt_in_train)
            # save model
            torch.save({
            'net_state_dict': dre_net.state_dict(),
            }, drefile_fullpath)
            PlotLoss(avg_train_loss, dre_loss_file_fullpath)
        else:
            # if already trained, load pre-trained DR model
            checkpoint_dre_net = torch.load(drefile_fullpath)
            dre_net.load_state_dict(checkpoint_dre_net['net_state_dict'])
        stop = timeit.default_timer()
        print("cDRE training finished; Time elapses: {}s".format(stop - start))

        # function for computing a bunch of images in a numpy array
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
            dre_precnn_net.eval()
            # print("\n Begin computing density ratio for images >>")
            with torch.no_grad():
                n_imgs_got = 0
                while n_imgs_got < n_imgs:
                    batch_images = imgs[n_imgs_got:(n_imgs_got+batch_size)]
                    batch_labels = labels[n_imgs_got:(n_imgs_got+batch_size)]
                    batch_images = batch_images.type(torch.float).cuda()
                    batch_labels = batch_labels.type(torch.long).cuda()
                    _, batch_features = dre_precnn_net(batch_images)
                    batch_ratios = dre_net(batch_features, batch_labels)
                    density_ratios.append(batch_ratios.cpu().detach())
                    n_imgs_got += batch_size
                ### while n_imgs_got
            density_ratios = torch.cat(density_ratios)
            density_ratios = density_ratios[0:n_imgs].numpy()
            return density_ratios

        # Enhanced sampler based on the trained DR model
        # Rejection Sampling:"Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
        def fn_enhancedSampler_given_label(nfake, given_label, batch_size=args.samp_batch_size):
            ## Burn-in Stage
            n_burnin = args.samp_burnin_size
            burnin_imgs, burnin_labels = fn_sampleGAN_given_label(n_burnin, given_label, batch_size, to_numpy=False)
            burnin_densityratios = comp_cond_density_ratio(burnin_imgs, burnin_labels)
            # print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
            M_bar = np.max(burnin_densityratios)
            del burnin_imgs, burnin_densityratios; gc.collect()
            ## Rejection sampling
            enhanced_imgs = []
            pb = SimpleProgressBar()
            # pbar = tqdm(total=nfake)
            num_imgs = 0
            while num_imgs < nfake:
                batch_imgs, batch_labels = fn_sampleGAN_given_label(batch_size, given_label, batch_size, to_numpy=False)
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
                pb.update(np.min([float(num_imgs)*100/nfake,100]))
                # pbar.update(len(indx_accept))
            # pbar.close()
            enhanced_imgs = np.concatenate(enhanced_imgs, axis=0)
            enhanced_imgs = enhanced_imgs[0:nfake]
            return enhanced_imgs, given_label*np.ones(nfake) #remove the first all zero array

    ##################################################
    # Train 10 unconditional DR models
    else:

        dre_net_list = []
        for i in range(args.num_classes):

            #dataloader
            trainloader_dre_i = trainloader_dre_list[i]

            # load G for current class
            netG = netG_list[i]
            netG = netG.cuda()

            # train i-th DR model
            start = timeit.default_timer()
            ## dre filename
            drefile_fullpath_i = save_models_folder + '/ckpt_DRE-F-SP_{}_epochs_{}_lambda_{}_{}_{}_epochs_{}_ntrain_{}_transform_{}_class_{}_seed_{}.pth'.format(args.dre_net, args.dre_epochs, args.dre_lambda, args.gan_name, args.gan_loss, args.gan_epochs, args.gan_transform, args.ntrain, i, args.seed)
            print('\n' + drefile_fullpath_i)

            path_to_ckpt_in_train = save_models_folder + '/ckpt_DRE-F-SP_{}_lambda_{}_{}_{}_epochs_{}_transform_{}_ntrain_{}_seed_{}'.format(args.dre_net, args.dre_lambda, args.gan_name, args.gan_loss, args.gan_epochs, args.gan_transform, args.ntrain, args.seed)
            os.makedirs(path_to_ckpt_in_train, exist_ok=True)

            dre_loss_file_fullpath_i = save_traincurves_folder + '/DRE-F-SP_{}_epochs_{}_lambda_{}_{}_{}_epochs_{}_ntrain_{}_transform_{}_class_{}_seed_{}.png'.format(args.dre_net, args.dre_epochs, args.dre_lambda, args.gan_name, args.gan_loss, args.gan_epochs, args.gan_transform, args.ntrain, i, args.seed)
            # initialize i-th density ratio net
            dre_net = DR_MLP(args.dre_net, p_dropout=0.4, init_in_dim = args.num_channels*args.img_size*args.img_size).cuda()
            dre_net = nn.DataParallel(dre_net)
            #if DR model exists, then load the pretrained model; otherwise, start training the model.
            if not os.path.isfile(drefile_fullpath_i):
                print("\n Begin Training DR for Class {} in Feature Space: >>>".format(i))
                dre_net, avg_train_loss = train_dre(i, trainloader_dre_i, dre_net, dre_precnn_net, netG, path_to_ckpt=path_to_ckpt_in_train)
                # save model
                torch.save({
                'net_state_dict': dre_net.state_dict(),
                }, drefile_fullpath_i)
                PlotLoss(avg_train_loss, dre_loss_file_fullpath_i)
            else:
                # if already trained, load pre-trained DR model
                checkpoint_dre_net_i = torch.load(drefile_fullpath_i)
                dre_net.load_state_dict(checkpoint_dre_net_i['net_state_dict'])
            dre_net_list.append(dre_net.cpu())
            stop = timeit.default_timer()
            print("DRE fitting for Class {} finished; Time elapses: {}s".format(i, stop - start))

        #end for i

        # function for computing a bunch of images in a numpy array
        def comp_density_ratio_given_label(imgs, given_label):
            #imgs: an numpy array
            n_imgs = imgs.shape[0]
            if args.samp_batch_size<n_imgs:
                batch_size_tmp = args.samp_batch_size
            else:
                batch_size_tmp = n_imgs
            dataset_tmp = IMGs_dataset(imgs)
            dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=NCPU)
            data_iter = iter(dataloader_tmp)

            density_ratios = []
            dre_net = dre_net_list[given_label]; dre_net = dre_net.cuda()
            dre_net.eval()
            dre_precnn_net.eval()
            # print("\n Begin computing density ratio for images >>")
            with torch.no_grad():
                tmp = 0
                while tmp < n_imgs:
                    batch_imgs = data_iter.next()
                    batch_imgs = batch_imgs.type(torch.float).cuda()
                    _, batch_features = dre_precnn_net(batch_imgs)
                    batch_weights = dre_net(batch_features)
                    density_ratios.append(batch_weights.view(-1,1).cpu().detach().numpy())
                    tmp += batch_size_tmp
                #end while
                density_ratios = np.concatenate(density_ratios, axis=0)
                return density_ratios

        # enhanced_sampler
        # Rejection Sampling:"Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
        def fn_enhancedSampler_given_label(nfake, given_label, batch_size=args.samp_batch_size):
            ## Burn-in Stage
            n_burnin = args.samp_burnin_size # originally 50000
            burnin_imgs, burnin_labels = fn_sampleGAN_given_label(n_burnin, given_label, batch_size)
            burnin_densityratios = comp_density_ratio_given_label(burnin_imgs, given_label)
            # print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
            M_bar = np.max(burnin_densityratios)
            while M_bar<=1e-20:
                print("M_bar too small: %f; regenerate %d burnin images" % (M_bar, n_burnin))
                n_burnin = args.samp_burnin_size # originally 50000
                burnin_imgs, burnin_labels = fn_sampleGAN_given_label(n_burnin, given_label, batch_size)
                burnin_densityratios = comp_density_ratio_given_label(burnin_imgs, given_label)
                # print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
                M_bar = np.max(burnin_densityratios)
            assert M_bar>0
            # print("M_bar is %f" % M_bar)
            del burnin_imgs, burnin_densityratios; gc.collect()

            ## Rejection sampling
            enhanced_imgs = []
            pb = SimpleProgressBar()
            num_imgs = 0
            while num_imgs < nfake:
                batch_imgs, _ = fn_sampleGAN_given_label(batch_size, given_label, batch_size)
                batch_ratios = comp_density_ratio_given_label(batch_imgs, given_label)
                M_bar = np.max([M_bar, np.max(batch_ratios)])
                # print("M_bar is %f" % M_bar)
                #threshold
                batch_p = batch_ratios/M_bar
                batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
                indx_accept = np.where(batch_psi<=batch_p)[0]
                if len(indx_accept)>0:
                    enhanced_imgs.append(batch_imgs[indx_accept])
                num_imgs+=len(indx_accept)
                del batch_imgs, batch_ratios; gc.collect()
                pb.update(np.min([float(num_imgs)*100/nfake,100]))
            enhanced_imgs = np.concatenate(enhanced_imgs, axis=0)
            return enhanced_imgs, given_label*np.ones(nfake) #remove the first all zero array

    # end if gan_name[0]
#end if subsampling_name


#--------------------------------------------------------------------------------------
''' Synthetic Data Generation '''
print("\n -----------------------------------------------------------------------------------------")
print("\n Start Generating Synthetic Data >>>")

fake_h5file_fullpath = os.path.join(args.root_path, 'data', args.unfiltered_fake_dataset_filename)
if os.path.isfile(fake_h5file_fullpath) and args.samp_filter_ce_percentile_threshold < 1:
    print("\n Loading exiting unfiltered fake data >>>")
    hf = h5py.File(fake_h5file_fullpath, 'r')
    fake_images = hf['fake_images'][:]
    fake_labels = hf['fake_labels'][:]
    hf.close()
else:
    ## sample from GAN or cGAN without subsampling
    if not args.subsampling:
        for i in range(args.num_classes):
            print("\n Generate {} fake images for class {}/{}.".format(args.samp_nfake_per_class, i+1, args.num_classes))
            fake_images_i, fake_labels_i = fn_sampleGAN_given_label(args.samp_nfake_per_class, i, args.samp_batch_size)
            if i == 0:
                fake_images, fake_labels = fake_images_i, fake_labels_i
            else:
                fake_images = np.concatenate((fake_images, fake_images_i), axis=0)
                fake_labels = np.concatenate((fake_labels, fake_labels_i))
        # end for i
        del fake_images_i; gc.collect()
    ## sample from GAN or cGAN with subsampling
    else:
        for i in range(args.num_classes):
            print("\n Generate {} fake images for class {}/{}.".format(args.samp_nfake_per_class, i+1, args.num_classes))
            fake_images_i, fake_labels_i = fn_enhancedSampler_given_label(args.samp_nfake_per_class, i, args.samp_batch_size)
            if i == 0:
                fake_images, fake_labels = fake_images_i, fake_labels_i
            else:
                fake_images = np.concatenate((fake_images, fake_images_i), axis=0)
                fake_labels = np.concatenate((fake_labels, fake_labels_i))
        # end for i
        del fake_images_i; gc.collect()

    ### denormlize: [-1,1]--->[0,255]
    fake_images = (fake_images*0.5+0.5)*255.0
    fake_images = fake_images.astype(np.uint8)
## end if



#--------------------------------------------------------------------------------------
''' Filtered and Adjusted by a pre-trained CNN '''
if args.samp_filter_ce_percentile_threshold < 1:
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Start Filtering Synthetic Data >>>")

    transform_fake_dataset = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(train_means, train_stds),
                ])
    fake_dataset = IMGs_dataset(fake_images, fake_labels, transform=transform_fake_dataset)
    fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=200, shuffle=False, num_workers=NCPU)

    ## initialize pre-trained CNN for filtering
    if args.samp_filter_precnn_net == "ResNet110":
        filter_precnn_net = ResNet110_custom(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "ResNet101":
        filter_precnn_net = ResNet101(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "ResNet152":
        filter_precnn_net = ResNet152(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "DenseNet121":
        filter_precnn_net = DenseNet121(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "PreActResNet101":
        filter_precnn_net = PreActResNet101(num_classes=args.num_classes)
    filter_precnn_net = nn.DataParallel(filter_precnn_net)
    filter_precnn_net = filter_precnn_net.cuda()
    ## load ckpt
    checkpoint = torch.load(os.path.join(save_models_folder, args.samp_filter_precnn_net_ckpt_filename))
    filter_precnn_net.load_state_dict(checkpoint['net_state_dict'])

    ## evaluate on fake data
    fake_CE_loss = []
    fake_labels_pred = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    filter_precnn_net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    pbar = tqdm(total=len(fake_images))
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(fake_dataloader):
            images = images.type(torch.float).cuda()
            labels = labels.type(torch.long).cuda()
            outputs = filter_precnn_net(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            fake_labels_pred.append(predicted.cpu().numpy())
            fake_CE_loss.append(loss.cpu().numpy())
            pbar.update(len(images))
        print('\n Test Accuracy of {} on the {} fake images: {} %'.format(args.samp_filter_precnn_net, len(fake_images), 100.0 * correct / total))
    fake_CE_loss = np.concatenate(fake_CE_loss)
    fake_labels_pred = np.concatenate(fake_labels_pred)

    CE_cutoff_point = np.quantile(fake_CE_loss, q=args.samp_filter_ce_percentile_threshold)
    indx_sel = np.where(fake_CE_loss<CE_cutoff_point)[0]
    fake_images = fake_images[indx_sel]
    # fake_labels = fake_labels[indx_sel]
    fake_labels = fake_labels_pred[indx_sel] #adjust the labels of fake data by using the pre-trained big CNN


## if args.samp_filter_ce_percentile_threshold



#--------------------------------------------------------------------------------------
''' Dump synthetic data '''
nfake = len(fake_images)

fake_dataset_name = '{}_{}_epochs_{}_transform_{}_subsampling_{}_FilterCEPct_{}_nfake_{}_seed_{}'.format(args.gan_name, args.gan_loss, args.gan_epochs, args.gan_transform, args.subsampling, args.samp_filter_ce_percentile_threshold, nfake, args.seed)

fake_h5file_fullpath = args.root_path + '/data/CIFAR{}_ntrain_{}_{}.h5'.format(args.num_classes, args.ntrain, fake_dataset_name)

if os.path.isfile(fake_h5file_fullpath):
    os.remove(fake_h5file_fullpath)

## dump fake iamges into h5 file
with h5py.File(fake_h5file_fullpath, "w") as f:
    f.create_dataset('fake_images', data = fake_images)
    f.create_dataset('fake_labels', data = fake_labels)


## histogram
fig = plt.figure()
ax = plt.subplot(111)
n, bins, patches = plt.hist(fake_labels, 100, density=False, facecolor='g', alpha=0.75)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Histogram of labels')
plt.grid(True)
#plt.show()
plt.savefig(os.path.join(args.root_path, '{}.png'.format(fake_dataset_name)))


## output some example fake images
n_row_show = 10
for i in range(args.num_classes):
    indx_i = np.where(fake_labels==i)[0]
    np.random.shuffle(indx_i)
    indx_i = indx_i[0:100]
    example_fake_images_i = fake_images[indx_i]
    if i == 0:
        example_fake_images = example_fake_images_i
    else:
        example_fake_images = np.concatenate((example_fake_images, example_fake_images_i), axis=0)
example_fake_images = example_fake_images / 255.0
example_fake_images = torch.from_numpy(example_fake_images)
save_image(example_fake_images.data, save_images_folder +'/example_fake_images_CIFAR{}_ntrain_{}_{}.png'.format(args.num_classes, args.ntrain, fake_dataset_name), nrow=n_row_show, normalize=True)


print("\n ===================================================================================================")
