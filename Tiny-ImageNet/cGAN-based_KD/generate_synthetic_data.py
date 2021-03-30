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
from train_cdre import train_cdre


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)

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
save_models_folder = args.root_path + '/output/saved_models'
os.makedirs(save_models_folder, exist_ok=True)

save_images_folder = args.root_path + '/output/saved_images'
os.makedirs(save_images_folder, exist_ok=True)

save_traincurves_folder = args.root_path + '/output/training_loss_fig'
os.makedirs(save_traincurves_folder, exist_ok=True)

save_fake_data_folder = args.root_path + '/output/fake_data'
os.makedirs(save_fake_data_folder, exist_ok=True)


#######################################################################################
'''                                  Load Data                                      '''
#######################################################################################
## generate subset
trainset_h5py_file = args.root_path + '/data/tiny-imagenet-200.h5'
hf = h5py.File(trainset_h5py_file, 'r')
images_train = hf['imgs'][:]
labels_train = hf['labels'][:]
images_test = hf['imgs_val'][:]
labels_test = hf['labels_val'][:]
hf.close()

### compute the mean and std for normalization
### Note that: In GAN-based KD, use computed mean and stds to normalize images for precnn training is better than using [0.5,0.5,0.5]
assert images_train.shape[1]==3
train_means = []
train_stds = []
for i in range(3):
    images_i = images_train[:,i,:,:]
    images_i = images_i/255.0
    train_means.append(np.mean(images_i))
    train_stds.append(np.std(images_i))
## for i

print("\n Training set shape: {}x{}x{}x{}; Testing set shape: {}x{}x{}x{}.".format(images_train.shape[0], images_train.shape[1], images_train.shape[2], images_train.shape[3], images_test.shape[0], images_test.shape[1], images_test.shape[2], images_test.shape[3]))
print("\r Normalization constants: {}, {}".format(train_means, train_stds))


''' transformations '''
if args.dre_precnn_transform:
    transform_precnn_train = transforms.Compose([
                transforms.RandomCrop((args.img_size, args.img_size), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(train_means, train_stds),
                ])
else:
    transform_precnn_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(train_means, train_stds),
                ])

transform_dre = transforms.Compose([
            # transforms.RandomCrop((args.img_size, args.img_size), padding=4), ##note that GAN training does not involve cropping!!!
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
            ])

# test set for cnn
transform_precnn_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_means, train_stds),
            ])
testset_precnn = IMGs_dataset(images_test, labels_test, transform=transform_precnn_test)
testloader_precnn = torch.utils.data.DataLoader(testset_precnn, batch_size=100, shuffle=False, num_workers=args.num_workers)



#######################################################################################
'''                             GAN and DRE Training                                '''
#######################################################################################
#--------------------------------------------------------------------------------------
''' Load Pre-trained BigGAN to Memory (not GPU) '''
ganfile_fullpath = save_models_folder + '/BigGAN_weights/G_ema.pth'
print(ganfile_fullpath)
assert os.path.exists(ganfile_fullpath)
ckpt_g = torch.load(ganfile_fullpath)
netG = BigGAN_Generator(dim_z=args.gan_dim_g, resolution=args.img_size, G_attn='32', n_classes=args.num_classes, G_shared=True, shared_dim=128, hier=True)
netG.load_state_dict(ckpt_g)

def fn_sampleGAN_given_label(nfake, given_label, batch_size, pretrained_netG=netG, to_numpy=True):
    raw_fake_images = []
    raw_fake_labels = []
    pretrained_netG = pretrained_netG.cuda()
    pretrained_netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < nfake:
            z = torch.randn(batch_size, args.gan_dim_g, dtype=torch.float).cuda()
            labels = (given_label*torch.ones(batch_size)).type(torch.long).cuda()
            batch_fake_images = nn.parallel.data_parallel(pretrained_netG, (z, pretrained_netG.shared(labels)))
            raw_fake_images.append(batch_fake_images.cpu())
            raw_fake_labels.append(labels.cpu().view(-1))
            tmp += batch_size

    raw_fake_images = torch.cat(raw_fake_images, dim=0)
    raw_fake_labels = torch.cat(raw_fake_labels)

    if to_numpy:
        raw_fake_images = raw_fake_images.numpy()
        raw_fake_labels = raw_fake_labels.numpy()

    return raw_fake_images[0:nfake], raw_fake_labels[0:nfake]

#--------------------------------------------------------------------------------------
''' Pre-trained CNN for feature extraction '''
if args.subsampling:
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Pre-trained CNN for feature extraction")
    # data loader
    trainset_dre_precnn = IMGs_dataset(images_train, labels_train, transform=transform_precnn_train)
    trainloader_dre_precnn = torch.utils.data.DataLoader(trainset_dre_precnn, batch_size=args.dre_precnn_batch_size_train, shuffle=True, num_workers=args.num_workers)
    # Filename
    filename_precnn_ckpt = save_models_folder + '/ckpt_PreCNNForDRE_{}_epoch_{}_transform_{}_seed_{}.pth'.format(args.dre_precnn_net, args.dre_precnn_epochs, args.dre_precnn_transform, args.seed)
    print('\n' + filename_precnn_ckpt)

    path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_PreCNNForDRE_{}_seed_{}'.format(args.dre_precnn_net, args.seed)
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


    ##############################################
    ''' cDRE Training '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n cDRE training")
    trainset_dre = IMGs_dataset(images_train, labels_train, transform=transform_dre)
    trainloader_dre = torch.utils.data.DataLoader(trainset_dre, batch_size=args.dre_batch_size, shuffle=True, num_workers=args.num_workers)

    ## dre filename
    drefile_fullpath = save_models_folder + '/ckpt_cDRE-F-SP_{}_epochs_{}_lambda_{}_seed_{}.pth'.format(args.dre_net, args.dre_epochs, args.dre_lambda, args.seed)
    print('\n' + drefile_fullpath)

    path_to_ckpt_in_train = save_models_folder + '/ckpt_cDRE-F-SP_{}_lambda_{}_seed_{}'.format(args.dre_net, args.dre_lambda, args.seed)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    dre_loss_file_fullpath = save_traincurves_folder + '/train_loss_cDRE-F-SP_{}_epochs_{}_lambda_{}_seed_{}.png'.format(args.dre_net, args.dre_epochs, args.dre_lambda, args.seed)

    ## init DRE model
    dre_net = cDR_MLP(args.dre_net, p_dropout=0.5, init_in_dim = args.num_channels*args.img_size*args.img_size, num_classes = args.num_classes).cuda()
    dre_net = nn.DataParallel(dre_net)
    ##if DR model exists, then load the pretrained model; otherwise, start training the model.
    if not os.path.isfile(drefile_fullpath):
        print("\n Begin Training conditional DR in Feature Space: >>>")
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
    def fn_enhancedSampler_given_label(nfake, given_label, batch_size=args.samp_batch_size, verbose=True):
        ## Burn-in Stage
        n_burnin = args.samp_burnin_size
        burnin_imgs, burnin_labels = fn_sampleGAN_given_label(n_burnin, given_label, batch_size, to_numpy=False)
        burnin_densityratios = comp_cond_density_ratio(burnin_imgs, burnin_labels)
        # print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
        M_bar = np.max(burnin_densityratios)
        del burnin_imgs, burnin_densityratios; gc.collect()
        ## Rejection sampling
        enhanced_imgs = []
        if verbose:
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
            if verbose:
                pb.update(np.min([float(num_imgs)*100/nfake,100]))
                # pbar.update(len(indx_accept))
        # pbar.close()
        enhanced_imgs = np.concatenate(enhanced_imgs, axis=0)
        enhanced_imgs = enhanced_imgs[0:nfake]
        return enhanced_imgs, given_label*np.ones(nfake)


#--------------------------------------------------------------------------------------
''' Synthetic Data Generation '''
print("\n -----------------------------------------------------------------------------------------")
print("\n Start Generating Synthetic Data >>>")

fake_h5file_fullpath = os.path.join(save_fake_data_folder, args.unfiltered_fake_dataset_filename)
if os.path.isfile(fake_h5file_fullpath) and args.samp_filter_ce_percentile_threshold < 1:
    print("\n Loading exiting unfiltered fake data >>>")
    hf = h5py.File(fake_h5file_fullpath, 'r')
    fake_images = hf['fake_images'][:]
    fake_labels = hf['fake_labels'][:]
    hf.close()
else:
    fake_images = []
    fake_labels = []
    ## sample from cGAN without subsampling
    if not args.subsampling:
        for i in range(args.num_classes):
            print("\n Generate {} fake images for class {}/{}.".format(args.samp_nfake_per_class, i+1, args.num_classes))
            fake_images_i, fake_labels_i = fn_sampleGAN_given_label(args.samp_nfake_per_class, i, args.samp_batch_size)
            ### denormlize: [-1,1]--->[0,255]
            fake_images_i = (fake_images_i*0.5+0.5)*255.0
            fake_images_i = fake_images_i.astype(np.uint8)
            fake_labels_i = fake_labels_i.astype(int)
            fake_images.append(fake_images_i)
            fake_labels.append(fake_labels_i)
        # end for i
        del fake_images_i; gc.collect()
    else: ## sample from cGAN with subsampling
        for i in range(args.num_classes):
            print("\n Generate {} fake images for class {}/{}.".format(args.samp_nfake_per_class, i+1, args.num_classes))
            fake_images_i, fake_labels_i = fn_enhancedSampler_given_label(args.samp_nfake_per_class, i, args.samp_batch_size)
            ### denormlize: [-1,1]--->[0,255]
            fake_images_i = (fake_images_i*0.5+0.5)*255.0
            fake_images_i = fake_images_i.astype(np.uint8)
            fake_labels_i = fake_labels_i.astype(int)
            fake_images.append(fake_images_i)
            fake_labels.append(fake_labels_i)
        # end for i
        del fake_images_i; gc.collect()
    fake_images = np.concatenate(fake_images, axis=0)
    fake_labels = np.concatenate(fake_labels, axis=0)
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
    fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers)

    ## initialize pre-trained CNN for filtering
    filter_precnn_net = cnn_initialization(args.samp_filter_precnn_net, num_classes=args.num_classes, img_size=args.img_size)
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

fake_dataset_name = 'BigGAN_subsampling_{}_FilterCEPct_{}_nfake_{}_seed_{}'.format(args.subsampling, args.samp_filter_ce_percentile_threshold, nfake, args.seed)

fake_h5file_fullpath = os.path.join(save_fake_data_folder, 'Tiny-ImageNet_{}.h5'.format(fake_dataset_name))

if os.path.isfile(fake_h5file_fullpath):
    os.remove(fake_h5file_fullpath)

## dump fake iamges into h5 file
assert fake_images.max()>1
with h5py.File(fake_h5file_fullpath, "w") as f:
    f.create_dataset('fake_images', data = fake_images, dtype='uint8')
    f.create_dataset('fake_labels', data = fake_labels, dtype='int')

## histogram
fig = plt.figure()
ax = plt.subplot(111)
n, bins, patches = plt.hist(fake_labels, 100, density=False, facecolor='g', alpha=0.75)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Histogram of labels')
plt.grid(True)
#plt.show()
plt.savefig(os.path.join(save_fake_data_folder, '{}.png'.format(fake_dataset_name)))


## output some example fake images
n_row_show = 10
n_classes_show = 10
sel_classes = np.random.choice(np.arange(args.num_classes), size=n_classes_show, replace=True)
sel_classes = np.sort(sel_classes)
example_fake_images = []
for i in range(n_classes_show):
    class_i = sel_classes[i]
    indx_i = np.where(fake_labels==class_i)[0]
    np.random.shuffle(indx_i)
    indx_i = indx_i[0:n_row_show]
    example_fake_images_i = fake_images[indx_i]
    example_fake_images.append(example_fake_images_i)
example_fake_images = np.concatenate(example_fake_images, axis=0)
example_fake_images = example_fake_images / 255.0
example_fake_images = torch.from_numpy(example_fake_images)
save_image(example_fake_images.data, save_images_folder +'/example_fake_images_Tiny-ImageNet_{}.png'.format(fake_dataset_name), nrow=n_row_show, normalize=True)






print("\n ===================================================================================================")
