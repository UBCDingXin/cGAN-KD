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
import shutil


#----------------------------------------
from opts import gen_synth_data_opts
from utils import *
from models import *
from train_cnn import train_cnn, test_cnn
from train_cdre import train_cdre
from eval_metrics import compute_FID, compute_IS



#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)


## subsampling?
if args.subsampling:
    subsampling_method = "sampling_cDR-RS_precnn_{}_lambda_{:.3f}_DR_{}_lambda_{:.3f}".format(args.dre_precnn_net, args.dre_precnn_lambda, args.dre_net, args.dre_lambda)
else:
    subsampling_method = "sampling_None"

## filter??
if args.filter:
    subsampling_method = subsampling_method + "_filter_{}_perc_{:.2f}".format(args.samp_filter_precnn_net, args.samp_filter_ce_percentile_threshold)
else:
    subsampling_method = subsampling_method + "_filter_None"

## adjust labels??
subsampling_method = subsampling_method + "_adjust_{}".format(args.adjust)


# path_torch_home = os.path.join(args.root_path, 'torch_cache')
# os.makedirs(path_torch_home, exist_ok=True)
# os.environ['TORCH_HOME'] = path_torch_home

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
precnn_models_directory = os.path.join(args.root_path, 'output/precnn_models')
os.makedirs(precnn_models_directory, exist_ok=True)

output_directory = os.path.join(args.root_path, 'output/Setting_{}'.format(args.gan_net))
os.makedirs(output_directory, exist_ok=True)

save_models_folder = os.path.join(output_directory, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)

save_traincurves_folder = os.path.join(output_directory, 'training_curves')
os.makedirs(save_traincurves_folder, exist_ok=True)

save_evalresults_folder = os.path.join(output_directory, 'eval_results')
os.makedirs(save_evalresults_folder, exist_ok=True)

dump_fake_images_folder = os.path.join(args.root_path, 'fake_data')
os.makedirs(dump_fake_images_folder, exist_ok=True)



#######################################################################################
'''                                  Load Data                                      '''
#######################################################################################
## generate subset
cifar_trainset = torchvision.datasets.CIFAR100(root = args.data_path, train=True, download=True)
images_train = cifar_trainset.data
images_train = np.transpose(images_train, (0, 3, 1, 2))
labels_train = np.array(cifar_trainset.targets)

cifar_testset = torchvision.datasets.CIFAR100(root = args.data_path, train=False, download=True)

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
# train_means = [0.5,0.5,0.5]
# train_stds = [0.5,0.5,0.5]

images_test = cifar_testset.data
images_test = np.transpose(images_test, (0, 3, 1, 2))
labels_test = np.array(cifar_testset.targets)

print("\n Training set shape: {}x{}x{}x{}; Testing set shape: {}x{}x{}x{}.".format(images_train.shape[0], images_train.shape[1], images_train.shape[2], images_train.shape[3], images_test.shape[0], images_test.shape[1], images_test.shape[2], images_test.shape[3]))

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

if args.dre_transform:
    transform_dre = transforms.Compose([
                transforms.Resize(int(args.img_size*1.1)),
                transforms.RandomCrop(args.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]), ##do not use other normalization constants!!!
                ])
else:
    transform_dre = transforms.Compose([
                # transforms.RandomCrop((args.img_size, args.img_size), padding=4), ## note that some GAN training does not involve cropping!!!
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]), ##do not use other normalization constants!!!
                ])

# test set for cnn
transform_precnn_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(train_means, train_stds),
                ])
testset_precnn = IMGs_dataset(images_test, labels_test, transform=transform_precnn_test)
testloader_precnn = torch.utils.data.DataLoader(testset_precnn, batch_size=100, shuffle=False, num_workers=args.num_workers)



#######################################################################################
'''                  Load pre-trained GAN to Memory (not GPU)                       '''
#######################################################################################
ckpt_g = torch.load(args.gan_ckpt_path)
if args.gan_net=="BigGAN":
    netG = BigGAN_Generator(dim_z=args.gan_dim_g, resolution=args.img_size, G_attn='0', n_classes=args.num_classes, G_shared=False)
    netG.load_state_dict(ckpt_g)
    netG = nn.DataParallel(netG)
else:
    raise Exception("Not supported GAN!!")

def fn_sampleGAN_given_label(nfake, given_label, batch_size, pretrained_netG=netG, to_numpy=True, verbose=False):
    raw_fake_images = []
    raw_fake_labels = []
    pretrained_netG = pretrained_netG.cuda()
    pretrained_netG.eval()
    if verbose:
        pb = SimpleProgressBar()
    with torch.no_grad():
        tmp = 0
        while tmp < nfake:
            z = torch.randn(batch_size, args.gan_dim_g, dtype=torch.float).cuda()
            labels = (given_label*torch.ones(batch_size)).type(torch.long).cuda()
            batch_fake_images = pretrained_netG(z, labels)
            raw_fake_images.append(batch_fake_images.cpu())
            raw_fake_labels.append(labels.cpu().view(-1))
            tmp += batch_size
            if verbose:
                pb.update(np.min([float(tmp)*100/nfake,100]))

    raw_fake_images = torch.cat(raw_fake_images, dim=0)
    raw_fake_labels = torch.cat(raw_fake_labels)

    if to_numpy:
        raw_fake_images = raw_fake_images.numpy()
        raw_fake_labels = raw_fake_labels.numpy()

    return raw_fake_images[0:nfake], raw_fake_labels[0:nfake]



#######################################################################################
'''                                  DRE Training                                   '''
#######################################################################################
if args.subsampling:
    ##############################################
    ''' Pre-trained CNN for feature extraction '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Pre-trained CNN for feature extraction")
    # data loader
    trainset_dre_precnn = IMGs_dataset(images_train, labels_train, transform=transform_precnn_train)
    trainloader_dre_precnn = torch.utils.data.DataLoader(trainset_dre_precnn, batch_size=args.dre_precnn_batch_size_train, shuffle=True, num_workers=args.num_workers)
    # Filename
    filename_precnn_ckpt = precnn_models_directory + '/ckpt_PreCNNForDRE_{}_lambda_{}_epoch_{}_transform_{}_ntrain_{}_seed_{}.pth'.format(args.dre_precnn_net, args.dre_precnn_lambda, args.dre_precnn_epochs, args.dre_precnn_transform, args.ntrain, args.seed)
    print('\n' + filename_precnn_ckpt)

    path_to_ckpt_in_train = precnn_models_directory + '/ckpts_in_train_PreCNNForDRE_{}_lambda_{}_ntrain_{}_seed_{}'.format(args.dre_precnn_net, args.dre_precnn_lambda, args.ntrain, args.seed)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    # initialize cnn
    dre_precnn_net = cnn_extract_initialization(args.dre_precnn_net, num_classes=args.num_classes)
    num_parameters = count_parameters(dre_precnn_net)
    # training
    if not os.path.isfile(filename_precnn_ckpt):
        print("\n Start training CNN for feature extraction in the DRE >>>")
        dre_precnn_net = train_cnn(dre_precnn_net, 'PreCNNForDRE_{}'.format(args.dre_precnn_net), trainloader_dre_precnn, testloader_precnn, epochs=args.dre_precnn_epochs, resume_epoch=args.dre_precnn_resume_epoch, lr_base=args.dre_precnn_lr_base, lr_decay_factor=args.dre_precnn_lr_decay_factor, lr_decay_epochs=dre_precnn_lr_decay_epochs, weight_decay=args.dre_precnn_weight_decay, extract_feature=True, net_decoder=None, lambda_reconst=args.dre_precnn_lambda, train_means=train_means, train_stds=train_stds, path_to_ckpt = path_to_ckpt_in_train)

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

    ### dataloader
    trainset_dre = IMGs_dataset(images_train, labels_train, transform=transform_dre)
    trainloader_dre = torch.utils.data.DataLoader(trainset_dre, batch_size=args.dre_batch_size, shuffle=True, num_workers=args.num_workers)

    ### dr model filename
    drefile_fullpath = save_models_folder + "/ckpt_cDRE-F-cSP_precnn_{}_lambda_{:.3f}_DR_{}_lambda_{:.3f}_epochs_{}_ntrain_{}_seed_{}.pth".format(args.dre_precnn_net, args.dre_precnn_lambda, args.dre_net, args.dre_lambda, args.dre_epochs, args.ntrain, args.seed)
    print('\n' + drefile_fullpath)

    path_to_ckpt_in_train = save_models_folder + '/ckpt_cDRE-F-cSP_precnn_{}_lambda_{:.3f}_DR_{}_lambda_{:.3f}_ntrain_{}_seed_{}'.format(args.dre_precnn_net, args.dre_precnn_lambda, args.dre_net, args.dre_lambda, args.ntrain, args.seed)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    dre_loss_file_fullpath = save_traincurves_folder + '/train_loss_cDRE-F-cSP_precnn_{}_lambda_{:.3f}_DR_{}_epochs_{}_lambda_{}_ntrain_{}_seed_{}.png'.format(args.dre_precnn_net, args.dre_precnn_lambda, args.dre_net, args.dre_epochs, args.dre_lambda, args.ntrain, args.seed)

    ### dre training
    dre_net = cDR_MLP(args.dre_net, p_dropout=0.5, init_in_dim = args.num_channels*args.img_size*args.img_size, num_classes = args.num_classes).cuda()
    num_parameters_DR = count_parameters(dre_net)
    dre_net = nn.DataParallel(dre_net)
    #if DR model exists, then load the pretrained model; otherwise, start training the model.
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





#######################################################################################
'''                          Filtering by teacher                                   '''
#######################################################################################
if args.filter or args.adjust:
    ## initialize pre-trained CNN for filtering
    if args.samp_filter_precnn_net == "densenet121":
        filter_precnn_net = DenseNet121(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "ResNet18":
        filter_precnn_net = ResNet18(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "ResNet34":
        filter_precnn_net = ResNet34(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "ResNet50":
        filter_precnn_net = ResNet50(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "resnet56":
        filter_precnn_net = resnet56(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "resnet32x4":
        filter_precnn_net = resnet32x4(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "wrn_40_2":
        filter_precnn_net = wrn_40_2(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "vgg13":
        filter_precnn_net = vgg13_bn(num_classes=args.num_classes)
    elif args.samp_filter_precnn_net == "vgg19":
        filter_precnn_net = vgg19_bn(num_classes=args.num_classes)
    else:
        raise Exception("Not supported CNN for the filtering or adjustment!!!")
    ## load ckpt
    checkpoint = torch.load(args.samp_filter_precnn_net_ckpt_path)
    filter_precnn_net.load_state_dict(checkpoint['model'])
    # filter_precnn_net = nn.DataParallel(filter_precnn_net)    

    
    def fn_filter_or_adjust(fake_images, fake_labels, filter_precnn_net=filter_precnn_net, filter=args.filter, adjust=args.adjust, CE_cutoff_point=1e30, burnin_mode=False, verbose=False, visualize_filtered_images=False, filtered_images_path=None):
        #fake_images: numpy array
        #fake_labels: numpy array
        
        filter_precnn_net = filter_precnn_net.cuda()
        filter_precnn_net.eval()
        
        assert fake_images.max()>=1.0 and fake_images.min()>=0
        fake_dataset = IMGs_dataset(fake_images, fake_labels, transform=transform_precnn_train)
        fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=args.samp_filter_batch_size, shuffle=False)
        
        ## evaluate on fake data
        fake_CE_loss = []
        fake_labels_pred = []
        criterion = nn.CrossEntropyLoss(reduction='none')
        filter_precnn_net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        if verbose:
            pbar = tqdm(total=len(fake_images))
        with torch.no_grad():
            correct = 0
            total = 0
            for _, (images, labels) in enumerate(fake_dataloader):
                images = images.type(torch.float).cuda()
                labels = labels.type(torch.long).cuda()
                outputs = filter_precnn_net(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                fake_labels_pred.append(predicted.cpu().numpy())
                fake_CE_loss.append(loss.cpu().numpy())
                if verbose:
                    pbar.update(len(images))
            if verbose:
                print('\n Test Accuracy of {} on the {} fake images: {:.3f} %'.format(args.samp_filter_precnn_net, len(fake_images), 100.0 * correct / total))
        fake_CE_loss = np.concatenate(fake_CE_loss)
        fake_labels_pred = np.concatenate(fake_labels_pred)
        
        if not burnin_mode:
            indx_label_diff = np.where(np.abs(fake_labels_pred-fake_labels)>1e-10)[0]
            print('\r Class {}: adjust the labels of {}/{} images before the filtering.'.format(int(fake_labels[0]+1), len(indx_label_diff), len(fake_labels)))
        
        filter_precnn_net = filter_precnn_net.cpu()
        
        if filter:
            if burnin_mode:
                CE_cutoff_point = np.quantile(fake_CE_loss, q=args.samp_filter_ce_percentile_threshold)
                print("\r Cut-off point for the filtering is {:.4f}. Test Accuracy of {} on the {} fake images: {:.3f} %.".format(CE_cutoff_point, args.samp_filter_precnn_net, len(fake_images), 100.0 * correct / total))
                return CE_cutoff_point
            
            if visualize_filtered_images: ## show some exampled filtered images
                indx_drop = np.where(fake_CE_loss>=CE_cutoff_point)[0]
                example_filtered_images = fake_images[indx_drop]
                n_row = int(np.sqrt(min(100, len(example_filtered_images))))
                example_filtered_images = example_filtered_images[0:n_row**2]        
                example_filtered_images = torch.from_numpy(example_filtered_images)
                if example_filtered_images.max()>1.0:
                    example_filtered_images = example_filtered_images/255.0
                                    
                filename_filtered_images = filtered_images_path + '/class_{}_dropped.png'.format(int(fake_labels[0]+1))
                save_image(example_filtered_images.data, filename_filtered_images, nrow=n_row, normalize=True)
                
            
            ## do the filtering
            indx_sel = np.where(fake_CE_loss<CE_cutoff_point)[0]
            fake_images = fake_images[indx_sel]
            fake_labels = fake_labels[indx_sel]
            fake_labels_pred = fake_labels_pred[indx_sel] #adjust the labels of fake data by using the pre-trained big CNN        
            
            
            if visualize_filtered_images: ## show kept images as reference
                example_selected_images = fake_images[0:n_row**2]        
                example_selected_images = torch.from_numpy(example_selected_images)
                if example_selected_images.max()>1.0:
                    example_selected_images = example_selected_images/255.0
                                    
                filename_filtered_images = filtered_images_path + '/class_{}_selected.png'.format(fake_labels[0])
                save_image(example_selected_images.data, filename_filtered_images, nrow=n_row, normalize=True)
            
            if not burnin_mode:
                indx_label_diff = np.where(np.abs(fake_labels_pred-fake_labels)>1e-10)[0]
                print('\r Class {}: adjust the labels of {}/{} images after the filtering.'.format(int(fake_labels[0]+1), len(indx_label_diff), len(fake_labels)))

        if adjust:
            return fake_images, fake_labels_pred
        else:
            return fake_images, fake_labels





#######################################################################################
'''                               Final Sampler                                     '''
#######################################################################################
def fn_final_sampler(nfake, label_i, batch_size=args.samp_batch_size, split_div=2, filter_nburnin=args.samp_filter_burnin_size, verbose=False, visualize_filtered_images=False, filtered_images_path=None):
    if verbose:
        pbar = tqdm(total=nfake)
        
    ### burning stage: compute the cut off point for the filtering
    if args.filter:
        if args.subsampling:
            brunin_images, brunin_labels = fn_enhancedSampler_given_label(nfake=filter_nburnin, given_label=label_i, batch_size=batch_size, verbose=False)
        else:
            brunin_images, brunin_labels = fn_sampleGAN_given_label(nfake=filter_nburnin, given_label=label_i, batch_size=batch_size, verbose=False)
        ## denormalize images
        brunin_images = (brunin_images*0.5+0.5)*255.0
        brunin_images = brunin_images.astype(np.uint8)
        
        ## compute the cut-off point
        CE_cutoff_point = fn_filter_or_adjust(brunin_images, brunin_labels, burnin_mode=True, verbose=False)
        
    fake_images_i = []
    fake_labels_i = []
    num_got = 0
    while num_got<nfake:
        if args.subsampling:
            batch_images, batch_labels = fn_enhancedSampler_given_label(nfake=nfake//split_div, given_label=label_i, batch_size=batch_size, verbose=False)
        else:
            batch_images, batch_labels = fn_sampleGAN_given_label(nfake=nfake//split_div, given_label=label_i, batch_size=batch_size, verbose=False)
        ## denormalize images
        batch_images = (batch_images*0.5+0.5)*255.0
        batch_images = batch_images.astype(np.uint8)
        
        ## filtering and adjustment
        if args.filter or args.adjust:
            batch_images, batch_labels = fn_filter_or_adjust(fake_images=batch_images, fake_labels=batch_labels, CE_cutoff_point=CE_cutoff_point, verbose=False, visualize_filtered_images=visualize_filtered_images, filtered_images_path=filtered_images_path)

        num_got += len(batch_images)
        fake_images_i.append(batch_images)
        fake_labels_i.append(batch_labels)
        
        if verbose:
            pbar.update(len(batch_images))
    ##end while
    fake_images_i = np.concatenate(fake_images_i, axis=0)
    fake_labels_i = np.concatenate(fake_labels_i, axis=0)
    fake_images_i = fake_images_i[0:nfake]
    fake_labels_i = fake_labels_i[0:nfake]
    return fake_images_i, fake_labels_i


###############################################################################
'''                             Generate fake data                          '''
###############################################################################
print("\n Geneating fake data: {}+{}...".format(args.gan_net, subsampling_method))

### generate fake images
dump_fake_images_filename = os.path.join(dump_fake_images_folder, 'cifar100_fake_images_{}_{}_NfakePerClass_{}_seed_{}.h5'.format(args.gan_net, subsampling_method, args.samp_nfake_per_class, args.seed))
print(dump_fake_images_filename)

if args.visualize_filtered_images:
    dump_filtered_fake_images_folder = os.path.join(dump_fake_images_folder, 'cifar100_fake_images_{}_{}_NfakePerClass_{}_seed_{}_example_filtered_images'.format(args.gan_net, subsampling_method, args.samp_nfake_per_class, args.seed))
    os.makedirs(dump_filtered_fake_images_folder, exist_ok=True)
else:
    dump_filtered_fake_images_folder=None

if not os.path.isfile(dump_fake_images_filename):
    print('\n Start generating fake data...')
    fake_images = []
    fake_labels = []
    start_time = timeit.default_timer()
    for i in range(args.num_classes):
        print("\n Generate {} fake images for class {}/{}.".format(args.samp_nfake_per_class, i+1, args.num_classes))
        fake_images_i, fake_labels_i = fn_final_sampler(nfake=args.samp_nfake_per_class, label_i=i, visualize_filtered_images=args.visualize_filtered_images, filtered_images_path=dump_filtered_fake_images_folder)
        fake_images.append(fake_images_i)
        fake_labels.append(fake_labels_i.reshape(-1))
        print("\n End generating {} fake images for class {}/{}. Time elapse: {:.3f}".format(args.samp_nfake_per_class, i+1, args.num_classes, timeit.default_timer()-start_time))
    fake_images = np.concatenate(fake_images, axis=0)
    fake_labels = np.concatenate(fake_labels, axis=0)
    del fake_images_i, fake_labels_i; gc.collect()
    print('\n End generating fake data!')

    with h5py.File(dump_fake_images_filename, "w") as f:
        f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=6)
        f.create_dataset('fake_labels', data = fake_labels, dtype='int')
else:
    print('\n Start loading generated fake data...')
    with h5py.File(dump_fake_images_filename, "r") as f:
        fake_images = f['fake_images'][:]
        fake_labels = f['fake_labels'][:]
assert len(fake_images) == len(fake_labels)

### visualize data distribution
frequencies = []
for i in range(args.num_classes):
    indx_i = np.where(fake_labels==i)[0]
    frequencies.append(len(indx_i))
frequencies = np.array(frequencies)
width = 0.8
x = np.arange(1,args.num_classes+1)
# plot data in grouped manner of bar type
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.grid(color='lightgrey', linestyle='--', zorder=0)
ax.bar(x, frequencies, width, align='center', color='tab:green', zorder=3)
ax.set_xlabel("Class")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(dump_fake_images_folder, "cifar100_fake_images_{}_{}_NfakePerClass_{}_class_dist.pdf".format(args.gan_net, subsampling_method, args.samp_nfake_per_class)))
plt.close()

print('\n Frequence of each class: MIN={}, MEAN={}, MAX={}.'.format(np.min(frequencies),np.mean(frequencies),np.max(frequencies)))




###############################################################################
'''                             Compute FID and IS                          '''
###############################################################################
if args.eval:
    #load pre-trained InceptionV3 (pretrained on CIFAR-100)
    PreNetFID = Inception3(num_classes=args.num_classes, aux_logits=True, transform_input=False)
    checkpoint_PreNet = torch.load(args.eval_ckpt_path)
    PreNetFID = nn.DataParallel(PreNetFID).cuda()
    PreNetFID.load_state_dict(checkpoint_PreNet['net_state_dict'])    

    ##############################################
    ''' Compute FID between real and fake images '''
    start = timeit.default_timer()
    
    ## normalize images
    assert fake_images.max()>1
    fake_images = (fake_images/255.0-0.5)/0.5
    assert images_train.max()>1
    images_train = (images_train/255.0-0.5)/0.5
    assert -1.0<=images_train.max()<=1.0 and -1.0<=images_train.min()<=1.0
    
    #####################
    ## Compute Intra-FID: real vs fake
    print("\n Start compute Intra-FID between real and fake images...")
    start_time = timeit.default_timer()
    intra_fid_scores = np.zeros(args.num_classes)
    for i in range(args.num_classes):
        indx_train_i = np.where(labels_train==i)[0]
        images_train_i = images_train[indx_train_i]
        indx_fake_i = np.where(fake_labels==i)[0]
        fake_images_i = fake_images[indx_fake_i]
        ##compute FID within each class
        intra_fid_scores[i] = compute_FID(PreNetFID, images_train_i, fake_images_i, batch_size = args.eval_FID_batch_size, resize = (299, 299))
        print("\r Evaluating: Class:{}; Real:{}; Fake:{}; FID:{}; Time elapses:{}s.".format(i+1, len(images_train_i), len(fake_images_i), intra_fid_scores[i], timeit.default_timer()-start_time))
    ##end for i
    # average over all classes
    print("\n Evaluating: Intra-FID: {}({}); min/max: {}/{}.".format(np.mean(intra_fid_scores), np.std(intra_fid_scores), np.min(intra_fid_scores), np.max(intra_fid_scores)))

    # dump FID versus class to npy
    dump_fids_filename = save_evalresults_folder + "/{}_subsampling_{}_fids".format(args.gan_net, subsampling_method)
    np.savez(dump_fids_filename, fids=intra_fid_scores)

    #####################
    ## Compute FID: real vs fake
    print("\n Start compute FID between real and fake images...")
    indx_shuffle_real = np.arange(len(images_train)); np.random.shuffle(indx_shuffle_real)
    indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
    fid_score = compute_FID(PreNetFID, images_train[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = args.eval_FID_batch_size, resize = (299, 299))
    print("\n Evaluating: FID between {} real and {} fake images: {}.".format(len(images_train), len(fake_images), fid_score))
    
    #####################
    ## Compute IS
    print("\n Start compute IS of fake images...")
    indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
    is_score, is_score_std = compute_IS(PreNetFID, fake_images[indx_shuffle_fake], batch_size = args.eval_FID_batch_size, splits=10, resize=(299,299))
    print("\n Evaluating: IS of {} fake images: {}({}).".format(len(fake_images), is_score, is_score_std))

    #####################
    # Dump evaluation results
    eval_results_fullpath = os.path.join(save_evalresults_folder, '{}_subsampling_{}.txt'.format(args.gan_net, subsampling_method))
    if not os.path.isfile(eval_results_fullpath):
        eval_results_logging_file = open(eval_results_fullpath, "w")
        eval_results_logging_file.close()
    with open(eval_results_fullpath, 'a') as eval_results_logging_file:
        eval_results_logging_file.write("\n===================================================================================================")
        eval_results_logging_file.write("\n Separate results for Subsampling {} \n".format(subsampling_method))
        print(args, file=eval_results_logging_file)
        eval_results_logging_file.write("\n Intra-FID: {}({}); min/max: {}/{}.".format(np.mean(intra_fid_scores), np.std(intra_fid_scores), np.min(intra_fid_scores), np.max(intra_fid_scores)))
        eval_results_logging_file.write("\n FID: {}.".format(fid_score))
        eval_results_logging_file.write("\n IS: {}({}).".format(is_score, is_score_std))    


#######################################################################################
'''               Visualize fake images of the trained GAN                          '''
#######################################################################################
if args.visualize_fake_images:
    
    # First, visualize conditional generation # vertical grid
    ## 10 rows; 10 columns (10 samples for each class)
    n_row = args.num_classes
    n_col = 10

    fake_images_view = []
    fake_labels_view = []
    for i in range(args.num_classes):
        fake_labels_i = i*np.ones(n_col)
        if args.subsampling:
            fake_images_i, _ = fn_enhancedSampler_given_label(nfake=n_col, given_label=i, batch_size=100, verbose=False)
        else:
            fake_images_i, _ = fn_sampleGAN_given_label(nfake=n_col, given_label=i, batch_size=100, pretrained_netG=netG, to_numpy=True)
        fake_images_view.append(fake_images_i)
        fake_labels_view.append(fake_labels_i)
    ##end for i
    fake_images_view = np.concatenate(fake_images_view, axis=0)
    fake_labels_view = np.concatenate(fake_labels_view, axis=0)

    ### output fake images from a trained GAN
    filename_fake_images = save_evalresults_folder + '/{}_subsampling_{}_fake_image_grid_{}x{}.png'.format(args.gan_net, subsampling_method, n_row, n_col)
    
    images_show = np.zeros((n_row*n_col, args.num_channels, args.img_size, args.img_size))
    for i_row in range(n_row):
        indx_i = np.where(fake_labels_view==i_row)[0]
        for j_col in range(n_col):
            curr_image = fake_images_view[indx_i[j_col]]
            images_show[i_row*n_col+j_col,:,:,:] = curr_image
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=True)

### end if args.visualize_fake_images


print("\n ===================================================================================================")