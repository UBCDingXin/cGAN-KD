# cGAN-KD: Distill and Transfer Knowledge via cGAN-generated Samples

This repository provides the source codes for the experiments in our paper on the CIFAR-10 and RC-49 datasets. <br />
If you use this code, please cite
```text

TO BE DONE...

```


<p align="center">
  <img src="images/workflow_cGAN-based_KD.png">
  The workflow of cGAN-KD.
</p>


-------------------------------

## Requirements
argparse>=1.1 <br />
h5py>=2.10.0 <br />
matplotlib>=3.2.1 <br />
numpy>=1.18.5 <br />
Pillow>=7.0.0 <br />
python=3.8.5 <br />
torch>=1.5.0 <br />
torchvision>=0.6.0 <br />
tqdm>=4.46.1 <br />



-------------------------------

## Datasets (h5 files) and necessary checkpoints
Download and unzip `cGAN-KD_data_and_ckpts.7z`:  <br />
https://1drv.ms/u/s!Arj2pETbYnWQsdtpYbSbbb7ntilamQ?e=AKaVHF <br />

Put `./C-X0K/CIFAR10_trainset_X0000_seed_2020.h5` at `./CIFAR_X0K/cGAN-based_KD/data/`. <br />
Put `./C-X0K/C10_2020.hdf5` at `./CIFAR_X0K/BigGAN/data/`. <br />
Put `./C-X0K/UDA_pretrained_teachers/*.pth` at `./CIFAR_X0K/Distiller/pretrained/`. <br />
Put `./C-X0K/ckpt_BigGAN_cifar10_ntrain_X0000_seed_2020` at `./CIFAR_X0K/cGAN-based_KD/Output_CIFAR10/saved_models/`. <br />
X stands for 5, 2, 1, representing C-50K, C-20K, and C-10K respectively. <br />

Put `./RC-49/dataset` at `./RC-49`. <br />
Put `./RC-49/output` at `./RC-49/CcGAN-based_KD`. <br />

-------------------------------
## Sample Usage
### CIFAR-10
We only take C-50K as an example to show how to conduct the experiment on CIFAR-10.

#### BigGAN training


####


### RC-49



-------------------------------
## Some Results
* **CIFAR-10**
<p align="center">
  <img src="images/cifar10_main_results.png">
  <img src="images/cifar10_main_SSKD.png">
  <img src="images/cifar10_main_UDA.png">
  <img src="images/cifar_ablation_effect_of_components_VGG11.png">
  <img src="images/cifar_ablation_error_vs_nfake.png">
</p>

* **RC-49**
<p align="center">
  <img src="images/rc49_main_results.png">
  <img src="images/rc49_ablation_effect_of_components_ShuffleNet.png">
  <img src="images/rc49_ablation_error_vs_nfake.png">
</p>
