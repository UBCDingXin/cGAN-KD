# Distilling and Transferring Knowledge via cGAN-generated Samples for Image Classification and Regression

This repository provides the source codes for the experiments in our paper. <br />
If you use this code, please cite
```text

@misc{ding2021distilling,
      title={Distilling and Transferring Knowledge via cGAN-generated Samples for Image Classification and Regression},
      author={Xin Ding and Z. Jane Wang and Zuheng Xu and Yongwei Wang and William J. Welch},
      year={2022},
      eprint={2104.03164v3},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```


<p align="center">
  <img src="images/workflow_cGAN-based_KD.png">
  The workflow of cGAN-KD.
</p>


<p align="center">
  <img src="images/evolution_of_fake_distribution.png">
  Evolution of fake samples' distributions and datasets.
</p>

-------------------------------
## To Do List

- [x] CIFAR-100
- [] ImageNet-100
- [] Steering Angle
- [] UTKFace


-------------------------------

## 1. Requirements
argparse>=1.1, h5py>=2.10.0, matplotlib>=3.2.1, numpy>=1.18.5, Pillow>=7.0.0, python=3.8.5, torch>=1.5.0, torchvision>=0.6.0,
tqdm>=4.46.1


-------------------------------

## 2. Datasets (h5 files) and necessary checkpoints

### 2.1. CIFAR-100
Download `eval_and_gan_ckpts.7z`:  <br />
https://1drv.ms/u/s!Arj2pETbYnWQuqt036MJ2KdVMKRXAw?e=4mo5SI <br />
Unzip `eval_and_gan_ckpts.7z` you will get `eval_and_gan_ckpts`. Then, put `eval_and_gan_ckpts` under `./CIFAR-100` <br />

### 2.2. ImageNet-100


### 2.3. Steering Angle


### 2.4. UTKFace






-------------------------------
## 3. Sample Usage

**Remember to correctly set all paths in .sh files correctly!**  <br />

### 3.1. CIFAR-100
#### BigGAN training
The implementation of BigGAN is mainly based on [3].  <br />
Run `./CIFAR-100/BigGAN/scripts/launch_cifar100_ema.sh`.  <br />
Checkpoints of BigGAN used in our experiments are already in `cGAN-KD_data_and_ckpts.7z`.  <br />

#### Fake data generation
Run `./CIFAR-100/make_fake_datasets/scripts/run.sh`.  <br />

#### Train cnns without KD
Run `./CIFAR-100/RepDistiller/scripts/vanilla/run_vanilla.sh`

#### Implement existing KD except SSKD, ReviewKD and TAKD



#### Implement TAKD



#### Implement SSKD



#### Implement ReviewKD



#### Implement cGAN-KD-based methods








### 3.2. ImageNet-100


### 3.3. Steering Angle


### 3.4. UTKFace




-------------------------------
## 4. Some Results
### 4.1. ImageNet-100
<p align="center">
  <img src="images/ImageNet100_main_results.png">
</p>

### 4.2. Steering Angle and UTKFace
<p align="center">
  <img src="images/steeringangle_and_utkface_main_results.png">
</p>



### 4.3. Ablation Study: CIFAR-100
<p align="center">
  <img src="images/cifar100_ablation_effect_of_components_grouped_error.png">
  <img src="images/cifar100_ablation_error_vs_nfake.png">
</p>


### 4.4. Ablation Study: Steering Angle
<p align="center">
  <img src="images/steeringangle_ablation_effect_of_components_grouped.png">
  <img src="images/steeringangle_ablation_error_vs_nfake.png">
</p>


-------------------------------
## 5. References
[1] X. Ding, Y. Wang, Z. Xu, W. J. Welch, and Z. J. Wang, “CcGAN: Continuous conditional generative adversarial networks for image generation,” in International Conference on Learning Representations, 2021.  <br />
[2] X. Ding, Y. Wang, Z. Xu, W. J. Welch, and Z. J. Wang, “Continuous conditional generative adversarial networks for image generation: Novel losses and label input mechanisms,” arXiv preprint arXiv:2011.07466, 2020. https://github.com/UBCDingXin/improved_CcGAN  <br />
[3] https://github.com/ajbrock/BigGAN-PyTorch <br />
[4] X. Ding, Y. Wang, Z. J. Wang, and W. J. Welch, "Efficient Subsampling of Realistic Images From GANs Conditional on a Class or a Continuous Variable." arXiv preprint arXiv:2103.11166v5 (2022). https://github.com/UBCDingXin/cDR-RS
