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

## Requirements
argparse>=1.1, h5py>=2.10.0, matplotlib>=3.2.1, numpy>=1.18.5, Pillow>=7.0.0, python=3.8.5, torch>=1.5.0, torchvision>=0.6.0,
tqdm>=4.46.1


-------------------------------

## Datasets (h5 files) and necessary checkpoints

### CIFAR-100


### ImageNet-100


### Steering Angle


### UTKFace






-------------------------------
## Sample Usage

### CIFAR-100
#### BigGAN training
The implementation of BigGAN is mainly based on [3].  <br />
Run `./CIFAR-100/BigGAN/scripts/launch_cifar100_ema.sh`.  <br />
Checkpoints of BigGAN used in our experiments are in `cGAN-KD_data_and_ckpts.7z`.  <br />

#### Fake data generation


### ImageNet-100


### Steering Angle


### UTKFace




-------------------------------
## Some Results
* **ImageNet-100**
<p align="center">
  <img src="images/ImageNet100_main_results.png">
</p>

* **Steering Angle and UTKFace**
<p align="center">
  <img src="images/steeringangle_and_utkface_main_results.png">
</p>



* **Ablation Study: CIFAR-100**
<p align="center">
  <img src="images/cifar100_ablation_effect_of_components_grouped_error.png">
  <img src="images/cifar100_ablation_error_vs_nfake.png">
</p>


* **Ablation Study: Steering Angle**
<p align="center">
  <img src="images/steeringangle_ablation_effect_of_components_grouped.png">
  <img src="images/steeringangle_ablation_error_vs_nfake.png">
</p>


-------------------------------
## References
[1] X. Ding, Y. Wang, Z. Xu, W. J. Welch, and Z. J. Wang, “CcGAN: Continuous conditional generative adversarial networks for image generation,” in International Conference on Learning Representations, 2021.  <br />
[2] X. Ding, Y. Wang, Z. Xu, W. J. Welch, and Z. J. Wang, “Continuous conditional generative adversarial networks for image generation: Novel losses and label input mechanisms,” arXiv preprint arXiv:2011.07466, 2020. https://github.com/UBCDingXin/improved_CcGAN  <br />
[3] https://github.com/ajbrock/BigGAN-PyTorch <br />
[4] X. Ding, Y. Wang, Z. J. Wang, and W. J. Welch, "Efficient Subsampling of Realistic Images From GANs Conditional on a Class or a Continuous Variable." arXiv preprint arXiv:2103.11166v5 (2022). https://github.com/UBCDingXin/cDR-RS
