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





-------------------------------
## Sample Usage





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
