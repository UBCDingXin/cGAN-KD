#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=nI_SK_VA_F
#SBATCH --output=%x-%j.out


module load arch/avx2 StdEnv/2020
module load cuda/11.0
module load python/3.9.6
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req

ROOT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD"
DATA_PATH="/scratch/dingx92/datasets/ImageNet-100"
FAKE_DATA_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/make_fake_datasets/fake_data/cifar100_fake_images_BigGAN_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_densenet121_perc_0.90_adjust_True_NfakePerClass_5000_seed_2021.h5"
NFAKE=100000

ARCH="wrn_40_2"
INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="resnet32x4"
INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="ResNet50"
INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="vgg13"
INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="vgg19"
INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla.txt

# ARCH="resnet56"
# INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
# python teacher.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
#     --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_model_path $INIT_PATH \
#     2>&1 | tee output_${ARCH}_vanilla.txt


# ARCH="wrn_40_1"
# INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
# python teacher.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
#     --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_model_path $INIT_PATH \
#     2>&1 | tee output_${ARCH}_vanilla.txt

# ARCH="resnet20"
# INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
# python teacher.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
#     --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_model_path $INIT_PATH \
#     2>&1 | tee output_${ARCH}_vanilla.txt

# ARCH="vgg8"
# INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
# python teacher.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
#     --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_model_path $INIT_PATH \
#     2>&1 | tee output_${ARCH}_vanilla.txt

# ARCH="resnet8x4"
# INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
# python teacher.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
#     --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_model_path $INIT_PATH \
#     2>&1 | tee output_${ARCH}_vanilla.txt

# ARCH="MobileNetV2"
# INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
# python teacher.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
#     --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_model_path $INIT_PATH \
#     2>&1 | tee output_${ARCH}_vanilla.txt

# ARCH="ShuffleV1"
# INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
# python teacher.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
#     --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_model_path $INIT_PATH \
#     2>&1 | tee output_${ARCH}_vanilla.txt

# ARCH="ShuffleV2"
# INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
# python teacher.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
#     --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_model_path $INIT_PATH \
#     2>&1 | tee output_${ARCH}_vanilla.txt

# ARCH="wrn_16_2"
# INIT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
# python teacher.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
#     --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_model_path $INIT_PATH \
#     2>&1 | tee output_${ARCH}_vanilla.txt