#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-01:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out


module load arch/avx2 StdEnv/2020
module load cuda/11.0
module load python/3.9.6
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req

ROOT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/RepDistiller"
DATA_PATH="/scratch/dingx92/datasets/ImageNet-100"

FILTER_NET="vgg19"
PERC="0.90"
FAKE_DATA_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/make_fake_datasets/fake_data/cifar100_fake_images_BigGAN_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_${FILTER_NET}_perc_${PERC}_adjust_True_NfakePerClass_3000_seed_2021.h5"
NFAKE=100000


MODEL="resnet20"
INIT_MODEL_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${MODEL}_epoch_240_last.pth"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_MODEL_PATH \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size 128 --learning_rate 0.01 --lr_decay_epochs "150,180,210" \
    2>&1 | tee output_${MODEL}_vanilla_fake_finetune_${FILTER_NET}_${PERC}_${NFAKE}.txt