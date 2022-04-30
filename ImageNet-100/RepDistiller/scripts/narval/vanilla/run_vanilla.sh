#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=I_vanilla
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


MODEL="wrn_40_2"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="resnet56"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="resnet32x4"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="ResNet50"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="resnet110"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="vgg13"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="vgg19"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="densenet121"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt

MODEL="wrn_16_2"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="wrn_40_1"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


# MODEL="resnet20"
# python train_teacher.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
#     2>&1 | tee output_${MODEL}_vanilla.txt


# MODEL="resnet8x4"
# python train_teacher.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
#     2>&1 | tee output_${MODEL}_vanilla.txt


# MODEL="MobileNetV2"
# python train_teacher.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
#     2>&1 | tee output_${MODEL}_vanilla.txt


# MODEL="vgg8"
# python train_teacher.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
#     2>&1 | tee output_${MODEL}_vanilla.txt


# MODEL="ShuffleV1"
# python train_teacher.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
#     2>&1 | tee output_${MODEL}_vanilla.txt


# MODEL="ShuffleV2"
# python train_teacher.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
#     2>&1 | tee output_${MODEL}_vanilla.txt


# MODEL="efficientnetb0"
# python train_teacher.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
#     2>&1 | tee output_${MODEL}_vanilla.txt


# MODEL="vgg11"
# python train_teacher.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
#     2>&1 | tee output_${MODEL}_vanilla.txt


# MODEL="ResNet18"
# python train_teacher.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
#     2>&1 | tee output_${MODEL}_vanilla.txt


# MODEL="ResNet34"
# python train_teacher.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
#     2>&1 | tee output_${MODEL}_vanilla.txt
