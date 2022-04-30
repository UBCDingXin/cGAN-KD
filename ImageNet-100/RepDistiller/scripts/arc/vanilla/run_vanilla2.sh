#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=I_vanilla2
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_old.req

ROOT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/RepDistiller"
DATA_PATH="/scratch/dingx92/datasets/ImageNet-100"


MODEL="resnet20"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="resnet8x4"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="MobileNetV2"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="vgg8"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="ShuffleV1"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="ShuffleV2"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="efficientnetb0"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="vgg11"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="ResNet18"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt


MODEL="ResNet34"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    2>&1 | tee output_${MODEL}_vanilla.txt
