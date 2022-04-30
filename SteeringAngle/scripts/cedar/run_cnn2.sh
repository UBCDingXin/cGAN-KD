#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=SA_cnn2
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_old.req

ROOT_PATH="/scratch/dingx92/cGAN-KD/SteeringAngle"
DATA_PATH="/scratch/dingx92/datasets/SteeringAngle/regression"



ARCH="vgg8"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="vgg11"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="vgg13"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="vgg16"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="vgg19"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt





ARCH="ShuffleV1"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="ShuffleV2"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="MobileNetV2"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="efficientnetb0"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt
