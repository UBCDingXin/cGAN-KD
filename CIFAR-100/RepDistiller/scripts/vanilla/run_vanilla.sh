#!/bin/bash

ROOT_PATH="./CIFAR-100/RepDistiller"
DATA_PATH="./datasets/CIFAR-100/data"


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
