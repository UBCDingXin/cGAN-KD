#!/bin/bash

ROOT_PATH="./ImageNet-100/SSKD"
DATA_PATH="./datasets/ImageNet-100"


ARCH="wrn_40_2"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="resnet110"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="vgg13"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="vgg19"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="ResNet34"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt


ARCH="resnet32x4"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="ResNet50"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="vgg13"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt


ARCH="resnet56"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt


ARCH="wrn_40_1"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="resnet20"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="vgg8"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="resnet8x4"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="MobileNetV2"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="ShuffleV1"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="ShuffleV2"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="wrn_16_2"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 20 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt
