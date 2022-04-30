#!/bin/bash

ROOT_PATH="./UTKFace"
DATA_PATH="./datasets/UTKFace"


ARCH="resnet20"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="resnet56"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="resnet110"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="resnet8x4"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="resnet32x4"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="ResNet18"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="ResNet34"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="ResNet50"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt



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





ARCH="wrn_16_1"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="wrn_16_2"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="wrn_40_1"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="wrn_40_2"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt





ARCH="densenet121"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="densenet161"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="densenet169"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt

ARCH="densenet201"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --epochs 350 --resume_epoch 0 --save_freq 50 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    2>&1 | tee output_${ARCH}_vanilla.txt
