#!/bin/bash

ROOT_PATH="./SteeringAngle"
DATA_PATH="./datasets/SteeringAngle"

FAKE_DATA_PATH="${ROOT_PATH}/output/fake_data/steeringangle_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_vgg19_perc_0.90_adjust_False_Nlabel_2000_NFakePerLabel_50_seed_2020.h5"
NFAKE=50000


ARCH="resnet20"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_Nfake_${NFAKE}.txt

ARCH="resnet56"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_Nfake_${NFAKE}.txt


ARCH="wrn_16_1"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_Nfake_${NFAKE}.txt



ARCH="wrn_40_1"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_Nfake_${NFAKE}.txt



ARCH="resnet8x4"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_Nfake_${NFAKE}.txt



ARCH="ShuffleV1"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_Nfake_${NFAKE}.txt


ARCH="MobileNetV2"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_Nfake_${NFAKE}.txt
