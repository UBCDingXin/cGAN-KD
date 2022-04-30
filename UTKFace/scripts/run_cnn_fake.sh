#!/bin/bash

ROOT_PATH="./UTKFace"
DATA_PATH="./datasets/UTKFace"


FILTER_NET="vgg11"
PERC="0.70"
FAKE_DATA_PATH="./UTKFace/output/fake_data/utkface_fake_images_SAGAN_cDR-RS_presae_epochs_200_sparsity_0.001_regre_1.000_DR_MLP5_epochs_200_lambda_0.010_kappa_-6.0_filter_${FILTER_NET}_perc_${PERC}_adjust_True_NfakePerLable_10000_seed_2020.h5"
NFAKE_PER_LABEL=1000


ARCH="resnet20"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake_per_label $NFAKE_PER_LABEL \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_180_210" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_NfakePerLabel_${NFAKE_PER_LABEL}.txt


ARCH="wrn_40_1"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake_per_label $NFAKE_PER_LABEL \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_180_210" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_NfakePerLabel_${NFAKE_PER_LABEL}.txt


ARCH="wrn_16_1"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake_per_label $NFAKE_PER_LABEL \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_180_210" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_NfakePerLabel_${NFAKE_PER_LABEL}.txt


ARCH="resnet56"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake_per_label $NFAKE_PER_LABEL \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_180_210" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_NfakePerLabel_${NFAKE_PER_LABEL}.txt


ARCH="MobileNetV2"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake_per_label $NFAKE_PER_LABEL \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_180_210" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_NfakePerLabel_${NFAKE_PER_LABEL}.txt


ARCH="ShuffleV1"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake_per_label $NFAKE_PER_LABEL \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_180_210" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${ARCH}_NfakePerLabel_${NFAKE_PER_LABEL}.txt
