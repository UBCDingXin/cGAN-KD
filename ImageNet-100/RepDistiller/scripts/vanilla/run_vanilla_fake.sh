#!/bin/bash

ROOT_PATH="./ImageNet-100/RepDistiller"
DATA_PATH="./datasets/ImageNet-100"

FILTER_NET="densenet161"
PERC="0.90"
FAKE_DATA_PATH="./ImageNet-100/make_fake_datasets/fake_data/imagenet100_fake_images_BigGANdeep_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_${FILTER_NET}_perc_${PERC}_adjust_False_NfakePerClass_3000_seed_2021.h5"
NFAKE=100000



MODEL="resnet20"
INIT_MODEL_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${MODEL}_epoch_240_last.pth"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_MODEL_PATH \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size 256 --learning_rate 0.01 --lr_decay_epochs "150,180,210" \
    2>&1 | tee output_${MODEL}_vanilla_fake_finetune_${FILTER_NET}_${PERC}_${NFAKE}.txt


MODEL="wrn_40_1"
INIT_MODEL_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${MODEL}_epoch_240_last.pth"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_MODEL_PATH \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size 256 --learning_rate 0.01 --lr_decay_epochs "150,180,210" \
    2>&1 | tee output_${MODEL}_vanilla_fake_finetune_${FILTER_NET}_${PERC}_${NFAKE}.txt


MODEL="wrn_16_2"
INIT_MODEL_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${MODEL}_epoch_240_last.pth"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_MODEL_PATH \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size 256 --learning_rate 0.01 --lr_decay_epochs "150,180,210" \
    2>&1 | tee output_${MODEL}_vanilla_fake_finetune_${FILTER_NET}_${PERC}_${NFAKE}.txt


MODEL="resnet8x4"
INIT_MODEL_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${MODEL}_epoch_240_last.pth"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_MODEL_PATH \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size 256 --learning_rate 0.01 --lr_decay_epochs "150,180,210" \
    2>&1 | tee output_${MODEL}_vanilla_fake_finetune_${FILTER_NET}_${PERC}_${NFAKE}.txt


MODEL="resnet56"
INIT_MODEL_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${MODEL}_epoch_240_last.pth"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_MODEL_PATH \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size 256 --learning_rate 0.01 --lr_decay_epochs "150,180,210" \
    2>&1 | tee output_${MODEL}_vanilla_fake_finetune_${FILTER_NET}_${PERC}_${NFAKE}.txt


MODEL="MobileNetV2"
INIT_MODEL_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${MODEL}_epoch_240_last.pth"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_MODEL_PATH \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size 256 --learning_rate 0.01 --lr_decay_epochs "150,180,210" \
    2>&1 | tee output_${MODEL}_vanilla_fake_finetune_${FILTER_NET}_${PERC}_${NFAKE}.txt


MODEL="ShuffleV1"
INIT_MODEL_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${MODEL}_epoch_240_last.pth"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_MODEL_PATH \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size 256 --learning_rate 0.01 --lr_decay_epochs "150,180,210" \
    2>&1 | tee output_${MODEL}_vanilla_fake_finetune_${FILTER_NET}_${PERC}_${NFAKE}.txt


MODEL="vgg8"
INIT_MODEL_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${MODEL}_epoch_240_last.pth"
python train_teacher.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --model $MODEL --resume_epoch 0 \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_MODEL_PATH \
    --epochs 240 --resume_epoch 0 --save_freq 20 \
    --batch_size 256 --learning_rate 0.01 --lr_decay_epochs "150,180,210" \
    2>&1 | tee output_${MODEL}_vanilla_fake_finetune_${FILTER_NET}_${PERC}_${NFAKE}.txt
