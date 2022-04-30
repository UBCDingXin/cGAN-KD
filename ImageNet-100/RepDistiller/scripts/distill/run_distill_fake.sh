#!/bin/bash

ROOT_PATH="./ImageNet-100/RepDistiller"
DATA_PATH="./datasets/ImageNet-100"

FILTER_NET="densenet161"
PERC="0.90"
FAKE_DATA_PATH="./ImageNet-100/make_fake_datasets/fake_data/imagenet100_fake_images_BigGANdeep_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_${FILTER_NET}_perc_${PERC}_adjust_False_NfakePerClass_3000_seed_2021.h5"
NFAKE=100000



TEACHER="resnet110"
TEACHER_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
STUDENT="resnet20"
INIT_STUDENT_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${STUDENT}_epoch_240_last.pth"

# KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill kd --model_s $STUDENT -r 0.1 -a 0.9 -b 0 --resume_epoch 0 \
    --batch_size 256 --learning_rate 0.01 --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_kd_fake.txt

# FitNet
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill hint --model_s $STUDENT -a 1 -b 100 --resume_epoch 0 \
    --batch_size 256 --learning_rate 0.01 --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_FitNet_fake.txt

# VID+KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill vid --model_s $STUDENT -a 1 -b 1 --resume_epoch 0 \
    --batch_size 256 --learning_rate 0.01 --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_VID_fake.txt

# RKD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill rkd --model_s $STUDENT -a 1 -b 1 --resume_epoch 0 \
    --batch_size 256 --learning_rate 0.01 --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_rkd_fake.txt

# CRD
resume_epoch=0
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill crd --model_s $STUDENT -a 1 -b 0.8 --resume_epoch $resume_epoch \
    --batch_size 256 --learning_rate 0.01 --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_CRD_fake.txt
