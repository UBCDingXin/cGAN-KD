#!/bin/bash

ROOT_PATH="./ImageNet-100/RepDistiller"
DATA_PATH="./datasets/ImageNet-100"


#######################################################################
TEACHER="resnet110"
TEACHER_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
STUDENT="resnet20"

# BLKD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill kd --model_s $STUDENT -r 0.1 -a 0.9 -b 0 --resume_epoch 0 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_BLKD.txt

# FitNet+KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill hint --model_s $STUDENT -a 1 -b 100 --resume_epoch 0 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_FitNet.txt

# VID+KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill vid --model_s $STUDENT -a 1 -b 1 --resume_epoch 0 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_VID.txt

# RKD+KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill rkd --model_s $STUDENT -a 1 -b 1 --resume_epoch 0 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_RKD.txt

# CRD+KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill crd --model_s $STUDENT -a 1 -b 0.8 --resume_epoch 0 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_CRD.txt
