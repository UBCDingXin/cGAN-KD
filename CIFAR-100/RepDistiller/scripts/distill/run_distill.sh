#!/bin/bash

ROOT_PATH="./CIFAR-100/RepDistiller"
DATA_PATH="./datasets/CIFAR-100/data"

TEACHER="wrn_40_2"
TEACHER_PATH="./CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
STUDENT="ShuffleV2"

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

# AT+KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill attention --model_s $STUDENT -a 1 -b 1000 --resume_epoch 0 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_AT.txt

# SP+KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill similarity --model_s $STUDENT -a 1 -b 3000 --resume_epoch 0 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_SP.txt

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

# PKT+KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill pkt --model_s $STUDENT -a 1 -b 30000 --resume_epoch 0 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_PKT.txt

# AB+KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill abound --model_s $STUDENT -a 1 -b 1 --resume_epoch 0 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_AB.txt

# FT+KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill factor --model_s $STUDENT -a 1 -b 200 --resume_epoch 0 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_FT.txt

# CRD+KD
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill crd --model_s $STUDENT -a 1 -b 0.8 --resume_epoch 0 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_CRD.txt
