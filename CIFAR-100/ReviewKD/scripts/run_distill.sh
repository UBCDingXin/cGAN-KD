#!/bin/bash

ROOT_PATH="./CIFAR-100/ReviewKD"
DATA_PATH="./datasets/CIFAR-100/data"


TEACHER="wrn_40_2"
STUDENT="wrn_40_1"
TEACHER_PATH="${ROOT_PATH}/output/vanilla/models/ckpt_${TEACHER}_epoch_240_last.pth"
python train.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --model $STUDENT --teacher $TEACHER --teacher-weight $TEACHER_PATH \
    --kd-loss-weight 5.0 --mode "distill" \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_ReviewKD.txt
