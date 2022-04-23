#!/bin/bash

ROOT_PATH="./CIFAR-100/SSKD"
DATA_PATH="./datasets/CIFAR-100/data"


TEACHER="wrn_40_2"
STUDENT="ShuffleV1"
TEACHER_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
python student.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --s_arch $STUDENT --t_path $TEACHER_PATH \
    --t_epochs 60 --epochs 240 --resume_epoch 0 --save_interval 40 \
    --batch_size 64 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_vanilla.txt
