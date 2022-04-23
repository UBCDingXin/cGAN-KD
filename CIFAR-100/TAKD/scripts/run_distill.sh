#!/bin/bash

ROOT_PATH="./CIFAR-100/TAKD"
DATA_PATH="./datasets/CIFAR-100/data"

## ===========================================================================================================================================
TEACHER="wrn_40_2"
TEACHER_PATH="./CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
ASSISTANT="vgg19"
STUDENT="ShuffleV1"

python takd.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --student $STUDENT --assistant $ASSISTANT --teacher_ckpt_path $TEACHER_PATH \
    --epochs 240 --resume_epoch_1 0 --resume_epoch_2 0 \
    --batch_size_train 64 --lr_base1 0.05 --lr_base2 0.01 --lr_decay_epochs "150_180_210" --transform \
    2>&1 | tee output_S_${STUDENT}_TA_${ASSISTANT}_T_${TEACHER}_vanilla.txt
