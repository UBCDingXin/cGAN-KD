#!/bin/bash

ROOT_PATH="./ImageNet-100/SSKD"
DATA_PATH="./datasets/ImageNet-100"
FAKE_DATA_PATH="./ImageNet-100/make_fake_datasets/fake_data/imagenet100_fake_images_BigGANdeep_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_densenet161_perc_0.90_adjust_False_NfakePerClass_3000_seed_2021.h5"
NFAKE=100000



##########################################################################
resume_epoch=0
TEACHER="resnet110"
STUDENT="resnet20"
TEACHER_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
INIT_STUDENT_PATH="${ROOT_PATH}/output/students/vanilla/ckpt_S_${STUDENT}_T_${TEACHER}_lr_0.05_decay_0.0005_epoch_240_last.pth"
python student.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --s_arch $STUDENT --t_path $TEACHER_PATH \
    --t_epochs 60 --epochs 240 --resume_epoch $resume_epoch --save_interval 10 \
    --batch_size 256 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 1e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_fake_nfake_${NFAKE}.txt
