#!/bin/bash

ROOT_PATH="./CIFAR-100/TAKD"
DATA_PATH="./datasets/CIFAR-100/data"
FILTER_NET="densenet121"
PERC="0.90"
FAKE_DATA_PATH="./CIFAR-100/make_fake_datasets/fake_data/cifar100_fake_images_BigGAN_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_${FILTER_NET}_perc_${PERC}_adjust_True_NfakePerClass_5000_seed_2021.h5"
NFAKE=100000


## ===========================================================================================================================================
TEACHER="wrn_40_2"
ASSISTANT="vgg8"
STUDENT="MobileNetV2"
TEACHER_PATH="./CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
INIT_ASSISTANT_PATH="./CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${ASSISTANT}_epoch_240_last.pth"
INIT_STUDENT_PATH="./CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${STUDENT}_epoch_240_last.pth"

python takd.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --student $STUDENT --assistant $ASSISTANT --teacher_ckpt_path $TEACHER_PATH \
    --epochs 240 --resume_epoch_1 0 --resume_epoch_2 0 \
    --batch_size_train 128 --lr_base1 0.01 --lr_base2 0.01 --lr_decay_epochs "150_180_210" --transform \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_assistant_path $INIT_ASSISTANT_PATH --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_TA_${ASSISTANT}_T_${TEACHER}_fake.txt
