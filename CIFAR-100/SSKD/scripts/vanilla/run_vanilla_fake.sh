#!/bin/bash

ROOT_PATH="./CIFAR-100/SSKD"
DATA_PATH="./datasets/CIFAR-100/data"
FAKE_DATA_PATH="./CIFAR-100/make_fake_datasets/fake_data/cifar100_fake_images_BigGAN_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_densenet121_perc_0.90_adjust_False_NfakePerClass_5000_seed_2021.h5"
NFAKE=100000



ARCH="wrn_40_2"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="ResNet50"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="resnet32x4"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="vgg13"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="vgg19"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="resnet56"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt


ARCH="wrn_40_1"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="resnet20"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="vgg8"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="resnet8x4"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="MobileNetV2"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="ShuffleV1"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="ShuffleV2"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt

ARCH="wrn_16_2"
INIT_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${ARCH}_epoch_240_last.pth"
python teacher.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --arch $ARCH --epochs 240 --resume_epoch 0 --save_interval 30 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    --fake_data $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_model_path $INIT_PATH \
    2>&1 | tee output_${ARCH}_vanilla_nfake_${NFAKE}.txt
