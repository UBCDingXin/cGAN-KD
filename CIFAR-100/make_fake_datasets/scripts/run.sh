#!/bin/bash

ROOT_PATH="./CIFAR-100/make_fake_datasets"
DATA_PATH="./datasets/CIFAR-100/data"
EVAL_PATH="./CIFAR-100/eval_and_gan_ckpts/ckpt_PreCNNForEval_InceptionV3_epoch_200_SEED_2021_Transformation_True.pth"
GAN_CKPT_PATH="./CIFAR-100/eval_and_gan_ckpts/BigGAN_38K/G_ema.pth"


SEED=2021
GAN_NET="BigGAN"
DRE_PRECNN="ResNet34"
DRE_PRECNN_EPOCHS=350
DRE_PRECNN_BS=256
DRE_DR="MLP5"
DRE_DR_EPOCHS=200
DRE_DR_LR_BASE=1e-4
DRE_DR_BS=256
DRE_DR_LAMBDA=0.01

SAMP_BS=600
SAMP_BURNIN=5000
SAMP_NFAKE_PER_CLASS=5000

PRECNN_NET="densenet121"
PRECNN_CKPT_PATH="./CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${PRECNN_NET}_epoch_240_last.pth"

## None
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED \
    --gan_net $GAN_NET --gan_ckpt_path $GAN_CKPT_PATH \
    --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN \
    --samp_nfake_per_class $SAMP_NFAKE_PER_CLASS \
    2>&1 | tee output_None.txt

## cDR-RS
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED \
    --gan_net $GAN_NET --gan_ckpt_path $GAN_CKPT_PATH \
    --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN \
    --samp_nfake_per_class $SAMP_NFAKE_PER_CLASS \
    --subsampling \
    --dre_precnn_net $DRE_PRECNN --dre_precnn_epochs $DRE_PRECNN_EPOCHS --dre_precnn_resume_epoch 0 \
    --dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs "150_250" \
    --dre_precnn_batch_size_train $DRE_PRECNN_BS --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
    --dre_net $DRE_DR --dre_epochs $DRE_DR_EPOCHS --dre_resume_epoch 0 \
    --dre_lr_base $DRE_DR_LR_BASE --dre_batch_size $DRE_DR_BS --dre_lambda $DRE_DR_LAMBDA \
    --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs "80_150" \
    2>&1 | tee output_cDR-RS.txt

## cDR-RS + filtering
FILTER_THRESH=0.9
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED \
    --gan_net $GAN_NET --gan_ckpt_path $GAN_CKPT_PATH \
    --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN \
    --samp_nfake_per_class $SAMP_NFAKE_PER_CLASS \
    --subsampling \
    --dre_precnn_net $DRE_PRECNN --dre_precnn_epochs $DRE_PRECNN_EPOCHS --dre_precnn_resume_epoch 0 \
    --dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs "150_250" \
    --dre_precnn_batch_size_train $DRE_PRECNN_BS --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
    --dre_net $DRE_DR --dre_epochs $DRE_DR_EPOCHS --dre_resume_epoch 0 \
    --dre_lr_base $DRE_DR_LR_BASE --dre_batch_size $DRE_DR_BS --dre_lambda $DRE_DR_LAMBDA \
    --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs "80_150" \
    --filter \
    --samp_filter_precnn_net $PRECNN_NET --samp_filter_precnn_net_ckpt_path $PRECNN_CKPT_PATH \
    --samp_filter_ce_percentile_threshold $FILTER_THRESH --samp_filter_batch_size $SAMP_BS \
    2>&1 | tee output_cDR-RS_filtering_${PRECNN_NET}_thresh_${FILTER_THRESH}_adjust_False.txt

## cDR-RS + filtering + adjustment
FILTER_THRESH=0.9
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED \
    --gan_net $GAN_NET --gan_ckpt_path $GAN_CKPT_PATH \
    --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN \
    --samp_nfake_per_class $SAMP_NFAKE_PER_CLASS \
    --subsampling \
    --dre_precnn_net $DRE_PRECNN --dre_precnn_epochs $DRE_PRECNN_EPOCHS --dre_precnn_resume_epoch 0 \
    --dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs "150_250" \
    --dre_precnn_batch_size_train $DRE_PRECNN_BS --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
    --dre_net $DRE_DR --dre_epochs $DRE_DR_EPOCHS --dre_resume_epoch 0 \
    --dre_lr_base $DRE_DR_LR_BASE --dre_batch_size $DRE_DR_BS --dre_lambda $DRE_DR_LAMBDA \
    --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs "80_150" \
    --filter --adjust \
    --samp_filter_precnn_net $PRECNN_NET --samp_filter_precnn_net_ckpt_path $PRECNN_CKPT_PATH \
    --samp_filter_ce_percentile_threshold $FILTER_THRESH --samp_filter_batch_size $SAMP_BS \
    2>&1 | tee output_cDR-RS_filtering_${PRECNN_NET}_thresh_${FILTER_THRESH}_adjust_True.txt
