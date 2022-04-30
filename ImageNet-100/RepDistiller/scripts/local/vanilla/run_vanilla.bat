@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/ImageNet-100/RepDistiller"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/ImageNet-100"
set FAKE_DATA_PATH="G:/OneDrive/Working_directory/cGAN-KD/ImageNet-100/make_fake_datasets/fake_data/imagenet100_fake_images_BigGAN_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_densenet121_perc_0.90_adjust_False_NfakePerClass_5000_seed_2021.h5"
set NFAKE=100000


@REM Vanilla
python train_teacher.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --model resnet20 --resume_epoch 0 --batch_size 64 ^ %*


@REM @REM cGAN-KD but w/o finetune
@REM python train_teacher.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% --model vgg8 --resume_epoch 0 ^
@REM     --use_fake_data --fake_data_path %FAKE_DATA_PATH% --nfake %NFAKE% ^ %*


@REM @REM cGAN-KD but w/ finetune
@REM set INIT_MODEL_PATH="G:/OneDrive/Working_directory/cGAN-KD/ImageNet-100/RepDistiller/output/teacher_models/vanilla/ckpt_vgg8_epoch_240_last.pth"
@REM python train_teacher.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% --model vgg8 --resume_epoch 0 ^
@REM     --use_fake_data --fake_data_path %FAKE_DATA_PATH% --nfake %NFAKE% ^
@REM     --finetune --init_model_path %INIT_MODEL_PATH% ^
@REM     --epochs 60 --resume_epoch 0 --save_freq 20 ^
@REM     --batch_size 64 --learning_rate 1e-4 --lr_decay_epochs "30,60" ^ %*