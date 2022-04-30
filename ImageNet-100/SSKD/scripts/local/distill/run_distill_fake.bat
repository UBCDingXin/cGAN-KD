@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/ImageNet-100/SSKD"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/ImageNet-100"

set FAKE_DATA_PATH="G:/OneDrive/Working_directory/cGAN-KD/ImageNet-100/make_fake_datasets/fake_data/imagenet100_fake_images_BigGAN_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_vgg19_perc_0.90_adjust_False_NfakePerClass_3000_seed_2021.h5"
set NFAKE=100000

set TEACHER="resnet110"
set STUDENT="resnet20"
set TEACHER_PATH="%ROOT_PATH%/output/teachers/vanilla/ckpt_%TEACHER%_epoch_240_last.pth"
set INIT_STUDENT_PATH="%ROOT_PATH%/output/students/vanilla/ckpt_S_%STUDENT%_T_%TEACHER%_lr_0.01_decay_0.0005_epoch_240_last.pth"
python student.py ^
    --root_path %ROOT_PATH% --real_data %DATA_PATH% ^
    --s_arch %STUDENT% --t_path %TEACHER_PATH% ^
    --t_epochs 60 --epochs 240 --resume_epoch 0 --save_interval 20 ^
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 1e-4 ^
    --fake_data %FAKE_DATA_PATH% --nfake %NFAKE% ^
    --finetune --init_student_path %INIT_STUDENT_PATH% ^ %*