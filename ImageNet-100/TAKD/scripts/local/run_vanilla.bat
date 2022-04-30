@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/CIFAR-100/TAKD"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/CIFAR-100/data"


set TEACHER="wrn_40_2"
set ASSISTANT="wrn_16_2"
set STUDENT="wrn_40_1"
set TEACHER_PATH="G:/OneDrive/Working_directory/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_%TEACHER%_epoch_240_last.pth"

python takd.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --student %STUDENT% --assistant %ASSISTANT% --teacher_ckpt_path %TEACHER_PATH% ^
    --epochs 2 --resume_epoch_1 0 --resume_epoch_2 0 ^
    --batch_size_train 64 --lr_base1 0.05 --lr_base2 0.05 --lr_decay_epochs "150_180_210" --transform ^ %*




set TEACHER="ResNet50"
set ASSISTANT="resnet110"
set STUDENT="resnet20"
set TEACHER_PATH="G:/OneDrive/Working_directory/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_%TEACHER%_epoch_240_last.pth"

python takd.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --student %STUDENT% --assistant %ASSISTANT% --teacher_ckpt_path %TEACHER_PATH% ^
    --epochs 2 --resume_epoch_1 0 --resume_epoch_2 0 ^
    --batch_size_train 64 --lr_base1 0.05 --lr_base2 0.05 --lr_decay_epochs "150_180_210" --transform ^ %*