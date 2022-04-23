@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/CIFAR-100/ReviewKD"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/CIFAR-100/data"


set TEACHER="vgg13"
set STUDENT="vgg8"
set TEACHER_PATH="%ROOT_PATH%/output/vanilla/models/ckpt_%TEACHER%_epoch_240_last.pth"
python train.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --model %STUDENT% --teacher %TEACHER% --teacher-weight %TEACHER_PATH% ^
    --kd-loss-weight 5.0 --mode "distill" ^ %*