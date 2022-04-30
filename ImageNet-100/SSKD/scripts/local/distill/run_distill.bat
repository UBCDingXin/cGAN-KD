@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/ImageNet-100/SSKD"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/ImageNet-100"

set TEACHER="vgg13"
set STUDENT="MobileNetV2"
set TEACHER_PATH="%ROOT_PATH%/output/teachers/vanilla/ckpt_%TEACHER%_epoch_240_last.pth"
python student.py ^
    --root_path %ROOT_PATH% --real_data %DATA_PATH% ^
    --s_arch %STUDENT% --t_path %TEACHER_PATH% ^
    --t_epochs 60 --epochs 240 --resume_epoch 0 --save_interval 20 ^
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 ^ %*