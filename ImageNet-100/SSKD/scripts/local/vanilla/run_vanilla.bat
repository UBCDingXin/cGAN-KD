@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/ImageNet-100/SSKD"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/ImageNet-100"

set ARCH="ResNet34"
python teacher.py ^
    --root_path %ROOT_PATH% --real_data %DATA_PATH% ^
    --arch %ARCH% --epochs 120 --resume_epoch 0 --save_interval 20 ^
    --batch_size 32 --lr 0.01 --lr_decay_epochs "30_60_90" --weight_decay 5e-4 ^ %*