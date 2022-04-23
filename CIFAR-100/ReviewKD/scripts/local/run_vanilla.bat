@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/CIFAR-100/ReviewKD"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/CIFAR-100/data"

python train.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --model MobileNetV2 --mode vanilla ^ %*

