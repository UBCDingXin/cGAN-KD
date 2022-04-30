@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/SteeringAngle"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/SteeringAngle/regression"

set ARCH="ShuffleV2"
python baseline_cnn.py ^
    --cnn_name %ARCH% --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --epochs 240 --resume_epoch 0 --save_freq 40 ^
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_180_210" ^
    --weight_decay 5e-4 ^ %*