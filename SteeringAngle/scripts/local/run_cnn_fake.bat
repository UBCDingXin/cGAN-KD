@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/SteeringAngle"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/SteeringAngle/regression"


set FAKE_DATA_PATH="%ROOT_PATH%/output/fake_data/steeringangle_fake_images_SAGAN_None_filter_None_adjust_False_Nlabel_2000_NFakePerLabel_50_seed_2020.h5"
set NFAKE=50000

set ARCH="ShuffleV2"
set INIT_MODEL_PATH="%ROOT_PATH%/output/CNN/vanilla/ckpt_%ARCH%_epoch_240_last.pth"
python baseline_cnn.py ^
    --cnn_name %ARCH% --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --fake_data_path %FAKE_DATA_PATH% --nfake %NFAKE% ^
    --epochs 240 --resume_epoch 0 --save_freq 40 ^
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_180_210" ^
    --weight_decay 5e-4 ^
    --finetune --init_model_path %INIT_MODEL_PATH% ^ %*