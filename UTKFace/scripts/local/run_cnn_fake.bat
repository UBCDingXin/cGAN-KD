@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/UTKFace"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/UTKFace/regression"

set FAKE_DATA_PATH="%ROOT_PATH%/output/fake_data/utkface_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_vgg11_perc_0.90_adjust_True_NFakePerLabel_2000_seed_2020.h5"
set NFAKE=80000


set ARCH="resnet20"
set INIT_MODEL_PATH="%ROOT_PATH%/output/CNN/vanilla/ckpt_%ARCH%_epoch_240_last.pth"
python baseline_cnn.py ^
    --cnn_name %ARCH% --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --fake_data_path %FAKE_DATA_PATH% --nfake %NFAKE% ^
    --epochs 240 --resume_epoch 0 --save_freq 20 ^
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_180_210" ^
    --weight_decay 5e-4 ^
    --finetune --init_model_path %INIT_MODEL_PATH% ^