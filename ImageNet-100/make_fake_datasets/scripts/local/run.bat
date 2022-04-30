@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/ImageNet-100/make_fake_datasets"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/ImageNet-100"
set EVAL_PATH="G:/OneDrive/Working_directory/cGAN-KD/ImageNet-100/eval_and_gan_ckpts/ckpt_PreCNNForEval_InceptionV3_epoch_200_SEED_2021_Transformation_True.pth"
set GAN_CKPT_PATH="G:/OneDrive/Working_directory/cGAN-KD/ImageNet-100/eval_and_gan_ckpts/BigGAN_deep_96K/G_ema.pth"



set SEED=2021
set GAN_NET="BigGANdeep"
set DRE_PRECNN="ResNet34"
set DRE_PRECNN_EPOCHS=350
set DRE_PRECNN_BS=256
set DRE_DR="MLP5"
set DRE_DR_EPOCHS=200
set DRE_DR_LR_BASE=1e-4
set DRE_DR_BS=256
set DRE_DR_LAMBDA=0.01

set SAMP_BS=50
set SAMP_BURNIN=1000 
@REM set SAMP_NFAKE_PER_CLASS=3000
set SAMP_NFAKE_PER_CLASS=500

set PRECNN_NET="densenet161"
set PRECNN_CKPT_PATH="G:/OneDrive/Working_directory/cGAN-KD/ImageNet-100/RepDistiller/output/teacher_models/vanilla/ckpt_%PRECNN_NET%_epoch_240_last.pth"




@REM @REM None
@REM python main.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% ^
@REM     --gan_net %GAN_NET% --gan_ckpt_path %GAN_CKPT_PATH% ^
@REM     --samp_batch_size %SAMP_BS% --samp_burnin_size %SAMP_BURNIN% ^
@REM     --samp_nfake_per_class %SAMP_NFAKE_PER_CLASS% ^ %*





@REM @REM cDR-RS
@REM python main.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% ^
@REM     --gan_net %GAN_NET% --gan_ckpt_path %GAN_CKPT_PATH% ^
@REM     --samp_batch_size %SAMP_BS% --samp_burnin_size %SAMP_BURNIN% ^
@REM     --samp_nfake_per_class %SAMP_NFAKE_PER_CLASS% ^
@REM     --subsampling ^
@REM     --dre_precnn_net %DRE_PRECNN% --dre_precnn_epochs %DRE_PRECNN_EPOCHS% --dre_precnn_resume_epoch 0 ^
@REM     --dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs "150_250" ^
@REM     --dre_precnn_batch_size_train %DRE_PRECNN_BS% --dre_precnn_weight_decay 1e-4 ^
@REM     --dre_net %DRE_DR% --dre_epochs %DRE_DR_EPOCHS% --dre_resume_epoch 0 ^
@REM     --dre_lr_base %DRE_DR_LR_BASE% --dre_batch_size %DRE_DR_BS% --dre_lambda %DRE_DR_LAMBDA% ^
@REM     --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs "80_150" ^ %*





@REM cDR-RS + filtering
python main.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% ^
    --gan_net %GAN_NET% --gan_ckpt_path %GAN_CKPT_PATH% ^
    --samp_batch_size %SAMP_BS% --samp_burnin_size %SAMP_BURNIN% ^
    --samp_nfake_per_class %SAMP_NFAKE_PER_CLASS% ^
    --subsampling ^
    --dre_precnn_net %DRE_PRECNN% --dre_precnn_epochs %DRE_PRECNN_EPOCHS% --dre_precnn_resume_epoch 0 ^
    --dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs "150_250" ^
    --dre_precnn_batch_size_train %DRE_PRECNN_BS% --dre_precnn_weight_decay 1e-4 ^
    --dre_net %DRE_DR% --dre_epochs %DRE_DR_EPOCHS% --dre_resume_epoch 0 ^
    --dre_lr_base %DRE_DR_LR_BASE% --dre_batch_size %DRE_DR_BS% --dre_lambda %DRE_DR_LAMBDA% ^
    --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs "80_150" ^
    --filter ^
    --samp_filter_precnn_net %PRECNN_NET% --samp_filter_precnn_net_ckpt_path %PRECNN_CKPT_PATH% ^
    --samp_filter_ce_percentile_threshold 0.9 --samp_filter_batch_size %SAMP_BS% --visualize_filtered_images ^ %*