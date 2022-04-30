@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/UTKFace"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/UTKFace/regression"


@REM No subsampling
python generate_synthetic_data.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --gan_arch "SAGAN" --gan_niters 40000 --gan_resume_niters 0 ^
    --gan_batch_size_disc 128 --gan_batch_size_gene 128 --gan_d_niters 4 ^
    --gan_threshold_type soft --gan_kappa -2 ^
    --gan_DiffAugment ^
    --samp_batch_size 500 --samp_nfake_per_label 2000 ^ %*


@REM subsampling
set FILTER_NET="vgg11"
set FILTER_NET_PATH="%ROOT_PATH%/output/CNN/vanilla/ckpt_%FILTER_NET%_epoch_240_last.pth"
python generate_synthetic_data.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --gan_arch "SAGAN" --gan_niters 40000 --gan_resume_niters 0 ^
    --gan_batch_size_disc 128 --gan_batch_size_gene 128 --gan_d_niters 4 ^
    --gan_threshold_type soft --gan_kappa -2 ^
    --gan_DiffAugment ^
    --subsampling ^
    --samp_batch_size 500 --samp_burnin_size 2000 --samp_nfake_per_label 2000 ^ %*
    

@REM @REM subsampling+filtering
@REM set FILTER_NET="vgg11"
@REM set FILTER_NET_PATH="%ROOT_PATH%/output/CNN/vanilla/ckpt_%FILTER_NET%_epoch_240_last.pth"
@REM set UNFILTER_DATA_FILENAME="utkface_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_NFakePerLabel_2000_seed_2020.h5"
@REM python generate_synthetic_data.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
@REM     --gan_arch "SAGAN" --gan_niters 40000 --gan_resume_niters 0 ^
@REM     --gan_batch_size_disc 128 --gan_batch_size_gene 128 --gan_d_niters 4 ^
@REM     --gan_threshold_type soft --gan_kappa -2 ^
@REM     --gan_DiffAugment ^
@REM     --samp_batch_size 500 --samp_burnin_size 100 --samp_nfake_per_label 2000 ^
@REM     --subsampling ^
@REM     --filter ^
@REM     --samp_filter_precnn_net %FILTER_NET% --samp_filter_precnn_net_ckpt_path %FILTER_NET_PATH% ^
@REM     --samp_filter_mae_percentile_threshold 0.9 ^
@REM     --unfiltered_fake_dataset_filename %UNFILTER_DATA_FILENAME% ^ %*


@REM @REM subsampling+filtering+adjust
@REM set FILTER_NET="vgg11"
@REM set FILTER_NET_PATH="%ROOT_PATH%/output/CNN/vanilla/ckpt_%FILTER_NET%_epoch_240_last.pth"
@REM set UNFILTER_DATA_FILENAME="utkface_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_NFakePerLabel_2000_seed_2020.h5"
@REM python generate_synthetic_data.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
@REM     --gan_arch "SAGAN" --gan_niters 40000 --gan_resume_niters 0 ^
@REM     --gan_batch_size_disc 128 --gan_batch_size_gene 128 --gan_d_niters 4 ^
@REM     --gan_threshold_type soft --gan_kappa -2 ^
@REM     --gan_DiffAugment ^
@REM     --samp_batch_size 500 --samp_burnin_size 100 --samp_nfake_per_label 2000 ^
@REM     --subsampling ^
@REM     --filter --adjust ^
@REM     --samp_filter_precnn_net %FILTER_NET% --samp_filter_precnn_net_ckpt_path %FILTER_NET_PATH% ^
@REM     --samp_filter_mae_percentile_threshold 0.9 ^
@REM     --unfiltered_fake_dataset_filename %UNFILTER_DATA_FILENAME% ^ %*




@REM @REM subsampling+filtering+adjust
@REM set FILTER_NET="densenet121"
@REM set FILTER_NET_PATH="%ROOT_PATH%/output/CNN/vanilla/ckpt_%FILTER_NET%_epoch_240_last.pth"
@REM set UNFILTER_DATA_FILENAME="utkface_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_NFakePerLabel_2000_seed_2020.h5"
@REM python generate_synthetic_data.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
@REM     --gan_arch "SAGAN" --gan_niters 40000 --gan_resume_niters 0 ^
@REM     --gan_batch_size_disc 128 --gan_batch_size_gene 128 --gan_d_niters 4 ^
@REM     --gan_threshold_type soft --gan_kappa -2 ^
@REM     --gan_DiffAugment ^
@REM     --samp_batch_size 500 --samp_burnin_size 100 --samp_nfake_per_label 2000 ^
@REM     --subsampling ^
@REM     --filter --adjust ^
@REM     --samp_filter_precnn_net %FILTER_NET% --samp_filter_precnn_net_ckpt_path %FILTER_NET_PATH% ^
@REM     --samp_filter_mae_percentile_threshold 0.9 ^
@REM     --unfiltered_fake_dataset_filename %UNFILTER_DATA_FILENAME% ^ %*


@REM @REM subsampling+filtering+adjust
@REM set FILTER_NET="resnet32x4"
@REM set FILTER_NET_PATH="%ROOT_PATH%/output/CNN/vanilla/ckpt_%FILTER_NET%_epoch_240_last.pth"
@REM set UNFILTER_DATA_FILENAME="utkface_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_NFakePerLabel_2000_seed_2020.h5"
@REM python generate_synthetic_data.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
@REM     --gan_arch "SAGAN" --gan_niters 40000 --gan_resume_niters 0 ^
@REM     --gan_batch_size_disc 128 --gan_batch_size_gene 128 --gan_d_niters 4 ^
@REM     --gan_threshold_type soft --gan_kappa -2 ^
@REM     --gan_DiffAugment ^
@REM     --samp_batch_size 500 --samp_burnin_size 100 --samp_nfake_per_label 2000 ^
@REM     --subsampling ^
@REM     --filter --adjust ^
@REM     --samp_filter_precnn_net %FILTER_NET% --samp_filter_precnn_net_ckpt_path %FILTER_NET_PATH% ^
@REM     --samp_filter_mae_percentile_threshold 0.9 ^
@REM     --unfiltered_fake_dataset_filename %UNFILTER_DATA_FILENAME% ^ %*