#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-12:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=U_make_data
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_old.req

ROOT_PATH="/scratch/dingx92/cGAN-KD/UTKFace"
DATA_PATH="/scratch/dingx92/datasets/UTKFace/regression"



# python generate_synthetic_data.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH \
#     --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
#     --gan_batch_size_disc 512 --gan_batch_size_gene 512  --gan_d_niters 4 \
#     --gan_threshold_type soft --gan_kappa -2 \
#     --gan_DiffAugment \
#     --samp_batch_size 500 --samp_nfake_per_label 2000 \
#     2>&1 | tee output_CcGAN_SAGAN_soft_kappa-2_None.txt


## subsampling only
python generate_synthetic_data.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
    --gan_batch_size_disc 512 --gan_batch_size_gene 512  --gan_d_niters 4 \
    --gan_threshold_type soft --gan_kappa -2 \
    --gan_DiffAugment \
    --subsampling \
    --samp_batch_size 1000 --samp_burnin_size 2000 --samp_nfake_per_label 2000 \
    2>&1 | tee output_CcGAN_SAGAN_soft_kappa-2_Subsampling.txt


## subsampling + filtering
PERC=0.9
FILTER_NET="vgg11"
FILTER_NET_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${FILTER_NET}_epoch_350_last.pth"
UNFILTER_DATA_FILENAME="utkface_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_NFakePerLabel_2000_seed_2020.h5"
python generate_synthetic_data.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
    --gan_batch_size_disc 512 --gan_batch_size_gene 512  --gan_d_niters 4 \
    --gan_threshold_type soft --gan_kappa -2 \
    --gan_DiffAugment \
    --samp_batch_size 500 --samp_burnin_size 100 --samp_nfake_per_label 2000 \
    --subsampling \
    --filter \
    --samp_filter_precnn_net $FILTER_NET --samp_filter_precnn_net_ckpt_path $FILTER_NET_PATH \
    --samp_filter_mae_percentile_threshold $PERC \
    --unfiltered_fake_dataset_filename $UNFILTER_DATA_FILENAME \
    2>&1 | tee output_CcGAN_SAGAN_soft_kappa-2_Subsampling+Filtering_Perc${PERC}.txt


## subsampling + filtering + adjustment
PERC=0.9
FILTER_NET="vgg11"
FILTER_NET_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${FILTER_NET}_epoch_350_last.pth"
UNFILTER_DATA_FILENAME="utkface_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_NFakePerLabel_2000_seed_2020.h5"
python generate_synthetic_data.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
    --gan_batch_size_disc 512 --gan_batch_size_gene 512  --gan_d_niters 4 \
    --gan_threshold_type soft --gan_kappa -2 \
    --gan_DiffAugment \
    --samp_batch_size 500 --samp_burnin_size 100 --samp_nfake_per_label 2000 \
    --subsampling \
    --filter --adjust \
    --samp_filter_precnn_net $FILTER_NET --samp_filter_precnn_net_ckpt_path $FILTER_NET_PATH \
    --samp_filter_mae_percentile_threshold $PERC \
    --unfiltered_fake_dataset_filename $UNFILTER_DATA_FILENAME \
    2>&1 | tee output_CcGAN_SAGAN_soft_kappa-2_Subsampling+Filtering+Adjustment_Perc${PERC}.txt




####################################
# smaller filtering threshold 
## subsampling + filtering
PERC=0.7
FILTER_NET="vgg11"
FILTER_NET_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${FILTER_NET}_epoch_350_last.pth"
UNFILTER_DATA_FILENAME="utkface_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_NFakePerLabel_2000_seed_2020.h5"
python generate_synthetic_data.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
    --gan_batch_size_disc 512 --gan_batch_size_gene 512  --gan_d_niters 4 \
    --gan_threshold_type soft --gan_kappa -2 \
    --gan_DiffAugment \
    --samp_batch_size 500 --samp_burnin_size 100 --samp_nfake_per_label 2000 \
    --subsampling \
    --filter \
    --samp_filter_precnn_net $FILTER_NET --samp_filter_precnn_net_ckpt_path $FILTER_NET_PATH \
    --samp_filter_mae_percentile_threshold $PERC \
    --unfiltered_fake_dataset_filename $UNFILTER_DATA_FILENAME \
    2>&1 | tee output_CcGAN_SAGAN_soft_kappa-2_Subsampling+Filtering_Perc${PERC}.txt

## subsampling + filtering + adjustment
PERC=0.7
FILTER_NET="vgg11"
FILTER_NET_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${FILTER_NET}_epoch_350_last.pth"
UNFILTER_DATA_FILENAME="utkface_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_NFakePerLabel_2000_seed_2020.h5"
python generate_synthetic_data.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
    --gan_batch_size_disc 512 --gan_batch_size_gene 512  --gan_d_niters 4 \
    --gan_threshold_type soft --gan_kappa -2 \
    --gan_DiffAugment \
    --samp_batch_size 500 --samp_burnin_size 100 --samp_nfake_per_label 2000 \
    --subsampling \
    --filter --adjust \
    --samp_filter_precnn_net $FILTER_NET --samp_filter_precnn_net_ckpt_path $FILTER_NET_PATH \
    --samp_filter_mae_percentile_threshold $PERC \
    --unfiltered_fake_dataset_filename $UNFILTER_DATA_FILENAME \
    2>&1 | tee output_CcGAN_SAGAN_soft_kappa-2_Subsampling+Filtering+Adjustment_Perc${PERC}.txt