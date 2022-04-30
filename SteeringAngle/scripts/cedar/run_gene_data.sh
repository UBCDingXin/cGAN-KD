#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-01:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=SA_make_data
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_old.req

ROOT_PATH="/scratch/dingx92/cGAN-KD/SteeringAngle"
DATA_PATH="/scratch/dingx92/datasets/SteeringAngle/regression"




# python generate_synthetic_data.py \
#     --root_path ${ROOT_PATH} --data_path ${DATA_PATH} \
#     --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
#     --gan_batch_size_disc 512 --gan_batch_size_gene 512 --gan_d_niters 2 \
#     --gan_threshold_type soft --gan_kappa -5 \
#     --gan_DiffAugment \
#     --samp_batch_size 500 --samp_burnin_size 500 --samp_num_fake_labels 2000 --samp_nfake_per_label 50 \
#     2>&1 | tee output_make_data_None.txt



# python generate_synthetic_data.py \
#     --root_path ${ROOT_PATH} --data_path ${DATA_PATH} \
#     --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
#     --gan_batch_size_disc 512 --gan_batch_size_gene 512 --gan_d_niters 2 \
#     --gan_threshold_type soft --gan_kappa -5 \
#     --gan_DiffAugment \
#     --subsampling \
#     --samp_batch_size 500 --samp_burnin_size 500 --samp_num_fake_labels 2000 --samp_nfake_per_label 50 \
#     2>&1 | tee output_make_data_subsampling.txt


# PERC=0.9
# FILTER_NET="vgg19"
# FILTER_NET_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${FILTER_NET}_epoch_350_last.pth"
# UNFILTER_DATA_FILENAME="steeringangle_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_Nlabel_2000_NFakePerLabel_50_seed_2020.h5"
# python generate_synthetic_data.py \
#     --root_path ${ROOT_PATH} --data_path ${DATA_PATH} \
#     --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
#     --gan_batch_size_disc 512 --gan_batch_size_gene 512 --gan_d_niters 2 \
#     --gan_threshold_type soft --gan_kappa -5 \
#     --gan_DiffAugment \
#     --subsampling \
#     --samp_batch_size 500 --samp_burnin_size 500 --samp_num_fake_labels 2000 --samp_nfake_per_label 50 \
#     --filter \
#     --samp_filter_precnn_net ${FILTER_NET} --samp_filter_precnn_net_ckpt_path ${FILTER_NET_PATH} \
#     --samp_filter_mae_percentile_threshold $PERC \
#     --unfiltered_fake_dataset_filename ${UNFILTER_DATA_FILENAME} \
#     2>&1 | tee output_make_data_subsampling+filtering_perc${PERC}.txt

# PERC=0.9
# FILTER_NET="vgg19"
# FILTER_NET_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${FILTER_NET}_epoch_350_last.pth"
# UNFILTER_DATA_FILENAME="steeringangle_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_Nlabel_2000_NFakePerLabel_50_seed_2020.h5"
# python generate_synthetic_data.py \
#     --root_path ${ROOT_PATH} --data_path ${DATA_PATH} \
#     --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
#     --gan_batch_size_disc 512 --gan_batch_size_gene 512 --gan_d_niters 2 \
#     --gan_threshold_type soft --gan_kappa -5 \
#     --gan_DiffAugment \
#     --subsampling \
#     --samp_batch_size 500 --samp_burnin_size 500 --samp_num_fake_labels 2000 --samp_nfake_per_label 50 \
#     --filter --adjust \
#     --samp_filter_precnn_net ${FILTER_NET} --samp_filter_precnn_net_ckpt_path ${FILTER_NET_PATH} \
#     --samp_filter_mae_percentile_threshold $PERC \
#     --unfiltered_fake_dataset_filename ${UNFILTER_DATA_FILENAME} \
#     2>&1 | tee output_make_data_subsampling+filtering+adjustment_perc${PERC}.txt





# PERC=0.7
# FILTER_NET="vgg19"
# FILTER_NET_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${FILTER_NET}_epoch_350_last.pth"
# UNFILTER_DATA_FILENAME="steeringangle_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_Nlabel_2000_NFakePerLabel_50_seed_2020.h5"
# python generate_synthetic_data.py \
#     --root_path ${ROOT_PATH} --data_path ${DATA_PATH} \
#     --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
#     --gan_batch_size_disc 512 --gan_batch_size_gene 512 --gan_d_niters 2 \
#     --gan_threshold_type soft --gan_kappa -5 \
#     --gan_DiffAugment \
#     --subsampling \
#     --samp_batch_size 500 --samp_burnin_size 500 --samp_num_fake_labels 2000 --samp_nfake_per_label 50 \
#     --filter \
#     --samp_filter_precnn_net ${FILTER_NET} --samp_filter_precnn_net_ckpt_path ${FILTER_NET_PATH} \
#     --samp_filter_mae_percentile_threshold $PERC \
#     --unfiltered_fake_dataset_filename ${UNFILTER_DATA_FILENAME} \
#     2>&1 | tee output_make_data_subsampling+filtering_perc${PERC}.txt

# PERC=0.7
# FILTER_NET="vgg19"
# FILTER_NET_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${FILTER_NET}_epoch_350_last.pth"
# UNFILTER_DATA_FILENAME="steeringangle_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_Nlabel_2000_NFakePerLabel_50_seed_2020.h5"
# python generate_synthetic_data.py \
#     --root_path ${ROOT_PATH} --data_path ${DATA_PATH} \
#     --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
#     --gan_batch_size_disc 512 --gan_batch_size_gene 512 --gan_d_niters 2 \
#     --gan_threshold_type soft --gan_kappa -5 \
#     --gan_DiffAugment \
#     --subsampling \
#     --samp_batch_size 500 --samp_burnin_size 500 --samp_num_fake_labels 2000 --samp_nfake_per_label 50 \
#     --filter --adjust \
#     --samp_filter_precnn_net ${FILTER_NET} --samp_filter_precnn_net_ckpt_path ${FILTER_NET_PATH} \
#     --samp_filter_mae_percentile_threshold $PERC \
#     --unfiltered_fake_dataset_filename ${UNFILTER_DATA_FILENAME} \
#     2>&1 | tee output_make_data_subsampling+filtering+adjustment_perc${PERC}.txt







PERC=0.7
FILTER_NET="vgg8"
FILTER_NET_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${FILTER_NET}_epoch_350_last.pth"
UNFILTER_DATA_FILENAME="steeringangle_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_Nlabel_2000_NFakePerLabel_50_seed_2020.h5"
python generate_synthetic_data.py \
    --root_path ${ROOT_PATH} --data_path ${DATA_PATH} \
    --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
    --gan_batch_size_disc 512 --gan_batch_size_gene 512 --gan_d_niters 2 \
    --gan_threshold_type soft --gan_kappa -5 \
    --gan_DiffAugment \
    --subsampling \
    --samp_batch_size 500 --samp_burnin_size 500 --samp_num_fake_labels 2000 --samp_nfake_per_label 50 \
    --filter --adjust \
    --samp_filter_precnn_net ${FILTER_NET} --samp_filter_precnn_net_ckpt_path ${FILTER_NET_PATH} \
    --samp_filter_mae_percentile_threshold $PERC \
    --unfiltered_fake_dataset_filename ${UNFILTER_DATA_FILENAME} \
    2>&1 | tee output_make_data_subsampling+filtering+adjustment_perc${PERC}.txt

PERC=0.7
FILTER_NET="ResNet18"
FILTER_NET_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${FILTER_NET}_epoch_350_last.pth"
UNFILTER_DATA_FILENAME="steeringangle_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_None_adjust_False_Nlabel_2000_NFakePerLabel_50_seed_2020.h5"
python generate_synthetic_data.py \
    --root_path ${ROOT_PATH} --data_path ${DATA_PATH} \
    --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
    --gan_batch_size_disc 512 --gan_batch_size_gene 512 --gan_d_niters 2 \
    --gan_threshold_type soft --gan_kappa -5 \
    --gan_DiffAugment \
    --subsampling \
    --samp_batch_size 500 --samp_burnin_size 500 --samp_num_fake_labels 2000 --samp_nfake_per_label 50 \
    --filter --adjust \
    --samp_filter_precnn_net ${FILTER_NET} --samp_filter_precnn_net_ckpt_path ${FILTER_NET_PATH} \
    --samp_filter_mae_percentile_threshold $PERC \
    --unfiltered_fake_dataset_filename ${UNFILTER_DATA_FILENAME} \
    2>&1 | tee output_make_data_subsampling+filtering+adjustment_perc${PERC}.txt