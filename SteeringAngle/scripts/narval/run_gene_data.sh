#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=SA_make_data
#SBATCH --output=%x-%j.out


module load arch/avx2 StdEnv/2020
module load cuda/11.0
module load python/3.9.6
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req

ROOT_PATH="/scratch/dingx92/cGAN-KD/SteeringAngle"
DATA_PATH="/scratch/dingx92/datasets/SteeringAngle/regression"


FILTER_NET="vgg19"
FILTER_NET_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${FILTER_NET}_epoch_240_last.pth"
python generate_synthetic_data.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --gan_arch "SAGAN" --gan_niters 20000 --gan_resume_niters 0 \
    --gan_batch_size_disc 256 --gan_batch_size_gene 256 --gan_d_niters 2 \
    --gan_threshold_type soft --gan_kappa -5 \
    --gan_DiffAugment \
    --subsampling \
    --dre_kappa -10 \
    --samp_filter_precnn_net $FILTER_NET --samp_filter_precnn_net_ckpt_path $FILTER_NET_PATH \
    --samp_filter_mae_percentile_threshold 0.9 \
    --samp_filter_batch_size 100 --samp_filter_burnin_size 200 \
    --samp_batch_size 100 --samp_burnin_size 200 --samp_num_fake_labels 2000 --samp_nfake_per_label 50 \
    --filter --adjust \
    2>&1 | tee output_make_data.txt