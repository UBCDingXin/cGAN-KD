#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=I_make_data
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req

ROOT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/make_fake_datasets"
DATA_PATH="/scratch/dingx92/datasets/ImageNet-100"
EVAL_PATH="/scratch/dingx92/ImageNet-100/eval_and_gan_ckpts/ckpt_PreCNNForEval_InceptionV3_epoch_200_SEED_2021_Transformation_True.pth"
GAN_CKPT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/eval_and_gan_ckpts/BigGAN_deep_96K/G_ema.pth"


SEED=2021
GAN_NET="BigGANdeep"
DRE_PRECNN="ResNet34"
DRE_PRECNN_EPOCHS=350
DRE_PRECNN_BS=256
DRE_DR="MLP5"
DRE_DR_EPOCHS=200
DRE_DR_LR_BASE=1e-4
DRE_DR_BS=256
DRE_DR_LAMBDA=0.01

SAMP_BS=200
SAMP_BURNIN=5000
SAMP_NFAKE_PER_CLASS=3000

PRECNN_NET="densenet121"
PRECNN_CKPT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/RepDistiller/output/teacher_models/vanilla/ckpt_${PRECNN_NET}_epoch_240_last.pth"

# ## None
# python main.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED \
#     --gan_net $GAN_NET --gan_ckpt_path $GAN_CKPT_PATH \
#     --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN \
#     --samp_nfake_per_class $SAMP_NFAKE_PER_CLASS \
#     2>&1 | tee output_None.txt

# ## cDR-RS
# python main.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED \
#     --gan_net $GAN_NET --gan_ckpt_path $GAN_CKPT_PATH \
#     --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN \
#     --samp_nfake_per_class $SAMP_NFAKE_PER_CLASS \
#     --subsampling \
#     --dre_precnn_net $DRE_PRECNN --dre_precnn_epochs $DRE_PRECNN_EPOCHS --dre_precnn_resume_epoch 0 \
#     --dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs "150_250" \
#     --dre_precnn_batch_size_train $DRE_PRECNN_BS --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
#     --dre_net $DRE_DR --dre_epochs $DRE_DR_EPOCHS --dre_resume_epoch 0 \
#     --dre_lr_base $DRE_DR_LR_BASE --dre_batch_size $DRE_DR_BS --dre_lambda $DRE_DR_LAMBDA \
#     --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs "80_150" \
#     2>&1 | tee output_cDR-RS.txt

## cDR-RS + filtering
FILTER_THRESH=0.9
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED \
    --gan_net $GAN_NET --gan_ckpt_path $GAN_CKPT_PATH \
    --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN \
    --samp_nfake_per_class $SAMP_NFAKE_PER_CLASS \
    --subsampling \
    --dre_precnn_net $DRE_PRECNN --dre_precnn_epochs $DRE_PRECNN_EPOCHS --dre_precnn_resume_epoch 0 \
    --dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs "150_250" \
    --dre_precnn_batch_size_train $DRE_PRECNN_BS --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
    --dre_net $DRE_DR --dre_epochs $DRE_DR_EPOCHS --dre_resume_epoch 0 \
    --dre_lr_base $DRE_DR_LR_BASE --dre_batch_size $DRE_DR_BS --dre_lambda $DRE_DR_LAMBDA \
    --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs "80_150" \
    --filter \
    --samp_filter_precnn_net $PRECNN_NET --samp_filter_precnn_net_ckpt_path $PRECNN_CKPT_PATH \
    --samp_filter_ce_percentile_threshold $FILTER_THRESH --samp_filter_batch_size $SAMP_BS \
    2>&1 | tee output_cDR-RS_filtering_${PRECNN_NET}_thresh_${FILTER_THRESH}_adjust_False.txt



PRECNN_NET="vgg19"
PRECNN_CKPT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/RepDistiller/output/teacher_models/vanilla/ckpt_${PRECNN_NET}_epoch_240_last.pth"
## cDR-RS + filtering
FILTER_THRESH=0.9
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED \
    --gan_net $GAN_NET --gan_ckpt_path $GAN_CKPT_PATH \
    --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN \
    --samp_nfake_per_class $SAMP_NFAKE_PER_CLASS \
    --subsampling \
    --dre_precnn_net $DRE_PRECNN --dre_precnn_epochs $DRE_PRECNN_EPOCHS --dre_precnn_resume_epoch 0 \
    --dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs "150_250" \
    --dre_precnn_batch_size_train $DRE_PRECNN_BS --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
    --dre_net $DRE_DR --dre_epochs $DRE_DR_EPOCHS --dre_resume_epoch 0 \
    --dre_lr_base $DRE_DR_LR_BASE --dre_batch_size $DRE_DR_BS --dre_lambda $DRE_DR_LAMBDA \
    --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs "80_150" \
    --filter \
    --samp_filter_precnn_net $PRECNN_NET --samp_filter_precnn_net_ckpt_path $PRECNN_CKPT_PATH \
    --samp_filter_ce_percentile_threshold $FILTER_THRESH --samp_filter_batch_size $SAMP_BS \
    2>&1 | tee output_cDR-RS_filtering_${PRECNN_NET}_thresh_${FILTER_THRESH}_adjust_False.txt


PRECNN_NET="ResNet50"
PRECNN_CKPT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/RepDistiller/output/teacher_models/vanilla/ckpt_${PRECNN_NET}_epoch_240_last.pth"
## cDR-RS + filtering
FILTER_THRESH=0.9
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED \
    --gan_net $GAN_NET --gan_ckpt_path $GAN_CKPT_PATH \
    --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN \
    --samp_nfake_per_class $SAMP_NFAKE_PER_CLASS \
    --subsampling \
    --dre_precnn_net $DRE_PRECNN --dre_precnn_epochs $DRE_PRECNN_EPOCHS --dre_precnn_resume_epoch 0 \
    --dre_precnn_lr_base 0.1 --dre_precnn_lr_decay_factor 0.1 --dre_precnn_lr_decay_epochs "150_250" \
    --dre_precnn_batch_size_train $DRE_PRECNN_BS --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
    --dre_net $DRE_DR --dre_epochs $DRE_DR_EPOCHS --dre_resume_epoch 0 \
    --dre_lr_base $DRE_DR_LR_BASE --dre_batch_size $DRE_DR_BS --dre_lambda $DRE_DR_LAMBDA \
    --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs "80_150" \
    --filter \
    --samp_filter_precnn_net $PRECNN_NET --samp_filter_precnn_net_ckpt_path $PRECNN_CKPT_PATH \
    --samp_filter_ce_percentile_threshold $FILTER_THRESH --samp_filter_batch_size $SAMP_BS \
    2>&1 | tee output_cDR-RS_filtering_${PRECNN_NET}_thresh_${FILTER_THRESH}_adjust_False.txt