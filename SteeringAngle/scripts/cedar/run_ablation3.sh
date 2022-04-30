#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=SA_abla_3-1
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


SETTING="ablation3-1"
FAKE_DATA_PATH="${ROOT_PATH}/output/fake_data/steeringangle_fake_images_SAGAN_cDR-RS_presae_epochs_200_DR_MLP5_epochs_200_lambda_0.010_filter_vgg8_perc_0.70_adjust_True_Nlabel_2000_NFakePerLabel_50_seed_2020.h5"
NFAKE=50000


ARCH="resnet20"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${SETTING}_${ARCH}_nfake_${NFAKE}.txt


ARCH="resnet56"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${SETTING}_${ARCH}_nfake_${NFAKE}.txt


ARCH="resnet8x4"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${SETTING}_${ARCH}_nfake_${NFAKE}.txt


ARCH="wrn_16_1"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${SETTING}_${ARCH}_nfake_${NFAKE}.txt


ARCH="wrn_40_1"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${SETTING}_${ARCH}_nfake_${NFAKE}.txt


ARCH="ShuffleV1"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${SETTING}_${ARCH}_nfake_${NFAKE}.txt


ARCH="MobileNetV2"
INIT_MODEL_PATH="${ROOT_PATH}/output/CNN/vanilla/ckpt_${ARCH}_epoch_350_last.pth"
python baseline_cnn.py \
    --cnn_name $ARCH --root_path $ROOT_PATH --data_path $DATA_PATH \
    --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --epochs 350 --resume_epoch 0 --save_freq 25 \
    --batch_size_train 128 --lr_base 0.01 --lr_decay_epochs "150_250" \
    --weight_decay 5e-4 \
    --finetune --init_model_path $INIT_MODEL_PATH \
    2>&1 | tee output_${SETTING}_${ARCH}_nfake_${NFAKE}.txt
