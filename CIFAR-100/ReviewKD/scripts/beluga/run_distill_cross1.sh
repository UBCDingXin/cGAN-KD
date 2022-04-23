#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=C_Re_DisCro1
#SBATCH --output=%x-%j.out


module load arch/avx2 StdEnv/2020
module load cuda/11.0
module load python/3.9.6
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req

ROOT_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/ReviewKD"
DATA_PATH="/scratch/dingx92/datasets/CIFAR-100/data"


TEACHER="wrn_40_2"
STUDENT="ShuffleV1"
TEACHER_PATH="${ROOT_PATH}/output/vanilla/models/ckpt_${TEACHER}_epoch_240_last.pth"
python train.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --model $STUDENT --teacher $TEACHER --teacher-weight $TEACHER_PATH \
    --kd-loss-weight 5.0 --mode "distill" \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_ReviewKD.txt


TEACHER="wrn_40_2"
STUDENT="ShuffleV2"
TEACHER_PATH="${ROOT_PATH}/output/vanilla/models/ckpt_${TEACHER}_epoch_240_last.pth"
python train.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --model $STUDENT --teacher $TEACHER --teacher-weight $TEACHER_PATH \
    --kd-loss-weight 5.0 --mode "distill" \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_ReviewKD.txt