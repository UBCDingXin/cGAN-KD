#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=nI_SK_Dis1
#SBATCH --output=%x-%j.out


module load arch/avx2 StdEnv/2020
module load cuda/11.0
module load python/3.9.6
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req

ROOT_PATH="/scratch/dingx92/cGAN-KD/ImageNet-100/SSKD"
DATA_PATH="/scratch/dingx92/datasets/ImageNet-100"


# ##########################################################################
# resume_epoch=0
# TEACHER="resnet110"
# STUDENT="resnet20"
# TEACHER_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
# python student.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --s_arch $STUDENT --t_path $TEACHER_PATH \
#     --t_epochs 60 --epochs 240 --resume_epoch $resume_epoch --save_interval 10 \
#     --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_vanilla.txt



# ##########################################################################
# resume_epoch=0
# TEACHER="wrn_40_2"
# STUDENT="wrn_40_1"
# TEACHER_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
# python student.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --s_arch $STUDENT --t_path $TEACHER_PATH \
#     --t_epochs 60 --epochs 240 --resume_epoch $resume_epoch --save_interval 10 \
#     --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_vanilla.txt


##########################################################################
resume_epoch=20
TEACHER="vgg19"
STUDENT="ShuffleV1"
TEACHER_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
python student.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --s_arch $STUDENT --t_path $TEACHER_PATH \
    --t_epochs 60 --epochs 240 --resume_epoch $resume_epoch --save_interval 10 \
    --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_vanilla.txt



##########################################################################
TEACHER="ResNet34"
STUDENT="wrn_40_1"
TEACHER_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
python student.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --s_arch $STUDENT --t_path $TEACHER_PATH \
    --t_epochs 60 --epochs 240 --resume_epoch 0 --save_interval 10 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_vanilla.txt



# ##########################################################################
# TEACHER="vgg13"
# STUDENT="MobileNetV2"
# TEACHER_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
# python student.py \
#     --root_path $ROOT_PATH --real_data $DATA_PATH \
#     --s_arch $STUDENT --t_path $TEACHER_PATH \
#     --t_epochs 60 --epochs 240 --resume_epoch 0 --save_interval 10 \
#     --batch_size 128 --lr 0.01 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
#     2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_vanilla.txt



##########################################################################
TEACHER="ResNet50"
STUDENT="vgg8"
TEACHER_PATH="${ROOT_PATH}/output/teachers/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
python student.py \
    --root_path $ROOT_PATH --real_data $DATA_PATH \
    --s_arch $STUDENT --t_path $TEACHER_PATH \
    --t_epochs 60 --epochs 240 --resume_epoch 0 --save_interval 10 \
    --batch_size 128 --lr 0.05 --lr_decay_epochs "150_180_210" --weight_decay 5e-4 \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_vanilla.txt
