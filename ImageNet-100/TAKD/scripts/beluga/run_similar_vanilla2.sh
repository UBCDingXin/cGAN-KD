#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=C_TA_SIM_VA2
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req


ROOT_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/TAKD"
DATA_PATH="/scratch/dingx92/datasets/CIFAR-100/data"


## ===========================================================================================================================================
TEACHER="resnet32x4"
TEACHER_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
ASSISTANT="resnet110"
STUDENT="resnet8x4"

python takd.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --student $STUDENT --assistant $ASSISTANT --teacher_ckpt_path $TEACHER_PATH \
    --epochs 240 --resume_epoch_1 0 --resume_epoch_2 0 \
    --batch_size_train 64 --lr_base1 0.05 --lr_base2 0.05 --lr_decay_epochs "150_180_210" --transform \
    2>&1 | tee output_S_${STUDENT}_TA_${ASSISTANT}_T_${TEACHER}_vanilla.txt


## ===========================================================================================================================================
TEACHER="vgg13"
TEACHER_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
ASSISTANT="vgg11"
STUDENT="vgg8"

python takd.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --student $STUDENT --assistant $ASSISTANT --teacher_ckpt_path $TEACHER_PATH \
    --epochs 240 --resume_epoch_1 0 --resume_epoch_2 0 \
    --batch_size_train 64 --lr_base1 0.05 --lr_base2 0.05 --lr_decay_epochs "150_180_210" --transform \
    2>&1 | tee output_S_${STUDENT}_TA_${ASSISTANT}_T_${TEACHER}_vanilla.txt



## ===========================================================================================================================================
TEACHER="vgg19"
TEACHER_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
ASSISTANT="vgg11"
STUDENT="vgg8"

python takd.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --student $STUDENT --assistant $ASSISTANT --teacher_ckpt_path $TEACHER_PATH \
    --epochs 240 --resume_epoch_1 0 --resume_epoch_2 0 \
    --batch_size_train 64 --lr_base1 0.05 --lr_base2 0.05 --lr_decay_epochs "150_180_210" --transform \
    2>&1 | tee output_S_${STUDENT}_TA_${ASSISTANT}_T_${TEACHER}_vanilla.txt