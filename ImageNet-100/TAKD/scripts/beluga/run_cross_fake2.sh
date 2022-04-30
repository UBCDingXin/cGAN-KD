#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=C_TA_CR_Fake2
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
FILTER_NET="densenet121"
PERC="0.90"
FAKE_DATA_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/make_fake_datasets/fake_data/cifar100_fake_images_BigGAN_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_${FILTER_NET}_perc_${PERC}_adjust_True_NfakePerClass_5000_seed_2021.h5"
NFAKE=100000



## ===========================================================================================================================================
TEACHER="resnet32x4"
ASSISTANT="vgg11"
STUDENT="MobileNetV2"
TEACHER_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
INIT_ASSISTANT_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${ASSISTANT}_epoch_240_last.pth"
INIT_STUDENT_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${STUDENT}_epoch_240_last.pth"

python takd.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --student $STUDENT --assistant $ASSISTANT --teacher_ckpt_path $TEACHER_PATH \
    --epochs 240 --resume_epoch_1 0 --resume_epoch_2 0 \
    --batch_size_train 128 --lr_base1 0.01 --lr_base2 0.01 --lr_decay_epochs "150_180_210" --transform \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_assistant_path $INIT_ASSISTANT_PATH --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_TA_${ASSISTANT}_T_${TEACHER}_fake.txt



## ===========================================================================================================================================
TEACHER="resnet32x4"
ASSISTANT="vgg13"
STUDENT="vgg8"
TEACHER_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
INIT_ASSISTANT_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${ASSISTANT}_epoch_240_last.pth"
INIT_STUDENT_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${STUDENT}_epoch_240_last.pth"

python takd.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --student $STUDENT --assistant $ASSISTANT --teacher_ckpt_path $TEACHER_PATH \
    --epochs 240 --resume_epoch_1 0 --resume_epoch_2 0 \
    --batch_size_train 128 --lr_base1 0.01 --lr_base2 0.01 --lr_decay_epochs "150_180_210" --transform \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_assistant_path $INIT_ASSISTANT_PATH --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_TA_${ASSISTANT}_T_${TEACHER}_fake.txt



## ===========================================================================================================================================
TEACHER="resnet32x4"
ASSISTANT="wrn_40_2"
STUDENT="ShuffleV1"
TEACHER_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
INIT_ASSISTANT_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${ASSISTANT}_epoch_240_last.pth"
INIT_STUDENT_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${STUDENT}_epoch_240_last.pth"

python takd.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --student $STUDENT --assistant $ASSISTANT --teacher_ckpt_path $TEACHER_PATH \
    --epochs 240 --resume_epoch_1 0 --resume_epoch_2 0 \
    --batch_size_train 128 --lr_base1 0.01 --lr_base2 0.01 --lr_decay_epochs "150_180_210" --transform \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_assistant_path $INIT_ASSISTANT_PATH --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_TA_${ASSISTANT}_T_${TEACHER}_fake.txt



## ===========================================================================================================================================
TEACHER="resnet32x4"
ASSISTANT="wrn_40_2"
STUDENT="ShuffleV2"
TEACHER_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
INIT_ASSISTANT_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${ASSISTANT}_epoch_240_last.pth"
INIT_STUDENT_PATH="/scratch/dingx92/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_${STUDENT}_epoch_240_last.pth"

python takd.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --student $STUDENT --assistant $ASSISTANT --teacher_ckpt_path $TEACHER_PATH \
    --epochs 240 --resume_epoch_1 0 --resume_epoch_2 0 \
    --batch_size_train 128 --lr_base1 0.01 --lr_base2 0.01 --lr_decay_epochs "150_180_210" --transform \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_assistant_path $INIT_ASSISTANT_PATH --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_TA_${ASSISTANT}_T_${TEACHER}_fake.txt