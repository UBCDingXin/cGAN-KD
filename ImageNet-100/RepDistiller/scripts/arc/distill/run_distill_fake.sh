#!/bin/bash

#PBS -l walltime=48:00:00,select=1:ncpus=1:ompthreads=1:ngpus=1:gpu_mem=32gb:mem=72gb
#PBS -N I_DisFa1
#PBS -A st-zjanew-1-gpu
#PBS -m abe
#PBS -M xin.ding@stat.ubc.ca
#PBS -o output_DisFake1.txt
#PBS -e error_DisFake1.txt

################################################################################

module unuse /arc/software/spack/share/spack/lmod/linux-centos7-x86_64/Core
module use /arc/software/spack-0.14.0-110/share/spack/lmod/linux-centos7-x86_64/Core

module load gcc
module load cuda
module load openmpi/3.1.5
module load openblas/0.3.9
module load py-torch/1.4.1-py3.7.6
module load py-torchvision/0.5.0-py3.7.6
module load py-pyparsing/2.4.2-py3.7.6
module load py-tqdm/4.36.1-py3.7.6
module load py-pillow/7.0.0-py3.7.6
module load py-cycler/0.10.0-py3.7.6
module load freetype/2.10.1
module load libpng/1.6.37
module load py-setuptools/41.4.0-py3.7.6
module load py-python-dateutil/2.8.0-py3.7.6
module load py-kiwisolver/1.1.0-py3.7.6
module load py-matplotlib
module load py-h5py
module load py-scipy

cd $PBS_O_WORKDIR

ROOT_PATH="/scratch/st-zjanew-1/dingxin9/cGAN-KD/ImageNet-100/RepDistiller"
DATA_PATH="/scratch/st-zjanew-1/dingxin9/datasets/ImageNet-100"

FILTER_NET="densenet161"
PERC="0.90"
FAKE_DATA_PATH="/scratch/st-zjanew-1/dingxin9/cGAN-KD/ImageNet-100/make_fake_datasets/fake_data/imagenet100_fake_images_BigGANdeep_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_${FILTER_NET}_perc_${PERC}_adjust_False_NfakePerClass_3000_seed_2021.h5"
NFAKE=100000



TEACHER="resnet110"
TEACHER_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${TEACHER}_epoch_240_last.pth"
STUDENT="resnet20"
INIT_STUDENT_PATH="${ROOT_PATH}/output/teacher_models/vanilla/ckpt_${STUDENT}_epoch_240_last.pth"

# # KD
# python train_student.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH \
#     --path_t $TEACHER_PATH --distill kd --model_s $STUDENT -r 0.1 -a 0.9 -b 0 --resume_epoch 0 \
#     --batch_size 256 --learning_rate 0.01 --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_student_path $INIT_STUDENT_PATH \
#     2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_kd_fake.txt

# # FitNet
# python train_student.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH \
#     --path_t $TEACHER_PATH --distill hint --model_s $STUDENT -a 1 -b 100 --resume_epoch 0 \
#     --batch_size 256 --learning_rate 0.01 --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_student_path $INIT_STUDENT_PATH \
#     2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_FitNet_fake.txt

# # VID+KD
# python train_student.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH \
#     --path_t $TEACHER_PATH --distill vid --model_s $STUDENT -a 1 -b 1 --resume_epoch 0 \
#     --batch_size 256 --learning_rate 0.01 --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_student_path $INIT_STUDENT_PATH \
#     2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_VID_fake.txt

# # RKD
# python train_student.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH \
#     --path_t $TEACHER_PATH --distill rkd --model_s $STUDENT -a 1 -b 1 --resume_epoch 0 \
#     --batch_size 256 --learning_rate 0.01 --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
#     --finetune --init_student_path $INIT_STUDENT_PATH \
#     2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_rkd_fake.txt

# CRD
resume_epoch=60
python train_student.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --path_t $TEACHER_PATH --distill crd --model_s $STUDENT -a 1 -b 0.8 --resume_epoch $resume_epoch --save_freq 10 \
    --batch_size 256 --learning_rate 0.01 --use_fake_data --fake_data_path $FAKE_DATA_PATH --nfake $NFAKE \
    --finetune --init_student_path $INIT_STUDENT_PATH \
    2>&1 | tee output_S_${STUDENT}_T_${TEACHER}_CRD_fake.txt
