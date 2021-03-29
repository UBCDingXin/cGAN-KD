ROOT_PATH="./CIFAR/CIFAR_10K/cGAN-based_KD"

SEED=2020
NUM_CLASSES=10
NTRAIN=10000

FAKE_DATASET_NAME="BigGAN_vanilla_epochs_2000_transform_True_subsampling_True_FilterCEPct_0.7_nfake_349999"
NFAKE=100000
EPOCHS=350
LR_DECAY_EPOCHS="150_250"
SAVE_FREQ="50_100_150_200_250_300_350"
NCPU=0


######################################################################################################
# MobileNet V2 ---> VGG11/ShuffleNetV2/efficientnet-b0
teacher_net="MobileNet"

student_net="VGG11"
lambda_kd=0.5
t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "BLKD ${teacher_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 blkd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_${NTRAIN}_fake_None.pth \
--student $student_net --seed $SEED --num_workers $NCPU \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--lambda_kd $lambda_kd --T_kd $t_kd \
2>&1 | tee output_blkd_${teacher_net}+None+${student_net}_lambda_${lambda_kd}_T_${t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt


student_net="ShuffleNet"
lambda_kd=0.5
t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "BLKD ${teacher_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 blkd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_${NTRAIN}_fake_None.pth \
--student $student_net --seed $SEED --num_workers $NCPU \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--lambda_kd $lambda_kd --T_kd $t_kd \
2>&1 | tee output_blkd_${teacher_net}+None+${student_net}_lambda_${lambda_kd}_T_${t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt


student_net="efficientnet-b0"
lambda_kd=0.5
t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "BLKD ${teacher_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 blkd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_${NTRAIN}_fake_None.pth \
--student $student_net --seed $SEED --num_workers $NCPU \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--lambda_kd $lambda_kd --T_kd $t_kd \
2>&1 | tee output_blkd_${teacher_net}+None+${student_net}_lambda_${lambda_kd}_T_${t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt






######################################################################################################
# ResNet18 ---> VGG11/ShuffleNetV2/efficientnet-b0
teacher_net="ResNet18"

student_net="VGG11"
lambda_kd=0.5
t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "BLKD ${teacher_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 blkd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_${NTRAIN}_fake_None.pth \
--student $student_net --seed $SEED --num_workers $NCPU \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--lambda_kd $lambda_kd --T_kd $t_kd \
2>&1 | tee output_blkd_${teacher_net}+None+${student_net}_lambda_${lambda_kd}_T_${t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt


student_net="ShuffleNet"
lambda_kd=0.5
t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "BLKD ${teacher_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 blkd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_${NTRAIN}_fake_None.pth \
--student $student_net --seed $SEED --num_workers $NCPU \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--lambda_kd $lambda_kd --T_kd $t_kd \
2>&1 | tee output_blkd_${teacher_net}+None+${student_net}_lambda_${lambda_kd}_T_${t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt


student_net="efficientnet-b0"
lambda_kd=0.5
t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "BLKD ${teacher_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 blkd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_${NTRAIN}_fake_None.pth \
--student $student_net --seed $SEED --num_workers $NCPU \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--lambda_kd $lambda_kd --T_kd $t_kd \
2>&1 | tee output_blkd_${teacher_net}+None+${student_net}_lambda_${lambda_kd}_T_${t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt









######################################################################################################
# DenseNet121 ---> VGG11/ShuffleNetV2/efficientnet-b0
teacher_net="DenseNet121"

student_net="VGG11"
lambda_kd=0.5
t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "BLKD ${teacher_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 blkd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_${NTRAIN}_fake_None.pth \
--student $student_net --seed $SEED --num_workers $NCPU \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--lambda_kd $lambda_kd --T_kd $t_kd \
2>&1 | tee output_blkd_${teacher_net}+None+${student_net}_lambda_${lambda_kd}_T_${t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt


student_net="ShuffleNet"
lambda_kd=0.5
t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "BLKD ${teacher_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 blkd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_${NTRAIN}_fake_None.pth \
--student $student_net --seed $SEED --num_workers $NCPU \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--lambda_kd $lambda_kd --T_kd $t_kd \
2>&1 | tee output_blkd_${teacher_net}+None+${student_net}_lambda_${lambda_kd}_T_${t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt


student_net="efficientnet-b0"
lambda_kd=0.5
t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "BLKD ${teacher_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 blkd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_${NTRAIN}_fake_None.pth \
--student $student_net --seed $SEED --num_workers $NCPU \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--lambda_kd $lambda_kd --T_kd $t_kd \
2>&1 | tee output_blkd_${teacher_net}+None+${student_net}_lambda_${lambda_kd}_T_${t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt
