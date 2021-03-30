ROOT_PATH="./Tiny-ImageNet/cGAN-based_KD"
SEED=2020
NUM_CLASSES=200

FAKE_DATASET_NAME="None"
NFAKE=1e30
EPOCHS=350
LR_DECAY_EPOCHS="150_250"
SAVE_FREQ="25_50_75_100_125_150_175_200_225_250_275_300_325_350"
NCPU=0

teacher_net="DenseNet121"

student_net="VGG11"
lambda_kd=0.5
t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "BLKD ${teacher_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 blkd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_98179_fake_None.pth \
--student $student_net --seed $SEED --num_workers $NCPU \
--num_classes $NUM_CLASSES \
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
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_98179_fake_None.pth \
--student $student_net --seed $SEED --num_workers $NCPU \
--num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--lambda_kd $lambda_kd --T_kd $t_kd \
2>&1 | tee output_blkd_${teacher_net}+None+${student_net}_lambda_${lambda_kd}_T_${t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt
