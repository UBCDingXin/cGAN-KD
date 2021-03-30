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
TA_net="MobileNet"
assistant_lambda_kd=0.5
assistant_t_kd=5


student_net="VGG11"
student_lambda_kd=0.5
student_t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "TAKD ${teacher_net}-->${TA_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 takd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_98179_fake_None.pth \
--teacher_assistant $TA_net \
--student $student_net \
--seed $SEED --num_workers $NCPU \
--num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch_1 0 --resume_epoch_2 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--assistant_lambda_kd $assistant_lambda_kd --assistant_T_kd $assistant_t_kd \
--student_lambda_kd $student_lambda_kd --student_T_kd $student_t_kd \
2>&1 | tee output_takd_${teacher_net}+${TA_net}+${student_net}_TAlambda_${assistant_lambda_kd}_TAT_${assistant_t_kd}_Stulambda_${student_lambda_kd}_StuT_${student_t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt


student_net="ShuffleNet"
student_lambda_kd=0.5
student_t_kd=5
echo "-------------------------------------------------------------------------------------------------"
echo "TAKD ${teacher_net}-->${TA_net}-->${student_net}"
CUDA_VISIBLE_DEVICES=0 python3 takd.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME --nfake $NFAKE \
--teacher $teacher_net --teacher_ckpt_filename ckpt_baseline_${teacher_net}_epoch_350_transform_True_seed_2020_data_real_nreal_98179_fake_None.pth \
--teacher_assistant $TA_net \
--student $student_net \
--seed $SEED --num_workers $NCPU \
--num_classes $NUM_CLASSES \
--epochs $EPOCHS --resume_epoch_1 0 --resume_epoch_2 0 --save_freq $SAVE_FREQ --weight_decay 1e-4 --transform \
--lr_decay_epochs $LR_DECAY_EPOCHS \
--assistant_lambda_kd $assistant_lambda_kd --assistant_T_kd $assistant_t_kd \
--student_lambda_kd $student_lambda_kd --student_T_kd $student_t_kd \
2>&1 | tee output_takd_${teacher_net}+${TA_net}+${student_net}_TAlambda_${assistant_lambda_kd}_TAT_${assistant_t_kd}_Stulambda_${student_lambda_kd}_StuT_${student_t_kd}_nreal_${NTRAIN}_nfake_${NFAKE}.txt
