REAL_DATA="./CIFAR/CIFAR_20K/cGAN-based_KD/data/CIFAR10_trainset_20000_seed_2020.h5"
FAKE_DATA="./CIFAR/CIFAR_20K/cGAN-based_KD/data/CIFAR10_ntrain_20000_BigGAN_vanilla_epochs_2000_transform_True_subsampling_True_FilterCEPct_0.6_nfake_299969_seed_2020.h5"
NFAKE=100000
NCPU=0


KD_METHOD="uda"



######################################################################################################
# MobileNet V2 ---> VGG11/ShuffleNetV2/efficientnet-b0
TEACHER="mobilenetv2"
STUDENT="vgg11"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/mobilenetv2_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt

TEACHER="mobilenetv2"
STUDENT="shufflenetv2"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/mobilenetv2_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt

TEACHER="mobilenetv2"
STUDENT="efficientnet-b0"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/mobilenetv2_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt



######################################################################################################
# resnet18 ---> VGG11/ShuffleNetV2/efficientnet-b0
TEACHER="resnet18"
STUDENT="vgg11"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/mobilenetv2_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt

TEACHER="resnet18"
STUDENT="shufflenetv2"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/mobilenetv2_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt

TEACHER="resnet18"
STUDENT="efficientnet-b0"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/mobilenetv2_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt



######################################################################################################
# densenet121 ---> VGG11/ShuffleNetV2/efficientnet-b0
TEACHER="densenet121"
STUDENT="vgg11"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/mobilenetv2_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt

TEACHER="densenet121"
STUDENT="shufflenetv2"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/mobilenetv2_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt

TEACHER="densenet121"
STUDENT="efficientnet-b0"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/mobilenetv2_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt
