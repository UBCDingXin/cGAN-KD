REAL_DATA="./CIFAR/CIFAR_50K/cGAN-based_KD/data/CIFAR10_trainset_50000_seed_2020.h5"
FAKE_DATA="None"

NFAKE=1e30
NCPU=8


KD_METHOD="uda"


TEACHER="vgg11"
STUDENT="vgg11"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}.txt


TEACHER="shufflenetv2"
STUDENT="shufflenetv2"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}.txt


TEACHER="efficientnet-b0"
STUDENT="efficientnet-b0"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}.txt
