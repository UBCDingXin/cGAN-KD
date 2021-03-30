REAL_DATA="./Tiny-ImageNet/cGAN-based_KD/data/tiny-imagenet-200.h5"
FAKE_DATA="None"

NFAKE=1e30
NCPU=0
KD_METHOD="uda"

TEACHER="resnet50"
STUDENT="vgg11"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/resnet50_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt

TEACHER="resnet50"
STUDENT="shufflenetv2"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/resnet50_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt

TEACHER="densenet121"
STUDENT="vgg11"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/densenet121_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt

TEACHER="densenet121"
STUDENT="shufflenetv2"
echo "-------------------------------------------------------------------------------------------------"
echo "KD: ${KD_METHOD}; teacher: ${TEACHER}; student: ${STUDENT}"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_kd.py \
--real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --num_workers $NCPU \
--mode $KD_METHOD \
--teacher $TEACHER --student $STUDENT \
--teacher-checkpoint pretrained/densenet121_teacher_best.pth \
2>&1 | tee output_${KD_METHOD}_teacher_${TEACHER}_student_${STUDENT}_nfake_${NFAKE}.txt
