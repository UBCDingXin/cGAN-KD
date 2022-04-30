@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/cGAN-KD/CIFAR-100/RepDistiller"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/CIFAR-100/data"
set FAKE_DATA_PATH="G:/OneDrive/Working_directory/cGAN-KD/CIFAR-100/make_fake_datasets/fake_data/cifar100_fake_images_BigGAN_sampling_cDR-RS_precnn_ResNet34_lambda_0.000_DR_MLP5_lambda_0.010_filter_densenet121_perc_0.90_adjust_True_NfakePerClass_5000_seed_2021.h5"
set NFAKE=150000

set TEACHER="ResNet50"
set TEACHER_PATH="G:/OneDrive/Working_directory/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_%TEACHER%_epoch_240_last.pth"
set STUDENT="resnet20"
set INIT_STUDENT_PATH="G:/OneDrive/Working_directory/cGAN-KD/CIFAR-100/RepDistiller/output/teacher_models/vanilla/ckpt_%STUDENT%_epoch_240_last.pth"


@REM @REM vanilla: KD
@REM python train_student.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
@REM     --path_t %TEACHER_PATH% --distill kd --model_s %STUDENT% -r 0.1 -a 0.9 -b 0 --resume_epoch 0 ^ %*

@REM @REM vanilla: RKD+KD
@REM python train_student.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
@REM     --path_t %TEACHER_PATH% --distill rkd --model_s %STUDENT% -a 1 -b 1 --resume_epoch 0 ^ %*

@REM @REM vanilla: CRD+KD
@REM python train_student.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
@REM     --path_t %TEACHER_PATH% --distill crd --model_s %STUDENT% -a 1 -b 0.8 --resume_epoch 0 ^ %*

@REM @REM vanilla: AB+KD
@REM python train_student.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
@REM     --path_t %TEACHER_PATH% --distill abound --model_s %STUDENT% -a 1 -b 1 --resume_epoch 0 ^ %*



@REM fake: KD
python train_student.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --path_t %TEACHER_PATH% --distill kd --model_s %STUDENT% -r 0.1 -a 0.9 -b 0 --resume_epoch 0 ^
    --use_fake_data --fake_data_path %FAKE_DATA_PATH% --nfake %NFAKE% ^
    --finetune --init_student_path %INIT_STUDENT_PATH% %*

@REM fake: RKD+KD
python train_student.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --path_t %TEACHER_PATH% --distill rkd --model_s %STUDENT% -a 1 -b 1 --resume_epoch 0 ^
    --use_fake_data --fake_data_path %FAKE_DATA_PATH% --nfake %NFAKE% ^
    --finetune --init_student_path %INIT_STUDENT_PATH% %*

@REM fake: CRD+KD
python train_student.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --path_t %TEACHER_PATH% --distill crd --model_s %STUDENT% -a 1 -b 0.8 --resume_epoch 0 ^
    --use_fake_data --fake_data_path %FAKE_DATA_PATH% --nfake %NFAKE% ^
    --finetune --init_student_path %INIT_STUDENT_PATH% %*