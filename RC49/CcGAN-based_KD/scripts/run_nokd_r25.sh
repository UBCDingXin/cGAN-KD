ROOT_PATH="./RC-49/CcGAN-based_KD"
REAL_DATA_PATH="./RC-49/dataset"

SEED=2020
NTRAIN_PER_LABEL=25
NCPU=0

## for real
FAKE_DATASET_NAME="None"
CNN_EPOCHS=350
LR_BASE=0.01
LR_DECAY_EPOCHS="150_250"
SAVE_FREQ="50_100_150_200_250_300_350"
NFAKE=1e30



CNN_NAME="ShuffleNet"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
2>&1 | tee output_baseline_${CNN_NAME}_real_NTrainPerLabel_${NTRAIN_PER_LABEL}_fake_${FAKE_DATASET_NAME}_seed_${SEED}.txt


CNN_NAME="MobileNet"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
2>&1 | tee output_baseline_${CNN_NAME}_real_NTrainPerLabel_${NTRAIN_PER_LABEL}_fake_${FAKE_DATASET_NAME}_seed_${SEED}.txt


CNN_NAME="efficientnet-b0"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
2>&1 | tee output_baseline_${CNN_NAME}_real_NTrainPerLabel_${NTRAIN_PER_LABEL}_fake_${FAKE_DATASET_NAME}_seed_${SEED}.txt


CNN_NAME="VGG11"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
2>&1 | tee output_baseline_${CNN_NAME}_real_NTrainPerLabel_${NTRAIN_PER_LABEL}_fake_${FAKE_DATASET_NAME}_seed_${SEED}.txt


CNN_NAME="VGG13"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
2>&1 | tee output_baseline_${CNN_NAME}_real_NTrainPerLabel_${NTRAIN_PER_LABEL}_fake_${FAKE_DATASET_NAME}_seed_${SEED}.txt



CNN_NAME="VGG16"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
2>&1 | tee output_baseline_${CNN_NAME}_real_NTrainPerLabel_${NTRAIN_PER_LABEL}_fake_${FAKE_DATASET_NAME}_seed_${SEED}.txt


CNN_NAME="ResNet18"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
2>&1 | tee output_baseline_${CNN_NAME}_real_NTrainPerLabel_${NTRAIN_PER_LABEL}_fake_${FAKE_DATASET_NAME}_seed_${SEED}.txt


CNN_NAME="ResNet50"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
2>&1 | tee output_baseline_${CNN_NAME}_real_NTrainPerLabel_${NTRAIN_PER_LABEL}_fake_${FAKE_DATASET_NAME}_seed_${SEED}.txt


CNN_NAME="DenseNet121"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
2>&1 | tee output_baseline_${CNN_NAME}_real_NTrainPerLabel_${NTRAIN_PER_LABEL}_fake_${FAKE_DATASET_NAME}_seed_${SEED}.txt
