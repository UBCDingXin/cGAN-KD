ROOT_PATH="./UTKFace/CcGAN-based_KD"
REAL_DATA_PATH="./UTKFace/dataset"

NCPU=0
SEED=2020
CNN_EPOCHS=350
LR_BASE=0.01
LR_DECAY_EPOCHS="150_250"
SAVE_FREQ="50_100_150_200_250_300_350"
NFAKE=80000

###=====================================================================================================================
FAKE_DATASET_NAME="SAGAN_hinge_niters_40000_subsampling_cDRE-F-SP+RS_hard_1e-20_FilterMAEPct_0.7_nfake_210000"

CNN_NAME="ShuffleNet"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
--validaiton_mode \
2>&1 | tee output_baseline_${CNN_NAME}_fake_${FAKE_DATASET_NAME}_seed_${SEED}_CV.txt

CNN_NAME="MobileNet"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
--validaiton_mode \
2>&1 | tee output_baseline_${CNN_NAME}_fake_${FAKE_DATASET_NAME}_seed_${SEED}_CV.txt

CNN_NAME="efficientnet-b0"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --fake_dataset_name $FAKE_DATASET_NAME --num_workers $NCPU \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ \
--batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS \
--weight_decay 1e-4 \
--validaiton_mode \
2>&1 | tee output_baseline_${CNN_NAME}_fake_${FAKE_DATASET_NAME}_seed_${SEED}_CV.txt
