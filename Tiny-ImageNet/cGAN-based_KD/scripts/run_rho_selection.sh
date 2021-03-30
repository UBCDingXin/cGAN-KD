ROOT_PATH="./Tiny-ImageNet/cGAN-based_KD"
SEED=2020
NUM_CLASSES=200

## for real+fake
CNN_EPOCHS=350
LR_BASE=0.1
LR_DECAY_EPOCHS="150_250"
SAVE_FREQ="25_50_75_100_125_150_175_200_225_250_275_300_325_350"
NFAKE=100000
WD=1e-4



###################################################################################################################
FAKE_DATASET_NAME="BigGAN_subsampling_True_FilterCEPct_0.5_nfake_300000"

CNN_NAME="ShuffleNet"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \
--validaiton_mode \
2>&1 | tee output_baseline_${CNN_NAME}_fake_${FAKE_DATASET_NAME}_seed_${SEED}_CV.txt

CNN_NAME="VGG11"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base $LR_BASE --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \
--validaiton_mode \
2>&1 | tee output_baseline_${CNN_NAME}_fake_${FAKE_DATASET_NAME}_seed_${SEED}_CV.txt
