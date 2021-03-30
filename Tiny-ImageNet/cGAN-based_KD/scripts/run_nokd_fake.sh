ROOT_PATH="./Tiny-ImageNet/cGAN-based_KD"
SEED=2020
NUM_CLASSES=200

## for real+fake
FAKE_DATASET_NAME="BigGAN_subsampling_True_FilterCEPct_0.5_nfake_300000" ## Format: BigGAN_..._nfake_xxx; Strings between GAN name and nfake value.; do not include seed
CNN_EPOCHS=350
LR_BASE=0.1
LR_DECAY_EPOCHS="150_250"
SAVE_FREQ="50_100_150_200_250_300_350"
NFAKE=100000
WD=1e-4


CNN_NAME="ShuffleNet"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base 0.1 --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \
2>&1 | tee output_baseline_${CNN_NAME}_fake_None_seed_${SEED}.txt

CNN_NAME="VGG11"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base 0.1 --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \
2>&1 | tee output_baseline_${CNN_NAME}_fake_None_seed_${SEED}.txt
