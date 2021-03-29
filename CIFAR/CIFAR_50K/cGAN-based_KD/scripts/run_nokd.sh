ROOT_PATH="./CIFAR/CIFAR_50K/cGAN-based_KD"
SEED=2020
NUM_CLASSES=10
NTRAIN=50000

## for real
FAKE_DATASET_NAME="None"
CNN_EPOCHS=350
LR_BASE=0.1
LR_DECAY_EPOCHS="150_250"
SAVE_FREQ="50_100_150_200_250_300_350"
NFAKE=1e30
WD=1e-4



CNN_NAME="ShuffleNet"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--ntrain $NTRAIN --nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base 0.1 --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \

CNN_NAME="MobileNet"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--ntrain $NTRAIN --nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base 0.1 --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \

CNN_NAME="efficientnet-b0"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--ntrain $NTRAIN --nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base 0.1 --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \

CNN_NAME="VGG11"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--ntrain $NTRAIN --nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base 0.1 --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \

CNN_NAME="VGG13"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--ntrain $NTRAIN --nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base 0.1 --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \

CNN_NAME="VGG16"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--ntrain $NTRAIN --nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base 0.1 --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \

CNN_NAME="ResNet18"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--ntrain $NTRAIN --nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base 0.1 --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \

CNN_NAME="ResNet50"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--ntrain $NTRAIN --nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base 0.1 --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \

CNN_NAME="DenseNet121"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline ${CNN_NAME}"
CUDA_VISIBLE_DEVICES=0 python3 baseline_cnn.py \
--root_path $ROOT_PATH --fake_dataset_name $FAKE_DATASET_NAME \
--cnn_name $CNN_NAME --seed $SEED \
--ntrain $NTRAIN --nfake $NFAKE --num_classes $NUM_CLASSES \
--epochs $CNN_EPOCHS --resume_epoch 0 --save_freq $SAVE_FREQ --batch_size_train 128 --lr_base 0.1 --lr_decay_factor 0.1 --lr_decay_epochs $LR_DECAY_EPOCHS --weight_decay $WD --transform \
