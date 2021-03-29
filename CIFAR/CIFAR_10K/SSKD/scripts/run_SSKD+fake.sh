REAL_DATA="./CIFAR/CIFAR_10K/cGAN-based_KD/data/CIFAR10_trainset_10000_seed_2020.h5"
FAKE_DATA="./CIFAR/CIFAR_10K/cGAN-based_KD/data/CIFAR10_ntrain_10000_BigGAN_vanilla_epochs_2000_transform_True_subsampling_True_FilterCEPct_0.7_nfake_349999_seed_2020.h5"
NFAKE=100000

### ResNet18 --> vgg11 ;
python3 student.py --real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE \
--t-path ./experiments/teacher_ResNet18_seed0/ --s-arch vgg11 --lr 0.05 --weight-decay 5e-4 --gpu-id 0 \
2>&1 | tee output_t_ResNet18_s_vgg11_nfake_${NFAKE}.txt

### ResNet18 --> ShuffleV2 ;
python3 student.py --real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE \
--t-path ./experiments/teacher_ResNet18_seed0/ --s-arch ShuffleV2 --lr 0.01 --weight-decay 5e-4 --gpu-id 0 \
2>&1 | tee output_t_ResNet18_s_ShuffleV2_nfake_${NFAKE}.txt
