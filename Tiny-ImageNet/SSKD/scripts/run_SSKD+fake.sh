REAL_DATA="./Tiny-ImageNet/cGAN-based_KD/data/tiny-imagenet-200.h5"
FAKE_DATA="./Tiny-ImageNet/cGAN-based_KD/output/fake_data/Tiny-ImageNet_BigGAN_subsampling_True_FilterCEPct_0.5_nfake_300000_seed_2020.h5"
NFAKE=100000
BATCH_SIZE=32

### ResNet50 --> vgg11 ;
python3 student.py --real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --batch-size $BATCH_SIZE \
--t-path ./experiments/teacher_ResNet50_seed0/ --s-arch vgg11 --lr 0.05 --weight-decay 1e-4 --gpu-id 0 \
2>&1 | tee output_t_ResNet50_s_vgg11_nfake_${NFAKE}.txt

### ResNet50 --> ShuffleV2 ;
python3 student.py --real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --batch-size $BATCH_SIZE \
--t-path ./experiments/teacher_ResNet50_seed0/ --s-arch ShuffleV2 --lr 0.01 --weight-decay 1e-4 --gpu-id 0 \
2>&1 | tee output_t_ResNet50_s_ShuffleV2_nfake_${NFAKE}.txt
