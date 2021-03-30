REAL_DATA="./Tiny-ImageNet/cGAN-based_KD/data/tiny-imagenet-200.h5"
FAKE_DATA="None"
NFAKE=1e30
BATCH_SIZE=32

### ResNet50 --> vgg11 ;
python3 student.py --real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --batch-size $BATCH_SIZE \
--t-path ./experiments/teacher_ResNet50_seed0/ --s-arch vgg11 --lr 0.05 --weight-decay 1e-4 --gpu-id 0 \
2>&1 | tee output_t_ResNet50_s_vgg11.txt

### ResNet50 --> ShuffleV2 ;
python3 student.py --real_data $REAL_DATA --fake_data $FAKE_DATA --nfake $NFAKE --batch-size $BATCH_SIZE \
--t-path ./experiments/teacher_ResNet50_seed0/ --s-arch ShuffleV2 --lr 0.01 --weight-decay 1e-4 --gpu-id 0 \
2>&1 | tee output_t_ResNet50_s_ShuffleV2.txt
