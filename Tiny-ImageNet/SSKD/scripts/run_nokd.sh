REAL_DATA="./Tiny-ImageNet/cGAN-based_KD/data/tiny-imagenet-200.h5"
FAKE_DATA="None"
NFAKE=1e30

# teacher training;
python3 teacher.py \
--real_data $REAL_DATA \
--arch ResNet50 --lr 0.05 --weight-decay 1e-4 --gpu-id 0 \
2>&1 | tee output_ResNet50.txt

python3 teacher.py \
--real_data $REAL_DATA \
--arch vgg11 --lr 0.05 --weight-decay 1e-4 --gpu-id 0 \
2>&1 | tee output_vgg11.txt

python3 teacher.py \
--real_data $REAL_DATA \
--arch ShuffleV2 --lr 0.01 --weight-decay 1e-4 --gpu-id 0 \
2>&1 | tee output_ShuffleV2.txt
