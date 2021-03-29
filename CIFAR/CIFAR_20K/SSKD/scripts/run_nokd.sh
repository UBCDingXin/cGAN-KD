REAL_DATA="./CIFAR/CIFAR_20K/cGAN-based_KD/data/CIFAR10_trainset_20000_seed_2020.h5"
FAKE_DATA="None"
NFAKE=1e30

# teacher training;
python3 teacher.py \
--real_data $REAL_DATA \
--arch MobileNetV2 --lr 0.01 --weight-decay 5e-4 --gpu-id 0 \
2>&1 | tee output_MobileNetV2.txt

python3 teacher.py \
--real_data $REAL_DATA \
--arch ResNet18 --lr 0.05 --weight-decay 5e-4 --gpu-id 0 \
2>&1 | tee output_ResNet18.txt

python3 teacher.py \
--real_data $REAL_DATA \
--arch vgg11 --lr 0.05 --weight-decay 5e-4 --gpu-id 0 \
2>&1 | tee output_vgg11.txt

python3 teacher.py \
--real_data $REAL_DATA \
--arch ShuffleV2 --lr 0.01 --weight-decay 5e-4 --gpu-id 0 \
2>&1 | tee output_ShuffleV2.txt
