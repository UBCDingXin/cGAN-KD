ROOT_PATH="./CIFAR-100/BigGAN"

EPOCHS=2000
BATCHSIZE=512

### complete CIFAR100 dataset
python train.py \
--root_path $ROOT_PATH --seed 2021 \
--shuffle --batch_size $BATCHSIZE --parallel --num_workers 0  --no_pin_memory \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs $EPOCHS \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--data_root data/ --dataset C100 --augment \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 1000 --no_fid --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--DiffAugment_policy 'color,translation,cutout' \
2>&1 | tee output_biggan_cifar100_full_2020.txt
