ROOT_PATH="./CIFAR/CIFAR_10K/cGAN-based_KD"

SEED=2020
NTRAIN=10000
GAN_LOSS="hinge"
CGAN_BATCH_SIZE=512
GAN_BATCH_SIZE=512
DRE_LAMBDA=1e-3
filtering_threshold=0.7 ##make sure we have at least 100K samples after filtering
NUM_CLASSES=10
NFAKE_PER_CLASS=50000

echo "-------------------------------------------------------------------------------------------------"
echo "BigGAN: No Subsampling, No filtering"
CUDA_VISIBLE_DEVICES=0 python3 generate_synthetic_data.py \
--root_path $ROOT_PATH --seed $SEED \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--gan_name BigGAN --gan_epochs 2000 --gan_resume_epoch 0 --gan_transform \
--samp_nfake_per_class $NFAKE_PER_CLASS \
2>&1 | tee output_BigGAN_subsampling_False_filter_False_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "BigGAN: Subsampling, No filtering"
CUDA_VISIBLE_DEVICES=0 python3 generate_synthetic_data.py \
--root_path $ROOT_PATH --seed $SEED \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--gan_name BigGAN --gan_epochs 2000 --gan_resume_epoch 0 --gan_transform \
--dre_precnn_net DenseNet121 --dre_precnn_epochs 350 --dre_precnn_resume_epoch 0 --dre_precnn_lr_decay_epochs 150_250 --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
--dre_net MLP5 --dre_epochs 350 --dre_resume_epoch 0 --dre_batch_size 512 --dre_lr_base 1e-4 --dre_lr_decay_epochs 150_250 --dre_lambda $DRE_LAMBDA \
--subsampling --samp_nfake_per_class $NFAKE_PER_CLASS \
2>&1 | tee output_BigGAN_subsampling_True_filter_False_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "BigGAN: Subsampling, filtering ${filtering_threshold}, adjustment"
CUDA_VISIBLE_DEVICES=0 python3 generate_synthetic_data.py \
--root_path $ROOT_PATH --seed $SEED \
--ntrain $NTRAIN --num_classes $NUM_CLASSES \
--gan_name BigGAN --gan_epochs 2000 --gan_resume_epoch 0 --gan_transform \
--dre_precnn_net DenseNet121 --dre_precnn_epochs 350 --dre_precnn_resume_epoch 0 --dre_precnn_lr_decay_epochs 150_250 --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
--dre_net MLP5 --dre_epochs 350 --dre_resume_epoch 0 --dre_batch_size 512 --dre_lr_base 1e-4 --dre_lr_decay_epochs 150_250 --dre_lambda $DRE_LAMBDA \
--subsampling --samp_nfake_per_class $NFAKE_PER_CLASS \
--unfiltered_fake_dataset_filename "CIFAR10_ntrain_${NTRAIN}_BigGAN_vanilla_epochs_2000_transform_True_subsampling_True_FilterCEPct_1.0_nfake_500000_seed_${SEED}.h5" \
--samp_filter_precnn_net DenseNet121 --samp_filter_precnn_net_ckpt_filename "ckpt_baseline_DenseNet121_epoch_350_transform_True_seed_${SEED}_data_real_nreal_${NTRAIN}_fake_None.pth" --samp_filter_ce_percentile_threshold $filtering_threshold \
--adjust_label \
2>&1 | tee output_BigGAN_subsampling_True_filter_${filtering_threshold}_seed_${SEED}.txt
