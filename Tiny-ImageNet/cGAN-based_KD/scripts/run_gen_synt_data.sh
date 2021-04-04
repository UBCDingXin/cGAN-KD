ROOT_PATH="./Tiny-ImageNet/cGAN-based_KD"

DRE_LAMBDA=1e-3
filtering_threshold=0.5

NFAKE_PER_CLASS=3000
SAMP_BATCHSIZE=3000

SEED=2020

echo "-------------------------------------------------------------------------------------------------"
echo "BigGAN: No Subsampling, No filtering"
CUDA_VISIBLE_DEVICES=0,1 python3 generate_synthetic_data.py \
--root_path $ROOT_PATH --seed $SEED \
--samp_nfake_per_class $NFAKE_PER_CLASS --samp_batch_size $SAMP_BATCHSIZE \
2>&1 | tee output_BigGAN_subsampling_False_filter_False_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "BigGAN: Subsampling, No filtering"
CUDA_VISIBLE_DEVICES=0,1 python3 generate_synthetic_data.py \
--root_path $ROOT_PATH --seed $SEED \
--dre_precnn_net DenseNet121 --dre_precnn_epochs 350 --dre_precnn_resume_epoch 0 --dre_precnn_lr_decay_epochs 150_250 --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
--dre_net MLP5 --dre_epochs 350 --dre_resume_epoch 0 --dre_batch_size 512 --dre_lr_base 1e-4 --dre_lr_decay_epochs 150_250 --dre_lambda $DRE_LAMBDA \
--subsampling --samp_nfake_per_class $NFAKE_PER_CLASS --samp_batch_size $SAMP_BATCHSIZE \
2>&1 | tee output_BigGAN_subsampling_True_filter_False_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "BigGAN: Subsampling, filtering ${filtering_threshold}"
CUDA_VISIBLE_DEVICES=0,1 python3 generate_synthetic_data.py \
--root_path $ROOT_PATH --seed $SEED \
--dre_precnn_net DenseNet121 --dre_precnn_epochs 350 --dre_precnn_resume_epoch 0 --dre_precnn_lr_decay_epochs 150_250 --dre_precnn_weight_decay 1e-4 --dre_precnn_transform \
--dre_net MLP5 --dre_epochs 350 --dre_resume_epoch 0 --dre_batch_size 512 --dre_lr_base 1e-4 --dre_lr_decay_epochs 150_250 --dre_lambda $DRE_LAMBDA \
--subsampling --samp_nfake_per_class $NFAKE_PER_CLASS --samp_batch_size $SAMP_BATCHSIZE \
--unfiltered_fake_dataset_filename "Tiny-ImageNet_BigGAN_subsampling_True_FilterCEPct_1.0_nfake_600000_seed_${SEED}.h5" \
--samp_filter_precnn_net DenseNet121 --samp_filter_precnn_net_ckpt_filename "ckpt_baseline_DenseNet121_epoch_350_transform_True_seed_${SEED}_data_real_nreal_98179_fake_None.pth" --samp_filter_ce_percentile_threshold $filtering_threshold \
2>&1 | tee output_BigGAN_subsampling_True_filter_${filtering_threshold}_seed_${SEED}.txt
