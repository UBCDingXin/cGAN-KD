ROOT_PATH="./RC-49/CcGAN-based_KD"
REAL_DATA_PATH="./RC-49/dataset"

NCPU=0
SEED=2020
NTRAIN_PER_LABEL=5
NITERS=30000
BATCH_SIZE_D=256
BATCH_SIZE_G=256
SIGMA=-1.0
KAPPA=-2.0
LR_G=1e-4
LR_D=1e-4

SAE_LAMBDA=1e-3
DRE_LAMBDA=1e-3
CcGAN_type="soft"
DRE_type="hard"
DRE_kappa=1e-20

NFAKE_PER_LABEL=200
NUM_FAKE_LABELS=-1
filtering_threshold=0.9



echo "-------------------------------------------------------------------------------------------------"
echo "RC49-${NTRAIN_PER_LABEL}; SNGAN: No Subsampling, No filtering"
CUDA_VISIBLE_DEVICES=0,1 python3 generate_synthetic_data.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --seed $SEED --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--gan_dim_embed 128 --gan_embed_x2y_net_name ResNet34 \
--gan_embed_x2y_epoch 200 --gan_embed_x2y_resume_epoch 0 --gan_embed_x2y_batch_size 256 \
--gan_embed_y2h_epoch 500 --gan_embed_y2h_batch_size 256 \
--gan_loss_type hinge --gan_niters $NITERS --gan_resume_niters 0 --gan_save_niters_freq 5000 \
--gan_lr_g $LR_G --gan_lr_d $LR_D --gan_dim_g 256 \
--gan_gene_ch 64 --gan_disc_ch 64 \
--gan_batch_size_disc $BATCH_SIZE_D --gan_batch_size_gene $BATCH_SIZE_G \
--gan_kernel_sigma $SIGMA --gan_threshold_type $CcGAN_type --gan_kappa $KAPPA \
--samp_batch_size 1000 \
--samp_num_fake_labels $NUM_FAKE_LABELS --samp_nfake_per_label $NFAKE_PER_LABEL \
2>&1 | tee output_SNGAN_NImgPerLabel_${NTRAIN_PER_LABEL}_subsampling_False_filter_False_seed_${SEED}.txt


echo "-------------------------------------------------------------------------------------------------"
echo "RC49-${NTRAIN_PER_LABEL}; SNGAN: Subsampling, No filtering"
CUDA_VISIBLE_DEVICES=0,1 python3 generate_synthetic_data.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --seed $SEED --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--gan_dim_embed 128 --gan_embed_x2y_net_name ResNet34 \
--gan_embed_x2y_epoch 200 --gan_embed_x2y_resume_epoch 0 --gan_embed_x2y_batch_size 256 --gan_embed_x2y_lr_base 0.01 \
--gan_embed_y2h_epoch 500 --gan_embed_y2h_batch_size 256 \
--gan_loss_type hinge --gan_niters $NITERS --gan_resume_niters 0 --gan_save_niters_freq 5000 \
--gan_lr_g $LR_G --gan_lr_d $LR_D --gan_dim_g 256 \
--gan_gene_ch 64 --gan_disc_ch 64 \
--gan_batch_size_disc $BATCH_SIZE_D --gan_batch_size_gene $BATCH_SIZE_G \
--gan_kernel_sigma $SIGMA --gan_threshold_type $CcGAN_type --gan_kappa $KAPPA \
--dre_presae_epochs 200 --dre_presae_resume_epoch 0 \
--dre_presae_lr_base 0.01 --dre_presae_lr_decay_factor 0.1 --dre_presae_lr_decay_freq 50 \
--dre_presae_batch_size_train 128 --dre_presae_weight_decay 1e-4 --dre_presae_lambda_sparsity $SAE_LAMBDA \
--dre_epochs 350 --dre_resume_epoch 0 --dre_lr_base 1e-4 --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs 100_200 \
--dre_batch_size 256 --dre_lambda $DRE_LAMBDA \
--dre_threshold_type $DRE_type --dre_kappa $DRE_kappa \
--dre_no_vicinal \
--subsampling \
--samp_batch_size 1000 \
--samp_num_fake_labels $NUM_FAKE_LABELS --samp_nfake_per_label $NFAKE_PER_LABEL \
2>&1 | tee output_SNGAN_NImgPerLabel_${NTRAIN_PER_LABEL}_subsampling_True_filter_False_seed_${SEED}.txt


echo "-------------------------------------------------------------------------------------------------"
echo "RC49-${NTRAIN_PER_LABEL}; SNGAN: No Subsampling, filtering ${filtering_threshold}"
CUDA_VISIBLE_DEVICES=0,1 python3 generate_synthetic_data.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --seed $SEED --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--gan_dim_embed 128 --gan_embed_x2y_net_name ResNet34 \
--gan_embed_x2y_epoch 200 --gan_embed_x2y_resume_epoch 0 --gan_embed_x2y_batch_size 256 --gan_embed_x2y_lr_base 0.01 \
--gan_embed_y2h_epoch 500 --gan_embed_y2h_batch_size 256 \
--gan_loss_type hinge --gan_niters $NITERS --gan_resume_niters 0 --gan_save_niters_freq 5000 \
--gan_lr_g $LR_G --gan_lr_d $LR_D --gan_dim_g 256 \
--gan_gene_ch 64 --gan_disc_ch 64 \
--gan_batch_size_disc $BATCH_SIZE_D --gan_batch_size_gene $BATCH_SIZE_G \
--gan_kernel_sigma $SIGMA --gan_threshold_type $CcGAN_type --gan_kappa $KAPPA \
--samp_batch_size 1000 \
--samp_num_fake_labels $NUM_FAKE_LABELS --samp_nfake_per_label $NFAKE_PER_LABEL \
--samp_filter_precnn_net VGG16 --samp_filter_precnn_net_ckpt_filename ckpt_baseline_VGG16_epoch_350_seed_2020_data_real_nreal_2250_fake_None_validation_False.pth \
--samp_filter_mae_percentile_threshold $filtering_threshold --unfiltered_fake_dataset_filename fake_RC49_NTrainPerLabel_5_SNGAN_hinge_niters_30000_subsampling_None_FilterMAEPct_1.0_nfake_179800_seed_2020.h5 \
2>&1 | tee output_SNGAN_NImgPerLabel_${NTRAIN_PER_LABEL}_subsampling_False_filter_${filtering_threshold}_seed_${SEED}.txt


echo "-------------------------------------------------------------------------------------------------"
echo "RC49-${NTRAIN_PER_LABEL}; SNGAN: Subsampling, filtering ${filtering_threshold}"
CUDA_VISIBLE_DEVICES=0,1 python3 generate_synthetic_data.py \
--root_path $ROOT_PATH --data_path $REAL_DATA_PATH --seed $SEED --num_workers $NCPU \
--max_num_img_per_label $NTRAIN_PER_LABEL \
--gan_dim_embed 128 --gan_embed_x2y_net_name ResNet34 \
--gan_embed_x2y_epoch 200 --gan_embed_x2y_resume_epoch 0 --gan_embed_x2y_batch_size 256 --gan_embed_x2y_lr_base 0.01 \
--gan_embed_y2h_epoch 500 --gan_embed_y2h_batch_size 256 \
--gan_loss_type hinge --gan_niters $NITERS --gan_resume_niters 0 --gan_save_niters_freq 5000 \
--gan_lr_g $LR_G --gan_lr_d $LR_D --gan_dim_g 256 \
--gan_gene_ch 64 --gan_disc_ch 64 \
--gan_batch_size_disc $BATCH_SIZE_D --gan_batch_size_gene $BATCH_SIZE_G \
--gan_kernel_sigma $SIGMA --gan_threshold_type $CcGAN_type --gan_kappa $KAPPA \
--dre_presae_epochs 200 --dre_presae_resume_epoch 0 \
--dre_presae_lr_base 0.01 --dre_presae_lr_decay_factor 0.1 --dre_presae_lr_decay_freq 50 \
--dre_presae_batch_size_train 128 --dre_presae_weight_decay 1e-4 --dre_presae_lambda_sparsity $SAE_LAMBDA \
--dre_epochs 350 --dre_resume_epoch 0 --dre_lr_base 1e-4 --dre_lr_decay_factor 0.1 --dre_lr_decay_epochs 100_200 \
--dre_batch_size 256 --dre_lambda $DRE_LAMBDA \
--dre_threshold_type $DRE_type --dre_kappa $DRE_kappa \
--dre_no_vicinal \
--subsampling \
--samp_batch_size 1000 \
--samp_num_fake_labels $NUM_FAKE_LABELS --samp_nfake_per_label $NFAKE_PER_LABEL \
--samp_filter_precnn_net VGG16 --samp_filter_precnn_net_ckpt_filename ckpt_baseline_VGG16_epoch_350_seed_2020_data_real_nreal_2250_fake_None_validation_False.pth \
--samp_filter_mae_percentile_threshold $filtering_threshold --unfiltered_fake_dataset_filename fake_RC49_NTrainPerLabel_5_SNGAN_hinge_niters_30000_subsampling_cDRE-F-SP+RS_hard_1e-20_FilterMAEPct_1.0_nfake_179800_seed_2020.h5 \
2>&1 | tee output_SNGAN_NImgPerLabel_${NTRAIN_PER_LABEL}_subsampling_True_filter_${filtering_threshold}_seed_${SEED}.txt
