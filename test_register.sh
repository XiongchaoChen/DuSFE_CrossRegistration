python test.py \
--resume './outputs/train_register/checkpoints/model_4.pt' \
--experiment_name 'test_register_4' \
--model_type 'model_reg' \
--dataset 'CardiacSPECT_Reg' \
--data_root '../../Data/Dataset_filename/' \
--net_G 'DuRegister_DuSE' \
--net_filter 32 \
--batch_size 4 \
--n_patch_train 1 \
--patch_size_train 80 80 40 \
--n_patch_test 1 \
--patch_size_test 80 80 40 \
--n_patch_valid 1 \
--patch_size_valid 80 80 40 \
--gpu_ids 1



