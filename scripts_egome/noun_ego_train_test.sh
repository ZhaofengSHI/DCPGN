# train
CUDA_VISIBLE_DEVICES=0 \
bash scripts_szf_egome/anticipation_noun/ego_ego_train_val.sh

# test in domain
CUDA_VISIBLE_DEVICES=0 \
bash scripts_szf_egome/anticipation_noun/ego_ego_test_in.sh

# test out domain
CUDA_VISIBLE_DEVICES=0 \
bash scripts_szf_egome/anticipation_noun/ego_ego_test_out.sh
