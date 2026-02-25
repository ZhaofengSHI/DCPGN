# train val test
CUDA_VISIBLE_DEVICES=0 \
bash scripts_szf_eel/anticipation_noun/ego_ego_train_val.sh

# test in domain
CUDA_VISIBLE_DEVICES=0 \
bash scripts_szf_eel/anticipation_noun/ego_ego_test_in.sh

# test out domain
CUDA_VISIBLE_DEVICES=0 \
bash scripts_szf_eel/anticipation_noun/ego_ego_test_out.sh