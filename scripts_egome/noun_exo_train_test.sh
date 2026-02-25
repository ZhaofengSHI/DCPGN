# train
CUDA_VISIBLE_DEVICES=1 \
bash scripts_szf_egome/anticipation_noun/exo_exo_train_val.sh

# test in domain
CUDA_VISIBLE_DEVICES=1 \
bash scripts_szf_egome/anticipation_noun/exo_exo_test_in.sh

# test out domain
CUDA_VISIBLE_DEVICES=1 \
bash scripts_szf_egome/anticipation_noun/exo_exo_test_out.sh