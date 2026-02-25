#!/bin/bash

source /data1/zhaofeng/anaconda3/bin/activate
conda activate eel-planning

#====== parameters ======#

clip_path='/data1/zhaofeng/TTA/EEL-benchmark-features-szf/EEL-our-extraction/FrozenBiLM-main/ViT-L-14.pt'
category_txt='anticpation_annotation_egome/noun_vocabulary_final.txt'


test_log_name='test_results_out_tta_tent_align_coop_v2.txt'


dataset=egobridge # egobridge
class_file='../output/ego/ta3n'
training=true  # true | false
testing=true # true | false
modality=RGB
frame_type=feature # frame | feature
num_segments=5 # sample frame # of each video for training
test_segments=5
baseline_type=video
frame_aggregation=trn-m # method to integrate the frame-level features (avgpool | trn | trn-m | rnn | temconv)
add_fc=1 #1
fc_dim=512
arch=resnet101
use_target=none # none | Sv | uSv
share_params=Y # Y | N

if [ "$use_target" == "none" ] 
then
	exp_DA_name=baseline
else
	exp_DA_name=DA
fi

#====== select dataset ======#
path_data_root=anticpation_annotation_egome/noun/ # depend on users
path_exp_root=logs_final/EGOME-action-anticipation-noun-szf-exoonly-ep200/ # depend on users
path_feat_root=/data1/zhaofeng/TTA/anti-TTA/DCPGN-release/EgoMe-benchmark/EgoMe_final_features/EgoMe_Video_CLIP_features_5fps_pt/ # depend on users

dataset_source=exo # depend on users
dataset_target=exo # depend on users
dataset_val=exo  # depend on users
dataset_test=exo # depend on users
num_source=8336 # number of training data (source)
num_target=8336 # number of training data (target)

###
train_source_list=$path_data_root"Annotation_train_Exo_noun.txt"
train_target_list=$path_data_root"Annotation_train_Exo_noun.txt"
val_list=$path_data_root"Annotation_val_Ego_noun.txt"
test_list="anticpation_annotation_egome/noun_addtext/Annotation_test_Ego_noun.txt"
NUM_CLASSES=35

####

path_exp=$path_exp_root'Testexp'
#====== select dataset ======#

pretrained=none

#====== parameters for algorithms ======#
# parameters for DA approaches
dis_DA=none # none | DAN | JAN
alpha=0 # depend on users

adv_pos_0=Y # Y | N (discriminator for relation features)
adv_DA=RevGrad # none | RevGrad
beta_0=0.75 # U->H: 0.75 | H->U: 1
beta_1=0.75 # U->H: 0.75 | H->U: 0.75
beta_2=0.5 # U->H: 0.5 | H->U: 0.5

use_attn=TransAttn # none | TransAttn | general
n_attn=1
use_attn_frame=none # none | TransAttn | general

use_bn=none # none | AdaBN | AutoDIAL
add_loss_DA=attentive_entropy # none | target_entropy | attentive_entropy
gamma=0.003 # U->H: 0.003 | H->U: 0.3

ens_DA=none # none | MCD
mu=0

# parameters for architectures
bS=128 # batch size
bS_2=$((bS * num_target / num_source ))
echo '('$bS', '$bS_2')'

lr=3e-3
optimizer=SGD

if [ "$use_target" == "none" ] 
then
	dis_DA=none
	alpha=0
	adv_pos_0=N
	adv_DA=none
	beta_0=0
	beta_1=0
	beta_2=0
	use_attn=none
	use_attn_frame=none
	use_bn=none
	add_loss_DA=none
	gamma=0
	ens_DA=none
	mu=0
	j=0

	exp_path=$path_exp'-'$optimizer'-share_params_'$share_params'/'$dataset'-'$num_segments'seg_'$j'/'
else
	exp_path=$path_exp'-'$optimizer'-share_params_'$share_params'-lr_'$lr'-bS_'$bS'_'$bS_2'/'$dataset'-'$num_segments'seg-disDA_'$dis_DA'-alpha_'$alpha'-advDA_'$adv_DA'-beta_'$beta_0'_'$beta_1'_'$beta_2'-useBN_'$use_bn'-addlossDA_'$add_loss_DA'-gamma_'$gamma'-ensDA_'$ens_DA'-mu_'$mu'-useAttn_'$use_attn'-n_attn_'$n_attn'/'
fi

echo 'exp_path: '$exp_path

### szf
select_top_k=3
max_proto=500
balance_weight=1.0
step=1
lr_test=5e-4
conf_temp=2

if ($testing)
then
	# model=checkpoint_20 # checkpoint | model_best
	model=model_best # checkpoint | model_best
	echo $model
	echo $test_list

	# testing on the testing set
	echo 'testing on the validation set'
	python anticipation_test_models_tent_align_coop_v2.py $class_file $modality \
	$test_list $exp_path$modality'/'$model'.pth.tar' \
	--arch $arch --test_segments $test_segments \
	--save_scores $exp_path$modality'/scores_'$dataset_target'-'$model'-'$test_segments'seg' --save_confusion $exp_path$modality'/confusion_matrix_'$dataset_target'-'$model'-'$test_segments'seg' \
	--n_rnn 1 --rnn_cell LSTM --n_directions 1 --n_ts 5 \
	--use_attn $use_attn --n_attn $n_attn --use_attn_frame $use_attn_frame --use_bn $use_bn --share_params $share_params \
	-j 4 --bS 64 --top 1 3 5 --add_fc $add_fc --fc_dim $fc_dim --baseline_type $baseline_type --frame_aggregation $frame_aggregation  --num_classes $NUM_CLASSES \
	--feat_path $path_feat_root --test_log_name $test_log_name \
	--select_top_k $select_top_k --max_proto $max_proto \
	--clip_path $clip_path --category_txt $category_txt \
	--balance_weight $balance_weight \
	--step $step --lr_test $lr_test \
	--conf_temp $conf_temp \
	
fi

# ----------------------------------------------------------------------------------
exit 0
