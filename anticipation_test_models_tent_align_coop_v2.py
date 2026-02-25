import argparse
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

from anticipation_dataset_text import TSNDataSet
from anticipation_models_tent_align_coop_v2 import VideoModel ##
from utils.utils import plot_confusion_matrix

from colorama import init
from colorama import Fore, Back, Style
from tqdm import tqdm
from time import sleep

from sklearn.metrics import f1_score, recall_score, precision_score
from anticipation_main import mean_average_precision


import tent_align_coop_v2 as tent
from clip.custom_clip import get_coop #coop
from clip import clip
import random
import numpy as np
import torch
init(autoreset=True)

# options
parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('class_file', type=str, default="classInd.txt")
parser.add_argument(
    'modality',
    type=str,
    choices=['RGB', 'Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--test_segments', type=int, default=5)
parser.add_argument('--num_classes', type=int, default=11)
parser.add_argument(
    '--add_fc',
    default=1,
    type=int,
    metavar='M',
    help=
    'number of additional fc layers (excluding the last fc layer) (e.g. 0, 1, 2, ...)'
)
parser.add_argument('--fc_dim',
                    type=int,
                    default=512,
                    help='dimension of added fc')
parser.add_argument('--baseline_type',
                    type=str,
                    default='frame',
                    choices=['frame', 'video', 'tsn'])
parser.add_argument(
    '--frame_aggregation',
    type=str,
    default='avgpool',
    choices=['avgpool', 'rnn', 'temconv', 'trn-m', 'none'],
    help='aggregation of frame features (none if baseline_type is not video)')
parser.add_argument('--dropout_i', type=float, default=0)
parser.add_argument('--dropout_v', type=float, default=0)

#------ RNN ------
parser.add_argument('--n_rnn',
                    default=1,
                    type=int,
                    metavar='M',
                    help='number of RNN layers (e.g. 0, 1, 2, ...)')
parser.add_argument('--rnn_cell',
                    type=str,
                    default='LSTM',
                    choices=['LSTM', 'GRU'])
parser.add_argument('--n_directions',
                    type=int,
                    default=1,
                    choices=[1, 2],
                    help='(bi-) direction RNN')
parser.add_argument('--n_ts',
                    type=int,
                    default=5,
                    help='number of temporal segments')

# ========================= DA Configs ==========================
parser.add_argument('--share_params',
                    type=str,
                    default='Y',
                    choices=['Y', 'N'])
parser.add_argument('--use_bn',
                    type=str,
                    default='none',
                    choices=['none', 'AdaBN', 'AutoDIAL'])
parser.add_argument('--use_attn_frame',
                    type=str,
                    default='none',
                    choices=['none', 'TransAttn', 'general', 'DotProduct'],
                    help='attention-mechanism for frames only')
parser.add_argument('--use_attn',
                    type=str,
                    default='none',
                    choices=['none', 'TransAttn', 'general', 'DotProduct'],
                    help='attention-mechanism')
parser.add_argument('--n_attn',
                    type=int,
                    default=1,
                    help='number of discriminators for transferable attention')

# ========================= Monitor Configs ==========================
parser.add_argument('--top',
                    default=[1, 3, 5],
                    nargs='+',
                    type=int,
                    help='show top-N categories')
parser.add_argument('--verbose', default=False, action="store_true")

# ========================= Runtime Configs ==========================
parser.add_argument('--save_confusion', type=str, default=None)
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--save_attention', type=str, default=None)
parser.add_argument('--max_num',
                    type=int,
                    default=-1,
                    help='number of videos to test')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--bS',
                    default=2,
                    help='batch size',
                    type=int,
                    required=False)
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')
parser.add_argument('--feat_path',type=str)

# szf add
parser.add_argument('--test_log_name',type=str)
parser.add_argument('--select_top_k',
                    type=int,
                    default=5,
                    help='select_top_k')
parser.add_argument('--max_proto',
                    type=int,
                    default=250,
                    help='max_proto')
# coop
parser.add_argument('--clip_path',type=str)
parser.add_argument('--category_txt',type=str)

## balance param
parser.add_argument('--balance_weight',type=float)
parser.add_argument('--step',type=int)
parser.add_argument('--lr_test',type=float)
parser.add_argument('--conf_temp',type=float)
args = parser.parse_args()


num_class = args.num_classes



###### szf load category text and coop ###############
# szf category list
with open(args.category_txt, 'r', encoding='utf-8') as file:
    lines = file.readlines()

category_list = [line.strip() for line in lines]
#######################################################


######################COOP#########################
# coop
device = "cuda" if torch.cuda.is_available() else "cpu"
coop_model = get_coop(args.clip_path, category_list, device, n_ctx=4, ctx_init="a_photo_of_a")
# coop_model = get_coop(args.clip_path, category_list, device, n_ctx=0, ctx_init="")
print('CLIP CoOp model loaded')
print("=> Model created: visual backbone {}".format(args.clip_path))
coop_model.reset_classnames(category_list, args.clip_path)

# CLIP
clip_model, _, preprocess = clip.load(args.clip_path, device=device)
########################################################


#=== Data loading ===#
print(Fore.CYAN + 'loading data......')

data_length = 1 if args.modality == "RGB" else 5
num_test = sum(1 for i in open(args.test_list))

data_set = TSNDataSet(
    "",
    args.test_list,
    args.feat_path,
    num_dataload=num_test,
    num_segments=args.test_segments,
    num_classes = num_class,
    new_length=data_length,
    modality=args.modality,
    image_tmpl="img_{:05d}.t7" if args.modality
    in ['RGB', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus'] else args.flow_prefix +
    "{}_{:05d}.t7",
    test_mode=True,
)

data_loader = torch.utils.data.DataLoader(data_set,
                                          batch_size=args.bS,
                                          shuffle=False, #False
                                          num_workers=args.workers,
                                          pin_memory=False
                                          )

data_gen = tqdm(data_loader)

num_iters = len(data_gen)

#=== Load the network ===#
print(Fore.CYAN + 'preparing the model......')
net = VideoModel(
    num_class,
    args.baseline_type,
    args.frame_aggregation,
    args.modality,
    train_segments=args.test_segments if args.baseline_type == 'video' else 1,
    val_segments=args.test_segments if args.baseline_type == 'video' else 1,
    base_model=args.arch,
    add_fc=args.add_fc,
    fc_dim=args.fc_dim,
    share_params=args.share_params,
    dropout_i=args.dropout_i,
    dropout_v=args.dropout_v,
    use_bn=args.use_bn,
    partial_bn=False,
    n_rnn=args.n_rnn,
    rnn_cell=args.rnn_cell,
    n_directions=args.n_directions,
    n_ts=args.n_ts,
    use_attn=args.use_attn,
    n_attn=args.n_attn,
    use_attn_frame=args.use_attn_frame,
    verbose=args.verbose,
    select_top_k= args.select_top_k, ##
    max_proto = args.max_proto, ## 
    num_iters = num_iters, ##
    coop_model = coop_model, ##
    balance_weight = args.balance_weight,
    clip_model = clip_model,
    confidence_temp = args.conf_temp
    )

print(args.weights)

checkpoint = torch.load(args.weights)

validation_score = checkpoint['prec1']
print("model epoch {} prec@1: {}".format(checkpoint['epoch'],
                                         checkpoint['prec1']))

base_dict = {
    '.'.join(k.split('.')[1:]): v
    for k, v in list(checkpoint['state_dict'].items())
}
net.load_state_dict(base_dict,strict=False) ##


# #--- GPU processing ---#
net = torch.nn.DataParallel(net.cuda())
# net.eval()

# tent adaptation szf
############################
# net.train()
net.requires_grad_(False) ### do not require grad
# net = tent.configure_model(net)


## tent
for name, param in net.named_parameters():
    for item in [
        # 'module.fc_classifier_video_source',
        # 'feature_convert',
        'ctx',
        ]:
        if item in name:
            param.requires_grad_(True)

# for name,param in net.named_parameters():
#     print(name,param.requires_grad)
    
params = [p for p in net.parameters() if p.requires_grad]
params_names = [n for n, p in net.named_parameters() if p.requires_grad]

print('----')
print(len(params))
print(params_names)
print('----')

optimizer = torch.optim.SGD(
                            params,
                            args.lr_test,
                            momentum=0.9,
                            weight_decay=0.0000,
                            nesterov=True
                            )

tented_net = tent.Tent(net, optimizer, steps=args.step)
# ###########################

output = []
attn_values = torch.Tensor()

#############################################################
def eval_video(video_data):
    i, data, label, text_token = video_data


    data = data.cuda()
    label = label.cuda(non_blocking=True)  # pytorch 0.4.X

    num_crop = 1

    out = tented_net(data, text_token, is_train=False)

    return out, label

#############################################################

proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

count_correct_topK = [0 for i in range(len(args.top))]
count_total = 0
video_pred = [[] for i in range(max(args.top))]
video_labels = []

all_preds = []
all_labels = []
#=== Testing ===#
print(Fore.CYAN + 'start testing......')
for i, (data, label, text_token) in enumerate(data_gen):

    if i >= max_num:
        break

    preds, labels = eval_video((i, data, label, text_token))

    labels = labels
    all_preds.append(preds)
    all_labels.append(labels)


###################### evaluation metrics ######################

all_labels = torch.cat(all_labels)
all_preds = torch.cat(all_preds)
all_pred_top5 = torch.scatter(torch.zeros_like(all_preds, device=all_preds.device), 1,
                              all_preds.topk(5)[1], 1.0).cpu().numpy()
all_preds = all_preds.detach().cpu().numpy()
all_labels = all_labels.detach().cpu().numpy()
macro_recall = recall_score(all_labels, all_pred_top5, average='macro') * 100
micro_recall = recall_score(all_labels, all_pred_top5, average='micro') * 100
sample_recall = recall_score(all_labels, all_pred_top5, average='samples') * 100

print(('Testing Results: macro_recall top5 {macro_recall:.3f} micro_recall top5 {micro_recall:.3f} sample_recall top5 {sample_recall:.3f}'.format(
    macro_recall=macro_recall, micro_recall=micro_recall,sample_recall=sample_recall)))

print("validation score:", validation_score)
print("testing score:", macro_recall)


## szf modified
if not args.test_log_name:
    args.test_log_name = 'test_results.txt'
import  os
result_file_path = os.path.join('/'.join(args.weights.split('/')[:-1]),args.test_log_name)
with open(result_file_path,'w') as f:
    f.writelines('Testing Results: macro_recall top5 {macro_recall:.3f} micro_recall top5 {micro_recall:.3f} sample_recall top5 {sample_recall:.3f}\n'.format(
    macro_recall=macro_recall, micro_recall=micro_recall,sample_recall=sample_recall))
    f.writelines("testing score:"+ str(macro_recall)+'\n')

##############################################################
