from torch import nn

from torch.nn.init import *
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import TRNmodule
import math
import numpy as np
from colorama import init
from colorama import Fore, Back, Style

from anticipation_pseudo_data_update import DynamicFeatureSelector
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_proto_classifier_conf(all_features_list,all_labels_list, all_en_list, all_conf_list):
    ### prototype
    prototype_list = []
    for cls_idx in range(len(all_features_list)):

        class_features = all_features_list[cls_idx]
        class_labels = all_labels_list[cls_idx]
        class_confs = all_conf_list[cls_idx]
        class_conf_normalize = class_confs / class_confs.sum()
        # print(class_conf_normalize.max())
        # class_conf_normalize = class_conf_normalize.pow(1.0 / 2.0) # temp
        # print(class_conf_normalize.max())

        reweight_features = torch.sum(class_features * class_confs.cuda(), dim=0)
        # reweight_features = torch.sum(class_features * class_conf_normalize.cuda(), dim=0)

        prototype_list.append(reweight_features)

    return torch.stack(prototype_list,dim=0)

class FeatureConvert(nn.Module):
    def __init__(self, input_dim, hidden_dim, droprate):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.hidden_dim = hidden_dim

        self.layer_in = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim , kernel_size=1),
            nn.ReLU(),
        )

        self.layer_mid = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim , kernel_size=1),
            nn.ReLU(),
        )

        self.layer_out = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.output_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.features = [] 

    def forward(self, x):

        x = x.permute(0,2,1)

        x1 = self.layer_in(x)
        x2 = self.layer_mid(x1)
        x3 = self.layer_out(x2)

        out_feat = x3.permute(0,2,1).contiguous()

        return out_feat


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dim // 2),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_dim // 2, out_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        return self.output(x)


class VideoModel(nn.Module):

    def __init__(self,
                 num_class,
                 baseline_type,
                 frame_aggregation,
                 modality,
                 train_segments=5,
                 val_segments=25,
                 base_model='resnet101',
                 path_pretrained='',
                 new_length=None,
                 before_softmax=True,
                 dropout_i=0.5,
                 dropout_v=0.5,
                 use_bn='none',
                 ens_DA='none',
                 crop_num=1,
                 partial_bn=True,
                 verbose=True,
                 add_fc=1,
                 fc_dim=1024,
                 n_rnn=1,
                 rnn_cell='LSTM',
                 n_directions=1,
                 n_ts=5,
                 use_attn='TransAttn',
                 n_attn=1,
                 use_attn_frame='none',
                 share_params='Y',
                 select_top_k= 5,
                 max_proto = 250,
                 num_iters = None,
                 coop_model = None,
                 balance_weight = None,
                 clip_model = None,
                 confidence_temp = None
                 ):
        super(VideoModel, self).__init__()

        # szf add pseudo_selector############################################################
        self.pseudo_selector = DynamicFeatureSelector(feature_dims=[512], select_top_k=select_top_k, entropy_top_k=max_proto)
        self.max_proto = max_proto
        self.select_top_k = select_top_k
        self.conf_temp = confidence_temp
        ##################################################################################3

        # szf add CLIP CooP############################################################
        self.coop_model = coop_model
        self.balance_weight = balance_weight

        # clip
        self.clip_model = clip_model
        ##############################################################################
         
        self.modality = modality
        self.train_segments = train_segments
        self.val_segments = val_segments
        self.baseline_type = baseline_type
        self.frame_aggregation = frame_aggregation
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout_rate_i = dropout_i
        self.dropout_rate_v = dropout_v
        self.use_bn = use_bn
        self.ens_DA = ens_DA
        self.crop_num = crop_num
        self.add_fc = add_fc
        self.fc_dim = fc_dim
        self.share_params = share_params

        # RNN
        self.n_layers = n_rnn
        self.rnn_cell = rnn_cell
        self.n_directions = n_directions
        self.n_ts = n_ts  # temporal segment

        # Attention
        self.use_attn = use_attn
        self.n_attn = n_attn
        self.use_attn_frame = use_attn_frame

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        if verbose:
            print(("""
				Initializing TSN with base model: {}.
				TSN Configurations:
				input_modality:     {}
				num_segments:       {}
				new_length:         {}
				""".format(base_model, self.modality, self.train_segments,
               self.new_length)))

        # szf
        self._prepare_DA(num_class, base_model)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        # self._enable_pbn = False
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_DA(self, num_class,
                    base_model):  # convert the model to DA framework

        self.feature_dim = 768 #####

        std = 0.001
        feat_shared_dim = min(
            self.fc_dim, self.feature_dim
        ) if self.add_fc > 0 and self.fc_dim > 0 else self.feature_dim
        feat_frame_dim = feat_shared_dim

        ## szf add
        self.feature_convert = FeatureConvert(self.feature_dim,feat_shared_dim,self.dropout_rate_i)

        self.num_bottleneck = 512

        self.TRN = TRNmodule.RelationModuleMultiScale(
            feat_shared_dim, self.num_bottleneck, self.train_segments)

        #  trn
        feat_aggregated_dim = self.num_bottleneck
        feat_video_dim = feat_aggregated_dim

        # 3. classifiers (video-level)
        self.fc_classifier_video_source = ClassificationHead(feat_video_dim,feat_video_dim // 2,num_class)

    def train(self, mode=True):
        # not necessary in our setting
        """
		Override the default train() to freeze the BN parameters
		:return:
		"""
        super(VideoModel, self).train(mode)

    def partialBN(self, enable):
        self._enable_pbn = enable


    def final_output(self, pred, pred_video, num_segments):
        if self.baseline_type == 'video':
            base_out = pred_video
        else:
            base_out = pred

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        output = base_out

        if self.baseline_type == 'tsn':
            if self.reshape:
                base_out = base_out.view(
                    (-1, num_segments) +
                    base_out.size()[1:])  # e.g. 16 x 3 x 12 (3 segments)

            output = base_out.mean(1)  # e.g. 16 x 12

        return output



    def forward(self, input_source, text_token, is_train):
        batch_source = input_source.size()[0]

        num_segments = self.train_segments if is_train else self.val_segments
        sample_len = self.new_length
        feat_all_source = []

        # szf add
        input_source_convert = self.feature_convert(input_source)

        # input_data is a list of tensors --> need to do pre-processing
        feat_base_source = input_source_convert.view(
            -1,
            input_source_convert.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048
        

        feat_fc_source = feat_base_source

        # default trn-m
        # elif 'trn' in self.frame_aggregation: ### trn-m
        feat_fc_video_source = feat_fc_source.view(
            (-1, num_segments) + feat_fc_source.size()[-1:]
        )  # reshape based on the segments (e.g. 640x512 --> 128x5x512)


        # print((-1, num_segments) + feat_fc_source.size()[-1:])
        feat_fc_video_relation_source = self.TRN(
            feat_fc_video_source
        )  # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)


        # sum up relation features (ignore 1-relation)
        feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)

        pred_fc_video_source = self.fc_classifier_video_source(
            feat_fc_video_source)


        ########## coop ######################################################
        coop_out_logits, category_text_features = self.coop_model(input_source) 
        coop_out_logits_mean = coop_out_logits[:,-1,:]
        ## clip text logits
        text_feature = F.normalize(self.clip_model.encode_text(text_token.cuda()),dim=-1).to(torch.float32).detach()
        text_logits = text_feature @ category_text_features.t() 
        text_logits = text_logits / 2.0 # coefficient
        vlm_logits = coop_out_logits_mean + text_logits

        ####################################################################

        #######################选取 frame-level features############################################################

        frame_level_repre = torch.sum(input_source_convert,dim=1)

        classifier_logits = pred_fc_video_source

        #####################构建的特征池 local######################3
        save_feature_list = []
        # frame-level
        save_feature_list.append(frame_level_repre)
        # logits
        save_logits = classifier_logits

        # 更新特征池（两种level）
        current_results_all = self.pseudo_selector.update_with_batch(save_feature_list, save_logits, self.conf_temp)
        current_results = current_results_all

        # 存储所有特征和标签  
        all_features_list = []
        all_labels_list = []
        all_en_list = []
        all_conf_list = []
        
        # 遍历所有类别 0 ~ L
        for cls_idx in range(save_logits.shape[1]):  # 假设类别索引是 0, 1, ..., L
            if cls_idx in current_results:
                # 如果类别存在，使用原始数据
                data = current_results[cls_idx]
                features_list_temp = data['features']   # len=2 list
                labels_temp = torch.full((features_list_temp[-1].shape[0], 1), cls_idx, dtype=torch.long).cuda()
                en_temp = data['entropies'].unsqueeze(-1)
                conf_temp = data['confidences'].unsqueeze(-1)
            else:
                # 如果类别不存在，填充 1×512 的全零矩阵，并补上标签
                # features_list_temp = [torch.zeros(1, 512).cuda(), torch.zeros(1, 512).cuda()]  # [1, 512]
                features_list_temp = [torch.zeros(1, 512).cuda()]  # [1, 512]
                labels_temp = torch.tensor([[cls_idx]], dtype=torch.long).cuda()  # [1, 1]
                en_temp = torch.ones(1, 1).cuda()
                conf_temp = torch.ones(1, 1).cuda()
            
            all_features_list.append(features_list_temp)
            all_labels_list.append(labels_temp)
            all_en_list.append(en_temp)
            all_conf_list.append(conf_temp)

        # frame level saved features
        all_features_list_frame = [feature[0] for feature in all_features_list]

        # prototype_classifier frame
        prototype_classifier_frame = compute_proto_classifier_conf(all_features_list_frame, all_labels_list, all_en_list, all_conf_list)

        prototype_logits_frame = F.normalize(frame_level_repre,dim=-1) @ F.normalize(prototype_classifier_frame,dim=-1).t()

        # ################ ablation ####################
        # output_source = pred_fc_video_source
        # output_source = prototype_logits_frame
        # output_source = prototype_logits_frame  + coop_out_logits_mean
        # output_source = prototype_logits_frame  + (coop_out_logits_mean + text_logits * 0.5) / 2.0
        # output_source = prototype_logits_frame  + (coop_out_logits_mean + text_logits * 0.5) / 2.0 # learnable prompt
        # ############################################################

        output_source = prototype_logits_frame  +  vlm_logits / 2.0

        return output_source, text_logits, coop_out_logits_mean, prototype_logits_frame



