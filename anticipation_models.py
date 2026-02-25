from torch import nn

from torch.nn.init import *
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import TRNmodule
import math

from colorama import init
from colorama import Fore, Back, Style

# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)

# init(autoreset=True)


# definition of Gradient Reversal Layer
class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


# definition of Gradient Scaling Layer
class GradScale(Function):

    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * ctx.beta
        return grad_input, None


# # definition of Temporal-ConvNet Layer
# class TCL(nn.Module):

#     def __init__(self, conv_size, dim):
#         super(TCL, self).__init__()

#         self.conv2d = nn.Conv2d(dim,
#                                 dim,
#                                 kernel_size=(conv_size, 1),
#                                 padding=(conv_size // 2, 0))

#         # # initialization
#         # kaiming_normal_(self.conv2d.weight)

#     def forward(self, x):
#         x = self.conv2d(x)

#         return x


class FeatureConvert(nn.Module):
    def __init__(self, input_dim, hidden_dim, droprate):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.hidden_dim = hidden_dim

        self.layer_in = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim , kernel_size=1),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )

        self.layer_mid = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim , kernel_size=1),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )

        self.layer_out = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.output_dim, kernel_size=1),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )

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
            # nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            # nn.Dropout(0.5),
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
                 share_params='Y'):
        super(VideoModel, self).__init__()
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
        # count = 0
        # if self._enable_pbn:
        #     print("Freezing BatchNorm2D except the first one.")
        #     for m in self.base_model.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             count += 1
        #             if count >= (2 if self._enable_pbn else 1):
        #                 m.eval()

        #                 # shutdown update in frozen mode
        #                 m.weight.requires_grad = False
        #                 m.bias.requires_grad = False

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


    def forward(self, input_source, is_train):
        batch_source = input_source.size()[0]

        num_segments = self.train_segments if is_train else self.val_segments
        sample_len = self.new_length
        feat_all_source = []

        # szf add
        input_source = self.feature_convert(input_source)

        # input_data is a list of tensors --> need to do pre-processing
        feat_base_source = input_source.view(
            -1,
            input_source.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048
        

        feat_fc_source = feat_base_source

        ### save features ####
        feat_all_source.append(
            feat_fc_source.view((batch_source, num_segments) +
                                feat_fc_source.size()[-1:])
        )  # reshape ==> 1st dim is the batch size

        # default trn-m
        # elif 'trn' in self.frame_aggregation: ### trn-m
        feat_fc_video_source = feat_fc_source.view(
            (-1, num_segments) + feat_fc_source.size()[-1:]
        )  # reshape based on the segments (e.g. 640x512 --> 128x5x512)


        # print((-1, num_segments) + feat_fc_source.size()[-1:])
        feat_fc_video_relation_source = self.TRN(
            feat_fc_video_source
        )  # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)

        ### save attentions ####
        # transferable attention
        attn_relation_source = feat_fc_video_relation_source[:, :,
                                                                    0]  # assign random tensors to attention values to avoid runtime error

        # sum up relation features (ignore 1-relation)
        feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)

        ### save features ####
        if self.baseline_type == 'video':
            feat_all_source.append(
                feat_fc_video_source.view((batch_source, ) +
                                          feat_fc_video_source.size()[-1:]))

        #=== source layers (video-level) ===#
        # feat_fc_video_source = self.dropout_v(feat_fc_video_source)

        pred_fc_video_source = self.fc_classifier_video_source(
            feat_fc_video_source)
        
        ### save features ####
        if self.baseline_type == 'video':  # only store the prediction from classifier 1 (for now)
            feat_all_source.append(
                pred_fc_video_source.view((batch_source, ) +
                                          pred_fc_video_source.size()[-1:]))

        # #=== final output ===#
        # output_source = self.final_output(
        #     pred_fc_source, pred_fc_video_source,
        #     num_segments)  # select output from frame or video prediction
        
        output_source = pred_fc_video_source
        output_source_2 = output_source


        return attn_relation_source, output_source, output_source_2, feat_all_source[::-1] # reverse the order of feature list due to some multi-gpu issues
        
   
