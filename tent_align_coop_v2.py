from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import numpy as np
import torch.nn.functional as F

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
# class ContrastiveLoss(nn.Module):  
#     def __init__(self, temperature=0.5):  
#         super().__init__()  
#         self.temperature = temperature  
          
#     def forward(self, features, labels):  
#         # 计算特征之间的余弦相似度  
#         features = F.normalize(features, dim=1)  
#         similarity_matrix = torch.matmul(features, features.T) / self.temperature  
          
#         # 创建标签掩码：相同类别为1，不同为0  
#         labels_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  
          
#         # 计算对比损失  
#         logits = torch.log_softmax(similarity_matrix, dim=1)  
#         loss = -torch.sum(logits * labels_mask, dim=1) / (labels_mask.sum(dim=1) + 1e-8)  
#         loss = loss.mean()  
          
#         return loss  
  

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, text_token,  is_train):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, text_token, self.model, self.optimizer, is_train)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, text_token, model, optimizer, is_train):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    model.train()
    # forward
    outputs, text_logits, visual_logits , proto_logits = model(x, text_token, is_train)


    t2v_v2t_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(text_logits ,dim=1),F.softmax(visual_logits ,dim=1).detach()) + \
        nn.KLDivLoss(reduction='batchmean')(F.log_softmax(visual_logits,dim=1),F.softmax(text_logits  ,dim=1).detach())

    loss = t2v_v2t_loss * 100.0
    # loss = t2v_v2t_loss * 1000.0
    print(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return outputs


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
