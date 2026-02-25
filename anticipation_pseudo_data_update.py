import torch
import torch.nn.functional as F
from collections import defaultdict

class DynamicFeatureSelector:
    def __init__(self, feature_dims, select_top_k=5, entropy_top_k=5):
        """
        参数:
            feature_dims: List[int], 每个特征的维度（如 [C1, C2, ...]）
            select_top_k: 每个样本从 logits 中选择置信度最高的前 select_top_k 个类别
            entropy_top_k: 每个类别最终保留熵最小的前 entropy_top_k 个样本
        """
        self.feature_dims = feature_dims  # 修改为列表
        self.select_top_k = select_top_k
        self.entropy_top_k = entropy_top_k
        self.global_results = defaultdict(list)  # 维护全局最优样本
        
    def update_with_batch(self, inter_representations, logits, conf_temperature):
        """
        用新 batch 的数据更新全局结果
        
        参数:
            inter_representations: List[Tensor], 每个 Tensor 的形状为 [B, C_i] 或 [B, L, C_i]
            logits: [B, L] 模型输出 logits（L 是类别数或其他维度，与特征无关）
        """

        # 计算 置信度 熵
        #  1 1 1 1 
        #  1 2 2 2
        probs = F.softmax(logits / conf_temperature , dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # [B]
        
        # 获取每个样本的 top-select_top_k 类别（按置信度）
        top_probs, top_classes = torch.topk(probs, k=self.select_top_k, dim=1) # [B, select_top_k]

        # print(top_probs)


        # 收集所有样本的数据
        batch_data = []
        for i in range(logits.size(0)):
            for idx in range(len(top_classes[i])):
            # for cls_idx in top_classes[i]:  # 只遍历 select_top_k 个类别
                # 提取当前样本的所有特征（处理 List[Tensor]）
                cls_idx = top_classes[i][idx]
                conf = top_probs[i][idx]

                sample_features = []
                for feat in inter_representations:
                    if feat.dim() == 2:  # [B, C_i]
                        sample_features.append(feat[i])
                    elif feat.dim() == 3:  # [B, L, C_i]
                        sample_features.append(feat[i])  # 不依赖 cls_idx，直接取 [B, L, C_i] 的第 i 个样本
                    else:
                        raise ValueError("特征维度必须是 [B, C_i] 或 [B, L, C_i]")
                
                batch_data.append({
                    'features': sample_features,  # 存储为 List[Tensor]
                    'class': cls_idx.item(),
                    'entropy': entropy[i].item(),
                    'confidence': conf.item()
                })
        
        # 按类别组织新 batch 数据
        batch_class_data = defaultdict(list)
        for data in batch_data:
            batch_class_data[data['class']].append(data)
        
        # 更新全局结果
        for cls_idx, new_samples in batch_class_data.items():
            current_samples = self.global_results.get(cls_idx, [])
            combined = current_samples + new_samples
            combined.sort(key=lambda x: x['entropy'])
            updated = combined[:self.entropy_top_k]
            self.global_results[cls_idx] = updated
        
        return self._convert_to_tensors()
    
    def _convert_to_tensors(self):
        """将全局结果转换为张量格式"""
        result = {}
        for cls_idx, samples in self.global_results.items():
            if not samples:
                # 空特征：返回 List[Tensor], 每个 Tensor 的形状为 [0, C_i]
                result[cls_idx] = {
                    'features': [torch.zeros(0, dim) for dim in self.feature_dims],
                    'entropies': torch.zeros(0),
                    'confidence': torch.zeros(0)
                }
                continue
            
            # 按特征维度分组
            grouped_features = defaultdict(list)
            for s in samples:
                for feat_idx, feat in enumerate(s['features']):
                    grouped_features[feat_idx].append(feat)
            
            # 堆叠每个特征
            features_list = []
            for feat_idx in range(len(self.feature_dims)):
                features = torch.stack(grouped_features[feat_idx])  # [N, C_i]
                features_list.append(features)
            
            entropies = torch.tensor([s['entropy'] for s in samples], dtype=torch.float32)
            confidences = torch.tensor([s['confidence'] for s in samples], dtype=torch.float32)
            
            result[cls_idx] = {
                'features': features_list,  # List[Tensor], 每个 [N, C_i]
                'entropies': entropies,      # [N]
                'confidences': confidences   # [N]
            }
        return result
    
    def get_current_results(self):
        """获取当前全局结果"""
        return self._convert_to_tensors()