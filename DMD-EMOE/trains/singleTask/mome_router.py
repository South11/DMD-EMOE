"""
MoME Router Network - Attentive Version
引入注意力池化，让路由器能捕捉序列中的关键时刻
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """
    注意力池化层：(Seq, Batch, Dim) -> (Batch, Dim)
    自动学习序列中哪些时间步是重要的
    """
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        # x: (Seq, Batch, Dim) or (Batch, Dim)
        if x.dim() == 2:
            return x

        # 计算注意力权重
        # (Seq, Batch, 1)
        scores = self.attn(x)
        # 在序列维度(dim=0)上做Softmax
        alpha = F.softmax(scores, dim=0)

        # 加权求和
        # (Seq, Batch, Dim) * (Seq, Batch, 1) -> sum(dim=0) -> (Batch, Dim)
        pooled = (x * alpha).sum(dim=0)
        return pooled

class MoMERouter(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], num_modalities=3, temperature=1.0, dropout=0.2):
        super(MoMERouter, self).__init__()
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_modalities))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, modality_features):
        # modality_features: List of (Batch, Dim) tensors
        combined = torch.cat(modality_features, dim=-1)
        logits = self.mlp(combined)
        weights = F.softmax(logits / self.temperature, dim=-1)
        return weights

class MoMELosses(nn.Module):
    def __init__(self, alpha=0.1):
        super(MoMELosses, self).__init__()
        self.alpha = alpha
        self.eps = 1e-8

    def balance_loss(self, router_weights, modality_errors):
        batch_size = router_weights.size(0)
        entropy_loss = -torch.sum(router_weights * torch.log(router_weights + self.eps)) / batch_size
        matching_loss = torch.mean(torch.sum(router_weights * modality_errors, dim=1))
        return entropy_loss * 0.1 + matching_loss * self.alpha