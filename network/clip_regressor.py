import torch
import torch.nn as nn
import torch.nn.functional as F


# 회귀 모델 정의
class TtoTRegressor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(TtoTRegressor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 최적 T값 회귀 예측
        )

    def forward(self, x):
        batch_size, T, features = x.shape
        x = self.mlp(x)  # (batch_size, T, hidden_dim)
        
        x = x.permute(1, 0, 2)  # T, batch_size, hidden_dim)
        x = self.transformer(x)  # (T, batch_size, hidden_dim)
        x = x.permute(1, 0, 2).mean(dim=1)  # (batch_size, T, hidden_dim) 평균 풀링 수행 (batch, hidden_dim)

        T_pred = self.fc(x).squeeze(-1)  # (batch_size,)
        return T_pred