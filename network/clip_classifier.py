import torch
import torch.nn as nn

class TtoTClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super(TtoTClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, T, 3) 또는 (T, 3)
        if len(x.shape) == 2:
            # 단일 이미지에 대한 T개의 스코어 세트
            T, features = x.shape
            scores = self.mlp(x).squeeze(-1)  # (T,)
        else:
            # 배치 처리
            batch_size, T, features = x.shape
            x_reshaped = x.reshape(-1, features)  # (batch_size*T, 3)
            scores = self.mlp(x_reshaped).squeeze(-1)  # (batch_size*T,)
            scores = scores.reshape(batch_size, T)  # (batch_size, T)
        
        return scores
