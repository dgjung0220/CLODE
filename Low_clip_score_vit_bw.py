import torch
from tqdm import tqdm
from network.conv_node import NODE
from misc import *
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.multimodal import CLIPImageQualityAssessment
import time
import timm

# GPU 번호 지정
gpu_number = 3  # 원하는 GPU 번호로 변경 가능

# GPU 사용 제한
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

# 이제 GPU가 한 개만 보이므로 cuda:0으로 접근
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = timm.models.vit_large_patch16_224(pretrained=True).to(device)

file_path = Path('/home/lbw/data/our485')
# img_labels = sorted(os.listdir(file_path / 'low'))
img_labels = [f for f in sorted(os.listdir(file_path / 'low')) if f.lower().endswith('.png')]

vit_features = []
# contrast_scores = []

# T 값들을 먼저 텐서로 변환하여 반복 변환 방지
# T_tensors = [torch.tensor([0, T]).float().cuda() for T in T_values]

def load_image(idx):
    # lq_img = image_tensor(file_path / 'low' / img_labels[idx], size=(256, 256))
    # gt_img = image_tensor(file_path / 'high' / img_labels[idx], size=(256, 256))
    lq_img = image_tensor(file_path / 'low' / img_labels[idx], (224,224))
    # gt_img = image_tensor(file_path / 'high' / img_labels[idx])
    
    return lq_img.to(device)

with torch.no_grad():
    for idx in tqdm(range(len(img_labels))):
        lq_img = load_image(idx)
        # high_psnr = 0.0
        # best_T = 2.0
        
        # # 모든 T에 대한 예측을 한 번에 계산
        # preds = []
        # psnrs = []

        vit_output = model(lq_img)
        
        vit_features.append([vit_output.cpu().numpy()])

save_path = Path('/home/lbw/CLODE/vit_L_224_csv')
save_path.mkdir(parents=True, exist_ok=True)

vit_features = np.array(vit_features)
np.save(Path(save_path / 'vit_features_L.npy'), vit_features)
