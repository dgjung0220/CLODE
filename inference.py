import torch
from tqdm import tqdm
from network.conv_node import NODE
from misc import *
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.multimodal import CLIPImageQualityAssessment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NODE(device, (3, 256, 256), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)
model.load_state_dict(torch.load(f'pth/universal.pth', weights_only=True), strict=False)

file_path = Path('/data/soom/LSRW/eval/Huawei')
img_labels = sorted(os.listdir(file_path / 'low'))

def load_image(idx):
    lq_img = image_tensor(file_path / 'low' / img_labels[idx], size=(256, 256))
    gt_img = image_tensor(file_path / 'high' / img_labels[idx], size=(256, 256))
    
    return lq_img.to(device), gt_img.to(device)

psnr_results, ssim_results, clip_scores_0, clip_scores_1 = [], [], [], []
T_values = np.linspace(0.1, 6, 100)  # Adjust T value

prompts = (('good exposure', 'bad exposure'), ('good exposure', 'under or over exposure'))
clip_iqa = CLIPImageQualityAssessment(prompts=prompts).to(device)

def calculate_clip_score(pred):
    score = clip_iqa(pred.unsqueeze(0))
    
    return score['user_defined_0'].item()

# 30개 이미지에 대해, [0.1, 6.0] 구간 T 100개 계산 => 각 이미지의 최적 T값과, 그때의 PSNR / SSIM / CLIP IQA score 저장.
# inference -> clip_score 계산 -> score 상위 4개 T 값에 대해 T / PSNR / SSIM / CLIP IQA score 저장 (30 * 4 * 4)
total_psnr, total_ssim = 0.0, 0.0
best_Ts = np.zeros((len(img_labels), 4, 4), dtype=float)

with torch.no_grad():
    for idx in tqdm(range(len(img_labels))):
        lq_img, gt_img = load_image(idx)
        img_values = []

        for T in tqdm(T_values):
            integration_time = torch.tensor([0, T]).float().cuda()
            pred = model(lq_img, integration_time, inference=True)['output'][0]
            
            _psnr = calculate_psnr(pred, gt_img).item()
            _ssim = calculate_ssim(pred, gt_img).item()
            _clip_score = calculate_clip_score(pred)
            
            img_values.append((T, _psnr, _ssim, _clip_score))

        # score 상위 4개에 대한 T / PSNR / SSIM / CLIP IQA score 저장
        img_values = np.array(img_values)
        best_idx_0 = np.argsort(img_values[:, 3])[::-1][:4]
        best_Ts[idx] = np.array(img_values[best_idx_0, :])
        # score 1위 T에 대한 PSNR, SSIM은 따로 합산
        total_psnr += best_Ts[idx, 0, 1]
        total_ssim += best_Ts[idx, 0, 2]

# Plotting the results
np.save('best_Ts.npy', best_Ts)
print(f'Average PSNR: {total_psnr / len(img_labels):.2f}')
print(f'Average SSIM: {total_ssim / len(img_labels):.2f}')
