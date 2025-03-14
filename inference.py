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
batch_size = 16

def load_images(start_idx, batch_size):
    lq_imgs, gt_imgs = [], []
    for label in img_labels[start_idx:start_idx+batch_size]:
        lq_imgs.append(image_tensor(file_path / 'low' / label, size=(256, 256)))
        gt_imgs.append(image_tensor(file_path / 'high' / label, size=(256, 256)))
    
    return torch.stack(lq_imgs).cuda(), torch.stack(gt_imgs).cuda()

psnr_results, ssim_results, clip_scores, clip_av_scores = [], [], [], []
T_values = np.linspace(0.1, 10, 100)  # Adjust T value

clip_iqa = CLIPImageQualityAssessment(prompts=("quality", "brightness")).to(device)

def calculate_clip_score(preds):
    score = clip_iqa(preds)
    score_np = score['quality'].detach().cpu().numpy()
    score_np_av = score_np + score['brightness'].detach().cpu().numpy()
    
    return score_np, score_np_av

with torch.no_grad():
    for T in T_values:
        integration_time = torch.tensor([0, T]).float().cuda()
        temp_psnr, temp_ssim = [], []
        temp_clip, temp_clip_av = np.array([]), np.array([])
        
        for start_idx in tqdm(range(0, len(img_labels), batch_size)):
            lq_imgs, gt_imgs = load_images(start_idx, batch_size)
            out = model(lq_imgs, integration_time, inference=True)       
            preds = out['output']
            
            for i in range(len(preds)):
                val1 = calculate_psnr(preds[i], gt_imgs[i]).item()
                val2 = calculate_ssim(preds[i], gt_imgs[i]).item()
                temp_psnr.append(val1)
                temp_ssim.append(val2)
            val3, val4 = calculate_clip_score(preds)
            temp_clip = np.concatenate((temp_clip, val3))
            temp_clip_av = np.concatenate((temp_clip_av, val4))

        psnr_results.append(np.mean(temp_psnr))
        ssim_results.append(np.mean(temp_ssim))
        clip_scores.append(np.mean(temp_clip))
        clip_av_scores.append(np.mean(temp_clip_av) / 2)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(T_values, psnr_results, label='PSNR')
plt.plot(T_values, ssim_results, label='SSIM')
plt.plot(T_values, clip_scores, label='CLIP IQA')
plt.plot(T_values, clip_av_scores, label='CLIP IQA (avg)')
plt.xlabel('T values')
plt.ylabel('Scores')
plt.legend()
plt.title('PSNR, SSIM, and CLIP IQA Scores vs T values')
plt.show()
plt.savefig('score_result.png')

# Print the T value with the highest CLIP IQA score
max_clip_iqa_idx = np.argmax(clip_scores)
max_clip_iqa_idx_av = np.argmax(clip_av_scores)
print(f'Max CLIP IQA Score: {clip_scores[max_clip_iqa_idx]:.2f} at T={T_values[max_clip_iqa_idx]:.2f}')
print(f'Max CLIP IQA Score (avg): {clip_av_scores[max_clip_iqa_idx_av]:.2f} at T={T_values[max_clip_iqa_idx_av]:.2f}')