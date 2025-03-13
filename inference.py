import torch
from tqdm import tqdm
from network.conv_node import NODE
from misc import *
import os
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NODE(device, (3, 256, 256), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)
model.load_state_dict(torch.load(f'pth/universal.pth', weights_only=True), strict=False)

import argparse
parser = argparse.ArgumentParser(description="CLODE")
parser.add_argument('--T', type=float, required=False, default=3)
args = parser.parse_args()

integration_time = torch.tensor([0, args.T]).float().cuda()
file_path = Path('/data/soom/LSRW/eval/Huawei')

img_labels = sorted(os.listdir(file_path / 'low'))
batch_size = 16

def load_images(start_idx, batch_size):
    lq_imgs, gt_imgs = [], []
    for label in img_labels[start_idx:start_idx+batch_size]:
        lq_imgs.append(image_tensor(file_path / 'low' / label, size=(256, 256)))
        gt_imgs.append(image_tensor(file_path / 'high' / label, size=(256, 256)))
    
    return torch.stack(lq_imgs).cuda(), torch.stack(gt_imgs).cuda()

psnr_results, ssim_results = [], []

with torch.no_grad():    
    for start_idx in tqdm(range(0, len(img_labels), batch_size)):
        lq_imgs, gt_imgs = load_images(start_idx, batch_size)
        out = model(lq_imgs, integration_time, inference=True)       
        preds = out['output']
        
        for i in range(len(preds)):
            val1 = calculate_psnr(preds[i], gt_imgs[i]).item()
            val2 = calculate_ssim(preds[i], gt_imgs[i]).item()
            psnr_results.append(val1)
            ssim_results.append(val2)
if len(gt_imgs) != 0:
    print(f'PSNR: {np.mean(psnr_results):.2f}, SSIM: {np.mean(ssim_results):.2f}')