import torch
from tqdm import tqdm
from network.conv_node import NODE
from misc import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NODE(device, (3, 256, 256), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)
model.load_state_dict(torch.load(f'pth/universal.pth'), strict=False)

import argparse
parser = argparse.ArgumentParser(description="CLODE")
parser.add_argument('--T', type=float, required=False, default=3)
args = parser.parse_args()

integration_time = torch.tensor([0, args.T]).float().cuda()
file_path = '/path/to/your/input'
gt_path = '/path/to/your/corresponding_gt'

lq_imgs, gt_imgs = get_filelist(file_path, gt_path)
psnr_results, ssim_results = [], []

with torch.no_grad():    

    for idx, img in tqdm(enumerate(lq_imgs)):
        x = img.cuda()
        out = model(x, integration_time, inference=True)        
        pred = out['output']
        
        if len(gt_imgs) != 0:
            gt = gt_imgs[idx].cuda()
            val1 = calculate_psnr(pred, gt).item()
            val2 = calculate_ssim(pred, gt).item()
            psnr_results.append(val1)
            ssim_results.append(val2)

if len(gt_imgs) != 0:
    print('PSNR: ' ,np.mean(psnr_results), 'SSIM: ', np.mean(ssim_results))