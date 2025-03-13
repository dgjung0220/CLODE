import torch
import numpy as np
import torchvision
import cv2

from PIL import Image
from tqdm import tqdm
from glob import glob

def image_tensor(image_path, size=None):
    img = Image.open(image_path)
    
    if size != None:
        img = img.resize(size)

    img = (np.asarray(img)/255.0)
    img = torch.from_numpy(img).float()
    img = img.permute(2,0,1)
    # img = img.cuda().unsqueeze(0)
    return img

def get_filelist(input_path, gt_path=None, size=None):
    
    lq_tensors, gt_tensors = [], []
    
    file_list = sorted(glob(input_path+'/*'), key=lambda x : x.split('/')[-1].split('-')[0])
    for i in tqdm(file_list):
        lq_tensors.append(image_tensor(i, size))

    if gt_path != None:
        gt_list = sorted(glob(gt_path+'/*'), key=lambda x : x.split('/')[-1].split('-')[0])            
        for i in tqdm(gt_list):
            gt_tensors.append(image_tensor(i, size))
            
    return lq_tensors, gt_tensors

'''
# basicsr code (calculate_psnr, calculate_ssim)
https://github.com/XPixelGroup/BasicSR/tree/master/basicsr
'''

def calculate_psnr(target, ref):
    
    target = torchvision.transforms.ToPILImage()(target.squeeze(0))
    ref = torchvision.transforms.ToPILImage()(ref.squeeze(0))
    
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    
    target = torchvision.transforms.ToPILImage()(target.squeeze(0))
    ref = torchvision.transforms.ToPILImage()(ref.squeeze(0))
    
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')