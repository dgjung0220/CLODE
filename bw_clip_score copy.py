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

# # GPU 번호 지정
# gpu_number = 3  # 원하는 GPU 번호로 변경 가능

# # GPU 사용 제한
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

# 이제 GPU가 한 개만 보이므로 cuda:0으로 접근
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = NODE(device, (3, 400, 600), 32, augment_dim=0, time_dependent=True, adjoint=True)
model.eval()
model.to(device)
model.load_state_dict(torch.load(f'./pth/lowlight.pth', weights_only=True), strict=False)

file_path = Path('./data/LOLdataset/our485')
# img_labels = sorted(os.listdir(file_path / 'low'))
img_labels = [f for f in sorted(os.listdir(file_path / 'low')) if f.lower().endswith('.png')]

def load_image(idx):
    # lq_img = image_tensor(file_path / 'low' / img_labels[idx], size=(256, 256))
    # gt_img = image_tensor(file_path / 'high' / img_labels[idx], size=(256, 256))
    lq_img = image_tensor(file_path / 'low' / img_labels[idx], (224,224))
    # gt_img = image_tensor(file_path / 'high' / img_labels[idx])
    
    return lq_img.to(device)


prompts = ('brightness', 'noisiness', 'quality')  # 단순 튜플로 변경
# prompts = ('darkness')

clip_metric = CLIPImageQualityAssessment(
    model_name_or_path="openai/clip-vit-base-patch16",
    prompts=prompts  # 이미 적절한 형식을 가진 튜플
).to(device)

clip_vision_encoder = clip_metric.model.vision_model
clip_vision_encoder.eval()
clip_vision_encoder.to(device)

clip_visual_projection = clip_metric.model.visual_projection
clip_visual_projection.eval()
clip_visual_projection.to(device)

def calculate_clip_score(pred, prompts=prompts):    
    # 이미 배치 차원이 있는지 확인하고 없으면 추가
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    
    with torch.no_grad():
        # 한 번의 forward pass로 모든 프롬프트에 대한 점수를 계산
        encoder_feature = clip_vision_encoder(pred) # Pooler Output: torch.Size([1, 768])
        projection = clip_visual_projection(encoder_feature[1]) # image projection: torch.Size([1, 512])

    return projection

# 메인 루프 최적화
# T_values = np.linspace(2, 5, 31)

clip_features = []
# contrast_scores = []

# T 값들을 먼저 텐서로 변환하여 반복 변환 방지
# T_tensors = [torch.tensor([0, T]).float().cuda() for T in T_values]

with torch.no_grad():
    for idx in tqdm(range(len(img_labels))):
        lq_img = load_image(idx)
        # high_psnr = 0.0
        # best_T = 2.0
        
        # # 모든 T에 대한 예측을 한 번에 계산
        # preds = []
        # psnrs = []

        clip_output = calculate_clip_score(lq_img)
        
        clip_features.append([clip_output.cpu().numpy()])

save_path = Path('./scores_csv_4prompts_400600')
save_path.mkdir(parents=True, exist_ok=True)

clip_features = np.array(clip_features)
np.save(Path(save_path / 'clip_features.npy'), clip_features)
