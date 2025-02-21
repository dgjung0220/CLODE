import torch
import torch.nn as nn
import torch.nn.functional as F
import losses as loss_func
import matplotlib.pyplot as plt
from network.torchdiffeq import odeint_adjoint

MAX_NUM_STEPS = 1000 # 30 # 50 # 100
 
def normalize_minmax(x):
    normalized = (x-x.min()) / (x.max() - x.min())
    return normalized
    
class Conv2dTime(nn.Conv2d):
    
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):        
        t_img = torch.ones_like(x[:, :1, :, :]) * t # Shape (batch_size, 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)  # Shape (batch_size, channels + 1, height, width)
        return super(Conv2dTime, self).forward(t_and_x)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class network(nn.Module):
    def __init__(self,n_chan,chan_embed=48):
        super(network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
    
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x 
    
class EnhanceFunc(nn.Module):
    
    def __init__(self, num_filters):
        
        super(EnhanceFunc, self).__init__()
        self.nfe = 0
        self.num_filters = num_filters        
        self.conv1 = Conv2dTime(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
         
        self.norm32 = nn.GroupNorm(1, 32)
        self.norm64 = nn.GroupNorm(1, 64)        
        self.up_conv = Conv2dTime(6, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.down_conv = Conv2dTime(32, 3, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        self.conv_3_1 = Conv2dTime(self.num_filters, self.num_filters, kernel_size=3, padding=3//2, padding_mode='reflect')
        self.conv_5_1 = Conv2dTime(self.num_filters, self.num_filters, kernel_size=5, padding=5//2, padding_mode='reflect')
        self.conv_3_2 = Conv2dTime(self.num_filters * 2, self.num_filters * 2, kernel_size=3, padding=3//2, padding_mode='reflect')
        self.conv_5_2 = Conv2dTime(self.num_filters * 2, self.num_filters * 2, kernel_size=5, padding=5//2, padding_mode='reflect')
        self.confusion = Conv2dTime(self.num_filters * 4, self.num_filters, 1, padding=0, stride=1, padding_mode='reflect')
        
        number_f = 32
        self.e_conv1 = Conv2dTime(6, number_f,3,1,1,bias=True) 
        self.e_conv2 = Conv2dTime(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = Conv2dTime(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = Conv2dTime(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = Conv2dTime(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = Conv2dTime(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = Conv2dTime(number_f*2,3,3,1,1,bias=True) 

        self.denoise = network(3)
        self.tv_loss = loss_func.L_TV()
        
        self.pred_t = []
                
    def pair_downsampler(self, img):
        c = img.shape[1]
        filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
        filter1 = filter1.repeat(c,1, 1, 1)
        filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c,1, 1, 1)
        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)    
        return output1, output2

    def mse(self, gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
        loss = torch.nn.MSELoss()
        return loss(gt,pred)
    
    def loss_func(self, noisy_img):
        noisy1, noisy2 = self.pair_downsampler(noisy_img)
        pred1 =  noisy1 - self.denoise(noisy1)
        pred2 =  noisy2 - self.denoise(noisy2)
        loss_res = 1/2 * (self.mse(noisy1,pred2) + self.mse(noisy2,pred1))
        noisy_denoised =  noisy_img - self.denoise(noisy_img)
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        loss_cons = 1/2 * (self.mse(pred1,denoised1) + self.mse(pred2,denoised2))
        loss = loss_res + loss_cons
        return loss
    
    def add_noise(self, x, noise_level):
    
        noisy = x + torch.normal(0, noise_level/255, x.shape).cuda()
        noisy = torch.clamp(noisy,0,1)
        return noisy

    def forward(self, t, x):
        self.nfe += 1
        
        _x = x[:, :3 , :, :] 
        _, c, h, w = _x.shape
 
        noise_map = self.loss_func(_x)
        p_x = _x - self.denoise(_x)
        _in = torch.cat([p_x, 1-p_x], 1)
    
        input_1 = self.relu(self.norm32(self.up_conv(t, _in)))
        output_3_1 = self.relu(self.norm32(self.conv_3_1(t, input_1)))
        output_5_1 = self.relu(self.norm32(self.conv_5_1(t, input_1)))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.norm64(self.conv_3_2(t, input_2)))
        output_5_2 = self.relu(self.norm64(self.conv_5_2(t, input_2)))
        input_3 = torch.cat([output_3_2, output_5_2], 1)     
        output = self.relu(self.norm32(self.confusion(t, input_3)))
        
        _A = F.tanh(self.down_conv(t, output)) 
 
        pred = _A * (torch.pow(_x, 2) - _x)
        self.last_curve_map = _A
        L_tv = torch.ones_like(_A) * self.tv_loss(_A)        
        noise_map = torch.ones_like(_A) * noise_map
        self.pred_t.append(t.item())
    
        return torch.cat([pred, L_tv, noise_map], 1)
        
class ODEBlock(nn.Module):
   
    def __init__(self, device, odefunc, is_conv=False, tol=1e-3, adjoint=False):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        
        if eval_times == None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times
        
        self.odefunc.nfe = 0    
        x_aug = x
                
        out = odeint_adjoint(self.odefunc, 
                             x_aug, 
                             integration_time,
                             rtol=self.tol, 
                             atol=self.tol, 
                             method='dopri5', # 'dopri5', 'euler', 'rk4'
                             options={'max_num_steps': MAX_NUM_STEPS})
        
        return out
       
class NODE(nn.Module):

    def __init__(self, device, img_size, num_filters, augment_dim=0, time_dependent=False, tol=1e-5, adjoint=True):
        
        super(NODE, self).__init__()
        self.device = device
        self.img_size = img_size
        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.tol = tol
        
        self.odefunc = EnhanceFunc(num_filters)
        self.odeblock = ODEBlock(device, self.odefunc, is_conv=True, tol=tol, adjoint=adjoint)
        self.idx = 0
        
    def forward(self, x, eval_time=None, inference=False):
        
        _, c, h, w = x.shape
        _input = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], 1)
        preds = self.odeblock(_input, eval_time)
    
        pred = preds[-1]
        curve_map = self.odefunc.last_curve_map

        if inference:
            output = {
                'output': torch.clamp(pred[:, :3, :, :] - self.odefunc.denoise(pred[:, :3, :, :]), 0, 1),
                'curve_map' : normalize_minmax(curve_map),
                'all' : [torch.clamp(pred[:, :3, :, :] - self.odefunc.denoise(pred[:, :3, :, :]), 0, 1) for pred in preds]
            }
        else:
            output = {
                'output': pred[:, :3, :, :],
                'curve_map' : pred[:, 3:6, :, :],
                'noise_map' : pred[:, 6:9, :, :],
            }

        return output