import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models.vgg import vgg16
import numpy as np
from pylab import *
from torch.autograd import Variable

# ------------------------------------------------------------------------------
# This code is adapted from:
# https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
# Original implementation by Li Chongyi
# Modifications were made for specific use in this project.
# ------------------------------------------------------------------------------

def normalize_minmax(x):
    normalized = (x-x.min()) / (x.max() - x.min())
    return normalized

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class L_color_rate(nn.Module):

    def __init__(self):
        super(L_color_rate, self).__init__()

    def forward(self, pre, cur):

        mr_pre, mg_pre, mb_pre = torch.split(pre*255, 1, dim=1)
        mr_cur, mg_cur, mb_cur = torch.split(cur*255, 1, dim=1)

        Drg = torch.pow(mr_pre.int()//mg_pre.int() - mr_cur.int()//mg_cur.int(), 2).sum()/255.0**2
        Drb = torch.pow(mr_pre.int()//mb_pre.int() - mr_cur.int()//mb_cur.int(), 2).sum()/255.0**2
        Dgb = torch.pow(mg_pre.int()//mb_pre.int() - mg_cur.int()//mb_cur.int(), 2).sum()/255.0**2
        k = torch.pow(Drg + Drb + Dgb, 0.5)
        return k
			
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E

class L_exp_value(nn.Module):

    def __init__(self,patch_size):
        super(L_exp_value, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val
    
    def forward(self, x, mean_val):
        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).cuda(),2))
        return d

class L_exp(nn.Module):
    
    def __init__(self,patch_size):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(16)
        
    def forward(self, x, target):
        # b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        e_map = torch.mean(target,1,keepdim=True)
        e_map = self.pool(e_map)
        d = F.mse_loss(mean, e_map)
        return d

class GANLoss(nn.Module):
    def __init__(self, gan_type='gan', real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))
    
    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss
    
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
    
    
    def forward(self, x ):
        b,c,h,w = x.shape
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        return h_relu_4_3

def load_vgg19(index):
    vgg = vgg_19(index)
    return vgg

def histeq(im, nbr_bins = 256):    
    imhist, bins = histogram(im.flatten(), nbr_bins)
    cdf = imhist.cumsum()
    cdf = 1.0 * cdf / cdf[-1]
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape)

class vgg_19(nn.Module):
    def __init__(self, index):
        super(vgg_19, self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)
        self.feature_ext = nn.Sequential(*list(vgg_model.features.children())[:index])

    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat((x, x, x), 1)
        out = self.feature_ext(x)
        return out
    
class Perceptual_loss(nn.Module):
    def __init__(self, vgg):
        super(Perceptual_loss, self).__init__()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.vgg = vgg

    def forward(self, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = self.vgg(img_vgg)
        target_fea = self.vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

def vgg_preprocess(batch):
    tensor_type = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = batch * 255  # * 0.5  [-1, 1] -> [0, 255]
    mean = tensor_type(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch