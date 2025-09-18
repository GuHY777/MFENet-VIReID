import torch
from torch import nn
import torch.nn.functional as F

from .build import MODEL_REGISTRY
from timm import create_model

from copy import deepcopy

import math

import random

from torch.nn import init


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)
            
            
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
class CNL(nn.Module):
    def __init__(self, dim):
        super(CNL, self).__init__()
        self.dim = dim

        self.g = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(dim))
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x, x_h):
        B = x.size(0)
        g_x = self.g(x_h).view(B, self.dim, -1)

        theta_x = self.theta(x).view(B, self.dim, -1)
        phi_x = self.phi(x_h).view(B, self.dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)
        
        y = torch.matmul(attention, g_x)
        y = y.view(B, self.dim, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        
        return z

class PNL(nn.Module):
    def __init__(self, dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.dim = dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.dim, self.dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.dim, self.dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.dim, self.dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(nn.Conv2d(self.dim//self.reduc_ratio, self.dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(dim))
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x, x_h):
        B = x.size(0)
        g_x = self.g(x_h).view(B, self.dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(B, self.dim, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x_h).view(B, self.dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.dim//self.reduc_ratio, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z
            
            
class high_freq_enhanced(nn.Module):
    def __init__(self, shape, alpha=-math.log(0.5)*16):
        super().__init__()
        
        self.alpha = alpha
        in_channels, h, w = shape
        h_freqs = torch.fft.fftfreq(h)
        h_freqs = torch.fft.fftshift(h_freqs)
        w_freqs = torch.fft.rfftfreq(w)
        hw_freqs = torch.meshgrid(h_freqs, w_freqs)
        self.register_buffer('hw_freqs', torch.stack(hw_freqs, dim=0)) # [2, H, W]
        
        self.high_weight_log = nn.Parameter(torch.randn(1, 1, h, w//2+1) * 0.02)
        
        self.h_ratio = nn.Sequential(
            nn.AdaptiveMaxPool2d((h, 1)),
            nn.Flatten(2),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )
        self.h_ratio[-1].weight.data.normal_(0, 0.02)
        self.h_ratio[-1].bias.data.fill_(0.0)
        
        self.w_ratio = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, w)),
            nn.Flatten(2),
            nn.Linear(w, w),
            nn.ReLU(),
            nn.Linear(w, 1),
        )
        self.w_ratio[-1].weight.data.normal_(0, 0.02)
        self.w_ratio[-1].bias.data.fill_(0.0)
        
    def forward(self, x):
        _, _, h, w = x.shape
        
        x = x.to(torch.float32)
        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_fft = torch.fft.fftshift(x_fft, dim=(2))
        x_amp, x_pha = torch.abs(x_fft), torch.angle(x_fft)
        
        h_ratio = self.h_ratio(x_amp).exp().unsqueeze(3) # [B, C, 1, 1]
        w_ratio = self.w_ratio(x_amp).exp().unsqueeze(2) # [B, C, 1, 1]
        
        mask = torch.exp(-self.alpha * ((self.hw_freqs[0]) ** 2 / h_ratio + (self.hw_freqs[1]) ** 2  / w_ratio))
        hard_mask = (mask >= 0.5).float()
        dif_mask = mask + (hard_mask - mask).detach()
        
        x_amp_low = x_amp * dif_mask
        x_amp_high = x_amp * (1.0 - dif_mask) * self.high_weight_log.exp()
        
        x_fft = (x_amp_low + x_amp_high) * torch.exp(1j * x_pha)
        x_fft = torch.fft.ifftshift(x_fft, dim=(2))
        x = torch.fft.irfft2(x_fft, s=(h, w), norm='ortho')

        return x
 

@MODEL_REGISTRY.register()
class CAJ(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super(CAJ, self).__init__()
        bb = create_model('resnet50.tv_in1k', True, pretrained_cfg_overlay={'file': '/root/data/.cache/models/resnet50-0676ba61.pth'})
        bb.layer4[0].conv2.stride = (1, 1)
        bb.layer4[0].downsample[0].stride = (1, 1)
        
        self.visible_module = nn.Sequential(
            bb.conv1,
            bb.bn1,
            bb.act1,
            bb.maxpool
        )
        self.infrared_module = deepcopy(self.visible_module)
        
        self.backbone = nn.Sequential(
            # high_freq_guided([64, 96, 48]),
            bb.layer1, # [B, 256, 96, 32]
            # high_freq_guided([256, 96, 48]),
            bb.layer2, # [B, 512, 48, 16]
            # high_freq_guided([512, 48, 24]),
            Non_local(512),
            Non_local(512),
            bb.layer3,  # [B, 1024, 24, 12]
            Non_local(1024),
            Non_local(1024),
            Non_local(1024),
            bb.layer4,  # [B, 2048, 24, 12]
        )
        
        self.pool = GeM()
        
        self.bn_neck = nn.BatchNorm1d(2048)
        self.bn_neck.bias.requires_grad_(False)
        self.bn_neck.apply(weights_init_kaiming)
        
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x, cam_ids, *args, **kwargs):
        infrared_flags = (cam_ids == 3) | (cam_ids == 6)
        if torch.all(infrared_flags):
            # only infrared images
            x= self.infrared_module(x)
        elif torch.all(~infrared_flags):
            # only visible images
            x = self.visible_module(x)
        else:
            # both visible and infrared images
            # training mode
            # [V1, V2, V3, V4, I1, I2, I3, I4]
            bs = x.size(0)
            f0_v = self.visible_module(x[:bs//2])
            f0_i = self.infrared_module(x[bs//2:])
            x = torch.cat([f0_v, f0_i], dim=0)
        
        f0 = self.backbone(x)
        f1 = self.pool(f0).flatten(1)
        f2 = self.bn_neck(f1)
        if not self.training:
            return (f2, f1, torch.cat([F.normalize(f1), F.normalize(f2)], dim=1))
        
        logits = self.classifier(f2)
        return [logits], [f1]
    
    def get_params(self, *args, **kwargs): 
        return self.parameters()
        # ignored_params_id = list(map(id, self.classifier.parameters())) \
        #                 + list(map(id, self.bn_neck.parameters())) \
        #                 + list(map(id, self.pool.parameters()))

        # base_params = filter(lambda p: id(p) not in ignored_params_id, self.parameters())
        # ignored_params = filter(lambda p: id(p) in ignored_params_id, self.parameters())
        
        # params = [
        #     {"params": ignored_params, "lr": 0.1},
        #     {"params": base_params, "lr":0.01}
        # ]
        
        # return params
        
    def freeze_backbone(self):
        self.backbone[1].requires_grad_(False)
        self.backbone[3].requires_grad_(False)
        self.backbone[4].requires_grad_(False)
        self.backbone[5].requires_grad_(False)
    
    def unfreeze_backbone(self):
        self.backbone[1].requires_grad_(True)
        self.backbone[3].requires_grad_(True)
        self.backbone[4].requires_grad_(True)
        self.backbone[5].requires_grad_(True)
        
        