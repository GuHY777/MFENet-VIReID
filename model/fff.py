import torch
from torch import nn
import torch.nn.functional as F

from .build import MODEL_REGISTRY
from timm import create_model

from copy import deepcopy

import math

import random

from torch.nn import init


class MAM(nn.Module):
    def __init__(self, dim, r=16):
        super(MAM, self).__init__()
        
        self.channel_attention = nn.Sequential(
                nn.Conv2d(dim, dim // r, kernel_size=1, bias=False), 
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)

    def forward(self, x):
        pooled = F.avg_pool2d(x, x.size()[2:])
        mask = self.channel_attention(pooled)
        x = x * mask + self.IN(x) * (1 - mask)

        return x        
    
    
class FMAM(nn.Module):
    def __init__(self, dim, r=16):
        super().__init__()
        
        self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(dim, dim // r, kernel_size=1, bias=False), 
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)
        
    def forward(self, x):
        _, _, H, W = x.shape
        
        x_norm = self.IN(x)
        
        x_norm_fft = torch.fft.rfft2(x_norm)
        
        x_norm_amp = torch.abs(x_norm_fft)
        
        
        x_fft = torch.fft.rfft2(x)
        
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft + 1e-7)
        
        mask = self.channel_attention(x_amp)
        x_amp_ = x_amp * mask + x_norm_amp * (1 - mask)
        
        real = x_amp_ * torch.cos(x_pha)
        imag = x_amp_ * torch.sin(x_pha)
        x_fft_ = torch.complex(real, imag)

        x_ = torch.fft.irfft2(x_fft_)
        
        return x_
        
        


class FreBranch(nn.Module):
    def __init__(self, shape):
        super().__init__()
        
        C, H, W = shape
        W_ = int(W/2) + 1
        
        self.phi = nn.Parameter(torch.rand(C, H, W_))
        
    def forward(self, x):
        x_fft = torch.fft.rfft2(x)
        
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft)
        
        x_amp_ = x_amp * self.phi
        
        x_fft_ = x_amp_ * torch.exp(1j * x_pha)
        
        x_ = torch.fft.irfft2(x_fft_)
        
        return x_
    
    
class GlobalFilter(nn.Module):
    def __init__(self, shape):
        super().__init__()
        dim, h, w = shape
        w = int(w/2) + 1
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        _, _, a, b = x.shape
        
        x = x.to(torch.float32)

        x1 = torch.fft.rfft2(x, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x1 = x1 * weight + x1
        x1 = torch.fft.irfft2(x1, s=(a, b), norm='ortho')
        return x1
    
class LowFilter(nn.Module):
    def __init__(self, shape, mask_ratio=0.5):
        super().__init__()
        
        dim, h, w = shape
        w = int(w/2) + 1
        
        self.h_crop = int(h * mask_ratio)
        self.w_crop = int(w * mask_ratio)
        self.h_start = h // 2 - self.h_crop // 2
        self.w_start = w // 2 - self.w_crop // 2
        
        self.weight = nn.Parameter(torch.randn(dim, self.h_crop, self.w_crop, dtype=torch.float32) * 0.02)
        
    def forward(self, x):
        _, _, a, b = x.shape
        
        x = x.to(torch.float32)
        
        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_amp, x_pha = torch.abs(x_fft), torch.angle(x_fft + 1e-7)
        
        x_amp = torch.fft.fftshift(x_amp, dim=(2))
        tmp = x_amp[..., self.h_start:self.h_start+self.h_crop, self.w_start:self.w_start+self.w_crop].clone()
        x_amp[..., self.h_start:self.h_start+self.h_crop, self.w_start:self.w_start+self.w_crop] = \
            tmp * self.weight + tmp

        x_amp = torch.fft.ifftshift(x_amp, dim=(2))
        
        x_fft = x_amp * torch.exp(1j * x_pha)
        
        x_ = torch.fft.irfft2(x_fft, s=(a, b), norm='ortho')
        return x_
    
    
class LowPerturb(nn.Module):
    def __init__(self, shape, mask_ratio=0.5, perturb_prob=0.5, uncertainty_factor=1.0):
        super().__init__()
        
        dim, h, w = shape
        w = w//2 + 1
        
        self.h_crop = int(h * mask_ratio)
        self.w_crop = int(w * mask_ratio)
        self.h_start = h // 2 - self.h_crop // 2
        self.w_start = 0
        
        self.complex_weight = nn.Parameter(torch.ones(dim, h, w, 2, dtype=torch.float32))
        self.gate = nn.Parameter(torch.zeros(1))
        # self.gate2 = nn.Parameter(torch.ones(1))
        self.eps = 1e-6
        self.perturb_prob = perturb_prob
        self.uncertainty_factor = uncertainty_factor
        
    def _reparameterize(self, mu, std, epsilon_norm):
        # epsilon = torch.randn_like(std) * self.factor
        epsilon = epsilon_norm * self.uncertainty_factor
        mu_t = mu + epsilon * std
        return mu_t
        
    def forward(self, x):
        _, _, h, w = x.shape
        
        x = x.to(torch.float32)
        img_fft = torch.fft.rfft2(x, norm='ortho')
        
        if self.training and random.random() <= self.perturb_prob:
            img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)
            
            img_abs = torch.fft.fftshift(img_abs, dim=(2))
            img_abs_low = img_abs[..., self.h_start:self.h_start + self.h_crop, self.w_start:self.w_start + self.w_crop].clone()
            
            miu = torch.mean(img_abs_low, dim=(2, 3),
                                     keepdim=True)
            var = torch.var(img_abs_low, dim=(2, 3),
                            keepdim=True)
            sig = (var + self.eps).sqrt()  # BxCx1x1

            var_of_miu = torch.var(miu, dim=0, keepdim=True) 
            var_of_sig = torch.var(sig, dim=0, keepdim=True)
            sig_of_miu = (var_of_miu + self.eps).sqrt()
            sig_of_sig = (var_of_sig + self.eps).sqrt() # 1xCx1x1
        
            epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
            epsilon_norm_sig = torch.randn_like(sig_of_sig)

            miu_mean = miu
            sig_mean = sig

            beta = self._reparameterize(mu=miu_mean, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
            gamma = self._reparameterize(mu=sig_mean, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
            
            # adjust statistics for each sample
            img_abs[..., self.h_start:self.h_start + self.h_crop, self.w_start:self.w_start + self.w_crop] = gamma * (
                    img_abs[..., self.h_start:self.h_start + self.h_crop, self.w_start:self.w_start + self.w_crop] - miu) / sig + beta
        
            img_abs = torch.fft.ifftshift(img_abs, dim=(2))  # recover
            
            img_fft = img_abs * torch.exp(1j * img_pha)
            
        img_mix = img_fft * torch.view_as_complex(self.complex_weight)
        img_mix = torch.fft.irfft2(img_mix, s=(h, w), norm='ortho')
        
        return img_mix * self.gate + x# * self.gate2
    
    
class LowPerturb2(nn.Module):
    def __init__(self, shape, mask_ratio=0.5, perturb_prob=0.5, uncertainty_factor=1.0):
        super().__init__()
        
        dim, h, w = shape
        w = w//2 + 1
        
        self.h_crop = int(h * math.sqrt(mask_ratio))
        self.w_crop = int(w * math.sqrt(mask_ratio))
        self.h_start = h // 2 - self.h_crop // 2
        self.w_start = 0
        
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.bn = nn.BatchNorm2d(dim)
        nn.init.constant_(self.bn.weight, 0) 
        nn.init.constant_(self.bn.bias, 0) 
        # self.gate = nn.Parameter(torch.zeros(1))
        # self.gate2 = nn.Parameter(torch.ones(1))
        self.eps = 1e-6
        self.perturb_prob = perturb_prob
        self.uncertainty_factor = uncertainty_factor
        
        self.var_of_miu = None
        self.var_of_sig = None
        
    def _reparameterize(self, mu, std, epsilon_norm):
        # epsilon = torch.randn_like(std) * self.factor
        epsilon = epsilon_norm * self.uncertainty_factor
        mu_t = mu + epsilon * std
        return mu_t
        
    def forward(self, x):
        bs, _, h, w = x.shape
        
        x = x.to(torch.float32)
        img_fft = torch.fft.rfft2(x, norm='ortho')
        
        if self.training and random.random() <= self.perturb_prob:
            img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft + 1e-7)
            
            img_abs = torch.fft.fftshift(img_abs, dim=(2))
            img_abs_low = img_abs[..., self.h_start:self.h_start + self.h_crop, self.w_start:self.w_start + self.w_crop].clone()
            
            idxs = torch.randperm(bs)
            img_abs[..., self.h_start:self.h_start + self.h_crop, self.w_start:self.w_start + self.w_crop] = \
                img_abs_low[idxs, :, :, :]
            
            # miu = torch.mean(img_abs_low, dim=0,
            #                          keepdim=True)
            # var = torch.var(img_abs_low, dim=0,
            #                 keepdim=True)
            # sig = (var + self.eps).sqrt()  # 1xCxHxW
        
            # epsilon_norm_sig = torch.randn_like(img_abs_low)
            # gamma = epsilon_norm_sig * sig * self.uncertainty_factor

            # # adjust statistics for each sample
            # img_abs[..., self.h_start:self.h_start + self.h_crop, self.w_start:self.w_start + self.w_crop] = \
            #     img_abs[..., self.h_start:self.h_start + self.h_crop, self.w_start:self.w_start + self.w_crop] \
            #         + gamma
                    
            # img_abs = F.relu(img_abs)
        
            img_abs = torch.fft.ifftshift(img_abs, dim=(2))  # recover
            
            img_fft = img_abs * torch.exp(1j * img_pha)
            
        img_mix = img_fft * torch.view_as_complex(self.complex_weight)
        img_mix = torch.fft.irfft2(img_mix, s=(h, w), norm='ortho')
        
        return self.bn(img_mix) + x
    
    
class PartFre(nn.Module):
    def __init__(self, shape, n=2):
        super().__init__()
        
        dim, h, w = shape
        w = w//2 + 1
        self.n = n

        # self.bns = nn.ModuleList([nn.BatchNorm2d(dim) for _ in range(3)])
        # self.attention = nn.Sequential(
        #         nn.Conv2d(2*dim, 2*dim // 16, kernel_size=1, bias=False), 
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(2*dim // 16, n, kernel_size=1, bias=False),
        #         nn.Softmax(dim=1)
        #     )
        # torch.nn.init.constant_(self.attention.bias, 0.0)
        # self.act = nn.Sigmoid()
        self.register_buffer('masks', self._masks(h, w, n))
        self.complex_channel = nn.Parameter(torch.randn(dim, 1, 1, 2, dtype=torch.float32) * 0.02)
        self.complex_spatial = nn.Parameter(torch.randn(1, h, w, 2, dtype=torch.float32) * 0.02)
        
        bn = nn.BatchNorm2d(dim)
        nn.init.constant_(bn.weight, 0.0) 
        nn.init.constant_(bn.bias, 0)         
        self.bns = nn.ModuleList([deepcopy(bn) for _ in range(n)])
        
    def forward(self, x):
        _, _, h, w = x.shape
        
        x = x.to(torch.float32)
        img_fft = torch.fft.rfft2(x, norm='ortho')
        img_fft = torch.fft.fftshift(img_fft, dim=(2))
        img_fft = img_fft * torch.view_as_complex(self.complex_channel * self.complex_spatial)
        # img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft + 1e-7)
        
        # masks = self.attention(torch.cat([img_abs, img_pha], dim=1))
        # masks = self.act(masks)
        
        # masks = self.attention(torch.cat([img_fft.real, img_fft.imag], dim=1))
        
        feats = [x]
        for i in range(self.n):
            # img_abs_ = img_abs * self.masks[i,:,:]
            # img_fft_ = img_abs_ * torch.exp(1j * img_pha)
            img_fft_ = img_fft * self.masks[i,:,:]
            img_fft_ = torch.fft.ifftshift(img_fft_, dim=(2))
            feat = torch.fft.irfft2(img_fft_, s=(h, w), norm='ortho')
            feats.append(self.bns[i](feat) + x)
        
        return torch.cat(feats, dim=0)
    
    def _masks(self, h, w, n=3):
        masks = []
        
        h_s, h_e = h//2 - h//2//n, h//2 - h//2//n + h//n
        w_e = w//n
        for i in range(n):
            mask = torch.zeros((h, w))
            
            if i == 0:
                mask[h_s:h_e, :w_e] = 1.0
            elif i < n-1:
                mask[h_s-h//n//2:h_s, :w_e+w//n] = 1.0
                mask[h_e:h_e+h//n//2, :w_e+w//n] = 1.0
                mask[h_s:h_e, w_e:w_e+w//n] = 1.0
                
                h_s -= h//n//2
                h_e += h//n//2
                w_e += w//n
            else:
                mask[:h_s, :] = 1.0
                mask[h_e:, :] = 1.0
                mask[h_s:h_e, w_e:] = 1.0
            
            masks.append(mask)
            
        return torch.stack(masks, dim=0)  

    
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.ones_(m.weight.data)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
            
            
class high_freq_guided(nn.Module):
    def __init__(self, shape, mask_ratio=0.5, reduc_ratio=2):
        super().__init__()
        
        in_channels, h, w = shape
        inter_channels = in_channels // reduc_ratio
        w = w//2 + 1
        self.inter_channels = inter_channels
        
        self.h_crop = int(h * math.sqrt(mask_ratio))
        self.w_crop = int(w * math.sqrt(mask_ratio))
        self.h_start = h // 2 - self.h_crop // 2
        self.w_start = 0
        
        high_mask = torch.ones((h, w), dtype=torch.bool)
        high_mask[self.h_start:self.h_start + self.h_crop, self.w_start:self.w_start + self.w_crop] = False
        self.register_buffer('high_mask', high_mask.view(-1,))
        
        self.h_pos_embed = nn.Parameter(torch.randn([in_channels // 2, h//2+1]) * 0.02)
        self.w_pos_embed = nn.Parameter(torch.randn([in_channels // 2, w]) * 0.02)
        
        self.bn = nn.BatchNorm2d(in_channels)
        
        self.g_phi = nn.Conv2d(in_channels, inter_channels * 2, 1, 1, 0)
        
        self.theta = nn.Conv2d(in_channels, inter_channels, 1, 1, 0)

        self.W = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 1, 1, 0),
            nn.BatchNorm2d(in_channels),
        )
        init.constant_(self.W[1].weight, 0.0)
        init.constant_(self.W[1].bias, 0.0)
        
        self.processmag = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.LeakyReLU(0.1),
            SELayer(channel=in_channels),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0))
        init.constant_(self.processmag[-1].weight, 0.0)
        init.constant_(self.processmag[-1].bias, 0.0)
        
        self.processpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.LeakyReLU(0.1),
            SELayer(channel=in_channels),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0))
        init.constant_(self.processpha[-1].weight, 0.0)
        init.constant_(self.processpha[-1].bias, 0.0)
        
    def forward(self, x):
        bs, _, h, w = x.shape
        
        x = x.to(torch.float32)
        img_fft = torch.fft.rfft2(x, norm='ortho')
        
        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft + 1e-7)
        img_abs = torch.fft.fftshift(img_abs, dim=(2))
        
        img_abs_bn = self.bn(img_abs.clone()) + torch.cat([torch.cat([self.h_pos_embed, -torch.flip(self.h_pos_embed,(1,))[:, 1:-1]], dim=1).unsqueeze(2).repeat(1, 1, img_abs.size(3)), 
                                        self.w_pos_embed.unsqueeze(1).repeat(1, img_abs.size(2), 1)], dim=0)
        
        img_abs_low = img_abs_bn[..., self.h_start:self.h_start + self.h_crop, self.w_start:self.w_start + self.w_crop].clone()
        
        x_high_g_phi = self.g_phi(img_abs_bn).view(bs, self.inter_channels * 2, -1)[..., self.high_mask]
        x_high_g, x_high_phi = x_high_g_phi.chunk(2, dim=1) # [B, C, nh]
        x_high_g = x_high_g.permute(0, 2, 1) # [B, nh, C]        
        
        x_low_theta = self.theta(img_abs_low).view(bs, self.inter_channels, -1)
        x_low_theta = x_low_theta.permute(0, 2, 1) # [B, nl, C]
        
        f = torch.matmul(x_low_theta, x_high_phi) / (self.inter_channels ** 0.5) # [B, nl, nh]
        s = F.softmax(f, dim=-1) # [B, nl, nh]
        
        y_low = torch.matmul(s, x_high_g).permute(0, 2, 1).contiguous() # [B, C, nl]
        y_low = y_low.view(bs, self.inter_channels, self.h_crop, self.w_crop) # [B, C, h, w]
        W_y_low = self.W(y_low) # [B, C, h, w]
        
        img_abs[..., self.h_start:self.h_start + self.h_crop, self.w_start:self.w_start + self.w_crop] = \
            W_y_low + img_abs[..., self.h_start:self.h_start + self.h_crop, self.w_start:self.w_start + self.w_crop]
        
        img_abs = torch.fft.ifftshift(img_abs, dim=(2))  # recover
        
        img_abs = img_abs + self.processmag(img_pha)
        img_pha = img_pha + self.processpha(img_abs)
        
        img_fft = img_abs * torch.exp(1j * img_pha)
        
        return torch.fft.irfft2(img_fft, s=(h, w), norm='ortho')
        
        

@MODEL_REGISTRY.register()
class FFF(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super(FFF, self).__init__()
        
        bb = create_model('resnet50.tv_in1k', True, pretrained_cfg_overlay={'file': '/root/data/.cache/models/resnet50-0676ba61.pth'})
        bb.layer4[0].conv2.stride = (1, 1)
        bb.layer4[0].downsample[0].stride = (1, 1)
        
        self.backbone = nn.Sequential(
            bb.conv1,
            bb.bn1,
            bb.act1,
            bb.maxpool,
            high_freq_guided([64, 96, 48]),
            bb.layer1, # [B, 256, 96, 32]
            high_freq_guided([256, 96, 48]),
            bb.layer2, # [B, 512, 48, 16]
            # LowPerturb([512, 48, 24]),
            bb.layer3,  # [B, 1024, 24, 12]
            # GlobalFilter([1024, 24, 12]),
            bb.layer4,  # [B, 2048, 24, 12]
            # GlobalFilter([2048, 24, 8])
        )
        
        self.bn_neck = nn.BatchNorm1d(2048)
        self.bn_neck.bias.requires_grad_(False)
        self.bn_neck.apply(weights_init_kaiming)
        
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x, cam_ids, *args, **kwargs):
        f0 = self.backbone(x)
        f1 = F.adaptive_avg_pool2d(f0, (1, 1)).flatten(1)
        f2 = self.bn_neck(f1)
        if not self.training:
            return f2
        
        logits = self.classifier(f2)
        return [logits], [f1]
    
    
    def get_params(self, *args, **kwargs): 
        return self.parameters()
        
        
        
        
        