import torch
from torch import nn
import torch.nn.functional as F

from .build import MODEL_REGISTRY
from timm import create_model

from copy import deepcopy

import math

import random

from torch.nn import init

import matplotlib.pyplot as plt

import numpy as np

from timm.layers import trunc_normal_

import logging

logger = logging.getLogger(__name__)


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
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
            
def bn_inits(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)
        m.bias.requires_grad_(False)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    

class OSBlock(nn.Module):
    def __init__(self, shape, num_scales=4):
        super().__init__()
        in_channels = shape[0]
        
        self.num_scales = num_scales
        mid_channels = in_channels // num_scales
        
        lightconv3x3 = lambda channels: nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.ModuleList([
            nn.Sequential(*[lightconv3x3(mid_channels) for _ in range(i+1)]) for i in range(num_scales)
        ])
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True)
        )
        init.zeros_(self.conv3[1].weight)
        init.zeros_(self.conv3[1].bias)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.cat([self.conv2[i](x1) for i in range(self.num_scales)], dim=1)
        x3 = self.conv3(x2)
        return x3

    
class LPF(nn.Module):
    def __init__(self, shape):
        super(LPF, self).__init__()
        
        inc, h, w = shape
        
        h_freqs = torch.fft.fftfreq(h)
        h_freqs = torch.fft.fftshift(h_freqs)
        w_freqs = torch.fft.rfftfreq(w)
        hw_freqs = torch.meshgrid(h_freqs, w_freqs)
        self.register_buffer('hw_freqs', torch.stack(hw_freqs, dim=0)) # [2, H, W]
        
        m0 = 0.2
        self.t = 0.607
        self.k = 1.0
        self.alpha = - math.log(self.t) / 0.25
        
        r0 = - (0.5 * m0) ** 2 * self.alpha / math.log(self.t)
        r0 = math.log(r0 / (1 - r0)) / self.k
        
        self.f0 = nn.Sequential(
            nn.AdaptiveAvgPool2d([1, 1]),
            nn.Conv2d(inc, inc//4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inc//4),
            nn.LeakyReLU(inplace=True),
        )
        self.cs = nn.Conv2d(inc//4, inc, 1, 1, 0, bias=True)
        init.zeros_(self.cs.weight)
        init.constant_(self.cs.bias, r0 / 2)
        
        self.ss = nn.Conv2d(inc//4, 2, 1, 1, 0, bias=True)
        init.zeros_(self.ss.weight)
        init.constant_(self.ss.bias, r0 / 2)
        
    def forward(self, x):
        if self.training:
            _f0 = self.f0(x)
            _cs = self.cs(_f0)
            _ss = self.ss(_f0)
        else:
            self._f0 = self.f0(x).detach() # for stability
            _cs = self.cs(self._f0)
            _ss = self.ss(self._f0)
        
        self.h_ratio_log = _cs + _ss[:, 0:1, :, :]
        self.w_ratio_log = _cs + _ss[:, 1:2, :, :]
        
        self.h_ratio = F.sigmoid(self.h_ratio_log * self.k)
        self.w_ratio = F.sigmoid(self.w_ratio_log * self.k)
        mask = torch.exp(-self.alpha * ((self.hw_freqs[0]) ** 2 / self.h_ratio + \
                                        (self.hw_freqs[1]) ** 2  / self.w_ratio))
        hard_mask = mask + ((mask >= self.t).float() - mask).detach()
        return mask, hard_mask
    

class LHM(nn.Module):
    def __init__(self, shape, osb_num=4):
        super(LHM, self).__init__()
        in_dims, h, w = shape
        
        self.lpf = LPF(shape)

        # ! 使用复数，需要主要初始化为1+j0!
        self.ws = nn.Parameter(torch.randn(in_dims, h, w//2+1, 2) * 0.02)
        trunc_normal_(self.ws.data, std=0.02)
        self.hig_os = OSBlock(shape, osb_num)

        self.fft = lambda x: torch.fft.fftshift(torch.fft.rfft2(x, norm='ortho'), dim=(-2))
        self.ifft = lambda x: torch.fft.irfft2(torch.fft.ifftshift(x, dim=(-2)), s=(h, w), norm='ortho')
        
        self.x_low0 = None
        self.x_low1 = None
        self.weighted_output = None
        
    def forward(self, x, cam_ids=None):
        x = x.to(torch.float32)
        x_fft = self.fft(x)
        
        _, fmask = self.lpf(x)

        x_low = x_fft * fmask * torch.view_as_complex(self.ws)
        x_hig = x_fft * (1.0 - fmask)
        
        if self.training:
            x_ = torch.cat([x_low, x_hig], dim=0)
            x_ = self.ifft(x_)
            x_low, x_hig = x_.chunk(2, dim=0)
        else:
            x_ = torch.cat([x_low, x_hig, (x_fft * fmask).detach()], dim=0)
            x_ = self.ifft(x_)
            x_low, x_hig, self.x_low0 = x_.chunk(3, dim=0)
            self.x_low0 = self.x_low0.detach()
            self.x_low1 = x_low.clone().detach()
        
        x_hig = self.hig_os(x_hig)

        self.weighted_output = x_low + x_hig

        return x + self.weighted_output
    

class LowPerturb(nn.Module):
    def __init__(self, shape, ratio=0.5, perturb_prob=0.5, uncertainty_factor=0.5, id_modality=False):
        super().__init__()
        self.id_modality = id_modality
        dim, h, w = shape
        self.fft = lambda x: torch.fft.fftshift(torch.fft.rfft2(x, norm='ortho'), dim=(-2))
        self.ifft = lambda x: torch.fft.irfft2(torch.fft.ifftshift(x, dim=(-2)), s=(h, w), norm='ortho')
        
        h_freqs = torch.fft.fftfreq(h)
        h_freqs = torch.fft.fftshift(h_freqs)
        w_freqs = torch.fft.rfftfreq(w)
        hw_freqs = torch.meshgrid(h_freqs, w_freqs, indexing='ij')
        
        v = ratio * 0.5
        mask = torch.zeros_like(hw_freqs[0])
        flag = (hw_freqs[0].abs() <= v) & (hw_freqs[1].abs() <= v)
        mask[flag] = 1.0
        
        self.h_start = int(torch.argmax(mask, dim=0)[0].item())
        self.h_crop = int(mask.sum(0)[0].item())
        self.w_crop = int(mask.sum(1).max().item())

        self.eps = 1e-6
        self.perturb_prob = perturb_prob
        self.uncertainty_factor = uncertainty_factor
        
    def _reparameterize(self, mu, std, epsilon_norm):
        epsilon = epsilon_norm * self.uncertainty_factor
        mu_t = mu + epsilon * std
        return mu_t
        
    def forward(self, x):
        if not self.training or self.uncertainty_factor == 0.0:
            return x
        
        b = x.size(0)
        flag = torch.rand(b, device=x.device) < self.perturb_prob
        if not torch.any(flag):
            return x

        x = x.to(torch.float32)
        img_fft = self.fft(x)
        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft + self.eps)
        
        img_abs_low = img_abs[..., self.h_start:self.h_start + self.h_crop, :self.w_crop].clone()
        
        miu = torch.mean(img_abs_low, dim=(2, 3), keepdim=True)
        sig = torch.std(img_abs_low, dim=(2, 3), keepdim=True) # BxCx1x1
        
        img_abs_low_norm = (img_abs_low - miu) / (sig + self.eps) # BxCxHxW
        
        if self.id_modality:
            sig_of_miu = torch.cat([
                torch.std(miu[:b//2], dim=0, keepdim=True).repeat(b//2, 1, 1, 1),
                torch.std(miu[b//2:], dim=0, keepdim=True).repeat(b//2, 1, 1, 1)
            ], dim=0) # BxCx1x1
            sig_of_sig = torch.cat([
                torch.std(sig[:b//2], dim=0, keepdim=True).repeat(b//2, 1, 1, 1),
                torch.std(sig[b//2:], dim=0, keepdim=True).repeat(b//2, 1, 1, 1)
            ], dim=0) # BxCx1x1
        else:
            sig_of_miu = torch.std(miu, dim=0, keepdim=True).repeat(b, 1, 1, 1)
            sig_of_sig = torch.std(sig, dim=0, keepdim=True).repeat(b, 1, 1, 1) # BxCx1x1
        
        epsilon_norm_miu = torch.randn_like(sig[flag]) # N(0,1)
        epsilon_norm_sig = torch.randn_like(sig[flag]) # B'xCx1x1
        
        beta = self._reparameterize(mu=miu[flag], std=sig_of_miu[flag], epsilon_norm=epsilon_norm_miu)
        gamma = self._reparameterize(mu=sig[flag], std=sig_of_sig[flag], epsilon_norm=epsilon_norm_sig)
        
        # adjust statistics for each sample
        img_abs[flag, :, self.h_start:self.h_start + self.h_crop, :self.w_crop] = gamma * img_abs_low_norm[flag] + beta
    
        x = self.ifft(torch.polar(img_abs, img_pha))
        return x


@MODEL_REGISTRY.register()
class BaseLine4(nn.Module):
    def __init__(self, 
                 num_classes, 
                 dataset, 
                 img_size,
                 
                 num_parts=4,
                 
                 freq_ratio=0.2, 
                 
                 id_low_weights=False, 
                 
                 perturb=False,
                 perturb_prob=0.5,
                 perturb_factor=1.0,
                 perturb_id_mod=False,
                 
                 base_id=False, 
                 
                 *args, **kwargs):
        super().__init__()
        bb = create_model('resnet50.tv_in1k', True, pretrained_cfg_overlay={'file': '/root/data/.cache/models/resnet50-0676ba61.pth'})
        bb.layer4[0].conv2.stride = (1, 1)
        bb.layer4[0].downsample[0].stride = (1, 1)
        
        self.id_low_weights = id_low_weights
        self.id_base = base_id
        self.num_parts = num_parts

        if base_id:
            self.base_v = nn.Sequential(
                bb.conv1,
                bb.bn1,
                bb.act1,
                bb.maxpool, # [64, 96, 48]
                LowPerturb([64, *(img_size[0]//4, img_size[1]//4)], freq_ratio, perturb_prob, perturb_factor, perturb_id_mod) \
                    if perturb else nn.Identity()
            )
            self.base_i = deepcopy(self.base_v)
        else:
            self.base = nn.Sequential(
                bb.conv1,
                bb.bn1,
                bb.act1,
                bb.maxpool, # [64, 96, 48]
                LowPerturb([64, *(img_size[0]//4, img_size[1]//4)], freq_ratio, perturb_prob, perturb_factor, perturb_id_mod) \
                    if perturb else nn.Identity()
            )
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        
        self.lhms = nn.ModuleList([
            LHM([512,  *(img_size[0]//8, img_size[1]//8)], 4),
            LHM([512,  *(img_size[0]//8, img_size[1]//8)], 4),
            LHM([1024, *(img_size[0]//16, img_size[1]//16)], 4),
            LHM([1024, *(img_size[0]//16, img_size[1]//16)], 4),
            LHM([1024, *(img_size[0]//16, img_size[1]//16)], 4),
        ])
        
        if dataset == 'sysu':
            self.pool = GeM(3.0)
            self.pool.p.requires_grad_(False)
        else:
            self.pool = GeM(10.0)
            self.pool.p.requires_grad_(False)
        
        self.bn_necks = nn.BatchNorm1d(2048)
        self.bn_necks.apply(bn_inits)
        
        self.clss = nn.Linear(2048, num_classes, bias=False)
        self.clss.apply(weights_init_classifier)

    def forward(self, x, cam_ids, *args, **kwargs):
        if self.id_base:
            infrared_flags = (cam_ids == 3) | (cam_ids == 6)
            if torch.all(infrared_flags):
                # only infrared images
                x= self.base_i(x)
            elif torch.all(~infrared_flags):
                # only visible images
                x = self.base_v(x)
            else:
                # both visible and infrared images
                # training mode
                # [V1, V2, V3, V4, I1, I2, I3, I4]
                bs = x.size(0)
                f0_v = self.base_v(x[:bs//2])
                f0_i = self.base_i(x[bs//2:])
                x = torch.cat([f0_v, f0_i])
        else:
            x = self.base(x)
        
        x = self.layer1(x)
        
        x = self.layer2[0](x)
        x = self.layer2[1](x)
        x = self.layer2[2](x)
        x = self.lhms[0](x, cam_ids)
        x = self.layer2[3](x)
        x = self.lhms[1](x, cam_ids)
        
        x = self.layer3[0](x)
        x = self.layer3[1](x)
        x = self.layer3[2](x)
        x = self.layer3[3](x)
        x = self.lhms[2](x, cam_ids)
        x = self.layer3[4](x)
        x = self.lhms[3](x, cam_ids)
        x = self.layer3[5](x)
        x = self.lhms[4](x, cam_ids)
        
        f0 = self.layer4(x)

        f1 = self.pool(f0).flatten(1)
        
        f2 = self.bn_necks(f1)
        
        if not self.training:
            f12s = [f1, f2]
            return (f2, torch.cat([F.normalize(f12) for f12 in f12s], dim=1))
        
        logits = self.clss(f2)
        return [logits], [f1]
    
    def get_params(self, *args, **kwargs): 
        params = []
        for k, v in self.named_parameters():
            if not v.requires_grad:
                continue
            
            if k.endswith(".ws") or k.endswith("cs.bias") or k.endswith("ss.bias"):# or v.ndim <= 1 or k.endswith(".bias"):     
                params += [{"params": [v], "weight_decay": 0.0}]
                
            else:
                params += [{"params": [v]}]
        
        return params
    
    
    def eval_hooks(self, x, tb_writer, global_step):
        def denorm(x):
            return x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        
        def norm(x):
            x_ = torch.stack(x, dim=-1)
            n = x_.size(-1)
            xn = (x_ - x_.amin(dim=(-3, -2, -1), keepdim=True)) / (x_.amax(dim=(-3, -2, -1), keepdim=True) - x_.amin(dim=(-3, -2, -1), keepdim=True))
            return [xn[..., i] for i in range(n)]
        
        hook_handles = []
        
        # ! register hooks
        # * LPF masks and hard masks
        lpf_forward_fs = []
        lpf_h_ratios = []
        lpf_w_ratios = []
        lpf_f0s = []
        lpf_cs_ws = []
        lpf_cs_bs = []
        lpf_ss_ws = []
        lpf_ss_bs = []
        def lpf_forward_hook(module, input, output):
            mask, hard_mask = output # BxCxHxW, BxCxHxW
            lpf_forward_fs.append([mask.detach().cpu(), hard_mask.detach().cpu()])
            lpf_h_ratios.append(module.h_ratio.detach().cpu())
            lpf_w_ratios.append(module.w_ratio.detach().cpu())
            lpf_f0s.append(module._f0.detach().cpu())
            lpf_cs_ws.append(module.cs.weight.detach().cpu())
            lpf_cs_bs.append(module.cs.bias.detach().cpu())
            lpf_ss_ws.append(module.ss.weight.detach().cpu())
            lpf_ss_bs.append(module.ss.bias.detach().cpu())
            
        for lhm in self.lhms:
            _hdl = lhm.lpf.register_forward_hook(lpf_forward_hook)
            hook_handles.append(_hdl)
            
        # * LHM inputs， outputs, old、new low frequency and ws
        lhm_inputs = []
        lhm_outputs = []
        lhm_weighted_outputs = []
        x_low_olds = []
        x_low_news = []
        lhm_ws = []
        def lhm_forward_hook(module, input, output):
            lhm_inputs.append(input[0].detach().cpu()) # BxCxHxW
            lhm_outputs.append(output.detach().cpu()) # BxCxHxW
            x_low_olds.append(module.x_low0.detach().cpu()) # BxCxHxW
            x_low_news.append(module.x_low1.detach().cpu()) # BxCxHxW
            lhm_weighted_outputs.append(module.weighted_output.detach().cpu()) # BxCxHxW
            lhm_ws.append(module.ws.detach().cpu()) # BxCxHxW//2+1x2
            
        for lhm in self.lhms:
            _hdl = lhm.register_forward_hook(lhm_forward_hook)
            hook_handles.append(_hdl)
            
        # * old high frequency and new high frequency
        x_hig_olds = []
        x_hig_news = []
        def hig_forward_hook(module, input, output):
            x_hig_olds.append(input[0].detach().cpu()) # BxCxHxW
            x_hig_news.append(output.detach().cpu()) # BxCxHxW
        
        for lhm in self.lhms:
            _hdl = lhm.hig_os.register_forward_hook(hig_forward_hook)
            hook_handles.append(_hdl)

        # ! forward
        training = self.training
        self.eval()
        with torch.no_grad():
            self.forward(x, None)
        self.train(training)
            
        # ! show figures
        # * LLPF masks
        for i, (mask, hard_mask) in enumerate(lpf_forward_fs):
            # normalize all the feature maps to [0, 1] for better visualization
            n_lhm_inputs, n_lhm_outputs, n_lhm_weighted_outputs, n_x_low_olds, n_x_low_news, n_x_hig_olds, n_x_hig_news = norm([
                    lhm_inputs[i],
                    lhm_outputs[i],
                    lhm_weighted_outputs[i],
                    x_low_olds[i],
                    x_low_news[i],
                    x_hig_olds[i],
                    x_hig_news[i]])
            vmin, vmax = 0.0, 1.0
            for j in range(8):
                fig, axes = plt.subplots(8, 16, figsize=(16, 16))
                # img
                axes[0, 0].imshow(denorm(x[j].permute(1, 2, 0).cpu().numpy()))
                for k in range(8):
                    axes[k, 0].axis('off')
                # feature maps
                for k in range(15):
                    axes[0, k+1].imshow(n_lhm_inputs[j, k*20, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[0, k+1].axis('off')
                    if k == 0:
                        axes[0, k+1].set_title(f'feature map {k*20}')
                # hard mask
                for k in range(15):
                    axes[1, k+1].imshow(hard_mask[j, k*20, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[1, k+1].axis('off')
                    if k == 0:
                        axes[1, k+1].set_title(f'hard mask {k*20}')
                # old low frequency
                for k in range(15):
                    axes[2, k+1].imshow(n_x_low_olds[j, k*20, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[2, k+1].axis('off')
                    if k == 0:
                        axes[2, k+1].set_title(f'old low frequency {k*20}')
                # old high frequency
                for k in range(15):
                    axes[3, k+1].imshow(n_x_hig_olds[j, k*20, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[3, k+1].axis('off')
                    if k == 0:
                        axes[3, k+1].set_title(f'old high frequency {k*20}')
                # new low frequency
                for k in range(15):
                    axes[4, k+1].imshow(n_x_low_news[j, k*20, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[4, k+1].axis('off')
                    if k == 0:
                        axes[4, k+1].set_title(f'new low frequency {k*20}')
                # new high frequency
                for k in range(15):
                    axes[5, k+1].imshow(n_x_hig_news[j, k*20, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[5, k+1].axis('off')
                    if k == 0:
                        axes[5, k+1].set_title(f'new high frequency {k*20}')
                # weighted output
                for k in range(15):
                    axes[6, k+1].imshow(n_lhm_weighted_outputs[j, k*20, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[6, k+1].axis('off')
                    if k == 0:
                        axes[6, k+1].set_title(f'weighted output {k*20}')
                # output
                for k in range(15):
                    axes[7, k+1].imshow(n_lhm_outputs[j, k*20, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[7, k+1].axis('off')
                    if k == 0:
                        axes[7, k+1].set_title(f'output {k*20}')
                # save figure
                tb_writer.add_figure(f'fixed_imgs/LHM-{i}/person-{j}', fig, global_step)
                plt.close(fig)
                
        for i in range(5):
            # low weights
            ws = lhm_ws[i].view(-1, 2)
            tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/ws-real', ws[:,0], global_step)
            tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/ws-imag', ws[:,1], global_step)
            
            # cs and ss weights and biases
            tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/cs-weight', lpf_cs_ws[i].view(-1,), global_step)
            tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/cs-bias', lpf_cs_bs[i].view(-1,), global_step)
            tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/ss-weight', lpf_ss_ws[i].view(-1,), global_step)
            tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/ss-bias', lpf_ss_bs[i].view(-1,), global_step)
            
            # gate values
            for j in range(4):
                hr = lpf_h_ratios[i][j].view(-1,)
                tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/person-{j}/h_ratio', hr, global_step)
                
                wr = lpf_w_ratios[i][j].view(-1,)
                tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/person-{j}/w_ratio', wr, global_step)
                
                f0 = lpf_f0s[i][j].view(-1,)
                tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/person-{j}/f0', f0, global_step)
                
        # ! remove hooks
        for hdl in hook_handles:
            hdl.remove()
    
    # def eval_hooks(self, x, tb_writer, global_step):
    #     def denorm(x):
    #         return x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        
    #     def norm(x):
    #         return (x - x.min()) / (x.max() - x.min())
        
    #     hook_handles = []
        
    #     # ! register hooks
    #     # * LPF masks and hard masks
    #     lpf_forward_fs = []
    #     def lpf_forward_hook(module, input, output):
    #         mask, hard_mask = output # BxCxHxW, BxCxHxW
    #         lpf_forward_fs.append([mask.detach().cpu(), hard_mask.detach().cpu()])
            
    #     for lhm in self.lhms:
    #         _hdl = lhm.lpf.register_forward_hook(lpf_forward_hook)
    #         hook_handles.append(_hdl)
            
    #     # * LPF: h_ratio and w_ratio
    #     lpf_h_ratios = []
    #     lpf_w_ratios = []
            
    #     # * LHM inputs， outputs, old、new low frequency and ws
    #     lhm_inputs = []
    #     lhm_outputs = []
    #     x_low_olds = []
    #     x_low_news = []
    #     lhm_ws = []
    #     def lhm_forward_hook(module, input, output):
    #         lhm_inputs.append(input[0].detach().cpu()) # BxCxHxW
    #         lhm_outputs.append(output.detach().cpu()) # BxCxHxW
    #         x_low_olds.append(module.x_low0.detach().cpu()) # BxCxHxW
    #         x_low_news.append(module.x_low1.detach().cpu()) # BxCxHxW
    #         lhm_ws.append(module.ws.detach().cpu()) # BxCxHxW//2+1x2
            
    #     for lhm in self.lhms:
    #         _hdl = lhm.register_forward_hook(lhm_forward_hook)
    #         hook_handles.append(_hdl)
            
    #     # * old high frequency and new high frequency
    #     x_hig_olds = []
    #     x_hig_news = []
    #     def hig_forward_hook(module, input, output):
    #         x_hig_olds.append(input[0].detach().cpu()) # BxCxHxW
    #         x_hig_news.append(output.detach().cpu()) # BxCxHxW
        
    #     for lhm in self.lhms:
    #         _hdl = lhm.hig_os.register_forward_hook(hig_forward_hook)
    #         hook_handles.append(_hdl)
            
    #     # * gate values
    #     # gate_values = []
    #     # def gate_forward_hook(module, input, output):
    #     #     gate_values.append(output.detach().cpu()) # -1
            
    #     # for lhm in self.lhms:
    #     #     _hdl = lhm.gate.register_forward_hook(gate_forward_hook)
    #     #     hook_handles.append(_hdl)

    #     # ! forward
    #     training = self.training
    #     self.eval()
    #     for m in self.modules():
    #         if isinstance(m, LHM):
    #             m.vis_mode = True
    #     with torch.no_grad():
    #         self.forward(x, None)
    #     self.train(training)
    #     for m in self.modules():
    #         if isinstance(m, LHM):
    #             m.vis_mode = False
            
    #     # ! show figures
    #     # * LLPF masks
    #     for i, (mask, hard_mask) in enumerate(lpf_forward_fs):
    #         # normalize all the feature maps to [0, 1] for better visualization
    #         n_lhm_inputs, n_lhm_outputs, n_x_low_olds, n_x_low_news, n_x_hig_olds, n_x_hig_news = norm(torch.cat([
    #                 lhm_inputs[i],
    #                 lhm_outputs[i],
    #                 x_low_olds[i],
    #                 x_low_news[i],
    #                 x_hig_olds[i],
    #                 x_hig_news[i],
    #             ], dim=0)).chunk(6)
    #         vmin, vmax = 0.0, 1.0
    #         for j in range(4):
    #             fig, axes = plt.subplots(8, 16, figsize=(16, 16))
    #             # img
    #             axes[0, 0].imshow(denorm(x[j].permute(1, 2, 0).cpu().numpy()))
    #             for k in range(8):
    #                 axes[k, 0].axis('off')
                
    #             # axes[1,0].hist(gate_values[i][j].view(-1,).numpy(), bins=50, color='blue', edgecolor='black')
    #             # axes[1,0].set_title(f'gate value {j}')
                
    #             # axes[2,0].hist(mask[j].view(-1,).numpy(), bins=50, color='blue', edgecolor='black')
    #             # axes[2,0].set_title(f'soft mask {j}')
    #             # feature maps
    #             for k in range(15):
    #                 axes[0, k+1].imshow(n_lhm_inputs[j, k, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
    #                 axes[0, k+1].axis('off')
    #                 if k == 0:
    #                     axes[0, k+1].set_title(f'feature map {k}')
    #             # soft masks
    #             for k in range(15):
    #                 axes[1, k+1].imshow(mask[j, k, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
    #                 axes[1, k+1].axis('off')
    #                 if k == 0:
    #                     axes[1, k+1].set_title(f'soft mask {k}')
    #             # hard masks
    #             for k in range(15):
    #                 axes[2, k+1].imshow(hard_mask[j, k, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
    #                 axes[2, k+1].axis('off')
    #                 if k == 0:
    #                     axes[2, k+1].set_title(f'hard mask {k}')
    #             # old low frequency
    #             for k in range(15):
    #                 axes[3, k+1].imshow(n_x_low_olds[j, k, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
    #                 axes[3, k+1].axis('off')
    #                 if k == 0:
    #                     axes[3, k+1].set_title(f'old low frequency {k}')
    #             # new low frequency
    #             for k in range(15):
    #                 axes[3, k+1].imshow(n_x_low_news[j, k, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
    #                 axes[3, k+1].axis('off')
    #                 if k == 0:
    #                     axes[3, k+1].set_title(f'new low frequency {k}')
    #             # old high frequency
    #             for k in range(15):
    #                 axes[4, k+1].imshow(n_x_hig_olds[j, k, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
    #                 axes[4, k+1].axis('off')
    #                 if k == 0:
    #                     axes[4, k+1].set_title(f'old high frequency {k}')
    #             # new high frequency
    #             for k in range(15):
    #                 axes[6, k+1].imshow(n_x_hig_news[j, k, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
    #                 axes[6, k+1].axis('off')
    #                 if k == 0:
    #                     axes[6, k+1].set_title(f'new high frequency {k}')
    #             # output
    #             for k in range(15):
    #                 axes[7, k+1].imshow(n_lhm_outputs[j, k, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
    #                 axes[7, k+1].axis('off')
    #                 if k == 0:
    #                     axes[7, k+1].set_title(f'output {k}')
    #             # save figure
    #             tb_writer.add_figure(f'fixed_imgs/LHM-{i}/person-{j}', fig, global_step)
    #             # plt.save(f'fixed_imgs/LHM-{i}.png')
    #             plt.close(fig)
                
    #     for i in range(5):
    #         ws = lhm_ws[i].view(-1, 2)
    #         tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/ws-real', ws[:,0], global_step)
    #         tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/ws-imag', ws[:,1], global_step)
            
    #         for j in range(4):
    #             # gv = gate_values[i][j].view(-1,)
    #             # tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/person-{j}/gate-value', gv, global_step)
                
    #             sm = lpf_forward_fs[i][0][j].view(-1,)
    #             tb_writer.add_histogram(f'fixed_imgs/LHM-{i}/person-{j}/soft-mask', sm, global_step)

    #     # ! remove hooks
    #     for hdl in hook_handles:
    #         hdl.remove()
            
            
    # def register_train_hooks(self, enabled=False):
    #     def denorm(x):
    #         return x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        
    #     def norm(x):
    #         return (x - x.min()) / (x.max() - x.min())
        
    #     # ! HookManger template!
    #     class HookManager:
    #         def __init__(self, enabled=False):
    #             if not enabled:
    #                 self.hook = []
    #                 return

    #             # ! register hooks
    #             # * low frequency weights values
    #             self.lhm_low_weights = []
    #             def lhm_low_weights_forward_hook(module, input, output):
    #                 self.lhm_low_weights.append(module.ws.detach().cpu().view(-1,)) # -1
                    
    #             for lhm in self.lhms:
    #                 _hdl = lhm.register_forward_hook(lhm_low_weights_forward_hook)
    #                 self.hook.append(_hdl)
                    
    #             # * gated attention weights values
    #             self.lhm_gate_weights = []
    #             def lhm_gate_weights_forward_hook(module, input, output):
    #                 self.lhm_gate_weights.append(output.detach().cpu().view(-1,))# -1
                    
    #             for lhm in self.lhms:
    #                 _hdl = lhm.gate.register_forward_hook(lhm_gate_weights_forward_hook)
    #                 self.hook.append(_hdl)

                    
    #         def __call__(self, x, tb_writer, global_step):
    #             if len(self.hook) == 0:
    #                 return
                
    #             fig, axes

    #         def __enter__(self):
    #             return self

    #         def __exit__(self, exc_type, exc_val, exc_tb):
    #             for h in self.hook:
    #                 h.remove()
                
            
                
                
        
        
        
        