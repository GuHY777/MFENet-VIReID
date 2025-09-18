import torch
import random
import math
from torch import nn
import torchvision.transforms as T
import torch.nn.functional as F

def _get_masks(h, w, N=None, linspaces=None, mode='square'):
    assert mode in ['square', 'rhombus', 'circle'], f"mode should be'square', 'rhombus', or 'circle', but got {mode}"
    assert not (N is None and linspaces is None), "either N or linspaces should be provided"
    
    h_freqs = torch.fft.fftfreq(h)
    h_freqs = torch.fft.fftshift(h_freqs)
    w_freqs = torch.fft.rfftfreq(w)
    hw_freqs = torch.meshgrid(h_freqs, w_freqs, indexing='ij')
    
    if linspaces is None:
        rs = torch.linspace(0, 1, N+1)[1:-1].tolist()
    else:
        if isinstance(linspaces, (int, float)):
            rs = [linspaces]
        else:
            rs = linspaces

    if mode == 'square':
        masks = []
        for i in range(len(rs)):
            vi = 0.5 * rs[i]
            
            mask = torch.zeros_like(hw_freqs[0])
            flag = (hw_freqs[0].abs() <= vi) & (hw_freqs[1].abs() <= vi)

            mask[flag] = 1.0
            if i:
                for j in range(i):
                    mask = mask - masks[j]
            masks.append(mask)
        
        mask = torch.ones_like(hw_freqs[0]) - torch.stack(masks, dim=0).sum(0)
        masks.append(mask)
    elif mode == 'rhombus':
        masks = []
        for i in range(len(rs)):
            vi = 0.5 * rs[i]
            
            mask = torch.zeros_like(hw_freqs[0])
            flag = ((hw_freqs[0].abs() + hw_freqs[1].abs()) <= vi) &\
                    ((hw_freqs[0].abs() + hw_freqs[1].abs()) > vi_1) if i else\
                    ((hw_freqs[0].abs() + hw_freqs[1].abs()) <= vi)
            mask[flag] = 1.0
            
            masks.append(mask)
            
            vi_1 = vi
        
        mask = torch.ones_like(hw_freqs[0]) - torch.stack(masks, dim=0).sum(0)
        masks.append(mask)
    
    else:
        masks = []
        for i in range(len(rs)):
            vi = 0.5 * rs[i]
            
            mask = torch.zeros_like(hw_freqs[0])
            flag = ((hw_freqs[0]**2 + hw_freqs[1]**2) <= vi**2) &\
                    ((hw_freqs[0]**2 + hw_freqs[1]**2) > vi_1**2) if i else\
                    ((hw_freqs[0]**2 + hw_freqs[1]**2) <= vi**2)
            mask[flag] = 1.0
            
            masks.append(mask)
            
            vi_1 = vi
        
        mask = torch.ones_like(hw_freqs[0]) - torch.stack(masks, dim=0).sum(0)
        masks.append(mask)
        
    return torch.stack(masks, dim=0) # [N, H, W]

class MFBRandomWeightedAverage(torch.nn.Module):
    def __init__(self, img_size, N=5, high_exchange=False, prob=0.5):
        super().__init__()
        self.N = N
        self.high_exchange = high_exchange
        self.prob = prob
        self.dirichlet_dist = torch.distributions.Dirichlet(torch.tensor([1.0, 1.0, 1.0]))
        self.register_buffer('masks', _get_masks(*img_size, N, mode='square'))
        
        self.fft = lambda x: torch.fft.fftshift(torch.fft.rfft2(x, norm='ortho'), dim=(-2))
        self.ifft = lambda x: torch.fft.irfft2(torch.fft.ifftshift(x, dim=(-2)), s=img_size, norm='ortho')
        
    def forward(self, img):
        if torch.rand(1) > self.prob:
            return img 
        
        x_fft = self.fft(img.unsqueeze(0)).squeeze(0)
        
        x_amp, x_pha = x_fft.abs(), torch.angle(x_fft + 1e-6)
        
        x_amps = []
        for i in range(self.N):
            x_amps.append(x_amp * self.masks[i])
            
        x_amps = torch.stack(x_amps, dim=0) # N, 3, H, W
        if self.high_exchange:
            ws = self.dirichlet_dist.sample([1,]) # 1, 3
            
            idxs = torch.randint(low=0, high=3, size=(self.N-1,))
            one_hot_vectors = F.one_hot(idxs, num_classes=3)
            ws_o = one_hot_vectors.float()
            
            ws = torch.cat([ws, ws_o], dim=0) # N, 3
        else:
            ws = self.dirichlet_dist.sample([self.N,]) # N, 3
        
        x_amp = (x_amps * ws.unsqueeze(-1).unsqueeze(-1)).sum(0) # 3, H, W
        x_fft_rec = (x_amp * torch.exp(1j * x_pha)).sum(0, keepdim=True).unsqueeze(0) # 1, 1, H, W
        x_rec = self.ifft(x_fft_rec).squeeze(0).repeat(3, 1, 1)
        
        return x_rec

def batch_randint_torch(high):
    random_floats = torch.rand_like(high, dtype=torch.float32)
    # 将随机浮点数缩放到 [0, high) 范围
    random_integers = high * random_floats
    # 转换为整数（向下取整）
    return random_integers.floor().long()

class RandomWeightedAverage(torch.nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        
        self.prob = prob
        self.dirichlet_dist = torch.distributions.Dirichlet(torch.tensor([1.0, 1.0, 1.0]))
        
    def forward(self, img):
        if torch.rand(1) > self.prob:
            return img 
        
        weights = self.dirichlet_dist.sample((1,1)).transpose(0,2)
        return torch.sum(img * weights, dim=0, keepdim=True).repeat(3,1,1)
    
class BatchRandomErasing(torch.nn.Module):
    def __init__(self, p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.33)):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_masks(hw, scale, ratio, N, try_nums=10):
        img_h, img_w = hw
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        
        masks = torch.ones(N, 1, img_h, img_w, dtype=torch.float32)
        
        erase_area = area * torch.empty(N, try_nums).uniform_(scale[0], scale[1])
        aspect_ratio = torch.exp(torch.empty(N, try_nums).uniform_(log_ratio[0], log_ratio[1]))
        hs = (erase_area * aspect_ratio).sqrt().round().long() # [N, try_nums]
        ws = (erase_area / aspect_ratio).sqrt().round().long()
        
        flags = ((hs < img_h) & (ws < img_w)).long()
        idxs = torch.argmax(flags, dim=1)
        has_true = torch.where(flags.any(dim=1))[0]
        
        hs = hs[torch.arange(N)[has_true], idxs[has_true]]
        ws = ws[torch.arange(N)[has_true], idxs[has_true]]
        _is = batch_randint_torch(img_h - hs + 1)
        _js = batch_randint_torch(img_w - ws + 1)
        
        for idx,h,w,i,j in zip(has_true,hs,ws,_is,_js):
            masks[idx, 0, i:i+h, j:j+w] = 0.0
            
        return masks

    def forward(self, imgs):
        flags = torch.rand(imgs.size(0), device=imgs.device) < self.p
        if flags.sum() == 0:
            return imgs
        
        masks = self.get_masks(imgs.shape[-2:], self.scale, self.ratio, flags.sum().item()).to(imgs.device)
        
        auged_imgs = imgs.clone()
        auged_imgs[flags] = imgs[flags] * masks
        
        return auged_imgs


class BatchRandomWeightedAverage(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dirichlet_dist = torch.distributions.Dirichlet(torch.tensor([1.0, 1.0, 1.0]))
        self.p = p
        
    def forward(self, imgs):
        bs = imgs.size(0)
        
        flags = torch.rand(bs, device=imgs.device) < self.p
        if not flags.any():
            return imgs
        
        auged_imgs = imgs.clone()
        weights = self.dirichlet_dist.sample((flags.sum(),1,1)).transpose(1,3).to(imgs.device) # bs x 3 x 1 x 1
        auged_imgs[flags] = torch.sum(imgs[flags] * weights, dim=1, keepdim=True).repeat(1,3,1,1)
        return auged_imgs


class FreqAug(torch.nn.Module):
    def __init__(self, prob=0.5, ratio=1.0, alpha=1.0, rwa_prob=0.5, img_size=(288, 144), only_rgb_aug=True):
        super().__init__()
        
        h, w = img_size
        self.prob = prob
        self.alpha = alpha
        self.only_rgb = only_rgb_aug
        
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
        
        self.brwa = BatchRandomWeightedAverage(p=rwa_prob)
        self.norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.rea = BatchRandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.33))
    
    @torch.no_grad()
    def forward(self, imgs):
        b = imgs.size(0)
        
        aug_imgs = imgs.clone()
        
        x_fft = self.fft(imgs)
        x_amp = x_fft.abs()
        x_pha = x_fft.angle()
        
        x_amp_low = x_amp[..., self.h_start:self.h_start+self.h_crop, :self.w_crop].clone()
        
        x_amp_low_mu = x_amp_low.mean(dim=(-2, -1), keepdim=True) # (b, 3, 1, 1)
        x_amp_low_std = x_amp_low.std(dim=(-2, -1), keepdim=True) # (b, 3, 1, 1)
        x_amp_low_norm = (x_amp_low - x_amp_low_mu) / (x_amp_low_std + 1e-6) # (b, 3, H, W)
        
        x_amp_low_mu_s = x_amp_low_mu.view(2, b//2, 3, 1, 1).std(dim=1) # (2, 3, 1, 1)
        x_amp_low_std_s = x_amp_low_std.view(2, b//2, 3, 1, 1).std(dim=1) # (2, 3, 1, 1)
        
        i_x_amp_low_std = torch.cat([x_amp_low_std[b//2+b//4:], x_amp_low_std[b//2:b//2+b//4]])
        i_x_amp_low_mu = torch.cat([x_amp_low_mu[b//2+b//4:], x_amp_low_mu[b//2:b//2+b//4]])
        
        flag = torch.rand(b//2) < self.prob
        if flag.sum() > 0:
            # x_amp[:b//2, :, self.h_start:self.h_start+self.h_crop, :self.w_crop][flag] = x_amp_low_norm[:b//2][flag] * \
            #                 (torch.randn(flag.sum(),1,1,1,device=imgs.device) * self.alpha * x_amp_low_std_s[0] + x_amp_low_std[:b//2][flag]) + \
            #                 (torch.randn(flag.sum(),1,1,1,device=imgs.device) * self.alpha * x_amp_low_mu_s[0]  + x_amp_low_mu[:b//2][flag])   
            # aug_imgs[:b//2][flag] = self.ifft(torch.polar(x_amp_, x_pha[:b//2][flag]))#.clamp(min=0.0, max=1.0)
            x_amp[:b//2, :, self.h_start:self.h_start+self.h_crop, :self.w_crop][flag] = x_amp_low_norm[:b//2][flag] * \
                            (torch.randn(flag.sum(),1,1,1,device=imgs.device) * self.alpha * x_amp_low_std_s[1] + x_amp_low_std[b//2:][flag]) + \
                            (torch.randn(flag.sum(),1,1,1,device=imgs.device) * self.alpha * x_amp_low_mu_s[1]  + x_amp_low_mu[b//2:][flag])  
            aug_imgs[:b//2][flag] = self.brwa(self.ifft(torch.polar(x_amp[:b//2][flag], x_pha[:b//2][flag])).clamp(min=0.0, max=1.0))
            
            
        if not self.only_rgb:
            flag = torch.rand(b//2) < self.prob
            if flag.sum() > 0:
                x_amp[b//2:, :, self.h_start:self.h_start+self.h_crop, :self.w_crop][flag] = x_amp_low_norm[b//2:][flag] * \
                                (torch.randn(flag.sum(),1,1,1,device=imgs.device) * self.alpha * x_amp_low_std_s[1] + x_amp_low_std[b//2:][flag]) + \
                                (torch.randn(flag.sum(),1,1,1,device=imgs.device) * self.alpha * x_amp_low_mu_s[1]  + x_amp_low_mu[b//2:][flag])   
                aug_imgs[b//2:][flag] = self.brwa(self.ifft(torch.polar(x_amp[b//2:][flag], x_pha[b//2:][flag])).clamp(min=0.0, max=1.0))
        
        # aug_imgs = self.ifft(torch.polar(x_amp, x_pha))#.clamp(min=0.0, max=1.0)
        
        # return self.brwa(aug_imgs) # (b, 3, H, W)
        return self.rea(self.norm(aug_imgs))