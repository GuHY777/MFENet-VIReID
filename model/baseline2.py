import torch
from torch import nn
from timm import create_model
from copy import deepcopy
from .necks import BNNeck
from .build import MODEL_REGISTRY
import torch.nn.functional as F
from timm.layers import trunc_normal_
from scipy import stats
import logging
logger = logging.getLogger(__name__)

_backbones = {
   'resnet18': ['resnet18.tv_in1k', '/root/data/.cache/models/resnet18-5c106cde.pth'],
   'resnet34': ['resnet34.tv_in1k','/root/data/.cache/models/resnet34-333f7ec4.pth'],
   'resnet50': ['resnet50.tv_in1k','/root/data/.cache/models/resnet50-0676ba61.pth'],
   'resnet101': ['resnet101.tv_in1k','/root/data/.cache/models/resnet101-5d3b4d8f.pth'],
}

NN = lambda x: F.normalize(x)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    
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

    
class IMA(nn.Module):
    def __init__(self, shape, N, P=2):
        super().__init__()
        
        in_dims, h, w = shape
        self.N = N
        self.P = P

        self.fft = lambda x: torch.fft.fftshift(torch.fft.rfft2(x, norm='ortho'), dim=(-2))
        self.ifft = lambda x: torch.fft.irfft2(torch.fft.ifftshift(x, dim=(-2)), s=(h, w), norm='ortho')
        
        self.register_buffer('masks', _get_masks(h, w, N, mode='square'))
        
        self.ws = nn.Sequential(
            nn.Conv2d(in_dims, P, 1, 1, 0),
            nn.Sigmoid()
        )
        # nn.init.zeros_(self.ws[-2].bias)
        
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(in_dims) for _ in range(P)
        ])
        
        # self.i = 0
        
    def forward(self, x):
        x = x.to(torch.float32)
        x_fft = self.fft(x)
        x_amp, x_pha = x_fft.abs(), torch.angle(x_fft + 1e-6)
        b,c,h,w = x_amp.shape
        
        ws = []
        x_amps = []
        for i in range(self.masks.shape[0]):
            x_amps.append(x_amp * self.masks[i])
            n = self.masks[i].sum()
            ws.append(torch.sum(x_amps[-1], dim=(2,3)) / n)
        x_amps = torch.stack(x_amps, dim=2).unsqueeze(0) # 1,B,C,N,H,W
        ws = torch.stack(ws, dim=2).unsqueeze(3) # B,C,N,1
        ws = self.ws(ws).unsqueeze(4).unsqueeze(2).transpose(0,1) # P,B,1,N,1,1
        x_amp = (x_amps * ws).sum(3) # P,B,C,H,W
        x_ = self.ifft(torch.polar(x_amp.view(self.P*b,c,h,w), x_pha.repeat(self.P, 1, 1, 1)))
        
        # if self.i % 600 == 0:
        #     for p in range(self.P):
        #         logger.info(f"P-{p}:\n {stats.describe(ws[p].flatten().detach().cpu().numpy())}")
        # self.i += 1
        
        return [self.bns[i](xi) for i, xi in enumerate(x_.chunk(self.P, dim=0))]
            
        
        # ws = []
        # x_amps = []
        # for i in range(self.masks.shape[0]):
        #     x_amps.append(x_amp * self.masks[i])
        #     n = self.masks[i].sum()
        #     ws.append(torch.sum(x_amps[-1], dim=(2,3)) / n)
        # x_amps = torch.stack(x_amps, dim=2).unsqueeze(0) # 1,B,C,N,H,W
        # ws = torch.stack(ws, dim=2).unsqueeze(3) # B,C,N,1
        # ws = self.ws(ws).unsqueeze(4).unsqueeze(2).transpose(0,1) # P,B,1,N,1,1
        # x_amp = (x_amps * ws).sum(3) # P,B,C,H,W
        # x_ = self.ifft(torch.polar(x_amp.view(self.P*b,c,h,w), x_pha.repeat(self.P, 1, 1, 1)))
        
        # if self.i % 600 == 0:
        #     logger.info(f"P-0:\n {stats.describe(ws[0].flatten().detach().cpu().numpy())}")
        #     logger.info(f"P-1:\n {stats.describe(ws[1].flatten().detach().cpu().numpy())}")
        # self.i += 1
        
        # return x.repeat(self.P, 1, 1, 1) + x_
        
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
        
        self.conv3 = nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=False)

        self.share_attn = nn.Sequential(
            nn.Conv1d(2, 1, 5, 1, 2),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.share_attn[-2].bias)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = 0.0
        for i in range(self.num_scales):
            x2_ = self.conv2[i](x1)
            w = self.share_attn(
                    torch.cat([
                        x2_.mean((2, 3)).view(x.size(0), 1, -1),
                        x2_.amax((2, 3)).view(x.size(0), 1, -1)
                    ], dim=1)
                ).view(x.size(0), -1, 1, 1)
            x2 = x2 + x2_ * w

        x3 = self.conv3(x2)
        return x3
    

class LHM(nn.Module):
    def __init__(self, shape, ratio=0.2):
        super(LHM, self).__init__()
        self.shape = shape
        self.ratio = ratio
        
        in_dims, h, w = shape

        self.bn = nn.BatchNorm2d(in_dims)
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
        
        self.hig_os = OSBlock(shape, 4)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dims, 2, 3, 1, 1),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.conv2[-2].bias)

        self.fft = lambda x: torch.fft.fftshift(torch.fft.rfft2(x, norm='ortho'), dim=(-2))
        self.ifft = lambda x: torch.fft.irfft2(torch.fft.ifftshift(x, dim=(-2)), s=(h, w), norm='ortho')
        
        mask = _get_masks(h, w, linspaces=ratio, mode='square')[0]
        self.h_start = int(torch.argmax(mask, dim=0)[0].item())
        self.h_crop = int(mask.sum(0)[0].item())
        self.w_crop = int(mask.sum(1).max().item())
        self.register_buffer('low_mask', mask)
        
        self.low_weights = nn.Parameter(torch.randn(in_dims, self.h_crop, self.w_crop, 2) * 0.02)

    def forward(self, x):
        x = x.to(torch.float32)
        x_fft = self.fft(x)
        x_low = (x_fft * self.low_mask).clone()
        x_low[..., self.h_start:self.h_start+self.h_crop, :self.w_crop] = \
            x_fft[..., self.h_start:self.h_start+self.h_crop, :self.w_crop] * torch.view_as_complex(self.low_weights)
        
        x_hig = x_fft * (1.0 - self.low_mask)
        x_ = torch.cat([x_low, x_hig], dim=0)
        x_ = self.ifft(x_)
        
        x_low, x_hig = x_.chunk(2, dim=0)

        x_hig = self.hig_os(x_hig)

        sp_attn = self.conv2(x)
        x_ = x_low * sp_attn[:, :1] + x_hig * sp_attn[:, 1:]

        return x + self.bn(x_)


@MODEL_REGISTRY.register()
class BaseLine2(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=751, L=2, no_ima=False, *args, **kwargs):
        super().__init__()
        
        self.L = L
        self.no_ima = no_ima
        
        timm_model_name, timm_pretrained_path = _backbones[backbone_name]
        bb = create_model(timm_model_name, True, pretrained_cfg_overlay={'file': timm_pretrained_path})
        if backbone_name in ['resnet18','resnet34']:
            bb.layer4[0].conv1.stride = (1, 1)
            bb.layer4[0].downsample[0].stride = (1, 1)
        else:
            bb.layer4[0].conv2.stride = (1, 1)
            bb.layer4[0].downsample[0].stride = (1, 1)
            
        img_size = kwargs['img_size']
        
        self.share_module = nn.Sequential(
            bb.conv1,
            bb.bn1,
            bb.act1,
            bb.maxpool,
            
            bb.layer1,
            
            bb.layer2[0],
            bb.layer2[1],
            bb.layer2[2],
            LHM([512, *(img_size[0]//8, img_size[1]//8)], 0.2),
            bb.layer2[3],
            LHM([512, *(img_size[0]//8, img_size[1]//8)], 0.2),
            
            bb.layer3[0],
            bb.layer3[1],
            bb.layer3[2],
            bb.layer3[3],
            LHM([1024, *(img_size[0]//16, img_size[1]//16)], 0.2),
            bb.layer3[4],
            LHM([1024, *(img_size[0]//16, img_size[1]//16)], 0.2),
            bb.layer3[5],
            IMA([1024, *(img_size[0]//16, img_size[1]//16)], 7, L) if not no_ima else nn.Identity(),
        )
        
        self.layer4s = nn.ModuleList([
            deepcopy(bb.layer4) for _ in range(L)
        ])
        
        self.pools = nn.ModuleList([
            GeM(p=3.0) for _ in range(L)
        ])
        for pool in self.pools:
            pool.p.requires_grad_(False)
        
        bn_neck = nn.BatchNorm1d(2048)
        nn.init.zeros_(bn_neck.bias)
        bn_neck.bias.requires_grad_(False)
        self.bn_necks = nn.ModuleList([
            deepcopy(bn_neck) for _ in range(L)
        ])

        self.clss = nn.ModuleList([
            nn.Linear(2048, num_classes, bias=False) for _ in range(L) 
        ])
        
    def forward(self, x, cam_ids, *args, **kwargs):
        f0 = self.share_module(x) if not self.no_ima else [self.share_module(x)]
        f1 = [self.layer4s[i](f0[i]) for i in range(self.L)]
        pf = [self.pools[i](f1[i]).flatten(1) for i in range(self.L)]
        f = [self.bn_necks[i](pf[i]) for i in range(self.L)]

        # if not self.training:
        #     return (
        #         torch.cat([NN(fi) for fi in f], dim=1),
        #         torch.cat([*[NN(fi) for fi in f], *[NN(pfi) for pfi in pf]], dim=1)
        #     )
        
        logits = [self.clss[i](f[i]) for i in range(self.L)]
        
        return logits, pf, pf, [NN(fi) for fi in f], [pf, ], [[NN(fi) for fi in f],]
    
    def get_params(self, *args, **kwargs):    
        params = self.parameters()
        return params