import torch
import torch.nn as nn

import torch.nn.functional as F
from .build import MODEL_REGISTRY
from timm import create_model

from copy import deepcopy

_backbones = {
   'resnet18': ['resnet18.tv_in1k', '/root/data/.cache/models/resnet18-5c106cde.pth'],
   'resnet34': ['resnet34.tv_in1k','/root/data/.cache/models/resnet34-333f7ec4.pth'],
   'resnet50': ['resnet50.tv_in1k','/root/data/.cache/models/resnet50-0676ba61.pth'],
   'resnet101': ['resnet101.tv_in1k','/root/data/.cache/models/resnet101-5d3b4d8f.pth'],
}



class AGPM(nn.Module):
    def __init__(self, in_channels, k, *args, **kwargs):
        super().__init__()
        self.conv1x1s = nn.ModuleList([nn.Conv2d(in_channels, in_channels // k, 1, bias=False) for _ in range(k)])
        
        
    def forward(self, x_spa):
        # fft
        _, _, h, w = x_spa.shape
        
        x_spa = x_spa.to(torch.float32)
        x_fft = torch.fft.rfft2(x_spa, norm='ortho')
        
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft + 1e-7)
        
        x_avg = F.adaptive_avg_pool2d(x_amp, (1, 1))
        
        fs = [conv1x1(x_avg) for conv1x1 in self.conv1x1s]
        fs = torch.cat(fs, dim=1)
        sig = torch.sigmoid(fs)
        x_pha = x_pha * sig + x_pha
        
        x_spa = x_amp * torch.exp(1j * x_pha)
        x_spa = torch.fft.irfft2(x_spa, s=(h, w), norm='ortho')
        
        return x_spa
    
    
class ANMM(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__()
        
        self.conv1x1s = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1, bias=False) for _ in range(2)])
        self.ins = nn.ModuleList([nn.InstanceNorm2d(in_channels, affine=True) for _ in range(2)])
        self.bn = nn.BatchNorm2d(in_channels)
        
    
    def forward(self, f_spa):
        
        _, _, h, w = f_spa.shape
        
        f_spa = f_spa.to(torch.float32)
        # fft
        f_fft = torch.fft.rfft2(f_spa, norm='ortho')
        
        f_amp = torch.abs(f_fft)
        f_pha = torch.angle(f_fft + 1e-7)
        
        f1_amp = self.conv1x1s[0](f_amp)
        f2_amp = self.conv1x1s[1](f_amp)
        
        f1_amp_n = self.ins[0](f1_amp)
        f2_amp_n = self.ins[1](f2_amp)
        
        f1_spa = f1_amp_n * torch.exp(1j * f_pha)
        f2_spa = f2_amp_n * torch.exp(1j * f_pha)
        
        f1_spa = torch.fft.irfft2(f1_spa, s=(h, w), norm='ortho')
        f2_spa = torch.fft.irfft2(f2_spa, s=(h, w), norm='ortho')
        
        return self.bn(f1_spa), self.bn(f2_spa)
        
        
@MODEL_REGISTRY.register()
class FDNM(nn.Module):
    def __init__(self, num_classes, agp_k=4, *args, **kwargs):
        """
        Frequency domain nuances mining for visible-infrared person re-identification
        """
        super().__init__()
        
        bb = create_model('resnet50.tv_in1k', True, pretrained_cfg_overlay={'file': '/root/data/.cache/models/resnet50-0676ba61.pth'})
        bb.layer4[0].conv2.stride = (1, 1)
        bb.layer4[0].downsample[0].stride = (1, 1)
        
        self.visible_module = nn.Sequential(
            bb.conv1,
            bb.bn1,
            bb.act1,
            bb.maxpool,
        )
        self.infrared_module = deepcopy(self.visible_module)
        
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        
        self.agp1 = AGPM(256, agp_k)
        self.agp2 = AGPM(512, agp_k)
        
        self.anm = ANMM(2048)
        
        self.bn_neck = nn.BatchNorm1d(2048)
        nn.init.constant_(self.bn_neck.bias, 0) 
        self.bn_neck.bias.requires_grad_(False)
        
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        
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
            x = torch.cat([f0_v, f0_i])
        
        x = self.layer1(x)
        x = self.agp1(x)
        x = self.layer2(x)
        x = self.agp2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1, x2 = self.anm(x)
        
        f1, f2 = F.adaptive_avg_pool2d(x1, (1, 1)).flatten(1), F.adaptive_avg_pool2d(x2, (1, 1)).flatten(1)
        f1_bn, f2_bn = self.bn_neck(f1), self.bn_neck(f2)
        
        if not self.training:
            return torch.cat([F.normalize(f1_bn), F.normalize(f2_bn)], dim=1)
        
        logits1 = self.classifier(f1_bn)
        logits2 = self.classifier(f2_bn)
        
        return [f1, f2], [logits1, logits2], [(f1, f2),]
        
    def get_params(self, *args, **kwargs): 
        return self.parameters()
        
        
        
if __name__ == '__main__':
    m = AGPM(256, 4)
    x = torch.randn(2, 256, 8, 8)
    y = m(x)