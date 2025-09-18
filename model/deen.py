import torch
from torch import nn
import torch.nn.functional as F

from .build import MODEL_REGISTRY
from timm import create_model

from copy import deepcopy

import math

import random

from torch.nn import init

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
            
            
class DEE_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DEE_module, self).__init__()

        self.FC11 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)

        self.FC21 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC21.apply(weights_init_kaiming)
        self.FC22 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC22.apply(weights_init_kaiming)
        self.FC23 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC23.apply(weights_init_kaiming)
        self.FC2 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC2.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x))/3
        x1 = self.FC1(F.relu(x1))
        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x))/3
        x2 = self.FC2(F.relu(x2))
        out = torch.cat((x, x1, x2), 0)
        out = self.dropout(out)
        return out
    
class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)
        
        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        
        return z

class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(nn.Conv2d(self.low_dim//self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim//self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z


class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)
    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z

@MODEL_REGISTRY.register()
class DEEN(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__()
        
        bb = create_model('resnet50.tv_in1k', True, pretrained_cfg_overlay={'file': '/root/data/.cache/models/resnet50-0676ba61.pth'})
        bb.layer4[0].conv2.stride = (1, 1)
        bb.layer4[0].downsample[0].stride = (1, 1)
        
        self.infrared_module = nn.Sequential(
            bb.conv1,
            bb.bn1,
            bb.act1,
            bb.maxpool,
        )
        self.visible_module = deepcopy(self.infrared_module)
        self.base_resnet = nn.Sequential(
            bb.layer1,
            bb.layer2,
            bb.layer3,
            bb.layer4,
        )
        
        self.DEE = DEE_module(1024)
        self.MFA1 = MFA_block(256, 64, 0)
        self.MFA2 = MFA_block(512, 256, 1)
        self.MFA3 = MFA_block(1024, 512, 1)
        
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x, cam_ids, *args, **kwargs):
        bs = x.size(0)
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
            
        x_ = x
        x = self.base_resnet[0](x_)
        x_ = self.MFA1(x, x_)
        x = self.base_resnet[1](x_)
        x_ = self.MFA2(x, x_)
        x = self.base_resnet[2](x_)
        x_ = self.MFA3(x, x_)
        x_ = self.DEE(x_)
        x = self.base_resnet[3](x_)
        
        xp = F.adaptive_avg_pool2d(x, 1)
        
        x_pool = xp.view(xp.size(0), xp.size(1))
        
        feat = self.bottleneck(x_pool)

        if self.training:
            xps = xp.view(xp.size(0), xp.size(1), xp.size(2)).permute(0, 2, 1) # 3B x 1 x C
            xp1, xp2, xp3 = torch.chunk(xps, 3, 0) # B x 1 x C
            xpss = torch.cat((xp2, xp3), 1) # B x 2 x C
            # loss_ort = torch.triu(torch.bmm(xpss, xpss.permute(0, 2, 1)), diagonal = 1).sum() / (xp.size(0))
            return [*torch.chunk(self.classifier(feat), 3, 0)], [*torch.chunk(x_pool, 3, 0)], [xpss, ], [(xp1.squeeze(1), xp2.squeeze(1)), (xp1.squeeze(1), xp3.squeeze(1))]
        else:
            return torch.cat([F.normalize(feat[:bs]), F.normalize(feat[bs:2*bs]), F.normalize(feat[2*bs:]), F.normalize(x_pool[:bs]), F.normalize(x_pool[bs:2*bs]), F.normalize(x_pool[2*bs:])], 1)
        
    def get_params(self, *args, **kwargs):    
        ignored_params_id = list(map(id, self.classifier.parameters())) + \
                         list(map(id, self.bottleneck.parameters()))
                         
        base_params = filter(lambda p: id(p) not in ignored_params_id, self.parameters())
        ignored_params = filter(lambda p: id(p) in ignored_params_id, self.parameters())
        
        params = [
            {"params": base_params, "lr": 0.01},
            {"params": ignored_params, "lr": 0.1},
        ]
        
        # params = self.parameters()
        return params