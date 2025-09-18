import torch
import logging
from torch import nn

from ._vmamba_ori import vmamba_tiny_s1l8, vmamba_small_s2l15, vmamba_base_s2l15
from .necks import BNNeck

from .build import MODEL_REGISTRY
from timm.layers import trunc_normal_
from copy import deepcopy

import torch.nn.functional as F

import random

from torch.nn import init

logger = logging.getLogger(__name__)


_backbones = {
    'vmamba_tiny': '/root/data/.cache/models/vssm1_tiny_0230s_ckpt_epoch_264.pth',
    'vmamba_small': '/root/data/.cache/models/vssm_small_0229_ckpt_epoch_222.pth',
    'vmamba_base': '/root/data/.cache/models/vssm_base_0229_ckpt_epoch_237.pth'
}


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

@MODEL_REGISTRY.register()
class VMambaVI(nn.Module):
    def __init__(self, backbone_name='vmamba_tiny', num_classes=751, drop_path_rate=0.3, last_stride=1, 
                 iid_patch_embed=False, use_sie=0, num_sies=0, sie_xishu =3.0, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.iid_patch_embed = iid_patch_embed
        
        path = _backbones[backbone_name]
        if backbone_name == 'vmamba_tiny':
            bb = vmamba_tiny_s1l8()
            depths = [2, 2, 8, 2]
        elif backbone_name == 'vmamba_small':
            bb = vmamba_small_s2l15()
            depths = [2, 2, 15, 2]
        elif backbone_name == 'vmamba_base':
            bb = vmamba_base_s2l15()
            depths = [2, 2, 15, 2]
        else:
            raise ValueError(f"Invalid backbone name: {backbone_name}")
        bb.load_state_dict(torch.load(path)['model'])

        self.use_sie = use_sie
        self.sie_xishu = sie_xishu
        if self.use_sie:
            self.sie_embed = nn.Parameter(torch.zeros(num_sies, bb.dims[0], 1, 1))
            trunc_normal_(self.sie_embed, std=.02)
            if self.use_sie == 1:
                logger.info('camera number is : {}'.format(num_sies))
            elif self.use_sie == 2:
                logger.info('modality number is : {}'.format(num_sies))
            else:
                raise ValueError(f"Invalid use_sie value: {self.use_sie}")
            logger.info('using SIE_Lambda is : {}'.format(sie_xishu))
        
        if self.iid_patch_embed:
            self.patch_embed_V = bb.patch_embed
            self.patch_embed_I = deepcopy(self.patch_embed_V)
        else:
            self.patch_embed = bb.patch_embed
            
        self.layers = bb.layers
        
        if last_stride == 1:
            self.layers[-2]._modules['downsample'][1].stride = (1, 1)
        
        logger.info(f"Drop path rate: {drop_path_rate}")
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for lyr in self.layers:
            for blk in lyr.blocks:
                blk.drop_path.drop_prob = dpr.pop(0)
                
        self.norm = bb.classifier.norm
        
        self.pool = GeM()
        
        self.bn_neck = nn.BatchNorm1d(768)
        self.bn_neck.bias.requires_grad_(False)
        self.bn_neck.apply(weights_init_kaiming)
        
        self.classifier = nn.Linear(768, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x, cam_ids, *args, **kwargs):
        if self.iid_patch_embed:
            infrared_flags = (cam_ids == 3) | (cam_ids == 6)
            if torch.all(infrared_flags):
                # only infrared images
                x= self.patch_embed_I(x)
            elif torch.all(~infrared_flags):
                # only visible images
                x = self.patch_embed_V(x)
            else:
                # both visible and infrared images
                # training mode
                # [V1, V2, V3, V4, I1, I2, I3, I4]
                bs = x.size(0)
                f0_v = self.patch_embed_V(x[:bs//2])
                f0_i = self.patch_embed_I(x[bs//2:])
                x = torch.cat([f0_v, f0_i])
        else:     
            x = self.patch_embed(x)
        
        if self.use_sie == 1:
            # camera-wise SIE
            sie_ids = cam_ids - 1 if self.num_classes == 395 else cam_ids - 2
        elif self.use_sie == 2:
            sie_ids = ((cam_ids == 3) | (cam_ids == 6)).long()
            x = x + self.sie_embed[sie_ids] * self.sie_xishu
            
        for i,lyr in enumerate(self.layers):
            x = lyr(x)
        f0 = self.norm(x)
        
        f1 = self.pool(f0).flatten(1)
        f2 = self.bn_neck(f1)
        if not self.training:
            return f2
        
        logits = self.classifier(f2)
        return [logits], [f1]
        
        # f = F.avg_pool2d(x, x.size()[2:]).flatten(1)
        
        # if not self.training:
        #     return f
        
        
        # f_bn = self.bn_neck(f)
        # logits = self.cls(f_bn)
        # return [logits,], [f,]
    
    def get_params(self, *args, **kwargs):
        return self.parameters()
    
    
    def freeze_backbone(self):
        self.patch_embed.requires_grad_(False)
        for lyr in self.layers:
            lyr.requires_grad_(False)
        self.norm.requires_grad_(False)
        
    
    def unfreeze_backbone(self):
        self.patch_embed.requires_grad_(True)
        for lyr in self.layers:
            lyr.requires_grad_(True)
        self.norm.requires_grad_(True)

    
if __name__ == '__main__':
    model = VMambaVI()
    tmp = list(model.named_parameters())
    
    print(1)
        
        
        