import torch
from torch import nn

import torch.nn.functional as F
from .build import MODEL_REGISTRY
from timm import create_model

_backbones = {
   'resnet18': ['resnet18.tv_in1k', '/root/data/.cache/models/resnet18-5c106cde.pth'],
   'resnet34': ['resnet34.tv_in1k','/root/data/.cache/models/resnet34-333f7ec4.pth'],
   'resnet50': ['resnet50.tv_in1k','/root/data/.cache/models/resnet50-0676ba61.pth'],
   'resnet101': ['resnet101.tv_in1k','/root/data/.cache/models/resnet101-5d3b4d8f.pth'],
}

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
        x1 = x1 * weight
        x1 = torch.fft.irfft2(x1, s=(a, b), norm='ortho')
        return x + x1


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


@MODEL_REGISTRY.register()
class MPANet2(nn.Module):
    def __init__(self, num_classes, num_parts=6, alpha=0.2, *args, **kwargs):
        super(MPANet2, self).__init__()
        """
        Discover cross-modality nuances for visible-infrared person re-identification
        """

        self.alpha = alpha
        
        bb = create_model('resnet50.tv_in1k', True, pretrained_cfg_overlay={'file': '/root/data/.cache/models/resnet50-0676ba61.pth'})
        bb.layer4[0].conv2.stride = (1, 1)
        bb.layer4[0].downsample[0].stride = (1, 1)
            
        self.backbone = nn.Sequential(
            bb.conv1,
            bb.bn1,
            bb.act1,
            bb.maxpool,
            
            bb.layer1,
            GlobalFilter([256, 96, 32]),
            bb.layer2,
            GlobalFilter([512, 48, 16]),
            bb.layer3,
            MAM(1024),
            bb.layer4,
            MAM(2048)
        )
        
        self.base_dim = 2048
        self.dim = 2048
        self.part_num = num_parts
        self.spatial_attention = nn.Conv2d(self.base_dim, self.part_num, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.constant_(self.spatial_attention.bias, 0.0)
        self.activation = nn.Sigmoid()
        
        self.visible_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)

        self.visible_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.visible_classifier_.weight.requires_grad_(False)
        self.visible_classifier_.weight.data = self.visible_classifier.weight.data

        self.infrared_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier_.weight.requires_grad_(False)
        self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data
        
        self.bn_neck = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck.bias, 0) 
        self.bn_neck.bias.requires_grad_(False)
        
        self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num , num_classes, bias=False)
        
    def forward(self, x, cam_ids, *args, **kwargs):        
        global_feat = self.backbone(x)
        b, c, w, h = global_feat.shape
        
        masks = global_feat
        masks = self.spatial_attention(masks)
        masks = self.activation(masks)

        feats = []
        for i in range(self.part_num):
            mask = masks[:, i:i+1, :, :]
            feat = mask * global_feat

            feat = F.avg_pool2d(feat, feat.size()[2:])
            feat = feat.view(feat.size(0), -1) # b x 2048

            feats.append(feat)

        global_feat = F.avg_pool2d(global_feat, global_feat.size()[2:])
        global_feat = global_feat.view(global_feat.size(0), -1) # b x 2048

        feats.append(global_feat)
        feats = torch.cat(feats, 1) # b x (2048+2048*6)
        feats_bn = self.bn_neck(feats)

        if self.training:
            logits = self.classifier(feats_bn)
            logits_v = self.visible_classifier(feats_bn[:b//2])
            logits_i = self.infrared_classifier(feats_bn[b//2:])
            logits_vi = torch.cat([logits_v, logits_i], 0)

            with torch.no_grad():
                self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.alpha) \
                                                 + self.infrared_classifier.weight.data * self.alpha
                self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.alpha) \
                                                 + self.visible_classifier.weight.data * self.alpha

                logits_v_ = self.infrared_classifier_(feats_bn[:b//2])
                logits_i_ = self.visible_classifier_(feats_bn[b//2:])

                logits_vi_ = torch.cat([logits_v_, logits_i_], 0).float()
            
            
            
            return [masks, ], [feats, ], [logits, ], [logits_vi, ], [(F.log_softmax(logits_vi, dim=1), F.softmax(logits_vi_, dim=1)), ]
        
        else:
            return feats_bn
        
    def get_params(self, *args, **kwargs): 
        return self.parameters()
        
        