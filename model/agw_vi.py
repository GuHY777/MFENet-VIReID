import torch
from torch import nn
from timm import create_model
from copy import deepcopy
from .necks import BNNeck
from .build import MODEL_REGISTRY

_backbones = {
   'resnet18': ['resnet18.tv_in1k', '/root/data/.cache/models/resnet18-5c106cde.pth'],
   'resnet34': ['resnet34.tv_in1k','/root/data/.cache/models/resnet34-333f7ec4.pth'],
   'resnet50': ['resnet50.tv_in1k','/root/data/.cache/models/resnet50-0676ba61.pth'],
   'resnet101': ['resnet101.tv_in1k','/root/data/.cache/models/resnet101-5d3b4d8f.pth'],
}



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

@MODEL_REGISTRY.register()
class AGW_VI(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=751, non_local= 'on', pool = 'gem', *args, **kwargs):
        super(AGW_VI, self).__init__()
        
        timm_model_name, timm_pretrained_path = _backbones[backbone_name]
        bb = create_model(timm_model_name, True, pretrained_cfg_overlay={'file': timm_pretrained_path})
        if backbone_name in ['resnet18','resnet34']:
            bb.layer4[0].conv1.stride = (1, 1)
            bb.layer4[0].downsample[0].stride = (1, 1)
        else:
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
        
        self.non_local = non_local
        if self.non_local == 'on':
            assert backbone_name == 'resnet50'
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])
        
        self.cls = BNNeck(2048, num_classes, bias=False, pool=pool, neck_feat='after')
        
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
        
        # shared block
        if self.non_local == 'on':
            # Layer 1
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.layer1)):
                x = self.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.layer2)):
                x = self.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.layer3)):
                x = self.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.layer4)):
                x = self.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
        return self.cls(x)
    
    def get_params(self, *args, **kwargs):    
        # ignored_params = list(map(id, self.cls.parameters()))
        # base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        # params = [
        #     {"params": self.cls.parameters(), "lr": 0.1},
        #     {"params": base_params, "lr":0.01}
        # ]
        
        params = self.parameters()
        return params