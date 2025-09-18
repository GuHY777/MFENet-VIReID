import torch
from torch import nn
import torch.nn.functional as F
from .build import LOSS_REGISTRY
from einops import rearrange
import logging
from scipy import stats
from torchsort import soft_rank
from functools import partial

logger = logging.getLogger(__name__)


__all__ = ['center_guided_pair_mining_loss', 'id_loss', 'triplet_loss', 'wrt_loss', 
           'part_reg_loss', 'center_loss', 'center_cluster_loss', 'kl_div_loss', 'diverse_loss', 'cross_modality_triplet_loss',
           'triplet_loss_wrt2', 'global_center_loss', 'local_center_loss1', 'local_center_loss2', 'modality_aware_loss', 'reg_loss',
           'modality_aware_loss2', 'cmsr_loss', 'cmrr_loss']


@LOSS_REGISTRY.register()
class cmrr_loss(nn.Module):
    '''
        Cross-Modal Rank Regularization Loss (CMRR Loss)
    '''
    def __init__(self, r=2.0):
        super().__init__()
        
        self.rank = partial(soft_rank, regularization_strength=r, regularization='kl')
        
        self.mod_idxs = None
        
        
    def spearmanr(self, pred, target, **kw):
        pred = pred - torch.mean(pred, dim=1, keepdim=True)
        pred = pred / torch.norm(pred, dim=1, keepdim=True)
        target = target - torch.mean(target, dim=1, keepdim=True)
        target = target / torch.norm(target, dim=1, keepdim=True)
        return (pred * target).sum(1)
        
    def forward(self, input, target):
        '''
            input: (f0, f1, f2, ...)
        
        '''
        N, _ = input[0].size()
        pos_idxs = target == target.unsqueeze(1)
        
        if self.mod_idxs is None:
            mod_idxs = torch.cat([
                    torch.zeros(N//2, dtype=torch.long), 
                    torch.ones(N//2, dtype=torch.long)])
            mod_idxs = (mod_idxs == mod_idxs.unsqueeze(1)).to(input[0].device)
            self.mod_idxs = mod_idxs
        
        ranks = []
        for i in range(len(input)):
            f = input[i]
            dists = ((f.unsqueeze(0) - f.unsqueeze(1))**2).sum(dim=2).clamp(min=1e-12).sqrt() # N x N
            
            dist_intra = dists[pos_idxs & (~self.mod_idxs)].view(N, -1)

            r_intra = self.rank(dist_intra)

            ranks.append(r_intra)
            
        loss_intra = 0.0
        for i in range(len(input)):
            for j in range(i+1, len(input)):
                loss_intra += (self.spearmanr(ranks[i], ranks[j]) + 1.0) / 2.0

        loss = loss_intra / (len(input) * (len(input)-1) / 2.0)
        
        return loss.mean()
            
            


@LOSS_REGISTRY.register()
class cmsr_loss(nn.Module):
    '''
        Cross-Modal Soft Retrieval Loss (CMSR Loss)
    '''
    def __init__(self, m=0.1, alpha=0.1):
        super().__init__()
        
        self.m = m
        self.alpha = alpha
        
        self.mod_idxs = None
        
    def forward(self, input, target):
        N, _ = input.size()
        
        dists = input.pow(2).sum(dim=1, keepdim=True) +\
                input.pow(2).sum(dim=1, keepdim=True).t() -\
                2 * torch.mm(input, input.t())
        dists = dists.clamp(min=1e-12).sqrt() # N x N
        
        pos_idxs = target == target.unsqueeze(1)
        
        if self.mod_idxs is None:
            mod_idxs = torch.cat([
                    torch.zeros(N//2, dtype=torch.long), 
                    torch.ones(N//2, dtype=torch.long)])
            mod_idxs = (mod_idxs == mod_idxs.unsqueeze(1)).to(input[0].device)
            self.mod_idxs = mod_idxs
        
        _pos_idxs = pos_idxs & (~torch.eye(N, dtype=torch.bool, device=input.device))
        
        dists_ap_intra = rearrange(dists[_pos_idxs & self.mod_idxs], '(N k) -> N k ', N = N)
        dists_ap_inter = rearrange(dists[_pos_idxs & (~self.mod_idxs)], '(N k) -> N k ', N = N)
        
        weights_ap_intra = F.softmax(-dists_ap_intra / self.alpha, dim=1)
        weights_ap_inter = F.softmax(dists_ap_inter / self.alpha, dim=1)

        dists_ap_intra = (weights_ap_intra * dists_ap_intra).sum(dim=1)
        dists_ap_inter = (weights_ap_inter * dists_ap_inter).sum(dim=1)
        
        loss = F.relu(dists_ap_inter - dists_ap_intra + self.m)
        return loss.mean()


@LOSS_REGISTRY.register()
class modality_aware_loss2(nn.Module):
    def __init__(self, s=64.0, m=0.3, gamma=1.0):
        super().__init__()
        
        self.gamma = gamma
        self.s = s
        self.m = m
        self.target = None
        self.i = 0
        
    def forward(self, input, target):
        '''
            input: (logits1, logits2)
            target: (bs) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        '''
        bs, n_cls = input[0].size()
        
        mask = torch.zeros_like(input[0], dtype=torch.bool)
        mask[torch.arange(bs), target] = True
        
        Sp_v = input[0][mask].view(bs, 1)
        Sp_i = input[1][mask].view(bs, 1)
        Sn_v = input[0][~mask].view(bs, n_cls - 1)
        Sn_i = input[1][~mask].view(bs, n_cls - 1)
        
        Sp = torch.cat([Sp_v, Sp_i], dim=1) # bs x 2
        Sn = torch.stack([Sn_v, Sn_i], dim=2) # bs x (n_cls-1) x 2
        
        Wp = F.softmax(-Sp/self.gamma, dim=-1) # softmin
        Wn = F.softmax(Sn/self.gamma, dim=-1) # softmax
        
        if self.i % 600 == 0:
            logger.info(f"\nWp:\n{stats.describe(Wp.flatten().detach().cpu().numpy())}\nWn:\n{stats.describe(Wn.flatten().detach().cpu().numpy())}")
            logger.info(f"\nWp[0]:\n{Wp[0]}\nWp[32]:\n{Wp[32]}\nWn[0,0]:\n{Wn[0,0]}\nWn[32,0]:\n{Wn[32,0]}")
        self.i += 1
        
        Sp = (Sp * Wp).sum(1, keepdim=True) # bs x 1
        Sn = (Sn * Wn).sum(2) # bs x (n_cls-1)
        
        S = torch.cat([Sp-self.m, Sn], dim=1) # bs x (n_cls)
        S = self.s * S
        
        if self.target is None:
            self.target = torch.zeros(bs, device=input[0].device, dtype=torch.long)
        
        loss = F.cross_entropy(S, self.target)
        
        return loss


@LOSS_REGISTRY.register()
class modality_aware_loss(nn.Module):
    def __init__(self, s=64.0, m=0.3, gamma=1.0):
        super().__init__()
        
        self.gamma = gamma
        self.s = s
        self.m = m
        self.target = None
        self.i = 0
        
    def forward(self, input, target):
        '''
            input: (logits1, logits2)
            target: (bs) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        '''
        bs, n_cls = input[0].size()
        
        mask = torch.zeros_like(input[0], dtype=torch.bool)
        mask[torch.arange(bs), target] = True
        
        Sp_v = input[0][mask].view(bs, 1)
        Sp_i = input[1][mask].view(bs, 1)
        Sn_v = input[0][~mask].view(bs, n_cls - 1)
        Sn_i = input[1][~mask].view(bs, n_cls - 1)
        
        Sp = torch.cat([Sp_v, Sp_i], dim=1) # bs x 2
        Sn = torch.stack([Sn_v, Sn_i], dim=2) # bs x (n_cls-1) x 2
        
        Wp = F.softmax(-Sp/self.gamma, dim=-1) # softmin
        Wn = F.softmax(Sn/self.gamma, dim=-1) # softmax
        
        if self.i % 600 == 0:
            logger.info(f"\nWp:\n{stats.describe(Wp.flatten().detach().cpu().numpy())}\nWn:\n{stats.describe(Wn.flatten().detach().cpu().numpy())}")
            logger.info(f"\nWp[0]:\n{Wp[0]}\nWp[32]:\n{Wp[32]}\nWn[0,0]:\n{Wn[0,0]}\nWn[32,0]:\n{Wn[32,0]}")
        self.i += 1
        
        Sp = (Sp * Wp).sum(1, keepdim=True) # bs x 1
        Sn = (Sn * Wn).sum(2) # bs x (n_cls-1)
        
        S = torch.cat([Sp-self.m, Sn], dim=1) # bs x (n_cls)
        S = self.s * S
        
        if self.target is None:
            self.target = torch.zeros(bs, device=input[0].device, dtype=torch.long)
        
        loss = F.cross_entropy(S, self.target)
        
        return loss
        
        
@LOSS_REGISTRY.register()
class reg_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, input, target):
        """
        input: ((n_cls, f), (n_cls, f))
        
        """
        input1, input2 = input
        loss = ((input1 - input2)**2).sum(dim=-1).clamp(min=1e-12).sqrt().mean()
        return loss        
        

@LOSS_REGISTRY.register()
class cross_modality_wrt(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        
        self.margin = margin
        
    def forward(self, input, target):
        N, _ = input.size()
        
        dists = input.pow(2).sum(dim=1, keepdim=True) +\
                input.pow(2).sum(dim=1, keepdim=True).t() -\
                2 * torch.mm(input, input.t())
        dists = dists.clamp(min=1e-12).sqrt() # N x N
        
        pos_idxs = target == target.unsqueeze(1)
        
        mod_idxs = torch.cat([
                torch.zeros(N//2, dtype=torch.long, device=input.device), 
                torch.ones(N//2, dtype=torch.long, device=input.device)])
        mod_idxs = mod_idxs == mod_idxs.unsqueeze(1)
        
        _pos_idxs = pos_idxs & (~torch.eye(N, dtype=torch.bool, device=input.device))
        
        dists_ap_intra = rearrange(dists[_pos_idxs & mod_idxs], '(N k) -> N k ', N = N)
        dists_ap_inter = rearrange(dists[_pos_idxs & (~mod_idxs)], '(N k) -> N k ', N = N)
        
        weights_ap_intra = F.softmax(dists_ap_intra, dim=1)
        weights_ap_inter = F.softmax(dists_ap_inter, dim=1)
        
        dists_ap_intra = (weights_ap_intra * dists_ap_intra).sum(dim=1)
        dists_ap_inter = (weights_ap_inter * dists_ap_inter).sum(dim=1)
        
        # loss = F.relu(torch.abs(dists_ap_intra - dists_ap_inter) - self.margin)
        loss = F.softplus(torch.abs(dists_ap_intra - dists_ap_inter))
        
        return loss.mean()


@LOSS_REGISTRY.register()
class local_center_loss1(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        '''
        # ! inputs: [V1, V1, V2, V2, I1, I1, I2, I2]
        # ! target: (bs) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        '''

        bs = inputs.size(0)
        mask = targets == targets[0]
        k = mask.sum().item()
        p = bs // k
        
        cs = inputs.view(2, p, k//2, -1).mean(2) # 2 x p x f
        vis_ctrs, inf_ctrs = cs[0], cs[1] # p x f
        # share_ctrs = cs.mean(0) # p x f
        
        # ctrs = torch.cat([vis_ctrs, inf_ctrs, share_ctrs], dim=0) # 3p x f
        ctrs = torch.cat([vis_ctrs, inf_ctrs], dim=0) # 2p x f
        dists = ((inputs.unsqueeze(1) - ctrs.unsqueeze(0))**2).sum(2).clamp(min=1e-12).sqrt() # bs x 3p
        
        unique_targets = targets.view(2, p, k//2)[0,:,0]
        # pos_idxs = targets.unsqueeze(1) == unique_targets.unsqueeze(0).repeat(1, 3)
        pos_idxs = targets.unsqueeze(1) == unique_targets.unsqueeze(0).repeat(1, 2)
        
        # dists_ap = rearrange(dists[pos_idxs], '(N k) -> N k ', N = bs, k = 3)
        dists_ap = rearrange(dists[pos_idxs], '(N k) -> N k ', N = bs, k = 2)
        dists_an = rearrange(dists[~pos_idxs], '(N k) -> N k ', N = bs)
        
        weights_ap = F.softmax(dists_ap * self.alpha, dim=1)
        weights_an = F.softmax(-dists_an * self.alpha, dim=1)
        
        dists_ap = (weights_ap * dists_ap).sum(dim=1)
        dists_an = (weights_an * dists_an).sum(dim=1)
        
        loss = F.softplus(self.gamma * (dists_ap - dists_an))
        
        return loss.mean()

@LOSS_REGISTRY.register()
class local_center_loss2(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        
        self.margin = margin
        
    def forward(self, inputs, targets):
        '''
        # ! inputs: [V1, V1, V2, V2, I1, I1, I2, I2]
        # ! target: (bs) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        '''

        bs = inputs.size(0)
        mask = targets == targets[0]
        k = mask.sum().item()
        p = bs // k
        
        cs = inputs.view(2, p, k//2, -1).mean(2) # 2 x p x f
        vis_ctrs, inf_ctrs = cs[0], cs[1] # p x f
        ctrs = torch.cat([vis_ctrs, inf_ctrs], dim=0) # 3p x f
        dists = ((ctrs.unsqueeze(1) - ctrs.unsqueeze(0))**2).sum(2).clamp(min=1e-12).sqrt() # 3p x 3p x f
        pos_idxs = torch.eye(p, dtype=torch.bool, device=inputs.device).repeat(2, 2)
        _pos_idxs = pos_idxs & (~torch.eye(2*p, dtype=torch.bool, device=inputs.device))
        
        dists_ap = dists[_pos_idxs]
        dists_an = rearrange(dists[~pos_idxs], '(N k) -> N k ', N = 2*p)
        
        # weights_ap = F.softmax(dists_ap, dim=1)
        weights_an = F.softmax(-dists_an, dim=1)
        
        # dists_ap = (weights_ap * dists_ap).sum(dim=1)
        dists_an = (weights_an * dists_an).sum(dim=1)
        
        loss = F.softplus(dists_ap - dists_an)
        
        return loss.mean()
        
        # share_ctrs = cs.mean(0) # p x f
        # ctrs = torch.cat([vis_ctrs, inf_ctrs, share_ctrs], dim=0) # 3p x f
        
        # dists = ((ctrs.unsqueeze(1) - ctrs.unsqueeze(0))**2).sum(2).clamp(min=1e-12).sqrt() # 3p x 3p x f
        # pos_idxs = torch.eye(p, dtype=torch.bool, device=inputs.device).repeat(3, 3)
        # _pos_idxs = pos_idxs & (~torch.eye(3*p, dtype=torch.bool, device=inputs.device))
        
        # dists_ap = rearrange(dists[_pos_idxs], '(N k) -> N k ', N = 3*p, k = 2)
        # dists_an = rearrange(dists[~pos_idxs], '(N k) -> N k ', N = 3*p)
        
        # weights_ap = F.softmax(dists_ap, dim=1)
        # weights_an = F.softmax(-dists_an, dim=1)
        
        # dists_ap = (weights_ap * dists_ap).sum(dim=1)
        # dists_an = (weights_an * dists_an).sum(dim=1)
        
        # return F.softplus(dists_ap - dists_an).mean()
        

@LOSS_REGISTRY.register()
class global_center_loss(nn.Module):
    def __init__(self, num_clss, feat_dims, alpha=1.0, gamma=1.0, momentum=0.9):
        super().__init__()
        
        self.momentum = momentum
        self.alpha = alpha
        self.gamma = gamma
        
        self.register_buffer('share_ctrs', torch.zeros(num_clss, feat_dims))
        self.register_buffer('vis_ctrs', torch.zeros(num_clss, feat_dims))
        self.register_buffer('inf_ctrs', torch.zeros(num_clss, feat_dims))
        self.register_buffer('cls_targets', torch.arange(num_clss).to(dtype=torch.long))
        
        self.start_iter = 5000
        self.i = 0
        
    def forward(self, inputs, targets):
        '''
        # ! inputs: [V1, V1, V2, V2, I1, I1, I2, I2]
        # ! target: (bs) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        '''

        bs = inputs.size(0)
        mask = targets == targets[0]
        k = mask.sum().item()
        p = bs // k
        
        cs = inputs.view(2, p, k//2, -1).mean(2) # 2 x p x f
        local_vis_ctrs, local_inf_ctrs = cs[0], cs[1] # p x f
        local_share_ctrs = cs.mean(0) # p x f
        
        unique_targets = targets.view(2, p, k//2)[0,:,0]
        self.share_ctrs[unique_targets] = self.momentum * self.share_ctrs[unique_targets] + (1 - self.momentum) * local_share_ctrs
        self.vis_ctrs[unique_targets] = self.momentum * self.vis_ctrs[unique_targets] + (1 - self.momentum) * local_vis_ctrs
        self.inf_ctrs[unique_targets] = self.momentum * self.inf_ctrs[unique_targets] + (1 - self.momentum) * local_inf_ctrs
        
        if self.i >= self.start_iter:
            ctrs = torch.cat([self.vis_ctrs, self.inf_ctrs, self.share_ctrs], dim=0)
            
            s_vis = inputs @ self.vis_ctrs.t() # bs x num_clss
            s_inf = inputs @ self.inf_ctrs.t() # bs x num_clss
            s_share = inputs @ self.share_ctrs.t() # bs x num_clss
            dists = torch.cat([s_vis, s_inf, s_share], dim=1) # bs x 3*num_clss
            dists = (2 - 2 * dists).clamp(min=1e-12).sqrt() # bs x 3*num_clss
            
            pos_idxs = targets.unsqueeze(1) == self.cls_targets.unsqueeze(0).repeat(1, 3)
            dists_ap = rearrange(dists[pos_idxs], '(N k) -> N k ', N = bs, k = 3)
            dists_an = rearrange(dists[~pos_idxs], '(N k) -> N k ', N = bs)
            
            weights_ap = F.softmax(dists_ap * self.alpha, dim=1)
            weights_an = F.softmax(-dists_an * self.alpha, dim=1)
            
            dists_ap = (weights_ap * dists_ap).sum(dim=1)
            dists_an = (weights_an * dists_an).sum(dim=1)
            
            loss = F.softplus(self.gamma * (dists_ap - dists_an)).mean() \
                    + 1. / 3 * (F.relu(torch.norm(self.share_ctrs - self.vis_ctrs, dim=1) - 0.1) \
                        + F.relu(torch.norm(self.share_ctrs - self.inf_ctrs, dim=1) - 0.1) \
                            + F.relu(torch.norm(self.inf_ctrs - self.vis_ctrs, dim=1) - 0.1)).mean()
            
            loss = loss.mean()
        else:
            loss = torch.tensor([0.0]).mean().to(inputs.device)
            

        self.share_ctrs = self.share_ctrs.detach()
        self.vis_ctrs = self.vis_ctrs.detach()
        self.inf_ctrs = self.inf_ctrs.detach()
        
        self.i += 1
        
        return loss
        
        
        
    

@LOSS_REGISTRY.register()
class center_guided_pair_mining_loss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=0.2)

    def forward(self, inputs, targets):
        '''
        # ! inputs: [[V1, V1, V2, V2, I1, I1, I2, I2], [V1, V1, V2, V2, I1, I1, I2, I2]]
        # ! target: (bs) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        '''
        f, f_p = inputs
        
        bs = f.size(0)
        mask = targets == targets[0]
        k = mask.sum().item()
        p = bs // k
        
        classes = torch.arange(p).to(device=f.device, dtype=torch.long)
        mask = classes.unsqueeze(1) == classes.unsqueeze(0)

        cs = f.view(2, p, k//2, -1).mean(2) # 2 x p x f
        c_v, c_i = cs[0], cs[1] # p x f
        
        cs_p = f_p.view(2, p, k//2, -1).mean(2) # 2 x p x f
        c_vp, c_ip = cs_p[0], cs_p[1] # p x f
        
        dist_i_vp = (c_i - c_vp).pow(2).sum(1).clamp(min=1e-12).sqrt() # p
        dist_v_vp = (c_v - c_vp).pow(2).sum(1).clamp(min=1e-12).sqrt() # p
        dist_vp_vp_neg = (c_vp.unsqueeze(1) - c_vp.unsqueeze(0)).pow(2).sum(2).clamp(min=1e-12).sqrt() # p x p
        dist_vp_vp_neg = dist_vp_vp_neg[~mask].view(p, p-1).min(dim=1)[0] # p
        dist_v_v_neg = (c_v.unsqueeze(1) - c_v.unsqueeze(0)).pow(2).sum(2).clamp(min=1e-12).sqrt() # p x p
        dist_v_v_neg = dist_v_v_neg[~mask].view(p, p-1).min(dim=1)[0] # p
        
        loss1 = self.ranking_loss(dist_v_vp.detach(), dist_i_vp, torch.ones_like(dist_v_vp)) + \
                self.ranking_loss(dist_v_v_neg, dist_i_vp, torch.ones_like(dist_v_v_neg)) * 0.5 + \
                self.ranking_loss(dist_vp_vp_neg, dist_i_vp, torch.ones_like(dist_vp_vp_neg)) * 0.5
                
        dist_v_ip = (c_v - c_ip).pow(2).sum(1).clamp(min=1e-12).sqrt() # p
        dist_i_ip = (c_i - c_ip).pow(2).sum(1).clamp(min=1e-12).sqrt() # p
        dist_ip_ip_neg = (c_ip.unsqueeze(1) - c_ip.unsqueeze(0)).pow(2).sum(2).clamp(min=1e-12).sqrt() # p x p
        dist_ip_ip_neg = dist_ip_ip_neg[~mask].view(p, p-1).min(dim=1)[0] # p
        dist_i_i_neg = (c_i.unsqueeze(1) - c_i.unsqueeze(0)).pow(2).sum(2).clamp(min=1e-12).sqrt() # p x p
        dist_i_i_neg = dist_i_i_neg[~mask].view(p, p-1).min(dim=1)[0] # p

        loss2 = self.ranking_loss(dist_v_ip.detach(), dist_i_ip, torch.ones_like(dist_v_ip)) + \
                self.ranking_loss(dist_i_i_neg, dist_i_ip, torch.ones_like(dist_i_i_neg)) * 0.5 + \
                self.ranking_loss(dist_ip_ip_neg, dist_i_ip, torch.ones_like(dist_ip_ip_neg)) * 0.5
        
        return (loss1 + loss2)/2


@LOSS_REGISTRY.register()
class id_loss(nn.Module):
    def __init__(self, label_smoothing=0.0) :
        '''
        label_smoothing: float, default 0.0
            If greater than 0, smooth the labels by adding a small value to them.
            This can help to prevent overfitting.
            
            y_label_smoothing = (1 - label_smoothing) * y_onehot + label_smoothing / num_classes
        
        $$
        \begin{cases}
            \mathcal{L} _{ce}=-\frac{1}{bs}\sum_{i=1}^{bs}{\sum_{j=1}^c{y_{j}^{\left( i \right)}\log \frac{\exp \left( x_{j}^{\left( i \right)} \right)}{\sum_k{\exp \left( x_{k}^{\left( i \right)} \right)}}}}\\
            y_{j}^{\left( i \right)}=\left\{ \begin{aligned}
            1 or\,\,1-\epsilon &,  j=l^{\left( i \right)}\\
            0 or\,\,\frac{\epsilon}{c-1}&, j\ne l^{\left( i \right)}\\
        \end{aligned} \right.\\
        \end{cases}
        $$
        
        '''
        super().__init__()
        self.labal_smoothing = label_smoothing
        
    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=1)    
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= self.labal_smoothing / (input.size(1) - 1)
            targets.scatter_(1, target.data.unsqueeze(1), (1 - self.labal_smoothing))

        loss = (-targets * log_probs).sum(dim=1)
        if torch.isnan(loss.mean()):
            print(1)
        return loss.mean()
    
        
@LOSS_REGISTRY.register()
class triplet_loss(nn.Module):
    def __init__(self, margin=0.3, squared=False, normalize_embeddings=False):
        """
        $$
        \mathcal{L} _{triplet}=\begin{cases}
            \frac{1}{bs}\sum_{i=1}^{bs}{\left[ \max_{y_j=y_i} \,\,d\left( x_i,x_j \right) -\min_{y_k\ne y_i} \,\,d\left( x_i,x_k \right) +m \right] _+}, m>0\\
            \frac{1}{bs}\sum_{i=1}^{bs}{\log \left( 1+\exp \left( \max_{y_j=y_i} \,\,d\left( x_i,x_j \right) -\min_{y_k\ne y_i} \,\,d\left( x_i,x_k \right) \right) \right) , m=0}\\
        \end{cases}
        $$
        
        """
        super().__init__()
        
        self.margin = margin
        self.squared = squared
        self.normalize_embeddings = normalize_embeddings
        
    def forward(self, input, target):
        '''
            input: (N, D)
            target: (N) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        '''
        N, _ = input.size()
        
        if self.normalize_embeddings:
            input = F.normalize(input, p=2, dim=1)
        
        dists = input.pow(2).sum(dim=1, keepdim=True) +\
                input.pow(2).sum(dim=1, keepdim=True).t() -\
                2 * torch.mm(input, input.t())
        if not self.squared:
            dists = dists.clamp(min=1e-12).sqrt() # N x N
        
        pos_idxs = target == target.unsqueeze(1)
        dists_ap = rearrange(dists[pos_idxs], '(N k) -> N k ', N = N).max(dim=1)[0]
        dists_an = rearrange(dists[~pos_idxs], '(N k) -> N k ', N = N).min(dim=1)[0]
        
        if self.margin:
            loss = F.relu(dists_ap - dists_an + self.margin)
        else:
            loss = F.softplus(dists_ap - dists_an)
            
        if torch.isnan(loss.mean()):
            print(1)
        
        return loss.mean()


@LOSS_REGISTRY.register()
class wrt_loss(nn.Module):
    def __init__(self, alpha=1, gamma=1, squared_diff=False, squared=False, normalize_embeddings=False, excepted_self=False):
        """Weighted Triplet Loss
        Deep Learning for Person Re-identification: A Survey and Outlook
        Channel Augmented Joint Learning for Visible-Infrared Recognition (ICCV 2021)
        
        $$
        \begin{cases}
            \mathcal{L} _{triplet\_enhanced}=\frac{1}{bs}\sum_{i=1}^{bs}{\log \left( 1+\exp \left( \mu \left( \sum_{j:y_j=y_i}{w^+\left( x_i,x_j \right) d\left( x_i,x_j \right)}\,\,-\sum_{k:y_k=y_i}{w^-\left( x_i,x_k \right) d\left( x_i,x_k \right)} \right) \right) \right)}\\
            w^+\left( x_i,x_j \right) =\frac{\exp \left( d\left( x_i,x_j \right) \right)}{\sum_{k:y_k=y_i}{\exp \left( d\left( x_i,x_k \right) \right)}}\\
            w^-\left( x_i,x_j \right) =\frac{\exp \left( -d\left( x_i,x_j \right) \right)}{\sum_{k:y_k\ne y_i}{\exp \left( -d\left( x_i,x_k \right) \right)}}\\
            \mu \left( x \right) =\left\{ \begin{aligned}
            x&, squared\_diff=False\\
            \left\{ \begin{aligned}
            x^2&, x>0\\
            -x^2&, x\leqslant 0\\
        \end{aligned} \right. &, squared\_diff=True\\
        \end{aligned} \right.\\
        \end{cases}
        $$

        
        
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.squared_diff = squared_diff
        self.squared = squared
        self.normalize_embeddings = normalize_embeddings
        self.excepted_self = excepted_self
        
    def forward(self, input, target):
        '''
            input: (N, D)
            target: (N) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        '''
        N, _ = input.size()
        
        if self.normalize_embeddings:
            input = F.normalize(input, p=2, dim=1)
        
        dists = input.pow(2).sum(dim=1, keepdim=True) +\
                input.pow(2).sum(dim=1, keepdim=True).t() -\
                2 * torch.mm(input, input.t())
        if not self.squared:
            dists = dists.clamp(min=1e-12).sqrt() # N x N
            
        pos_idxs = target == target.unsqueeze(1)
        if self.excepted_self:
            _pos_idxs = pos_idxs & (~torch.eye(N, dtype=torch.bool, device=input.device))
            dists_ap = rearrange(dists[_pos_idxs], '(N k) -> N k ', N = N)
        else:
            dists_ap = rearrange(dists[pos_idxs], '(N k) -> N k ', N = N)
        dists_an = rearrange(dists[~pos_idxs], '(N k) -> N k ', N = N)
        
        weights_ap = F.softmax(dists_ap * self.alpha, dim=1)
        weights_an = F.softmax(-dists_an * self.alpha, dim=1)
        
        dists_ap = (weights_ap * dists_ap).sum(dim=1)
        dists_an = (weights_an * dists_an).sum(dim=1)
        
        if self.squared_diff:
            diff = dists_ap - dists_an
            diff_pow = diff.pow(2)
            diff_pow = torch.clamp(diff_pow, max=88.0)
            
            flag = diff > 0
            loss = (F.softplus(self.gamma * diff_pow[flag])).sum()
            loss += (F.softplus(- self.gamma * diff_pow[~flag])).sum()
            
            return loss / N
        else:
            loss = F.softplus(self.gamma * (dists_ap - dists_an))
            return loss.mean()
        
        
@LOSS_REGISTRY.register()
class part_reg_loss(nn.Module):
    def __init__(self, num_parts=2):
        """
        $$
        \mathcal{L} _{part\_reg}=\frac{1}{bs}\sum_{i=1}^{bs}{\frac{2}{p\left( p-1 \right)}\sum_{j=1}^p{\sum_{k=j+1}^p{f_{j}^{\left( i \right)}{f_{k}^{\left( i \right)}}^T}}}
        $$
        
        """
        super().__init__()
        
        self.num_parts = num_parts
        
    def forward(self, input, target):
        """
        input: (B, P, ...)
        
        """
        b = input.size(0)
        input = input.view(b, self.num_parts, -1)
        loss = torch.bmm(input, input.transpose(1, 2))#.abs()
        # loss = ((input.unsqueeze(2) - input.unsqueeze(1))**2).sum(dim=-1).clamp(min=1e-12).sqrt() # p x p
        loss = torch.triu(loss, diagonal=1).sum() / (b * self.num_parts * (self.num_parts - 1) / 2)
        return loss
    

@LOSS_REGISTRY.register()
class center_loss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        
    $$
    \mathcal{L} _{center}=\frac{1}{bs}\sum_{i=1}^{bs}{\left\| f^{\left( i \right)}-c^{\left( y_i \right)} \right\| _2}
    $$
    """
    # ! Set the nn.Parameter in the model!!!!
    def __init__(self, squared=True):
        super().__init__()
        self.squared = squared
        # ! ctrs = nn.Parameter(torch.randn(n_cls, feat_dim))

    def forward(self, input, target):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        x, ctrs = input # bs x feat_dim, n_cls x feat_dim
        
        bs = x.size(0)
        n_cls = ctrs.size(0)
        
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ctrs, 2).sum(dim=1, keepdim=True).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        if not self.squared:
            distmat = distmat.clamp(min=1e-12).sqrt()  # for numerical stability

        classes = torch.arange(n_cls).to(device=x.device, dtype=torch.long)
        target = target.unsqueeze(1).expand(bs, n_cls)
        mask = target.eq(classes.expand(bs, n_cls))

        loss = distmat[mask].mean()

        return loss
    
@LOSS_REGISTRY.register()
class center_cluster_loss(nn.Module):
    def __init__(self, margin=0, squared=False):
        super().__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, input, target):

        # input: (bs, f)
        # ! target: (bs) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 

        bs = input.size(0)
        mask = target == target[0]
        k = mask.sum().item()
        p = bs // k
        
        _input = input.view(2,p,k//2,-1).transpose(0,1).contiguous()
        _input = _input.view(p,k,-1) # p x k x f, V1,V1,I1,I1,V2,V2,I2,I2
        ctrs = _input.mean(1) # p x f
        # compute sample to centers
        _input = _input - ctrs.unsqueeze(1) # p x k x f
        dist_sc = _input.pow(2).sum(2) # p x k
        if not self.squared:
            dist_sc = dist_sc.clamp(min=1e-12).sqrt() # p x k
        loss_sc = dist_sc.mean()
        
        # compute centers to centers
        dist_cc = (ctrs.unsqueeze(1) - ctrs.unsqueeze(0)).pow(2).sum(2) # p x p
        if not self.squared:
            dist_cc = dist_cc.clamp(min=1e-12).sqrt() # p x p
        loss_cc = torch.triu((self.margin - dist_cc).clamp(min=0.0), diagonal=1).sum() / (p * (p - 1) / 2)
        
        # compute loss
        loss = loss_sc + loss_cc
        
        return loss
    
    
@LOSS_REGISTRY.register()
class kl_div_loss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        self.kldiv = nn.KLDivLoss(reduction=reduction)
        
    def forward(self, input, _):
        """
        # ! input: (log_pred, target)
        """
        log_pred, target = input
        loss = self.kldiv(log_pred, target)
        return loss
    
    
@LOSS_REGISTRY.register()
class center_guided_nuances_mining_loss(nn.Module):
    def __init__(self, m=0.2):
        super().__init__()
        
        self.m = m
        
    def forward(self, input, target):
        # ! target: (bs) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        f1, f2 = input
        
        bs = f1.size(0)
        mask = target == target[0]
        k = mask.sum().item()
        p = bs // k
        
        cs1 = f1.view(2, p, k//2, -1).mean(2) # 2 x p x f
        c_v1, c_i1 = cs1[0], cs1[1] # p x f
        
        cs2 = f2.view(2, p, k//2, -1).mean(2) # 2 x p x f
        c_v2, c_i2 = cs2[0], cs2[1] # p x f
        
        d_v1_i1 = (c_v1 - c_i1).pow(2).sum(1).clamp(min=1e-12).sqrt() # p
        d_v2_i2 = (c_v2 - c_i2).pow(2).sum(1).clamp(min=1e-12).sqrt() # p
        d_v1_v2 = (c_v1 - c_v2).pow(2).sum(1).clamp(min=1e-12).sqrt() # p
        d_i1_i2 = (c_i1 - c_i2).pow(2).sum(1).clamp(min=1e-12).sqrt() # p
        
        d_v1_v1 = (c_v1.unsqueeze(1) - c_v1.unsqueeze(0)).pow(2).sum(2).clamp(min=1e-12).sqrt() # p x p
        d_i1_i1 = (c_i1.unsqueeze(1) - c_i1.unsqueeze(0)).pow(2).sum(2).clamp(min=1e-12).sqrt() # p x p
        d_v2_v2 = (c_v2.unsqueeze(1) - c_v2.unsqueeze(0)).pow(2).sum(2).clamp(min=1e-12).sqrt() # p x p
        d_i2_i2 = (c_i2.unsqueeze(1) - c_i2.unsqueeze(0)).pow(2).sum(2).clamp(min=1e-12).sqrt() # p x p
        
        loss = 0.0
        loss += torch.triu(2 * d_v1_i1 - d_v1_v2 - d_v1_v1 + self.m, diagonal=1).sum() / (p * (p - 1) / 2)
        loss += torch.triu(2 * d_v2_i2 - d_v1_v2 - d_v2_v2 + self.m, diagonal=1).sum() / (p * (p - 1) / 2)
        loss += torch.triu(2 * d_v1_i1 - d_i1_i2 - d_i1_i1 + self.m, diagonal=1).sum() / (p * (p - 1) / 2)
        loss += torch.triu(2 * d_v2_i2 - d_i1_i2 - d_i2_i2 + self.m, diagonal=1).sum() / (p * (p - 1) / 2)
        loss /= 4.0
        
        return loss
    
    
@LOSS_REGISTRY.register()
class diverse_loss(nn.Module):
    def __init__(self, tau=0.01):
        super().__init__()
        
        self.tau = tau
        
    def forward(self, input, target):
        # input: logits of each view, [bxn_cls, ...]
        # ! target: (bs) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        n = len(input)
        bs, num_cls = input[0].size(0), input[0].size(1)
        
        
        # remove the target label
        dists_t = []
        mask = torch.ones_like(input[0], dtype=torch.bool)
        mask[torch.arange(bs), target] = False
        for i in range(n):
            logits = input[i][mask].view(bs, num_cls-1)
            dists_t.append(F.softmax(logits/self.tau, dim=1))
        dists_t = torch.stack(dists_t, dim=1) # bxnx(num_cls-1)    
        
        return torch.triu(torch.bmm(dists_t, dists_t.transpose(1,2)).clamp(min=1e-12), diagonal=1).sum() / (bs * n * (n - 1) / 2)
    
    
@LOSS_REGISTRY.register()
class cross_modality_triplet_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target):
        # ! target: (bs) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        
        bs, _ = input.size()
        
        dists = input.pow(2).sum(dim=1, keepdim=True) +\
                input.pow(2).sum(dim=1, keepdim=True).t() -\
                2 * torch.mm(input, input.t())
        dists = dists.clamp(min=1e-12).sqrt() # N x N
            
        pos_idxs = target == target.unsqueeze(1)
        mod_idxs = torch.cat([torch.zeros(bs//2, dtype=torch.long, device=input.device), torch.ones(bs//2, dtype=torch.long, device=input.device)])
        mod_idxs = mod_idxs == mod_idxs.unsqueeze(1)
        
        dists_ap = rearrange(dists[pos_idxs & (~mod_idxs)], '(N k) -> N k ', N = bs)
        dists_an = rearrange(dists[(~pos_idxs) & (~mod_idxs)], '(N k) -> N k ', N = bs)
        
        weights_ap = F.softmax(dists_ap, dim=1)
        weights_an = F.softmax(-dists_an, dim=1)
        
        dists_ap = (weights_ap * dists_ap).sum(dim=1)
        dists_an = (weights_an * dists_an).sum(dim=1)
        
        loss = F.softplus(dists_ap - dists_an)
        return loss.mean()
        
        
        

@LOSS_REGISTRY.register()
class triplet_loss_wrt2(nn.Module):
    def __init__(self, alpha=1, gamma=1, excepted_self=True, cross_modal=False):
        """Weighted Triplet Loss
        Deep Learning for Person Re-identification: A Survey and Outlook
        Channel Augmented Joint Learning for Visible-Infrared Recognition (ICCV 2021)
        
        $$
        \begin{cases}
            \mathcal{L} _{triplet\_enhanced}=\frac{1}{bs}\sum_{i=1}^{bs}{\log \left( 1+\exp \left( \mu \left( \sum_{j:y_j=y_i}{w^+\left( x_i,x_j \right) d\left( x_i,x_j \right)}\,\,-\sum_{k:y_k=y_i}{w^-\left( x_i,x_k \right) d\left( x_i,x_k \right)} \right) \right) \right)}\\
            w^+\left( x_i,x_j \right) =\frac{\exp \left( d\left( x_i,x_j \right) \right)}{\sum_{k:y_k=y_i}{\exp \left( d\left( x_i,x_k \right) \right)}}\\
            w^-\left( x_i,x_j \right) =\frac{\exp \left( -d\left( x_i,x_j \right) \right)}{\sum_{k:y_k\ne y_i}{\exp \left( -d\left( x_i,x_k \right) \right)}}\\
            \mu \left( x \right) =\left\{ \begin{aligned}
            x&, squared\_diff=False\\
            \left\{ \begin{aligned}
            x^2&, x>0\\
            -x^2&, x\leqslant 0\\
        \end{aligned} \right. &, squared\_diff=True\\
        \end{aligned} \right.\\
        \end{cases}
        $$
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.excepted_self = excepted_self
        self.cross_modal = cross_modal
        
    def forward(self, input, target):
        '''
            input: (N, D)
            target: (N) [1,1,2,2,1,1,2,2] - > [V1, V1, V2, V2, I1, I1, I2, I2] 
        '''
        N, _ = input.size()
        
        dists = input.pow(2).sum(dim=1, keepdim=True) +\
                input.pow(2).sum(dim=1, keepdim=True).t() -\
                2 * torch.mm(input, input.t())
        dists = dists.clamp(min=1e-12).sqrt() # N x N
        
        if self.cross_modal:
            pos_idxs = target == target.unsqueeze(1)
            mod_idxs = torch.cat([
                torch.zeros(N//2, dtype=torch.long, device=input.device), 
                torch.ones(N//2, dtype=torch.long, device=input.device)])
            mod_idxs = mod_idxs == mod_idxs.unsqueeze(1)
            
            if self.excepted_self:
                _pos_idxs = pos_idxs & (~torch.eye(N, dtype=torch.bool, device=input.device))
                dists_ap_intra = rearrange(dists[_pos_idxs & mod_idxs], '(N k) -> N k ', N = N)
                dists_an_intra = rearrange(dists[(~pos_idxs) & mod_idxs], '(N k) -> N k ', N = N)
                
                dists_ap_inter = rearrange(dists[_pos_idxs & (~mod_idxs)], '(N k) -> N k ', N = N)
                dists_an_inter = rearrange(dists[(~pos_idxs) & (~mod_idxs)], '(N k) -> N k ', N = N)
            else:
                dists_ap_intra = rearrange(dists[pos_idxs & mod_idxs], '(N k) -> N k ', N = N)
                dists_an_intra = rearrange(dists[(~pos_idxs) & mod_idxs], '(N k) -> N k ', N = N)
                
                dists_ap_inter = rearrange(dists[pos_idxs & (~mod_idxs)], '(N k) -> N k ', N = N)
                dists_an_inter = rearrange(dists[(~pos_idxs) & (~mod_idxs)], '(N k) -> N k ', N = N)
            
            weights_ap_intra = F.softmax(dists_ap_intra * self.alpha, dim=1)
            weights_an_intra = F.softmax(-dists_an_intra * self.alpha, dim=1)
            
            weights_ap_inter = F.softmax(dists_ap_inter * self.alpha, dim=1)
            weights_an_inter = F.softmax(-dists_an_inter * self.alpha, dim=1)
            
            dists_ap_intra = (weights_ap_intra * dists_ap_intra).sum(dim=1)
            dists_an_intra = (weights_an_intra * dists_an_intra).sum(dim=1)
            
            dists_ap_inter = (weights_ap_inter * dists_ap_inter).sum(dim=1)
            dists_an_inter = (weights_an_inter * dists_an_inter).sum(dim=1)
            
            loss = 0.5 * F.softplus(self.gamma * (dists_ap_intra - dists_an_intra)) + \
                   0.5 * F.softplus(self.gamma * (dists_ap_inter - dists_ap_intra))
            return loss.mean()
            
        else:
            pos_idxs = target == target.unsqueeze(1)
            if self.excepted_self:
                mod_idxs = torch.cat([
                        torch.zeros(N//2, dtype=torch.long, device=input.device), 
                        torch.ones(N//2, dtype=torch.long, device=input.device)])
                mod_idxs = mod_idxs == mod_idxs.unsqueeze(1)
                
                _pos_idxs = pos_idxs & (~torch.eye(N, dtype=torch.bool, device=input.device))
                dists_ap = rearrange(dists[_pos_idxs], '(N k) -> N k ', N = N)
                
                dists_ap_intra = rearrange(dists[_pos_idxs & mod_idxs], '(N k) -> N k ', N = N)
                dists_ap_inter = rearrange(dists[_pos_idxs & (~mod_idxs)], '(N k) -> N k ', N = N)
            else:
                dists_ap = rearrange(dists[pos_idxs], '(N k) -> N k ', N = N)
            dists_an = rearrange(dists[~pos_idxs], '(N k) -> N k ', N = N)
            
            weights_ap = F.softmax(dists_ap * self.alpha, dim=1)
            weights_an = F.softmax(-dists_an * self.alpha, dim=1)
            
            weights_ap_intra = F.softmax(dists_ap_intra * self.alpha, dim=1)
            weights_ap_inter = F.softmax(dists_ap_inter * self.alpha, dim=1)
            
            dists_ap = (weights_ap * dists_ap).sum(dim=1)
            dists_an = (weights_an * dists_an).sum(dim=1)
            
            dists_ap_intra = (weights_ap_intra * dists_ap_intra).sum(dim=1)
            dists_ap_inter = (weights_ap_inter * dists_ap_inter).sum(dim=1)
            
            loss = F.softplus(self.gamma * (dists_ap - dists_an)) + F.relu(torch.abs(dists_ap_intra - dists_ap_inter) - 0.1)
            return loss.mean()