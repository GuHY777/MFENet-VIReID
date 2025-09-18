from .rank_cylib import evaluate_rank
import logging
from tabulate import tabulate
import torch
import torch.nn.functional as F
import numpy as np
from .faiss_utils import faiss_search
from tqdm import tqdm
import os.path as osp
import scipy.io as sio
from collections import defaultdict


logger = logging.getLogger(__name__)


def get_gallery_names(perm, cams, ids, trial_id, num_shots=1):
    names = []
    for cam in cams:
        cam_perm = perm[cam - 1][0].squeeze()
        for i in ids:
            if (i - 1) < len(cam_perm) and len(cam_perm[i - 1]) > 0:
                instance_id = cam_perm[i - 1][trial_id][:num_shots]
                names.extend(['cam{}/{:0>4d}/{:0>4d}'.format(cam, i, ins) for ins in instance_id.tolist()])
    return names

class Evaluator:
    def __init__(self, args, model, dls, tb_writer=None):
        self.use_cython = args.use_cython
        self.search_option = args.search_option
        self.dataroot = args.dataroot
        self.dataset = args.dataset
        self.split_num = args.split_num
        self.dist_metric = args.dist_metric
        self.test_flip = args.test_flip
        if args.dataset == 'sysu':
            self.eval_metric = 'all-1'
        elif args.dataset =='regdb':
            self.eval_metric = 'Infrared->Visible'
        elif args.dataset == 'llcm':
            self.eval_metric = 'Infrared->Visible'
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        self.dls = dls
        self.model = model
        
        self.tb_writer = tb_writer
        self.header = ['Epoch', 'mAP', 'Rank-1', 'Rank-10', 'Rank-20', 'mINP']
        self.results = defaultdict(list)
        
        if self.dataset =='sysu':
            logger.info(f"\n# --- SYSU-MM01 Evaluator initialized --- #")
        elif self.dataset =='regdb':
            logger.info(f"\n# --- RegDB-{self.split_num} Evaluator initialized --- #")
        elif self.dataset == 'llcm':
            logger.info(f"\n# --- LLCM Evaluator initialized --- #")
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        logger.info(f'Cython={self.use_cython}\nSearch option={self.search_option}\nMetric={self.dist_metric}\nTest flip={self.test_flip}')
        
        
    @torch.no_grad()
    def __call__(self, epoch):
        q_featss, q_pids, q_camids, q_img_paths = self.extract_features(0)
        g_featss, g_pids, g_camids, g_img_paths = self.extract_features(1)
        
        if isinstance(q_featss, list):
            out_ = None
            for feat_idx, (q_feats, g_feats) in enumerate(zip(q_featss, g_featss)):
                if feat_idx == 1: # pooled + bn
                    out_ = self._processing(epoch, q_feats, q_pids, q_camids, q_img_paths, g_feats, g_pids, g_camids, g_img_paths, feat_idx=str(feat_idx))
                else:
                    self._processing(epoch, q_feats, q_pids, q_camids, q_img_paths, g_feats, g_pids, g_camids, g_img_paths, feat_idx=str(feat_idx))
            return out_
        else:
            assert isinstance(q_featss, torch.Tensor)
            return self._processing(epoch, q_featss, q_pids, q_camids, q_img_paths, g_featss, g_pids, g_camids, g_img_paths)
    
    def _processing(self, epoch, q_feats, q_pids, q_camids, q_img_paths, g_feats, g_pids, g_camids, g_img_paths, feat_idx=''):
        if self.dist_metric == 'cosine':
            q_feats = F.normalize(q_feats, dim=1)
            g_feats = F.normalize(g_feats, dim=1)
        
        pre_fix = f'Feat_{feat_idx}: ' if feat_idx else ''
        
        if self.dataset =='sysu':
            perm = sio.loadmat(osp.join(self.dataroot, 'SYSU-MM01/exp', 'rand_perm_cam.mat'))[
                'rand_perm_cam']
            
            for exp_name in ['all-1']:#, 'all-10', 'indoor-1', 'indoor-10']:
                mode, num_shots = [x if i == 0 else int(x) for i, x in enumerate(exp_name.split('-'))]
                mAP, mINP, cmc = self.eval_sysu(q_feats, q_pids, q_camids, g_feats, g_pids, g_camids, g_img_paths, 
                                        perm, mode, num_shots, num_trials=10)
                self._show(pre_fix+exp_name, epoch, mAP, cmc, mINP)
            return self.results[pre_fix+self.eval_metric][-1]
        elif self.dataset =='regdb':
            for exp_name in ['Infrared->Visible', 'Visible->Infrared']:
                mAP, mINP, cmc = self.eval_regdb(q_feats, q_pids, q_camids, g_feats, g_pids, g_camids, exp_name)
                self._show(pre_fix+exp_name, epoch, mAP, cmc, mINP)
            return self.results[pre_fix+self.eval_metric][-1]
        elif self.dataset == 'llcm':
            for exp_name in ['Infrared->Visible', 'Visible->Infrared']:
                mAP, mINP, cmc = self.eval_llcm(q_feats, q_pids, q_camids, q_img_paths,
                                          g_feats, g_pids, g_camids, g_img_paths,
                                          exp_name, num_trials=10)
                self._show(pre_fix+exp_name, epoch, mAP, cmc, mINP)
            return self.results[pre_fix+self.eval_metric][-1]
    
    
    def _show(self, exp_name, epoch, mAP, cmc, mINP):
        self.results[exp_name].append([
                    epoch,
                    mAP,
                    cmc[0],
                    cmc[9],
                    cmc[19],
                    mINP
                ])
                
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(f"Metrics_{exp_name}", {
                'mAP': mAP,
                'Rank-1': cmc[0], 
                'Rank-9': cmc[9], 
                'Rank-19': cmc[19], 
                'mINP': mINP,
            }, epoch)
        
        logger.info(f"\nResults ({exp_name}):\n{tabulate(self.results[exp_name], headers=self.header, tablefmt='orgtbl')}")
    
    def extract_features(self, idx=0):
        torch.cuda.empty_cache()
        self.model.eval()
        
        feats, pids, camids, img_paths = [], [], [], []
        N = int(np.ceil(self.dls[idx].dataset.__len__() / self.dls[idx].batch_size))     
        
        for (img, label, cam_id, img_path, _) in tqdm(self.dls[idx], desc=f"Extracting {'query' if idx==0 else 'gallery'} features", total=N):
            pids.append(label)
            camids.append(cam_id)
            img_paths.append(img_path)
            
            img = img.to('cuda')
            cam_id = cam_id.to('cuda')
            
            f = self.model(img, cam_id)
            if self.test_flip:
                f1 = self.model(torch.flip(img, [3]), cam_id)
                if isinstance(f, list|tuple):
                    f = [(_f + _f1) / 2 for _f, _f1 in zip(f, f1)]
                else:
                    f = (f + f1) / 2
            feats.append(f)
        
        if isinstance(feats[0], list|tuple):
            _feats = []
            for i in range(len(feats[0])):
                _feats.append(torch.cat([_f[i] for _f in feats], dim=0).detach().to('cpu'))
            feats = _feats
        else:
            feats = torch.cat(feats, dim=0).detach().to('cpu')
        pids = torch.cat(pids, dim=0).detach().to('cpu').numpy()
        camids = torch.cat(camids, dim=0).detach().to('cpu').numpy()
        img_paths = np.concatenate(img_paths, axis=0)
        
        return feats, pids, camids, img_paths
    
    def eval_sysu(self, q_feats, q_pids, q_camids, g_feats, g_pids, g_camids, g_img_paths,
                  perm, mode='all', num_shots=1, num_trials=10):
        gallery_cams = [1, 2] if mode == 'indoor' else [1, 2, 4, 5]
        gallery_indices = np.in1d(g_camids, gallery_cams)
        
        g_feats = g_feats[gallery_indices]
        g_pids = g_pids[gallery_indices]
        g_camids = g_camids[gallery_indices]
        g_img_paths = g_img_paths[gallery_indices]
        
        gallery_names = np.array(['/'.join(osp.splitext(path)[0].split('/')[-3:]) for path in g_img_paths])
        # cam1/0002/0001
        gallery_pids_set = np.unique(g_pids)
        
        mAP, mINP, cmc = 0, 0, np.zeros(20)
        for t in range(num_trials):
            names = get_gallery_names(perm, gallery_cams, gallery_pids_set, t, num_shots)
            flag = np.in1d(gallery_names, names)
            # assert flag.sum() == num_shots * len(gallery_pids_set) * len(gallery_cams)
            
            _g_feats = g_feats[flag]
            _g_pids = g_pids[flag]
            _g_camids = g_camids[flag]
            
            I = faiss_search(q_feats, _g_feats, -1, self.search_option)
            all_cmc, all_AP, all_INP = evaluate_rank(
                I, q_pids, _g_pids, q_camids, _g_camids, sysu=True, max_rank=20, new_cmc=True, # SYSU-MM01 uses new_cmc=True
            )
            mAP += np.mean(all_AP)
            mINP += np.mean(all_INP)
            cmc += all_cmc[:20]
            
        mAP /= num_trials / 100.
        mINP /= num_trials / 100.
        cmc /= num_trials / 100.
        
        return mAP, mINP, cmc
    
    def eval_regdb(self, q_feats, q_pids, q_camids, g_feats, g_pids, g_camids, mode='Infrared->Visible'):
        if mode == 'Infrared->Visible':
            I = faiss_search(q_feats, g_feats, -1, self.search_option)
            all_cmc, all_AP, all_INP = evaluate_rank(
                I, q_pids, g_pids, q_camids, g_camids, max_rank=20,# new_cmc=True,
            )
        else:
            I = faiss_search(g_feats, q_feats, -1, self.search_option)
            all_cmc, all_AP, all_INP = evaluate_rank(
                I, g_pids, q_pids, g_camids, q_camids, max_rank=20,# new_cmc=True,
            )
        return np.mean(all_AP) * 100., np.mean(all_INP) * 100., all_cmc * 100.
            
        
    def eval_llcm(self, q_feats, q_pids, q_camids, q_img_paths, g_feats, g_pids, g_camids, g_img_paths,
                  exp_name='Infrared->Visible', num_trials=10):

        if exp_name == 'Infrared->Visible':
            random_idxs = self.dls[1].dataset.random_idxs
            
            mAP, mINP, cmc = 0, 0, np.zeros(20)
            for trial in range(num_trials):
                gallery_indices = np.in1d(np.arange(len(g_img_paths)), random_idxs[:,trial])
                
                _g_feats = g_feats[gallery_indices]
                _g_pids = g_pids[gallery_indices]
                _g_camids = g_camids[gallery_indices]
                
                I = faiss_search(q_feats, _g_feats, -1, self.search_option)
                all_cmc, all_AP, all_INP = evaluate_rank(
                    I, q_pids, _g_pids, q_camids, _g_camids, sysu=False, max_rank=20, new_cmc=True, # LLCM uses new_cmc=True
                )
                mAP += np.mean(all_AP)
                mINP += np.mean(all_INP)
                cmc += all_cmc[:20]
                
            mAP /= num_trials / 100.
            mINP /= num_trials / 100.
            cmc /= num_trials / 100.
        else:
            random_idxs = self.dls[0].dataset.random_idxs
            
            mAP, mINP, cmc = 0, 0, np.zeros(20)
            for trial in range(num_trials):
                gallery_indices = np.in1d(np.arange(len(q_img_paths)), random_idxs[:,trial])
                
                _g_feats = q_feats[gallery_indices]
                _g_pids = q_pids[gallery_indices]
                _g_camids = q_camids[gallery_indices]
                
                I = faiss_search(g_feats, _g_feats, -1, self.search_option)
                all_cmc, all_AP, all_INP = evaluate_rank(
                    I, g_pids, _g_pids, g_camids, _g_camids, sysu=False, max_rank=20, new_cmc=True, # LLCM uses new_cmc=True
                )
                mAP += np.mean(all_AP)
                mINP += np.mean(all_INP)
                cmc += all_cmc[:20]
                
            mAP /= num_trials / 100.
            mINP /= num_trials / 100.
            cmc /= num_trials / 100.
        
        
        
        return mAP, mINP, cmc
            
        
        
        
