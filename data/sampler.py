import numpy as np

import copy
from torch.utils.data import Sampler
from collections import defaultdict


class EvenlyRandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        self.index_dic_V = defaultdict(list)
        self.index_dic_I = defaultdict(list)
        N_V, N_I = 0, 0
        for i, identity in enumerate(data_source.ids):
            if data_source.cam_ids[i] in [3, 6]:
                self.index_dic_I[identity].append(i)
                N_I += 1
            else:
                self.index_dic_V[identity].append(i)
                N_V += 1
        self.pids = list(self.index_dic_I.keys())
        self.N = max(N_V, N_I)
        self.iters_num = int(self.N / (batch_size // 2)) + 1
        # estimate number of examples in an epoch
        self.length = self.iters_num * self.batch_size
        
    def __iter__(self):
        for _ in range(self.iters_num):
            batch_idxs_V = []
            batch_idxs_I = []
            selected_pids = np.random.choice(self.pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs_V.extend(self._sample(self.index_dic_V[pid], self.num_instances // 2))
                batch_idxs_I.extend(self._sample(self.index_dic_I[pid], self.num_instances // 2))
            batch_idxs_V.extend(batch_idxs_I)
            yield from batch_idxs_V
            
    def __len__(self):
        return self.length
    
    @staticmethod
    def _sample(idxs, num):
        if len(idxs) > num:
            return np.random.choice(idxs, size=num, replace=False)
        else:
            n = num // len(idxs)
            out_idxs = []
            for _ in range(n):
                out_idxs.extend(idxs)
            if num - n*len(idxs) > 0:
                _idxs = np.random.choice(idxs, size=num - n*len(idxs), replace=False)
                out_idxs.extend(_idxs)
            np.random.shuffle(out_idxs)
            return out_idxs
            
    
class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_V = defaultdict(list)
        self.index_dic_I = defaultdict(list)
        for i, identity in enumerate(data_source.ids):
            if data_source.cam_ids[i] in [3, 6]:
                self.index_dic_I[identity].append(i)
            else:
                self.index_dic_V[identity].append(i)
        self.pids = list(self.index_dic_I.keys())

        # estimate number of examples in an epoch
        self.length = 0

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs_V = copy.deepcopy(self.index_dic_V[pid])
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            
            N_min = min(len(idxs_I), len(idxs_V))
            N_min = int(np.ceil(N_min / (self.num_instances // 2))) * (self.num_instances // 2)
            
            idxs_V = self._complete(idxs_V, N_min)
            idxs_I = self._complete(idxs_I, N_min)

            np.random.shuffle(idxs_V)
            np.random.shuffle(idxs_I)
            
            batch_idxs = []
            for n in range(N_min // (self.num_instances // 2)):
                batch_idxs.extend(idxs_V[n*self.num_instances//2:(n+1)*self.num_instances//2])
                batch_idxs.extend(idxs_I[n*self.num_instances//2:(n+1)*self.num_instances//2])
                batch_idxs_dict[pid].append(batch_idxs) #[[V, V, V, V, I, I, I, I],[...]]
                batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            # convert [V1, V1, I1, I1, V2, V2, I2, I2] to [V1, V1, V2, V2, I1, I1, I2, I2] 
            tmp_idxs_V = []
            tmp_idxs_I = []
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0) #[V, V, V, V, I, I, I, I]
                tmp_idxs_V.extend(batch_idxs[:self.num_instances//2])
                tmp_idxs_I.extend(batch_idxs[self.num_instances//2:])
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
            final_idxs.extend(tmp_idxs_V)
            final_idxs.extend(tmp_idxs_I)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
    
    def _complete(self, idxs, N):
        if len(idxs) < N:
            out_idxs = copy.deepcopy(idxs)
            _idxs = np.random.choice(idxs, size=N - len(idxs), replace=True)
            out_idxs.extend(_idxs)
        else:
            out_idxs = np.random.choice(idxs, size=N, replace=False)
        return out_idxs

