import os
import re
import os.path as osp
from glob import glob
import numpy as np
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

'''
    Specific dataset classes for person re-identification dataset. 
'''


class SYSUDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        if mode == 'train':
            train_ids = open(os.path.join(root, 'exp', 'train_id.txt')).readline()
            val_ids = open(os.path.join(root, 'exp', 'val_id.txt')).readline()

            train_ids = train_ids.strip('\n').split(',')
            val_ids = val_ids.strip('\n').split(',')
            selected_ids = train_ids + val_ids
        else:
            test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
            selected_ids = test_ids.strip('\n').split(',')

        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)

        img_paths = glob(os.path.join(root, '**/*.jpg'), recursive=True)
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]

        if mode == 'gallery':
            # RGB imgs
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]
        elif mode == 'query':
            # IR imgs
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            # map the selected ids to 0~num_ids for training
            self.id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [self.id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

        # lazy load
        self.index2img = {}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = self.__getImage(item)
        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)
        
        if self.transform is not None:
            if isinstance(self.transform, list):
                if cam.item() in (1, 2, 4, 5):
                    img = self.transform[0](img)
                elif cam.item() in (3, 6):
                    img = self.transform[1](img)
                else:
                    raise ValueError('Invalid camera id: {}'.format(cam.item()))
            else:
                img = self.transform(img)

        return img, label, cam, path, item

    def __getImage(self, index):
        if index not in self.index2img:
            path = self.img_paths[index]
            img = Image.open(path)
            self.index2img[index] = img
        else:
            img = self.index2img[index]

        return img
    

class RegDBDataset(Dataset):
    def __init__(self, root, mode='train', transform=None, split_num='10'):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        num = split_num
        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/train_thermal_'+num+'.txt','r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/test_thermal_'+num+'.txt','r'))

        if mode == 'gallery':
            img_paths = [root + '/' + path for path, _ in index_RGB]
        elif mode == 'query':
            img_paths = [root + '/' + path for path, _ in index_IR]
        else:
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]

        selected_ids = [int(path.split('/')[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'Thermal') + 2 for path in img_paths] 
        # the visible cams are 1 2 4 5 and thermal cams are 3 6 in sysu
        # to simplify the code, visible cam is 2 and thermal cam is 3 in regdb
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item
    
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

class LLCMDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']
        
        if mode == 'train':
            # Load training images (path) and labels
            train_color_list   = os.path.join(root, 'idx/train_vis.txt')
            train_thermal_list = os.path.join(root, 'idx/train_nir.txt')

            color_img_file, train_color_label = load_data(train_color_list)
            thermal_img_file, train_thermal_label = load_data(train_thermal_list)
            
            self.img_paths = [os.path.join(root, color_img_file[i]) for i in range(len(color_img_file))] +\
                             [os.path.join(root, thermal_img_file[i]) for i in range(len(thermal_img_file))]
            self.ids = train_color_label + train_thermal_label
            
            # self.cam_ids = [int(os.path.basename(fn).split('c')[1][:2]) for fn in self.img_paths]
            self.cam_ids = [2] * len(train_color_label) + [3] * len(train_thermal_label)
        else:
            if mode == 'gallery':
                # RGB imgs
                cameras = ['test_vis/cam1','test_vis/cam2','test_vis/cam3','test_vis/cam4','test_vis/cam5','test_vis/cam6','test_vis/cam7','test_vis/cam8','test_vis/cam9']
            else:
                # IR imgs
                cameras = ['test_nir/cam1','test_nir/cam2','test_nir/cam4','test_nir/cam5','test_nir/cam6','test_nir/cam7','test_nir/cam8','test_nir/cam9']
                
            self.img_paths = []
            self.ids = []
            self.cam_ids = []
            
            self.random_idxs = []
            
            file_path = os.path.join(root,'idx/test_id.txt')
            
            with open(file_path, 'r') as file:
                ids = file.read().splitlines()
                ids = [int(y) for y in ids[0].split(',')]
                ids = ["%04d" % x for x in ids]
                
            for id in sorted(ids):
                for cam in cameras:
                    img_dir = os.path.join(root,cam,id)
                    if os.path.isdir(img_dir):
                        new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                        
                        # During the testing stage, for each camera, we randomly choose one image
                        # from the images of each identity to form the gallery set for evaluation
                        # the performance of the models. We repeat the above evaluation 10 times with
                        # random split of the gallery set and report the average performance.
                        self.random_idxs.append([
                            len(self.img_paths) + random.randint(0, len(new_files)-1) for _ in range(10)
                        ])
                        
                        for img_path in new_files:
                            camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
                            
                            self.img_paths.append(img_path)
                            self.cam_ids.append(camid)
                            self.ids.append(pid)
                            
            self.random_idxs = np.array(self.random_idxs) # (num_ids, 10)

        
        self.num_ids = np.unique(self.ids).shape[0]
        self.transform = transform
        
        # lazy load
        self.index2img = {}
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, item):
        path = self.img_paths[item]
        
        img = self.__getImage(item)
        if self.transform is not None:
            img = self.transform(img)
            
        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)
        
        return img, label, cam, path, item
    
    def __getImage(self, index):
        if index not in self.index2img:
            path = self.img_paths[index]
            img = Image.open(path)
            self.index2img[index] = img
        else:
            img = self.index2img[index]

        return img

