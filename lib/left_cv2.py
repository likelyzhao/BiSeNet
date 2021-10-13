#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import lib.transform_cv2 as T
from lib.base_dataset import BaseDataset


labels_info = [
    {"hasInstances": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 0},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "leftobj", "ignoreInEval": False, "id": 1, "color": [0, 0, 0], "trainId": 1},
]



class Leftobj(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Leftobj, self).__init__(
                dataroot, annpath, trans_func, mode)
        print('dataroot', dataroot)
        print('annpath', annpath)
        self.n_cats = 2
        self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']

        print(self.lb_map)
        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223), # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )





if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = Leftobj('/data/zhaozhijian/BiSeNet/datasets/left/', annpath='/data/zhaozhijian/BiSeNet/datasets/left/list_all.txt', mode='train')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
