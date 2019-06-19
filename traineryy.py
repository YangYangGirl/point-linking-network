from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from torch import nn
import torch as t

from utils.config import opt

LossTuple = namedtuple('LossTuple',
                        ['pt_loc_loss',
                        'nopt_loss',
                        'total_loss'
                        ])

class PointLinkTrainer(nn.Module):
    def __init__(self, point_link):
        super(PointLinkTrainer, self).__init__()
        
        self.point_link = point_link
        self.grid_size = t.tensor(14)
        
        self.optimizer = self.point_link.get_optimizer()
        
    def forward(self, imgs, p, q, x, y, bboxes, labels, ):
        _, _, H, W = imgs.shape
        img_size = (H, W)
        out = self.point_link(imgs)
        gt = t.ones([14, 14, 204])
        for i in range(self.grid_size):
            for j in range(self.grid_size):
        for b in bboxes:
            gt[int(b[0]/W*self.grid_size), int(b[1]/H*self.grid_size), 0] = 1
            gt[int((b[0]+b[2])/W*self.grid_size), int(b[1]/H*self.grid_size), 0] = 1
            gt[int(b[0]/W*self.grid_size), int((b[1]+b[3])/H*self.grid_size), 0] = 1
            gt[int((b[0]+b[2])/W*self.grid_size), int((b[1]+b[3])/H*self.grid_size), 0] = 1
            

