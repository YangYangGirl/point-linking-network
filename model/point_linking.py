from __future__ import  absolute_import
from __future__ import division
import torch as t
import numpy as np
import cupy as cp
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
from model.utils.nms import non_maximum_suppression

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import opt

class Point_Linking(nn.Module):
    
    def __init__(self, inception_V2, dilation, inference)


    def predict(self, imgs, size=None, visualize=False):
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
             prepared_imgs = imgs
        existencess = list()
        #labels = list()
        scoress = list()        
        xy_positionss = list()
        linkss = list() 
        for img, size in zip(prepared_imgs, sizes):
            existences, scores, xy_positions, links = self(img, scale=scale)
            existencess.append(existences)
            scoress.append(scores)
            xy_positionss.append(xy_positions)
            linkss.append(links)
       self.use_preset('evaluate')
       self.train()
       return existencess, scoress, xy_positionss, linkss

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
            
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        elif opt.use_RMSProp:
            self.optimizer = t.optim.RMSProp(params, alpha=0.9)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer
