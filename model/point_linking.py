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

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Fourbranch(nn.Module):

    def __init__(self):
        super(Fourbranch, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1536, 1536, kernel_size=3, stride=1)
            BasicConv2d(1536, 204, kernel_size=3, stride=1)
            nn.Conv2d(in_channels, out_channels, 3, stride=(1, 1, 1, 1, 1, 1, 1), padding=(2, 2, 4, 8, 16, 1, 1), dilation=(2, 2, 4, 8, 16, 1, 1), bias=True)
       )
 
        self.branch1 = nn.Sequential(
            BasicConv2d(1536, 1536, kernel_size=3, stride=1)
            BasicConv2d(1536, 204, kernel_size=3, stride=1)
            nn.Conv2d(in_channels, out_channels, (3, 3, 3, 3, 3, 3), stride=(1, 1, 1, 1, 1, 1, 1), padding=(2, 2, 4, 8, 16, 1, 1), dilation=(2, 2, 4, 8, 16, 1, 1), bias=True)
       )

        self.branch2 = nn.Sequential(
            BasicConv2d(1536, 1536, kernel_size=3, stride=1)
            BasicConv2d(1536, 204, kernel_size=3, stride=1)
            nn.Conv2d(in_channels, out_channels, (3, 3, 3, 3, 3, 3), stride=(1, 1, 1, 1, 1, 1, 1), padding=(2, 2, 4, 8, 16, 1, 1), dilation=(2, 2, 4, 8, 16, 1, 1), bias=True)
       )

        self.branch3 = nn.Sequential(
            BasicConv2d(1536, 1536, kernel_size=3, stride=1)
            BasicConv2d(1536, 204, kernel_size=3, stride=1)
            nn.Conv2d(in_channels, out_channels, (3, 3, 3, 3, 3, 3), stride=(1, 1, 1, 1, 1, 1, 1), padding=(2, 2, 4, 8, 16, 1, 1), dilation=(2, 2, 4, 8, 16, 1, 1), bias=True)
       )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return torch.stack([x0, x1, x2, x3], 1)

class Point_Linking(nn.Module):
    
    def __init__(self, inception_V2, dilation, inference):
        super(Point_Linking, self).__init__()
        self.inception_V2 = inception_V2
        self.fourbranch = Fourbranch()
        self.inference = inference
        
        self.grid_size = 14
        self.classes = 20

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = t.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = t.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = t.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]
        grid_size  = x.size(3)
        f = self.inception_V2(x)
        four_out = self.fourbranch(f)
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda) 
        return four_out

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
        link_mnst = list()
        for img, size in zip(prepared_imgs, sizes):
            four_out = self(img)
            for i in range(4):
                m = 
                four_out[i][:, :, 0: 1]
                four_out[i][:, :, 1: 21]
                four_out[i][:, :, 21: 23]
                four_out[i][:, :, 23: 37]
                four_out[i][:, :, 37: 51]
                for t in range(self.grid_size):
                    for s in range(self.grid_size):
                        for n in range(self.grid_size):
                            for m in range(self.grid_size):
                                for c in range(self.classes):
                                link_mnst[c*14*14*14*21+t*14*14*14+s*14*14+n*14+m] = p_mn*p_st*q_cmn*q_cst*(l_mn_s*l_mn_t+l_st_m*l_st_n)/2
            	r = t.argmax(link_mnst)
            	m_ = r%14
            	n_ = r//14%14
            	s_ = r//14//14%14
            	t_ = r//14//14//14%14
            	c_ = r//14//14//14//21%21

            existencess.append(existences)
            scoress.append(scores)
            xy_positionss.append(xy_positions)
            linkss.append(links)
       self.use_preset('evaluate')
       self.train()
       return existencess, scoress, xy_positionss, linkss
    
    '''def inference(self, link_mnst):
        m_ = r%14
        n_ = r\14%14
        s_ = r\14\14%14
        t_ = r\14\14\14%14
        c_ = r\14\14\14\21%21
    '''
    def use_preset(self, preset):
        """Use the given preset during prediction.
        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.
        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.
        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.
        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

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

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
