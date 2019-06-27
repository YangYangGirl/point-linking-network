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
        self.B =2

        self.branch0 = nn.Sequential(
            BasicConv2d(1536, 1536, kernel_size=3, stride=1, padding=1),
            BasicConv2d(1536, 204, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(204, 204, 3, stride=1, padding=2, dilation=2, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=2, dilation=2, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=4, dilation=4, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=8, dilation=8, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=16, dilation=16, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=1, dilation=1, bias=True),
            
       ) 
 
        self.branch1 = nn.Sequential(
            BasicConv2d(1536, 1536, kernel_size=3, stride=1, padding=1),
            BasicConv2d(1536, 204, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(204, 204, 3, stride=1, padding=2, dilation=2, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=2, dilation=2, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=4, dilation=4, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=8, dilation=8, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=16, dilation=16, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=1, dilation=1, bias=True),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1536, 1536, kernel_size=3, stride=1, padding=1),
            BasicConv2d(1536, 204, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(204, 204, 3, stride=1, padding=2, dilation=2, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=2, dilation=2, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=4, dilation=4, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=8, dilation=8, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=16, dilation=16, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=1, dilation=1, bias=True),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(1536, 1536, kernel_size=3, stride=1, padding=1),
            BasicConv2d(1536, 204, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(204, 204, 3, stride=1, padding=2, dilation=2, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=2, dilation=2, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=4, dilation=4, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=8, dilation=8, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=16, dilation=16, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.Conv2d(204, 204, 3, stride=1, padding=1, dilation=1, bias=True),
        )

    def use_softmax(self, x):
        x = x.view([14, 14, 2 * self.B, 51])
        y = t.empty(14, 14, 2 * self.B, 51)
        y[:, :, :, 0] = t.sigmoid(x[:, :, :, 0].contiguous())
        y[:, :, :, 1: 21] = F.softmax(x[:, :, :, 1: 21].contiguous(), dim=3)
        y[:, :, :, 21: 23] = t.sigmoid(x[:, :, :, 21: 23].contiguous())
        y[:, :, :, 23: 37] = F.softmax(x[:, :, :, 23: 37].contiguous(), dim=3)
        y[:, :, :, 37: 51] = F.softmax(x[:, :, :, 37: 51].contiguous(), dim=3)
        return y

    def forward(self, x):
        
        x0 = self.use_softmax(self.branch0(x))
        x1 = self.use_softmax(self.branch1(x))
        x2 = self.use_softmax(self.branch2(x))
        x3 = self.use_softmax(self.branch3(x))
        
        return t.stack([x0, x1, x2, x3], 0)

class Point_Linking(nn.Module):
    
    def __init__(self, inception_V2):
        super(Point_Linking, self).__init__()
        self.inception_V2 = inception_V2
        self.fourbranch = Fourbranch()
        
        self.grid_size = 14
        self.classes = 20
        self.B =2
        self.img_dim = 448

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = t.cuda.FloatTensor if cuda else t.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = t.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = t.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]
        f = self.inception_V2(x)
        four_out = self.fourbranch(f)
        self.compute_grid_offsets(self.grid_size, cuda=x.is_cuda) 
        return four_out

    def compute_area(self, a, b):
        x_area = [[0, a], [a+1, self.grid_size], [0, a], [a+1, self.grid_size]]
        y_area = [[0, b], [0, b], [b+1, self.grid_size], [b+1, self.grid_size]]
        return x_area, y_area
        
    def predict(self, imgs, sizes=None, visualize=False):
        bboxes = list()
        labels = list()
        scores = list()
        
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

        #link_mnst = t.zeros(self.grid_size**4*self.classes)
        direction = 0
        results = list()
        for img, size in zip(prepared_imgs, sizes):
            bboxes_ = list()
            labels_ = list()
            scores_ = list()
            four_out = self(t.from_numpy(img).unsqueeze(0).cuda().float())
            for i in range(self.B):
                out_p = four_out[direction]#.reshape([self.grid_size,self.grid_size, 2 * self.B, 51])
                out_c = four_out[direction]#.reshape([self.grid_size, self.grid_size, 2 * self.B, 51])
                for b in range(self.grid_size):
                    for a in range(self.grid_size):
                        x_area, y_area = self.compute_area(a, b)
                        for n in range(y_area[i][0], y_area[i][1]):
                            for m in range(x_area[i][0], x_area[i][1]):
                                for c in range(self.classes):
                                    p_mn = out_p[m, n, i, 0]        #(m, n) center point; (a, b) point
                                    p_ab = out_c[a, b, i+self.B, 0]
                                    q_cmn = out_p[m, n, i, 1+c]
                                    q_cab = out_c[a, b, i+self.B, 1+c]
                                    l_mn_a = out_p[m, n, i, 23+a]
                                    l_mn_b = out_c[m, n, i+self.B, 37+b]
                                    l_ab_m = out_p[a, b, i, 23+m]
                                    l_ab_n = out_c[a, b, i+self.B, 37+n]
                                    #link_mnst[c*14*14*14*21+b*14*14*14+a*14*14+n*14+m] = p_mn*p_ab*q_cmn*q_cab*(l_mn_a*l_mn_b+l_ab_m*l_ab_n)/2
                                    score = p_mn*p_ab*q_cmn*q_cab*(l_mn_a*l_mn_b+l_ab_m*l_ab_n)/2
                                    if score > 0.3:
                                        print(score)
                                        results.append([m, n, a, b, c, score])
                for p in results: 
                    bbox = [p[0], p[1], 2*p[2]-p[0], 2*p[3] - p[1]]
                    bboxes_.append(bbox)
                    labels_.append(p[4])
                    scores_.append(p[5])  #result of a img
            bboxes.append(bboxes_)
            labels.append(labels_)
            scores.append(scores_) 

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

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
