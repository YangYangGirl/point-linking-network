from __future__ import  absolute_import
from __future__ import division
import torch as t
import numpy as np
import cupy as cp
from data import util
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
from model.utils.nms import non_maximum_suppression

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import opt
from trainer import gt_convert
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
        #print("===== y[0, 0, 0, 1: 21] =====")
        #print(y[0, 0, 0, 1: 21])
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

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score
        
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

    def eval_center(self, dataloader, sizes=None, visualize=True):
        pred_centers = list() 
        gt_bboxes = list()
        loss_center = 0
        for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(dataloader):
            sizes = np.array(sizes).tolist()
            in_size = (sizes[0], sizes[1])
            gt_bboxes_ = util.resize_bbox(gt_bboxes_[0].numpy(), in_size, (448, 448))
            gt_ps, gt_ps_d, gt_cs, gt_cs_d, gt_labels, gt_linkcs_x, gt_linkcs_y, gt_linkps_x, gt_linkps_y = gt_convert(t.from_numpy(gt_bboxes_).unsqueeze(0), gt_labels_, 448, 448, 14, 20)
            net_result = self(imgs.cuda().float())[0]
            for i_x in range(14):
                for i_y in range(14):
                    if [i_x, i_y] in gt_cs.tolist(): 
                        loss_center += (net_result[i_x, i_y, 2, 0] - 1) ** 2
                    else:
                        loss_center += net_result[i_x, i_y, 2, 0] ** 2
        print(loss_center)
        
    def predict_center(self, imgs, sizes=None, visualize=False):
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
             prepared_imgs = list()
             for img in imgs:
                 prepared_imgs.append(img.numpy())
        #link_mnst = t.zeros(self.grid_size**4*self.classes)
        direction = 0
        results = list()
        results_score = list()
        for img, size in zip(prepared_imgs, sizes):
            bboxes_ = list()
            labels_ = list()
            scores_ = list()
            '''print("img.shape", img.shape)
            print("size", size)
            scale = img.shape[3] / size[1]    #fit to raw image
            print("scale", scale)
            print("img.shape[3], size[1]", img.shape[3], size[1]) '''
            four_out = self(t.from_numpy(img).unsqueeze(0).cuda().float())
            for i in range(self.B):
                out_p = four_out[direction]#.reshape([self.grid_size,self.grid_size, 2 * self.B, 51])
                out_c = four_out[direction]#.reshape([self.grid_size, self.grid_size, 2 * self.B, 51])
                for b in range(self.grid_size):
                    for a in range(self.grid_size):
                        for c in range(self.classes):
                            p_ab = out_c[a, b, i+self.B, 0]
                            q_cab = out_c[a, b, i+self.B, 1+c]
                            score = p_ab * q_cab
                            if score > 0.01:
                                results.append([a, b, c, score])
                                results_score.append(score)
                
            if len(results_score) > 0:
                print("have detect:", len(results_score))
                max_score_index = np.argmax(results_score)
                rs = list(np.expand_dims(np.array(results[max_score_index]), axis=0))
                for p in rs: 
                    #bbox = [p[0], p[1], 2*p[2]-p[0], 2*p[3] - p[1]]
                    #(y_{min}, x_{min}, y_{max}, x_{max})
                    bbox = [p[1], p[0], p[1]+1, p[0]+1]
                    #print(bbox)
                    bbox = [b * 32 for b in bbox] 
                    bboxes_.append(bbox)
                    labels_.append(p[2])
                    scores_.append(p[3]*1000)  #result of a img
            #bboxes_nms = self._suppress(bboxes_, scores_)
            bboxes.append(np.array(bboxes_))
            labels.append(np.array(labels_))
            scores.append(np.array(scores_))

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

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
             prepared_imgs = list()
             for img in imgs:
                 prepared_imgs.append(img.numpy())
        #link_mnst = t.zeros(self.grid_size**4*self.classes)
        direction = 0
        results = list()
        for img, size in zip(prepared_imgs, sizes):
            bboxes_ = list()
            labels_ = list()
            scores_ = list()
            '''print("img.shape", img.shape)
            print("size", size)
            scale = img.shape[3] / size[1]    #fit to raw image
            print("scale", scale)
            print("img.shape[3], size[1]", img.shape[3], size[1]) '''
            four_out = self(t.from_numpy(img).unsqueeze(0).cuda().float())
            for i in range(self.B):
                out_p = four_out[direction]#.reshape([self.grid_size,self.grid_size, 2 * self.B, 51])
                out_c = four_out[direction]#.reshape([self.grid_size, self.grid_size, 2 * self.B, 51])
                for b in range(self.grid_size):
                    for a in range(self.grid_size):
                        if out_c[a, b, i+self.B, 0] < 0.5: continue 
                        x_area, y_area = self.compute_area(a, b)
                        for n in range(y_area[direction][0], y_area[direction][1]):
                            for m in range(x_area[direction][0], x_area[direction][1]):
                                for c in range(self.classes):
                                    p_mn = out_p[m, n, i, 0]        #(a, b) center point; (m, n) point
                                    p_ab = out_c[a, b, i+self.B, 0]
                                    q_cmn = out_p[m, n, i, 1+c]
                                    q_cab = out_c[a, b, i+self.B, 1+c]
                                    l_mn_a = out_p[m, n, i, 23+a]
                                    l_mn_b = out_c[m, n, i+self.B, 37+b]
                                    l_ab_m = out_p[a, b, i, 23+m]
                                    l_ab_n = out_c[a, b, i+self.B, 37+n]
                                    #link_mnst[c*14*14*14*21+b*14*14*14+a*14*14+n*14+m] = p_mn*p_ab*q_cmn*q_cab*(l_mn_a*l_mn_b+l_ab_m*l_ab_n)/2
                                    score = p_mn*p_ab*q_cmn*q_cab*(l_mn_a*l_mn_b+l_ab_m*l_ab_n)/2
                                    #print(p_mn, p_ab, q_cmn, q_cab, l_mn_a, l_mn_b, l_ab_m, l_ab_n)
                                    #print(score)
                                    m_ , n_, a_, b_ = out_p[m, n, i, 21], out_p[m, n, i, 22], out_c[a, b, i+self.B, 21], out_c[a, b, i+self.B, 22]
                                    if score > 0.01:
                                        #print([a, b, m, n, c, score])
                                        #print([a_, b_, m_, n_, c, score])
                                        results.append([a+a_, b+b_, m+m_, n+n_, c, score])
                for p in results: 
                    #bbox = [p[0], p[1], 2*p[2]-p[0], 2*p[3] - p[1]]
                    bbox = [p[3], p[2], 2*p[1]-p[3], 2*p[0]-p[2]]
                    #print(bbox)
                    bbox = [b * 32 for b in bbox] 
                    bboxes_.append(bbox)
                    labels_.append(p[4])
                    scores_.append(p[5]*1000)  #result of a img
            #bboxes_nms = self._suppress(bboxes_, scores_)
            bboxes.append(np.array(bboxes_))
            labels.append(np.array(labels_))
            scores.append(np.array(scores_))

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
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0.00004}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.00004}]
            
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        elif opt.use_RMSprop:
            self.optimizer = t.optim.RMSprop(params, alpha=0.9)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
