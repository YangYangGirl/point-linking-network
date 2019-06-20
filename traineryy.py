from __future__ import absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from torch import nn
import torch as t

from utils.config import opt

'''LossTuple = namedtuple('LossTuple',
                       ['pt_loc_loss',
                        'nopt_loss',
                        'total_loss'
                        ])
'''
def gt_convert(bboxes, labels, H, W, grid_size, classes):
    gt = list()
    gt_ps = list()   #x
    gt_ps_d = list()  #x_ij
    gt_cs = list()    #中心点
    gt_cs_d = list()
    gt_labels = list()
    gt_linkcs_x = list()
    gt_linkcs_y = list()

    bboxes = bboxes / W * grid_size


    for which, b in enumerate(bboxes):
        x0 = int(b[0])
        y0 = int(b[1])
        x0_d = b[0] - int(b[0])
        y0_d = b[1] - int(b[1])

        x1 = int((b[0] + b[2]))
        y1 =  int(b[1])
        x1_d = (b[0] + b[2])- int((b[0] + b[2]))
        y1_d = b[1] - int(b[1])

        x2 = int(b[0])
        y2 = int((b[1] + H))
        x2_d = b[0] - int(b[0])
        y2_d = (b[1] + H) - int((b[1] + H))

        x3 = int((b[0] + b[2]))
        y3 = int((b[1] + H))
        x3_d = (b[0] + b[2]) - int((b[0] + b[2]))
        y3_d = (b[1] + H) - int((b[1] + H))

        xc = int((b[0] + b[2] / 2))
        yc = int((b[1] + b[3] / 2))
        xc_d = (b[0] + b[2] / 2) - int((b[0] + b[2] / 2))
        yc_d = (b[1] + b[3] / 2)  - int((b[1] + b[3] / 2))

        x0_ = (x0, y0)
        x1_ = (x1, y1)
        x2_ = (x2, y2)
        x3_ = (x3, y3)
        xc_ = (xc. yc)
        gt_ps.append(x0_, x1_, x2_, x3_)
        gt_ps_d.append((x0_d, y0_d), (x1_d, y1_d), (x2_d, y2_d), (x3_d, y3_d))
        gt_cs.append(xc_)
        gt_cs_d.append((xc_d, yc_d))

        gt_label = t.zeros((classes))
        gt_label[labels[which]] = 1
        gt_labels.append(gt_label)

        gt_linkc_x = t.zeros((grid_size))
        gt_linkc_x[xc] = 1
        gt_linkcs_x.append(gt_link_x)
        gt_linkc_y = t.zeros((grid_size))
        gt_linkc_y[yc] = 1
        gt_linkcs_y.append(gt_link_y)

    return gt_ps, gt_ps_d, gt_cs, gt_cs_d, gt_labels, gt_linkcs_x, gt_linkcs_y


class PointLinkTrainer(nn.Module):
    def __init__(self, point_link):
        super(PointLinkTrainer, self).__init__()

        self.point_link = point_link
        self.grid_size = 14
        self.B = 2
        self.optimizer = self.point_link.get_optimizer()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.classes = 20
        self.w_class = 1
        self.w_coord = 1
        self.w_link = 1
        self.total_loss = 0

    def forward(self, imgs, p, q, x, y, bboxes, labels, ):
        _, _, H, W = imgs.shape
        img_size = (H, W)
        out = self.point_link(imgs)
        out.reshape([14, 14, 2*self.B , 51])

        gt_ps, gt_ps_d, gt_cs, gt_cs_d, gt_labels, gt_linkcs_x, gt_linkcs_y= gt_convert(bboxes, labels, H, W, self.grid_size)

        for i_x in range(14):
            for i_y in range(14):
                for j in range(2  * self.B):
                    if j < self.B:
                        if (i_x,i_y) in gt_ps:
                            which = gt_ps.index((i_x, i_y))
                            x_ij, y_ij = gt_ps_d[which]
                            loss1 = (out[i_x, i_y, j, 0] - 1)**2
                            loss2 = self.w_class * self.mse_loss(out[i_x, i_y, j, 1: 1 + self.classes], gt_labels[int(which/4)])
                            loss3 = self.w_coord * (out[i_x, i_y, j, 1 + self.classes : 2 + self.classes] - x_ij)**2 +\
                                    (out[i_x, i_y, j, 1 + self.classes : 3 + self.classes] - y_ij)**2
                            loss4 = self.w_link * self.mse_loss(out[i_x, i_y, 3 + self.classes: 3 + self.classes + self.grid_size], gt_linkcs_x) +\
                            self.mse_loss(out[3 + self.classes + self.grid_size: 3 + self.classes + 2 * self.grid_size], gt_linkcs_y)
                            loss_pt = loss1 + loss2 + loss3 + loss4
                            self.total_loss += loss_pt
                        else:
                            loss_nopt = out[i_x, i_y, j, 0]**2
                            self.total_loss += loss_pt
                    if j >= self.B:
                        if (i_x,i_y) in gt_ps:
                            which = gt_ps.index((i_x, i_y))
                            x_ij, y_ij = gt_ps_d[which]
                            loss1 = (out[i_x, i_y, j, 0] - 1)**2
                            loss2 = self.w_class * self.mse_loss(out[i_x, i_y, j, 1: 1 + self.classes], gt_labels[int(which/4)])
                            loss3 = self.w_coord * (out[i_x, i_y, j, 1 + self.classes : 2 + self.classes] - x_ij)**2 +\
                                    (out[i_x, i_y, j, 1 + self.classes : 3 + self.classes] - y_ij)**2
                            loss4 = self.w_link * self.mse_loss(out[i_x, i_y, 3 + self.classes: 3 + self.classes + self.grid_size], gt_linkcs_x) +\
                            self.mse_loss(out[3 + self.classes + self.grid_size: 3 + self.classes + 2 * self.grid_size], gt_linkcs_y)
                            loss_pt = loss1 + loss2 + loss3 + loss4
                            self.total_loss += loss_pt
                        else:
                            loss_nopt = out[i_x, i_y, j, 0]**2
                            self.total_loss += loss_pt
                            
        '''gt = t.zeros([14, 14, 204])

        for which, b in enumerate(bboxes):
            # pij
            gt[int(b[0] / W * self.grid_size), int(b[1] / H * self.grid_size), 0] = 1
            gt[int((b[0] + b[2]) / W * self.grid_size), int(b[1] / H * self.grid_size), 0] = 1
            gt[int(b[0] / W * self.grid_size), int((b[1] + b[3]) / H * self.grid_size), 0] = 1
            gt[int((b[0] + b[2]) / W * self.grid_size), int((b[1] + b[3]) / H * self.grid_size), 0] = 1

            gt[int((b[0] + b[2]/2) / W * self.grid_size), int((b[1] + b[3]/2) / H * self.grid_size), 52 + 0] = 1

            # qij
            gt[int(b[0] / W * self.grid_size), int(b[1] / H * self.grid_size), 1 + labels[which]] = 1
            gt[int((b[0] + b[2]) / W * self.grid_size), int(b[1] / H * self.grid_size), 1 + labels[which]] = 1
            gt[int(b[0] / W * self.grid_size), int((b[1] + b[3]) / H * self.grid_size), 1 + labels[which]] = 1
            gt[int((b[0] + b[2]) / W * self.grid_size), int((b[1] + b[3]) / H * self.grid_size), 1 + labels[which]] = 1
+
            gt[int((b[0] + b[2] / 2) / W * self.grid_size), int((b[1] + b[3] / 2) / H * self.grid_size), 52 + 1 + labels[which]] = 1

            # xij， yij
            gt[int(b[0] / W * self.grid_size), int(b[1] / H * self.grid_size), 21] = b[0] / W * self.grid_size - int(
                b[0] / W * self.grid_size)
            gt[int(b[0] / W * self.grid_size), int(b[1] / H * self.grid_size), 22] = b[1] / H * self.grid_size - int(
                b[1] / H * self.grid_size)
            gt[int(b[0] / W * self.grid_size), int((b[1] + b[3]) / H * self.grid_size), 21] = b[0] / W * self.grid_size - int(
                b[0] / W * self.grid_size)
            gt[int(b[0] / W * self.grid_size), int(b[1] / H * self.grid_size), 22] = b[1] / H * self.grid_size - int(
                b[1] / H * self.grid_size)
            gt[int(b[0] / W * self.grid_size), int(b[1] / H * self.grid_size), 21] = b[0] / W * self.grid_size - int(
                b[0] / W * self.grid_size)
            gt[int(b[0] / W * self.grid_size), int(b[1] / H * self.grid_size), 22] = b[1] / H * self.grid_size - int(
                b[1] / H * self.grid_size)
            gt[int(b[0] / W * self.grid_size), int(b[1] / H * self.grid_size), 21] = b[0] / W * self.grid_size - int(
                b[0] / W * self.grid_size)
            gt[int(b[0] / W * self.grid_size), int(b[1] / H * self.grid_size), 22] = b[1] / H * self.grid_size - int(
                b[1] / H * self.grid_size)

            gt[int((b[0] + b[2] / 2) / W * self.grid_size), int((b[1] + b[3] / 2) / H * self.grid_size), 52 + 21] = (b[0] + b[2] / 2) / W * self.grid_size - int((b[0] + b[2] / 2) / W * self.grid_size)
            gt[int((b[0] + b[2] / 2) / W * self.grid_size), int((b[1] + b[3] / 2) / H * self.grid_size), 52 + 22] = (b[1] + b[3] / 2) / H * self.grid_size - int((b[1] + b[3] / 2) / H * self.grid_size)

            # l_x_ij,l_y_ij
            gt[int((b[0] + b[2] / 2) / W * self.grid_size), int((b[1] + b[3] / 2) / H * self.grid_size), 52 + 23 + int(b[0] / W * self.grid_size)] = 1
            gt[int((b[0] + b[2] / 2) / W * self.grid_size), int((b[1] + b[3] / 2) / H * self.grid_size), 52 + 37 + int(b[1] / H * self.grid_size)] = 1
            gt[int((b[0] + b[2] / 2) / W * self.grid_size), int((b[1] + b[3] / 2) / H * self.grid_size), 52 + 23 + int((b[0] + b[2]) / W * self.grid_size)] = 1
            gt[int((b[0] + b[2] / 2) / W * self.grid_size), int((b[1] + b[3] / 2) / H * self.grid_size), 52 + 37 +int(b[1] / H * self.grid_size)] = 1
            gt[int((b[0] + b[2] / 2) / W * self.grid_size), int((b[1] + b[3] / 2) / H * self.grid_size), 52 + 23 + int(b[0] / W * self.grid_size)] = 1
            gt[int((b[0] + b[2] / 2) / W * self.grid_size), int((b[1] + b[3] / 2) / H * self.grid_size), 52 + 37 + int(b[1] / H * self.grid_size)] = 1
            gt[int((b[0] + b[2] / 2) / W * self.grid_size), int((b[1] + b[3] / 2) / H * self.grid_size), 52 + 23 + int((b[0] + b[2]) / W * self.grid_size)] = 1
            gt[int((b[0] + b[2] / 2) / W * self.grid_size), int((b[1] + b[3] / 2) / H * self.grid_size), 52 + 37 + int((b[1] + b[3]) / H * self.grid_size)] = 1

            gt[int(b[0] / W * self.grid_size), int(b[1] / H * self.grid_size), 23 + int((b[0] + b[2] / 2) / W * self.grid_size)] = 1
            gt[int(b[0] / W * self.grid_size), int(b[1] / H * self.grid_size), 37 + int((b[1] + b[3] / 2) / H * self.grid_size)] = 1
            gt[int((b[0] + b[2]) / W * self.grid_size), int(b[1] / H * self.grid_size), 23 + int((b[0] + b[2] / 2) / W * self.grid_size)] = 1
            gt[int((b[0] + b[2]) / W * self.grid_size), int(b[1] / H * self.grid_size), 37 + int((b[1] + b[3] / 2) / H * self.grid_size)] = 1
            gt[int(b[0] / W * self.grid_size), int((b[1] + b[3]) / H * self.grid_size), 23 + int((b[0] + b[2] / 2) / W * self.grid_size)] = 1
            gt[int(b[0] / W * self.grid_size), int((b[1] + b[3]) / H * self.grid_size), 37 + int((b[1] + b[3] / 2) / H * self.grid_size)] = 1
            gt[int((b[0] + b[2]) / W * self.grid_size), int((b[1] + b[3]) / H * self.grid_size), 23 + int((b[0] + b[2] / 2) / W * self.grid_size)] = 1
            gt[int((b[0] + b[2]) / W * self.grid_size), int((b[1] + b[3]) / H * self.grid_size), 37 + int((b[1] + b[3] / 2) / H * self.grid_size)] = 1
        '''

        loss_pt = 0
        loss_nopt = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                out[i, j, 0]
