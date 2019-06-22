from __future__ import absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from torch import nn
import torch as t

from utils.config import opt
from utils.vis_tool import Visualizer

'''LossTuple = namedtuple('LossTuple',
                       ['pt_loc_loss',
                        'nopt_loss',
                        'total_loss'
                        ])
'''
def gt_convert(bboxes, labels, H, W, grid_size, classes):

    gt_ps = list()
    gt_ps_d = list()
    gt_cs = list()
    gt_cs_d = list()
    gt_labels = list()
    gt_linkcs_x = list()
    gt_linkcs_y = list()
    gt_linkps_x = list()
    gt_linkps_y = list()
    
    print("W" + str(W))
    bboxes = bboxes / W * grid_size
    print(bboxes)
    print(type(bboxes))
    bboxes = bboxes[0]
    print(bboxes.shape)
    for which, b in enumerate(bboxes):
        #b = b.squeeze()
        x0 = int(b[0])
        y0 = int(b[1])
        x0_d = b[0] - x0
        y0_d = b[1] - y0

        x1 = int(b[0] + b[2])
        y1 =  int(b[1])
        x1_d = (b[0] + b[2])- int(b[0] + b[2])
        y1_d = b[1] - int(b[1])

        x2 = int(b[0])
        y2 = int(b[1] + H)
        x2_d = b[0] - int(b[0])
        y2_d = (b[1] + H) - int(b[1] + H)

        x3 = int(b[0] + b[2])
        y3 = int((b[1] + H))
        x3_d = (b[0] + b[2]) - int(b[0] + b[2])
        y3_d = (b[1] + H) - int(b[1] + H)

        xc = int(b[0] + b[2] / 2)
        yc = int(b[1] + b[3] / 2)
        xc_d = (b[0] + b[2] / 2) - int(b[0] + b[2] / 2)
        yc_d = (b[1] + b[3] / 2)  - int(b[1] + b[3] / 2)

        x0_ = (x0, y0)
        x1_ = (x1, y1)
        x2_ = (x2, y2)
        x3_ = (x3, y3)
        xc_ = (xc, yc)
        gt_ps += [x0_, x1_, x2_, x3_]
        gt_ps_d += [(x0_d, y0_d), (x1_d, y1_d), (x2_d, y2_d), (x3_d, y3_d)]
        gt_cs.append(xc_)
        gt_cs_d.append((xc_d, yc_d))

        gt_label = t.zeros((classes))
        gt_label[labels[0][which]] = 1
        gt_labels.append(gt_label)

        gt_linkc_x = t.zeros((grid_size))
        gt_linkc_x[xc] = 1
        gt_linkcs_x.append(gt_linkc_x)
        gt_linkc_y = t.zeros((grid_size))
        gt_linkc_y[yc] = 1
        gt_linkcs_y.append(gt_linkc_y)

        gt_linkp_x = t.zeros((4, grid_size))
        gt_linkp_y = t.zeros((4, grid_size))
        print(gt_ps[0])
        for i, (p_x, p_y) in enumerate(gt_ps):
            print("p_x")
            print(p_x)
            gt_linkp_x[i][p_x] = 1
            gt_linkp_y[i][p_y] = 1
        gt_linkps_x.append(gt_linkp_x)
        gt_linkps_y.append(gt_linkp_y)

    return gt_ps, gt_ps_d, gt_cs, gt_cs_d, gt_labels, gt_linkcs_x, gt_linkcs_y, gt_linkps_x, gt_linkps_y


class PointLinkTrainer(nn.Module):
    def __init__(self, point_link):
        super(PointLinkTrainer, self).__init__()

        self.point_link = point_link
        self.grid_size = 14
        self.B = 2
        self.optimizer = self.point_link.get_optimizer()
        self.mse_loss = nn.MSELoss()
        self.classes = 20
        self.w_class = 1
        self.w_coord = 1
        self.w_link = 1
        self.total_loss = 0
        self.vis = Visualizer(env=opt.env)

    def compute_loss(self, out_four, bboxes, labels, H, W):
        """

        Args:
            out_four:
            bboxes:
            labels:
            H:
            W:

        Returns: losses of four branches

        """
        loss = list()
        gt_ps, gt_ps_d, gt_cs, gt_cs_d, gt_labels, gt_linkcs_x, gt_linkcs_y, gt_linkps_x, gt_linkps_y = \
            gt_convert(bboxes, labels, H, W, self.grid_size, self.classes)

        for direction in range(4):
            total_loss = 0
            out = out_four[direction].reshape([14, 14, 2 * self.B, 51])
            for i_x in range(14):
                for i_y in range(14):
                    for j in range(2 * self.B):
                        if j < self.B:
                            if (i_x, i_y) in gt_ps:
                                which = gt_ps.index((i_x, i_y))
                                x_ij, y_ij = gt_ps_d[which]
                                loss1 = (out[i_x, i_y, j, 0] - 1) ** 2
                                loss2 = self.w_class * self.mse_loss(out[i_x, i_y, j, 1: 1 + self.classes],
                                                                     gt_labels[int(which / 4)])
                                loss3 = self.w_coord * self.mse_loss(
                                    out[i_x, i_y, j, 1 + self.classes: 3 + self.classes], gt_ps_d[which])
                                loss4 = self.w_link * self.mse_loss(
                                    out[i_x, i_y, 3 + self.classes: 3 + self.classes + self.grid_size],
                                    gt_linkcs_x[which // 4]) + \
                                        self.mse_loss(out[
                                                      3 + self.classes + self.grid_size: 3 + self.classes + 2 * self.grid_size],
                                                      gt_linkcs_y[which // 4])

                                loss_pt = loss1 + loss2 + loss3 + loss4
                                total_loss += loss_pt
                            else:
                                loss_nopt = out[i_x, i_y, j, 0] ** 2
                                total_loss += loss_pt
                        if j >= self.B:
                            if (i_x, i_y) in gt_ps:
                                which = gt_ps.index((i_x, i_y))
                                x_ij, y_ij = gt_cs_d[which]
                                loss1 = (out[i_x, i_y, j, 0] - 1) ** 2
                                loss2 = self.w_class * self.mse_loss(out[i_x, i_y, j, 1: 1 + self.classes],
                                                                     gt_labels[which])
                                loss3 = self.w_coord * self.mse_loss(
                                    out[i_x, i_y, j, 1 + self.classes: 3 + self.classes] - gt_cs_d[which])
                                loss4 = self.w_link * self.mse_loss(
                                    out[i_x, i_y, 3 + self.classes: 3 + self.classes + self.grid_size],
                                    gt_linkps_x[which][direction]) + \
                                        self.mse_loss(out[
                                                      3 + self.classes + self.grid_size: 3 + self.classes + 2 * self.grid_size],
                                                      gt_linkps_y[which][direction])

                                loss_pt = loss1 + loss2 + loss3 + loss4
                                total_loss += loss_pt
                            else:
                                loss_nopt = out[i_x, i_y, j, 0] ** 2
                                total_loss += loss_pt
            loss[direction] = total_loss
        return loss


    def forward(self, imgs, bboxes, labels, direction):
        _, _, H, W = imgs.shape
        img_size = (H, W)
        print("yyyyyyyyyyyyyyyyyyyyyyy")
        print(imgs.shape)  #yy
        out_four = self.point_link(imgs)

        loss = self.compute_loss(out_four, bboxes, labels, H, W)

        return loss

    def train_step(self, imgs, bboxes, labels, direction):
        self.optimizer.zero_grad()
        total_loss = self.forward(imgs, bboxes, labels, direction)
        total_loss.backward()
        self.optimizer.step()

        return total_loss


    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.
        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.point_link.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/pointlink_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path


    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self
