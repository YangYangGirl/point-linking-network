from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from model.inceptionresnetv2 import InceptionResNetV2
from model.point_linking import Point_Linking
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt

def pretrained_inception():
    if opt.caffe_pretrain:
        model = InceptionResNetV2()
        if not opt.load_path:
            model_dict = model.state_dict()
            pretrained_dict = t.load_url(opt.caffe_pretrain_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    else:
        model = InceptionResNetV2() 
    return model

class PointLinkInception(Point_Linking):
    def __init__(self):
        inception = InceptionResNetV2()

        super(PointLinkInception, self).__init__(inception)
