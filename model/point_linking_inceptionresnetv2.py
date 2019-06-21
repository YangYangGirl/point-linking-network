from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from model.inceptionresnetv2 import InceptionResNetV2
from model.point_linking import Point_Linking
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt

class PointLinkInception(Point_Linking):
    def __init__(self, classes=20):
        inception = InceptionResNetV2()

        super(Point_Linking, self).__init__(inception)
