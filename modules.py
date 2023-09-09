#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from apex import amp
import copy
import os
from .utils import FrozenBatchNorm2d, ConvBNReLU, ReceptiveConv, ResidualConvBlock

eps = 1e-12

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (ConvBNReLU, nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            pass
            # m.initialize()


##################### PART3: ICE   ########################

class ICE(nn.Module):
    def __init__(self, num_channels=64, ratio=8):
        super(ICE, self).__init__()
        self.conv_cross = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_cross = nn.BatchNorm2d(num_channels)

        self.eps = 1e-5   

        self.conv_mask = nn.Conv2d(num_channels, 1, kernel_size=1)#context Modeling
        self.softmax = nn.Softmax(dim=2)
                                                                                                                                                                                                                                                                                                                       

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // ratio, kernel_size=1),
            nn.LayerNorm([num_channels // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // ratio, num_channels, kernel_size=1)
        )
        self.initialize()

    def forward(self, in1, in2=None, in3=None, flag=None):
        if in2!=None and in1.size()[2:] != in2.size()[2:]:
            in2 = F.interpolate(in2, size=in1.size()[2:], mode='bilinear')
        else: in2 = in1
        if in3!=None and in1.size()[2:] != in3.size()[2:]:
            in3 = F.interpolate(in3, size=in1.size()[2:], mode='bilinear')
        else: in3 = in1

        # x = torch.cat((in1,in2,in3), 1)
        x = in1 + in2 + in3
        x = F.relu(self.bn_cross(self.conv_cross(x))) #[B, C, H, W]

        context = (x.pow(2).sum((2,3), keepdim=True) + self.eps).pow(0.5) # [B, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)

        out = x * channel_add_term
        return out
    
    def initialize(self):
        weight_init(self)


class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(depth, depth, 1, 1)
        self.atrous_block2 = nn.Conv2d(depth*2, depth, 3, 1, padding=2, dilation=2)
        self.atrous_block3 = nn.Conv2d(depth*3, depth, 3, 1, padding=4, dilation=4)
        self.atrous_block4 = nn.Conv2d(depth*4, depth, 3, 1, padding=6, dilation=6)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
        x = self.conv(x)
        atrous_block1 = self.atrous_block1(x)
        x1 = torch.cat([atrous_block1, x], dim=1)
        atrous_block2 = self.atrous_block2(x1)
        x2 = torch.cat([atrous_block2, x1], dim=1)
        atrous_block3 = self.atrous_block3(x2)
        x3 = torch.cat([atrous_block3, x2], dim=1)
        atrous_block4 = self.atrous_block4(x3)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block2,
                                              atrous_block3, atrous_block4], dim=1))
        return net

    def initialize(self):
        weight_init(self)

