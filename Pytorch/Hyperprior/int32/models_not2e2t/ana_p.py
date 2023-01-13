#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
# from .basics import *
import math
import torch.nn as nn
import torch
import numpy as np
from .gradSTE import floorSTE, roundSTE, wSTE
import copy
import torch.nn.functional as F

class Analysis_prior_net(nn.Module):
    '''
    Analysis prior net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=323):
        super(Analysis_prior_net, self).__init__()
        self.conv1 = nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
        # self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        # self.relu2 = nn.ReLU()
        self.conv3 =  nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.in_scale = 4


    def forward(self, x, muls, relus, Bits=8):
        clp_k = 9
        stas_max = []
        for i in range(len(relus)):
            stas_max.append(round((math.pow(2.0, Bits) - 1) / relus[i] * (2 ** (16 + clp_k))))
        clp = copy.deepcopy(stas_max)
        scl = copy.deepcopy(relus)

        device = torch.device("cuda:0" if x.is_cuda else "cpu")
        x = torch.abs(x).clamp(0, 255)
        # self.conv1.weight.data = wSTE.apply(self.conv1.weight.data)
        # self.conv1.bias.data = roundSTE.apply(self.conv1.bias.data)
        # x = self.conv1(x * (2 ** self.in_scale))
        # x = roundSTE.apply(x * (2 ** self.in_scale)) #####
        x = F.conv2d(x, 
                    # wSTE.apply(self.conv1.weight), 
                    roundSTE.apply(self.conv1.weight), 
                    roundSTE.apply(self.conv1.bias), 
                    self.conv1.stride, 
                    self.conv1.padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[0].reshape(1, -1, 1, 1)
        x = floorSTE.apply((x + 2 ** (15 + self.in_scale - clp_k))/2 ** (16 + self.in_scale - clp_k))
        # x = x.permute(1, 0, 2, 3)
        x = torch.clip(x, 0, clp[0])
        scl0_k = 4
        if scl0_k > 0:
            scl0 = floorSTE.apply((torch.tensor(scl[0]) + 2 ** (scl0_k - 1)) / 2 ** scl0_k)
        x = floorSTE.apply((x * scl0 + 2 ** (15 + clp_k - scl0_k))/2 ** (16 + clp_k - scl0_k))

        # self.conv2.weight.data = wSTE.apply(self.conv2.weight.data)
        # self.conv2.bias.data = roundSTE.apply(self.conv2.bias.data)
        # x = self.conv2(x)
        x = F.conv2d(x, 
                    # wSTE.apply(self.conv2.weight), 
                    roundSTE.apply(self.conv2.weight), 
                    roundSTE.apply(self.conv2.bias), 
                    self.conv2.stride, 
                    self.conv2.padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[1].reshape(1, -1, 1, 1)
        x = floorSTE.apply((x + 2 ** (22 - clp_k))/2 ** (23 - clp_k))
        # x = x.permute(1, 0, 2, 3)
        x = torch.clip(x, 0, clp[1])
        scl1_k = 4
        if scl1_k > 0:
            scl1 = floorSTE.apply((torch.tensor(scl[1]) + 2 ** (scl1_k - 1)) / 2 ** scl1_k)
        x = floorSTE.apply((x * scl1 + 2 ** (15 + clp_k - scl1_k))/2 ** (16 + clp_k - scl1_k))

        # self.conv3.weight.data = wSTE.apply(self.conv3.weight.data)
        # self.conv3.bias.data = roundSTE.apply(self.conv3.bias.data)
        # x= self.conv3(x)
        x = F.conv2d(x, 
                    # wSTE.apply(self.conv3.weight), 
                    roundSTE.apply(self.conv3.weight), 
                    roundSTE.apply(self.conv3.bias), 
                    self.conv3.stride, 
                    self.conv3.padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[2].reshape(1, -1, 1, 1)
        x = floorSTE.apply((x + 2 ** 22)/2 ** 23)
        # x = x.permute(1, 0, 2, 3)

        return x