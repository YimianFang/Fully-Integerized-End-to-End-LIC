#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
# from .basics import *
import math
import torch.nn as nn
import torch
import numpy as np
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
        self.mulspE0 = torch.tensor(np.load("onnxdata/mulspE0.npy")).cuda()
        self.mulspE1 = torch.tensor(np.load("onnxdata/mulspE1.npy")).cuda()
        self.mulspE2 = torch.tensor(np.load("onnxdata/mulspE2.npy")).cuda()
        self.clp = torch.tensor(np.load("onnxdata/clppE.npy")).cuda()
        self.scl = torch.tensor(np.load("onnxdata/sclpE.npy")).cuda()
        self.w1_1 = nn.Parameter(torch.floor(self.conv1.weight  / 5))
        self.w1_2 = nn.Parameter(torch.floor((self.conv1.weight + 1) / 5))
        self.w1_3 = nn.Parameter(torch.floor((self.conv1.weight + 2) / 5))
        self.w1_4 = nn.Parameter(torch.floor((self.conv1.weight + 3) / 5))
        self.w1_5 = nn.Parameter(torch.floor((self.conv1.weight + 4) / 5))
        self.b1_1 = nn.Parameter(torch.floor(self.conv1.bias / 5))
        self.b1_2 = nn.Parameter(torch.floor((self.conv1.bias + 1) / 5))
        self.b1_3 = nn.Parameter(torch.floor((self.conv1.bias + 2) / 5))
        self.b1_4 = nn.Parameter(torch.floor((self.conv1.bias + 3) / 5))
        self.b1_5 = nn.Parameter(torch.floor((self.conv1.bias + 4) / 5))
        self.w2_1 = nn.Parameter(torch.floor(self.conv2.weight  / 5))
        self.w2_2 = nn.Parameter(torch.floor((self.conv2.weight + 1) / 5))
        self.w2_3 = nn.Parameter(torch.floor((self.conv2.weight + 2) / 5))
        self.w2_4 = nn.Parameter(torch.floor((self.conv2.weight + 3) / 5))
        self.w2_5 = nn.Parameter(torch.floor((self.conv2.weight + 4) / 5))
        self.b2_1 = nn.Parameter(torch.floor(self.conv2.bias / 5))
        self.b2_2 = nn.Parameter(torch.floor((self.conv2.bias + 1) / 5))
        self.b2_3 = nn.Parameter(torch.floor((self.conv2.bias + 2) / 5))
        self.b2_4 = nn.Parameter(torch.floor((self.conv2.bias + 3) / 5))
        self.b2_5 = nn.Parameter(torch.floor((self.conv2.bias + 4) / 5))
        self.w3_1 = nn.Parameter(torch.floor(self.conv3.weight  / 5))
        self.w3_2 = nn.Parameter(torch.floor((self.conv3.weight + 1) / 5))
        self.w3_3 = nn.Parameter(torch.floor((self.conv3.weight + 2) / 5))
        self.w3_4 = nn.Parameter(torch.floor((self.conv3.weight + 3) / 5))
        self.w3_5 = nn.Parameter(torch.floor((self.conv3.weight + 4) / 5))
        self.b3_1 = nn.Parameter(torch.floor(self.conv3.bias / 5))
        self.b3_2 = nn.Parameter(torch.floor((self.conv3.bias + 1) / 5))
        self.b3_3 = nn.Parameter(torch.floor((self.conv3.bias + 2) / 5))
        self.b3_4 = nn.Parameter(torch.floor((self.conv3.bias + 3) / 5))
        self.b3_5 = nn.Parameter(torch.floor((self.conv3.bias + 4) / 5))


    def forward(self, x):
        device = torch.device("cuda:0" if x.is_cuda else "cpu")
        x = torch.abs(x)
        # x = self.conv1(x) ### .to(torch.int32)
        x = F.conv2d(x, self.w1_1, self.b1_1, self.conv1.stride, self.conv1.padding) + F.conv2d(x, self.w1_2, self.b1_2, self.conv1.stride, self.conv1.padding) + \
            F.conv2d(x, self.w1_3, self.b1_3, self.conv1.stride, self.conv1.padding) + F.conv2d(x, self.w1_4, self.b1_4, self.conv1.stride, self.conv1.padding) + \
            F.conv2d(x, self.w1_5, self.b1_5, self.conv1.stride, self.conv1.padding)
        x = x * self.mulspE0
        x = torch.floor((x + 2 ** 10)/2 ** 11)
        x = torch.clip(x, 0, self.clp[0])
        x = torch.floor((x * self.scl[0] + 2 ** 20)/2 ** 21)

        # x = self.conv2(x) ### .to(torch.int32)
        x = F.conv2d(x, self.w2_1, self.b2_1, self.conv2.stride, self.conv2.padding) + F.conv2d(x, self.w2_2, self.b2_2, self.conv2.stride, self.conv2.padding) + \
            F.conv2d(x, self.w2_3, self.b2_3, self.conv2.stride, self.conv2.padding) + F.conv2d(x, self.w2_4, self.b2_4, self.conv2.stride, self.conv2.padding) + \
            F.conv2d(x, self.w2_5, self.b2_5, self.conv2.stride, self.conv2.padding)
        x = x * self.mulspE1
        x = torch.floor((x + 2 ** 13)/2 ** 14)
        x = torch.clip(x, 0, self.clp[1])
        x = x * self.scl[1]
        x = torch.floor((x + 2 ** 20)/2 ** 21)


        # x= self.conv3(x) ### .to(torch.int32)
        x = F.conv2d(x, self.w3_1, self.b3_1, self.conv3.stride, self.conv3.padding) + F.conv2d(x, self.w3_2, self.b3_2, self.conv3.stride, self.conv3.padding) + \
            F.conv2d(x, self.w3_3, self.b3_3, self.conv3.stride, self.conv3.padding) + F.conv2d(x, self.w3_4, self.b3_4, self.conv3.stride, self.conv3.padding) + \
            F.conv2d(x, self.w3_5, self.b3_5, self.conv3.stride, self.conv3.padding)
        x = x * self.mulspE2
        x = torch.floor((x + 2 ** 22)/2 ** 23)

        return x