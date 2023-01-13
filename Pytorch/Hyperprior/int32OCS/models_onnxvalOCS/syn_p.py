#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
# from .basics import *
import math
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
import numpy as np

class Synthesis_prior_net(nn.Module):
    '''
    Decode synthesis prior
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Synthesis_prior_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        # self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        # self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        self.mulspD0 = torch.tensor(np.load("onnxdata/mulspD0.npy")).cuda()
        self.mulspD1 = torch.tensor(np.load("onnxdata/mulspD1.npy")).cuda()
        self.mulspD2 = torch.tensor(np.load("onnxdata/mulspD2.npy")).cuda()
        self.clp = torch.tensor(np.load("onnxdata/clppD.npy")).cuda()
        self.scl = torch.tensor(np.load("onnxdata/sclpD.npy")).cuda()
        self.charts = torch.tensor(np.load("onnxdata/charts.npy")).cuda()
        self.w1_1 = nn.Parameter(torch.floor(self.deconv1.weight  / 5))
        self.w1_2 = nn.Parameter(torch.floor((self.deconv1.weight + 1) / 5))
        self.w1_3 = nn.Parameter(torch.floor((self.deconv1.weight + 2) / 5))
        self.w1_4 = nn.Parameter(torch.floor((self.deconv1.weight + 3) / 5))
        self.w1_5 = nn.Parameter(torch.floor((self.deconv1.weight + 4) / 5))
        self.b1_1 = nn.Parameter(torch.floor(self.deconv1.bias / 5))
        self.b1_2 = nn.Parameter(torch.floor((self.deconv1.bias + 1) / 5))
        self.b1_3 = nn.Parameter(torch.floor((self.deconv1.bias + 2) / 5))
        self.b1_4 = nn.Parameter(torch.floor((self.deconv1.bias + 3) / 5))
        self.b1_5 = nn.Parameter(torch.floor((self.deconv1.bias + 4) / 5))
        self.w2_1 = nn.Parameter(torch.floor(self.deconv2.weight  / 5))
        self.w2_2 = nn.Parameter(torch.floor((self.deconv2.weight + 1) / 5))
        self.w2_3 = nn.Parameter(torch.floor((self.deconv2.weight + 2) / 5))
        self.w2_4 = nn.Parameter(torch.floor((self.deconv2.weight + 3) / 5))
        self.w2_5 = nn.Parameter(torch.floor((self.deconv2.weight + 4) / 5))
        self.b2_1 = nn.Parameter(torch.floor(self.deconv2.bias / 5))
        self.b2_2 = nn.Parameter(torch.floor((self.deconv2.bias + 1) / 5))
        self.b2_3 = nn.Parameter(torch.floor((self.deconv2.bias + 2) / 5))
        self.b2_4 = nn.Parameter(torch.floor((self.deconv2.bias + 3) / 5))
        self.b2_5 = nn.Parameter(torch.floor((self.deconv2.bias + 4) / 5))
        self.w3_1 = nn.Parameter(torch.floor(self.deconv3.weight  / 5))
        self.w3_2 = nn.Parameter(torch.floor((self.deconv3.weight + 1) / 5))
        self.w3_3 = nn.Parameter(torch.floor((self.deconv3.weight + 2) / 5))
        self.w3_4 = nn.Parameter(torch.floor((self.deconv3.weight + 3) / 5))
        self.w3_5 = nn.Parameter(torch.floor((self.deconv3.weight + 4) / 5))
        self.b3_1 = nn.Parameter(torch.floor(self.deconv3.bias / 5))
        self.b3_2 = nn.Parameter(torch.floor((self.deconv3.bias + 1) / 5))
        self.b3_3 = nn.Parameter(torch.floor((self.deconv3.bias + 2) / 5))
        self.b3_4 = nn.Parameter(torch.floor((self.deconv3.bias + 3) / 5))
        self.b3_5 = nn.Parameter(torch.floor((self.deconv3.bias + 4) / 5))
                

    def forward(self, x):
        device = torch.device("cuda:0" if x.is_cuda else "cpu")
        # x = self.deconv1(x) ### .to(torch.int32)
        x = F.conv_transpose2d(x, self.w1_1, self.b1_1, self.deconv1.stride, self.deconv1.padding, self.deconv1.output_padding) + \
            F.conv_transpose2d(x, self.w1_2, self.b1_2, self.deconv1.stride, self.deconv1.padding, self.deconv1.output_padding) + \
            F.conv_transpose2d(x, self.w1_3, self.b1_3, self.deconv1.stride, self.deconv1.padding, self.deconv1.output_padding) + \
            F.conv_transpose2d(x, self.w1_4, self.b1_4, self.deconv1.stride, self.deconv1.padding, self.deconv1.output_padding) + \
            F.conv_transpose2d(x, self.w1_5, self.b1_5, self.deconv1.stride, self.deconv1.padding, self.deconv1.output_padding)
        x = x * self.mulspD0
        x = torch.floor((x + 2 ** 8)/2 ** 9)
        x = torch.clip(x, 0, self.clp[0])
        x = torch.floor((x * self.scl[0] + 2 ** 20)/2 ** 21)

        # x= self.deconv2(x) ### .to(torch.int32)
        x = F.conv_transpose2d(x, self.w2_1, self.b2_1, self.deconv2.stride, self.deconv2.padding, self.deconv2.output_padding) + \
            F.conv_transpose2d(x, self.w2_2, self.b2_2, self.deconv2.stride, self.deconv2.padding, self.deconv2.output_padding) + \
            F.conv_transpose2d(x, self.w2_3, self.b2_3, self.deconv2.stride, self.deconv2.padding, self.deconv2.output_padding) + \
            F.conv_transpose2d(x, self.w2_4, self.b2_4, self.deconv2.stride, self.deconv2.padding, self.deconv2.output_padding) + \
            F.conv_transpose2d(x, self.w2_5, self.b2_5, self.deconv2.stride, self.deconv2.padding, self.deconv2.output_padding)
        x = x * self.mulspD1
        x = torch.floor((x + 2 ** 9)/2 ** 10)
        x = torch.clip(x, 0, self.clp[1])
        x = torch.floor((x * self.scl[1] + 2 ** 20)/2 ** 21)

        # x= self.deconv3(x) ### .to(torch.int32)
        x = F.conv_transpose2d(x, self.w3_1, self.b3_1, self.deconv3.stride, self.deconv3.padding, self.deconv3.output_padding) + \
            F.conv_transpose2d(x, self.w3_2, self.b3_2, self.deconv3.stride, self.deconv3.padding, self.deconv3.output_padding) + \
            F.conv_transpose2d(x, self.w3_3, self.b3_3, self.deconv3.stride, self.deconv3.padding, self.deconv3.output_padding) + \
            F.conv_transpose2d(x, self.w3_4, self.b3_4, self.deconv3.stride, self.deconv3.padding, self.deconv3.output_padding) + \
            F.conv_transpose2d(x, self.w3_5, self.b3_5, self.deconv3.stride, self.deconv3.padding, self.deconv3.output_padding)
        x = x * self.mulspD2
        ### check charts ###
        r1 = 14
        r2 = 6
        x = torch.floor((x + 2 ** 13)/2 ** 14) #int 
        #####直接输出charts第1列中的结果，通过compression/charts.npy查表进行熵编码
        #####charts.npy的形成见charts.py
        #####具体的熵编码和熵解码步骤见rangec.py
        # idx = 0
        # for s in self.charts[:, 0]:
        #     idx += (x > s)
        # x = self.charts[idx, 1].to(torch.float)
        # x = x2 / (2 ** r2)

        return x
