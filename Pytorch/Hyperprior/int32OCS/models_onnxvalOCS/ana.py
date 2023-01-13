import math
import torch.nn as nn
import torch
from yaml import load
from quant_utils import *
import torch.nn.functional as F

class Analysis_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Analysis_net, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        # self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        # self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        # self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        self.mulsE0 = torch.tensor(np.load("onnxdata/mulsE0.npy")).cuda()
        self.mulsE1 = torch.tensor(np.load("onnxdata/mulsE1.npy")).cuda()
        self.mulsE2 = torch.tensor(np.load("onnxdata/mulsE2.npy")).cuda()
        self.mulsE3 = torch.tensor(np.load("onnxdata/mulsE3.npy")).cuda()
        self.clp = torch.tensor(np.load("onnxdata/clpE.npy")).cuda()
        self.scl = torch.tensor(np.load("onnxdata/sclE.npy")).cuda()
        self.w1_1 = nn.Parameter(torch.floor(self.conv1.weight / 5))
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
        self.w4_1 = nn.Parameter(torch.floor(self.conv4.weight  / 5))
        self.w4_2 = nn.Parameter(torch.floor((self.conv4.weight + 1) / 5))
        self.w4_3 = nn.Parameter(torch.floor((self.conv4.weight + 2) / 5))
        self.w4_4 = nn.Parameter(torch.floor((self.conv4.weight + 3) / 5))
        self.w4_5 = nn.Parameter(torch.floor((self.conv4.weight + 4) / 5))
        self.b4_1 = nn.Parameter(torch.floor(self.conv4.bias / 5))
        self.b4_2 = nn.Parameter(torch.floor((self.conv4.bias + 1) / 5))
        self.b4_3 = nn.Parameter(torch.floor((self.conv4.bias + 2) / 5))
        self.b4_4 = nn.Parameter(torch.floor((self.conv4.bias + 3) / 5))
        self.b4_5 = nn.Parameter(torch.floor((self.conv4.bias + 4) / 5))


    def forward(self, x):
        device = torch.device("cuda:0" if x.is_cuda else "cpu")
        # x = self.conv1(x) ### .to(torch.int32)
        x = F.conv2d(x, self.w1_1, self.b1_1, self.conv1.stride, self.conv1.padding) + F.conv2d(x, self.w1_2, self.b1_2, self.conv1.stride, self.conv1.padding) + \
            F.conv2d(x, self.w1_3, self.b1_3, self.conv1.stride, self.conv1.padding) + F.conv2d(x, self.w1_4, self.b1_4, self.conv1.stride, self.conv1.padding) + \
            F.conv2d(x, self.w1_5, self.b1_5, self.conv1.stride, self.conv1.padding)
        x = x * self.mulsE0
        x = torch.floor((x + 2 ** 19)/2 ** 20)
        # x = torch.div(x + 2 ** 19, 2 ** 20, rounding_mode='floor')
        x = torch.clip(x, 0, self.clp[0])
        x = torch.floor((x * self.scl[0] + 2 ** 18)/2 ** 19)
        # x = torch.div(x * self.scl[0] + 2 ** 18, 2 ** 19, rounding_mode='floor').to(torch.float32)
        
        # x = self.conv2(x) ### .to(torch.int32)
        x = F.conv2d(x, self.w2_1, self.b2_1, self.conv2.stride, self.conv2.padding) + F.conv2d(x, self.w2_2, self.b2_2, self.conv2.stride, self.conv2.padding) + \
            F.conv2d(x, self.w2_3, self.b2_3, self.conv2.stride, self.conv2.padding) + F.conv2d(x, self.w2_4, self.b2_4, self.conv2.stride, self.conv2.padding) + \
            F.conv2d(x, self.w2_5, self.b2_5, self.conv2.stride, self.conv2.padding)
        x = x * self.mulsE1
        x = torch.floor((x + 2 ** 15)/2 ** 16)
        x = torch.clip(x, 0, self.clp[1])
        x = torch.floor((x * self.scl[1] + 2 ** 18)/2 ** 19)

        # x = self.conv3(x) ### .to(torch.int32)
        x = F.conv2d(x, self.w3_1, self.b3_1, self.conv3.stride, self.conv3.padding) + F.conv2d(x, self.w3_2, self.b3_2, self.conv3.stride, self.conv3.padding) + \
            F.conv2d(x, self.w3_3, self.b3_3, self.conv3.stride, self.conv3.padding) + F.conv2d(x, self.w3_4, self.b3_4, self.conv3.stride, self.conv3.padding) + \
            F.conv2d(x, self.w3_5, self.b3_5, self.conv3.stride, self.conv3.padding)
        x = x * self.mulsE2
        x = torch.floor((x + 2 ** 15)/2 ** 16)
        x = torch.clip(x, 0, self.clp[2])
        x = torch.floor((x * self.scl[2] + 2 ** 18)/2 ** 19)

        # x= self.conv4(x) ### .to(torch.int32)
        x = F.conv2d(x, self.w4_1, self.b4_1, self.conv4.stride, self.conv4.padding) + F.conv2d(x, self.w4_2, self.b4_2, self.conv4.stride, self.conv4.padding) + \
            F.conv2d(x, self.w4_3, self.b4_3, self.conv4.stride, self.conv4.padding) + F.conv2d(x, self.w4_4, self.b4_4, self.conv4.stride, self.conv4.padding) + \
            F.conv2d(x, self.w4_5, self.b4_5, self.conv4.stride, self.conv4.padding)
        x = x * self.mulsE3
        x = torch.floor((x + 2 **14)/2 ** 15) #####针对ana最后一层的改进，少除2**4

        return x