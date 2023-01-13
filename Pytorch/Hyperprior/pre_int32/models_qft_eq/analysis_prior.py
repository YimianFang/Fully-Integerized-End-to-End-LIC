#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
# from .basics import *
from .analysis import Analysis_net
import math
import torch.nn as nn
from torch.nn.quantized import *
import torch
from quant_utils import *
from torch.quantization import QuantStub, DeQuantStub
import copy
import numpy as np

class Analysis_prior_net(nn.Module):
    '''
    Analysis prior net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Analysis_prior_net, self).__init__()
        self.conv1 = nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.conv3 =  nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        # self.quant1 = QuantStub()
        # self.quant2 = QuantStub()
        # self.quant3 = QuantStub()
        # self.dequant = DeQuantStub()
        # self.conv1 = priorAnalysis_subnet1(out_channel_M, out_channel_N)
        # self.conv2 = priorAnalysis_subnet2(out_channel_N, out_channel_N)
        # self.conv3 = priorAnalysis_subnet3(out_channel_N, out_channel_N)

    def forward(self, x):
        x = torch.abs(x)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)

        return x


def build_model():
    input_image = torch.zeros([5, 3, 256, 256])
    analysis_net = Analysis_net()
    analysis_prior_net = Analysis_prior_net()

    feature = analysis_net(input_image)
    z = analysis_prior_net(feature)
    
    print(input_image.size())
    print(feature.size())
    print(z.size())


if __name__ == '__main__':
    build_model()
