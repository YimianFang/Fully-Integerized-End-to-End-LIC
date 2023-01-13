import math
import torch.nn as nn
import torch
from quant_utils import *
import torch.nn.functional as F

class Synthesis_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_N=192, out_channel_M=323):
        super(Synthesis_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        # self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        # self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        # self.relu3 = nn.ReLU()
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1)
        self.mulsD0 = torch.tensor(np.load("onnxdata/mulsD0.npy")).cuda()
        self.mulsD1 = torch.tensor(np.load("onnxdata/mulsD1.npy")).cuda()
        self.mulsD2 = torch.tensor(np.load("onnxdata/mulsD2.npy")).cuda()
        self.mulsD3 = torch.tensor(np.load("onnxdata/mulsD3.npy")).cuda()
        self.clp = torch.tensor(np.load("onnxdata/clpD.npy")).cuda()
        self.scl = torch.tensor(np.load("onnxdata/sclD.npy")).cuda()
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
        self.w4_1 = nn.Parameter(torch.floor(self.deconv4.weight  / 5))
        self.w4_2 = nn.Parameter(torch.floor((self.deconv4.weight + 1) / 5))
        self.w4_3 = nn.Parameter(torch.floor((self.deconv4.weight + 2) / 5))
        self.w4_4 = nn.Parameter(torch.floor((self.deconv4.weight + 3) / 5))
        self.w4_5 = nn.Parameter(torch.floor((self.deconv4.weight + 4) / 5))
        self.b4_1 = nn.Parameter(torch.floor(self.deconv4.bias / 5))
        self.b4_2 = nn.Parameter(torch.floor((self.deconv4.bias + 1) / 5))
        self.b4_3 = nn.Parameter(torch.floor((self.deconv4.bias + 2) / 5))
        self.b4_4 = nn.Parameter(torch.floor((self.deconv4.bias + 3) / 5))
        self.b4_5 = nn.Parameter(torch.floor((self.deconv4.bias + 4) / 5))

        
    def forward(self, x):
        device = torch.device("cuda:0" if x.is_cuda else "cpu")
        x = torch.floor((x + (2 ** 3)) / (2 ** 4)) #####针对ana最后一层的改进，少除2**4
        # x = self.deconv1(x) ### .to(torch.int32)
        x = F.conv_transpose2d(x, self.w1_1, self.b1_1, self.deconv1.stride, self.deconv1.padding, self.deconv1.output_padding) + \
            F.conv_transpose2d(x, self.w1_2, self.b1_2, self.deconv1.stride, self.deconv1.padding, self.deconv1.output_padding) + \
            F.conv_transpose2d(x, self.w1_3, self.b1_3, self.deconv1.stride, self.deconv1.padding, self.deconv1.output_padding) + \
            F.conv_transpose2d(x, self.w1_4, self.b1_4, self.deconv1.stride, self.deconv1.padding, self.deconv1.output_padding) + \
            F.conv_transpose2d(x, self.w1_5, self.b1_5, self.deconv1.stride, self.deconv1.padding, self.deconv1.output_padding)
        x = x * self.mulsD0
        x = torch.floor((x + 2 ** 12)/2 ** 13)
        x = torch.clip(x, 0, self.clp[0])
        x = torch.floor((x * self.scl[0] + 2 ** 19)/2 ** 20)
        
        # x= self.deconv2(x) ### .to(torch.int32)
        x = F.conv_transpose2d(x, self.w2_1, self.b2_1, self.deconv2.stride, self.deconv2.padding, self.deconv2.output_padding) + \
            F.conv_transpose2d(x, self.w2_2, self.b2_2, self.deconv2.stride, self.deconv2.padding, self.deconv2.output_padding) + \
            F.conv_transpose2d(x, self.w2_3, self.b2_3, self.deconv2.stride, self.deconv2.padding, self.deconv2.output_padding) + \
            F.conv_transpose2d(x, self.w2_4, self.b2_4, self.deconv2.stride, self.deconv2.padding, self.deconv2.output_padding) + \
            F.conv_transpose2d(x, self.w2_5, self.b2_5, self.deconv2.stride, self.deconv2.padding, self.deconv2.output_padding)
        x = x * self.mulsD1
        x = torch.floor((x + 2 ** 12)/2 ** 13)
        x = torch.clip(x, 0, self.clp[1])
        x = torch.floor((x * self.scl[1] + 2 ** 17)/2 ** 18)

        # x= self.deconv3(x) ### .to(torch.int32)
        x = F.conv_transpose2d(x, self.w3_1, self.b3_1, self.deconv3.stride, self.deconv3.padding, self.deconv3.output_padding) + \
            F.conv_transpose2d(x, self.w3_2, self.b3_2, self.deconv3.stride, self.deconv3.padding, self.deconv3.output_padding) + \
            F.conv_transpose2d(x, self.w3_3, self.b3_3, self.deconv3.stride, self.deconv3.padding, self.deconv3.output_padding) + \
            F.conv_transpose2d(x, self.w3_4, self.b3_4, self.deconv3.stride, self.deconv3.padding, self.deconv3.output_padding) + \
            F.conv_transpose2d(x, self.w3_5, self.b3_5, self.deconv3.stride, self.deconv3.padding, self.deconv3.output_padding)
        x = x * self.mulsD2
        x = torch.floor((x + 2 ** 12)/2 ** 13)
        x = torch.clip(x, 0, self.clp[2])
        x = torch.floor((x * self.scl[2] + 2 ** 17)/2 ** 18)

        # x= self.deconv4(x) ### .to(torch.int32)
        x = F.conv_transpose2d(x, self.w4_1, self.b4_1, self.deconv4.stride, self.deconv4.padding, self.deconv4.output_padding) + \
            F.conv_transpose2d(x, self.w4_2, self.b4_2, self.deconv4.stride, self.deconv4.padding, self.deconv4.output_padding) + \
            F.conv_transpose2d(x, self.w4_3, self.b4_3, self.deconv4.stride, self.deconv4.padding, self.deconv4.output_padding) + \
            F.conv_transpose2d(x, self.w4_4, self.b4_4, self.deconv4.stride, self.deconv4.padding, self.deconv4.output_padding) + \
            F.conv_transpose2d(x, self.w4_5, self.b4_5, self.deconv4.stride, self.deconv4.padding, self.deconv4.output_padding)
        x = x * self.mulsD3
        x = torch.floor((x + 2 ** 21)/2 ** 22) #####/255 未除以255

        return x