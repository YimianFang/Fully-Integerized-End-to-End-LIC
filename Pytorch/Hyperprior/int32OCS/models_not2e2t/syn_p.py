#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
# from .basics import *
import math
import torch.nn as nn
import torch
from .gradSTE import floorSTE, roundSTE, wSTE
import copy
import torch.nn.functional as F

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
        self.charts = torch.load("compression/charts.pth.tar") #####165个元素
        # self.n = 0
                

    def forward(self, x, muls, relus, Bits=8):
        clp_k = 9
        stas_max = []
        for i in range(len(relus)):
            stas_max.append(round((math.pow(2.0, Bits) - 1) / relus[i] * (2 ** (16 + clp_k))))
        clp = copy.deepcopy(stas_max)
        scl = copy.deepcopy(relus)

        device = torch.device("cuda:0" if x.is_cuda else "cpu")
        # self.deconv1.weight.data = wSTE.apply(self.deconv1.weight.data)
        # self.deconv1.bias.data = roundSTE.apply(self.deconv1.bias.data)
        # x = self.deconv1(x)
        x = F.conv_transpose2d(x, 
                    # wSTE.apply(self.deconv1.weight), 
                    roundSTE.apply(self.deconv1.weight), 
                    roundSTE.apply(self.deconv1.bias), 
                    self.deconv1.stride, 
                    self.deconv1.padding,
                    self.deconv1.output_padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[0].reshape(1, -1, 1, 1)
        x = floorSTE.apply((x + 2 ** (17 - clp_k))/2 ** (18 - clp_k))
        # x = x.permute(1, 0, 2, 3)
        x = torch.clip(x, 0, clp[0])
        scl0_k = 4
        if scl0_k > 0:
            scl0 = floorSTE.apply((torch.tensor(scl[0]) + 2 ** (scl0_k - 1)) / 2 ** scl0_k)
        x = floorSTE.apply((x * scl0 + 2 ** (15 + clp_k - scl0_k))/2 ** (16 + clp_k - scl0_k))

        # self.deconv2.weight.data = wSTE.apply(self.deconv2.weight.data)
        # self.deconv2.bias.data = roundSTE.apply(self.deconv2.bias.data)
        # x= self.deconv2(x)
        x = F.conv_transpose2d(x, 
                    # wSTE.apply(self.deconv2.weight), 
                    roundSTE.apply(self.deconv2.weight), 
                    roundSTE.apply(self.deconv2.bias), 
                    self.deconv2.stride, 
                    self.deconv2.padding,
                    self.deconv2.output_padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[1].reshape(1, -1, 1, 1)
        x = floorSTE.apply((x + 2 ** (18 - clp_k))/2 ** (19 - clp_k))
        # x = x.permute(1, 0, 2, 3)
        x = torch.clip(x, 0, clp[1])
        scl1_k = 4
        if scl1_k > 0:
            scl1 = floorSTE.apply((torch.tensor(scl[1]) + 2 ** (scl1_k - 1)) / 2 ** scl1_k)
        x = floorSTE.apply((x * scl1 + 2 ** (15 + clp_k - scl1_k))/2 ** (16 + clp_k - scl1_k))

        # self.deconv3.weight.data = wSTE.apply(self.deconv3.weight.data)
        # self.deconv3.bias.data = roundSTE.apply(self.deconv3.bias.data)
        # x= self.deconv3(x)
        x = F.conv_transpose2d(x, 
                    # wSTE.apply(self.deconv3.weight), 
                    roundSTE.apply(self.deconv3.weight), 
                    roundSTE.apply(self.deconv3.bias), 
                    self.deconv3.stride, 
                    self.deconv3.padding,
                    self.deconv3.output_padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[2].reshape(1, -1, 1, 1)
        # x = x.permute(1, 0, 2, 3)
        if self.training:
        #####flt exp
            # x = floorSTE.apply((x + 2 ** 17)/2 ** 18) #int
            x = x / 2 ** 18 #flt
            x = torch.exp(x) #####check charts
        #####
        else:
        #####
            # r1 = 14
            # x = floorSTE.apply((x + 2 ** (r1 - 1))/2 ** r1) #int
            # x = x / 2 ** (18 - r1)
            # x = torch.exp(x) #####check charts
            # r2 = 6
            # x = torch.round(x * (2 ** r2))
            # x = x / (2 ** r2)
        #####
            r1 = 14
            r2 = 6
            x = torch.floor((x + 2 ** (r1 - 1))/2 ** r1) #int
            # torch.save(x, "compression/noexp/noexp"+str(self.n)+"_code.pth.tar")
            # self.n += 1
            #####
            idx = 0
            for s in self.charts[:, 0]:
                idx += (x > s)
            x = self.charts[idx, 1].to(torch.float)
            x = x / (2 ** r2)
            #####
        #####diff
            # r1 = 14
            # x = floorSTE.apply((x + 2 ** (r1 - 1))/2 ** r1) #int
            # x2 = copy.deepcopy(x)
            # for i in self.charts:
            #     x2[x == i] = self.charts[i]
            # x = x / 2 ** (18 - r1)
            # x = torch.exp(x) #####check charts
            # r2 = 6
            # x = torch.round(x * (2 ** r2))
            # x = x / (2 ** r2)

        return x