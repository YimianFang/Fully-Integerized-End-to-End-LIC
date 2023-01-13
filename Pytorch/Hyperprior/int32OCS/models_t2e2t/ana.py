import math
import torch.nn as nn
import torch
from quant_utils import *
from .gradSTE import floorSTE, roundSTE, wSTE, pchannel_intBit_STE, intBit_STE
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
        self.in_scale = 8 #####


    def forward(self, x, muls, relus, Bits=8, if_int16=False):
        clp_k = 7
        stas_max = []
        for i in range(len(relus)):
            stas_max.append(round((math.pow(2.0, Bits) - 1) / relus[i] * (2 ** (16 + clp_k))))
        clp = copy.deepcopy(stas_max)
        scl = copy.deepcopy(relus)

        device = torch.device("cuda:0" if x.is_cuda else "cpu")
        # self.conv1.weight.data = wSTE.apply(self.conv1.weight.data)
        # self.conv1.bias.data = roundSTE.apply(self.conv1.bias.data)
        # x = self.conv1(roundSTE.apply(x * (2 ** self.in_scale)))
        if not if_int16:
        ####################int32####################
            x = F.conv2d(torch.clip(roundSTE.apply(x * (2 ** self.in_scale)), 0, 255), 
                        wSTE.apply(self.conv1.weight), 
                        # roundSTE.apply(self.conv1.weight), 
                        roundSTE.apply(self.conv1.bias), 
                        self.conv1.stride, 
                        self.conv1.padding)
            # x = x.permute(1, 0, 2, 3)
            # x = x * muls[0].reshape(-1, 1, 1, 1)
            x = x * muls[0].reshape(1, -1, 1, 1)
            if x.abs().max()/2**31 >= 1:
                print("Overflow!")
            x = floorSTE.apply((x + 2 ** (16 + self.in_scale - clp_k))/2 ** (17 + self.in_scale - clp_k))
            # x = x.permute(1, 0, 2, 3)
            x = torch.clip(x, 0, clp[0])
            scl0_k = 4
            if scl0_k > 0:
                scl0 = floorSTE.apply((torch.tensor(scl[0]) + 2 ** (scl0_k - 1)) / 2 ** scl0_k)
            x = floorSTE.apply((roundSTE.apply(x * scl0) + 2 ** (15 + clp_k - scl0_k))/2 ** (16 + clp_k - scl0_k))
        else:
        ####################int32/int16####################
            x = F.conv2d(x, 
                        pchannel_intBit_STE.apply(self.conv1.weight), 
                        self.conv1.bias, 
                        self.conv1.stride, 
                        self.conv1.padding)
            clp_0 = (math.pow(2.0, Bits) - 1) / scl[0] * (2 ** 16)
            x = floorSTE.apply((torch.clip(x * scl[0], 0, clp_0 * scl[0]) + 2 ** 15) / (2 ** 16))
        ####################int16####################
        
        # self.conv2.weight.data = wSTE.apply(self.conv2.weight.data)
        # self.conv2.bias.data = roundSTE.apply(self.conv2.bias.data)
        # x= self.conv2(x)
        x = F.conv2d(x, 
                    wSTE.apply(self.conv2.weight), 
                    # roundSTE.apply(self.conv2.weight), 
                    roundSTE.apply(self.conv2.bias), 
                    self.conv2.stride, 
                    self.conv2.padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[1].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (22 - clp_k))/2 ** (23- clp_k))
        # x = x.permute(1, 0, 2, 3)
        x = torch.clip(x, 0, clp[1])
        scl1_k = 4
        if scl1_k > 0:
            scl1 = floorSTE.apply((torch.tensor(scl[1]) + 2 ** (scl1_k - 1)) / 2 ** scl1_k)
        x = floorSTE.apply((roundSTE.apply(x * scl1) + 2 ** (15 + clp_k - scl1_k))/2 ** (16 + clp_k - scl1_k))

        # self.conv3.weight.data = wSTE.apply(self.conv3.weight.data)
        # self.conv3.bias.data = roundSTE.apply(self.conv3.bias.data)
        # x= self.conv3(x)
        x = F.conv2d(x, 
                    wSTE.apply(self.conv3.weight), 
                    # roundSTE.apply(self.conv3.weight), 
                    roundSTE.apply(self.conv3.bias), 
                    self.conv3.stride, 
                    self.conv3.padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[2].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (22 - clp_k))/2 ** (23 - clp_k))
        # x = x.permute(1, 0, 2, 3)
        x = torch.clip(x, 0, clp[2])
        scl2_k = 4
        if scl2_k > 0:
            scl2 = floorSTE.apply((torch.tensor(scl[2]) + 2 ** (scl2_k - 1)) / 2 ** scl2_k)
        x = floorSTE.apply((roundSTE.apply(x * scl2) + 2 ** (15 + clp_k - scl2_k))/2 ** (16 + clp_k - scl2_k))

        # self.conv4.weight.data = wSTE.apply(self.conv4.weight.data)
        # self.conv4.bias.data = roundSTE.apply(self.conv4.bias.data)
        # x= self.conv4(x)
        x = F.conv2d(x, 
                    wSTE.apply(self.conv4.weight), 
                    # roundSTE.apply(self.conv4.weight), 
                    roundSTE.apply(self.conv4.bias), 
                    self.conv4.stride, 
                    self.conv4.padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[3].reshape(1, -1, 1, 1)
        # x = floorSTE.apply((x + 2 ** 18)/2 ** 19)
        delta_dvd = 0 #####
        x = floorSTE.apply((x + 2 **(14 - delta_dvd))/2 ** (15 - delta_dvd)) #####针对ana最后一层的改进，少除2**4       (x + 2 **(14 - delta_dvd))/2 ** (15 - delta_dvd) (x + 2 **(17 - delta_dvd))/2 ** (18 - delta_dvd)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        # x = x.permute(1, 0, 2, 3)

        return x