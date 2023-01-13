import math
import torch.nn as nn
import torch
from quant_utils import *
from .gradSTE import floorSTE, roundSTE, wSTE
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

        
    def forward(self, x, muls, relus, Bits=8):
        clp_k = 7 #####6
        stas_max = []
        for i in range(len(relus)):
            stas_max.append(round((math.pow(2.0, Bits) - 1) / relus[i] * (2 ** (16 + clp_k))))
        clp = copy.deepcopy(stas_max)
        scl = copy.deepcopy(relus)

        device = torch.device("cuda:0" if x.is_cuda else "cpu")
        # self.deconv1.weight.data = wSTE.apply(self.deconv1.weight.data)
        # self.deconv1.bias.data = roundSTE.apply(self.deconv1.bias.data)
        # x = self.deconv1(x)
        delta_dvd = 0 #####
        # x[x==8] = 7
        x = floorSTE.apply((x + (2 ** (3 + delta_dvd))) / (2 ** (4 + delta_dvd))) #####针对ana最后一层的改进，少除2**4
        x = F.conv_transpose2d(x, 
                    wSTE.apply(self.deconv1.weight), 
                    # roundSTE.apply(self.deconv1.weight), 
                    roundSTE.apply(self.deconv1.bias), 
                    self.deconv1.stride, 
                    self.deconv1.padding,
                    self.deconv1.output_padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[0].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x =floorSTE.apply((x + 2 ** (18 - clp_k))/2 ** (19  - clp_k)) #####
        # x = x.permute(1, 0, 2, 3)
        x = torch.clip(x, 0, clp[0])
        scl0_k = 2
        if scl0_k > 0:
            scl0 = floorSTE.apply((torch.tensor(scl[0]) + 2 ** (scl0_k - 1)) / 2 ** scl0_k)
        x = floorSTE.apply((x * scl0 + 2 ** (15 + clp_k - scl0_k))/2 ** (16 + clp_k - scl0_k))
        
        # self.deconv2.weight.data = wSTE.apply(self.deconv2.weight.data)
        # self.deconv2.bias.data = roundSTE.apply(self.deconv2.bias.data)
        # x= self.deconv2(x)
        x = F.conv_transpose2d(x, 
                    wSTE.apply(self.deconv2.weight), 
                    # roundSTE.apply(self.deconv2.weight), 
                    roundSTE.apply(self.deconv2.bias), 
                    self.deconv2.stride, 
                    self.deconv2.padding,
                    self.deconv2.output_padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[1].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (18 - clp_k))/2 ** (19 - clp_k)) #####
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
                    wSTE.apply(self.deconv3.weight), 
                    # roundSTE.apply(self.deconv3.weight), 
                    roundSTE.apply(self.deconv3.bias), 
                    self.deconv3.stride, 
                    self.deconv3.padding,
                    self.deconv3.output_padding)
        # x = x.permute(1, 0, 2, 3)
        x = x * muls[2].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (18 - clp_k))/2 ** (19 - clp_k)) #####
        # x = x.permute(1, 0, 2, 3)
        x = torch.clip(x, 0, clp[2])
        scl2_k = 4
        if scl2_k > 0:
            scl2 = floorSTE.apply((torch.tensor(scl[2]) + 2 ** (scl2_k - 1)) / 2 ** scl2_k)
        x = floorSTE.apply((x * scl2 + 2 ** (15 + clp_k - scl2_k))/2 ** (16 + clp_k - scl2_k))

        # self.deconv4.weight.data = wSTE.apply(self.deconv4.weight.data)
        # self.deconv4.bias.data = roundSTE.apply(self.deconv4.bias.data)
        # x= self.deconv4(x)
        x = F.conv_transpose2d(x, 
                    wSTE.apply(self.deconv4.weight), 
                    # roundSTE.apply(self.deconv4.weight), 
                    roundSTE.apply(self.deconv4.bias), 
                    self.deconv4.stride, 
                    self.deconv4.padding,
                    self.deconv4.output_padding)
        # x = x.permute(1, 0, 2, 3)
        # ############
        # x = x * (muls[3].reshape(1, -1, 1, 1) * 2 ** 7 ) #####7
        # if x.abs().max()/2**31 >= 1:
        #   print("Overflow!")
        # # x = x.permute(1, 0, 2, 3)
        # # x = floorSTE.apply((x * 2 ** 7 + 2 ** 22)/2 ** 23)/128
        # x = floorSTE.apply((x+ 2 ** 21)/2 ** 22)
        # x /= 255 #####
        # ############
        x = x * muls[3].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        # x = x.permute(1, 0, 2, 3)
        # x = floorSTE.apply((x * 2 ** 7 + 2 ** 22)/2 ** 23)/128
        x = floorSTE.apply((x + 2 ** 14)/2 ** 15)
        x /= 255 #####

        return x