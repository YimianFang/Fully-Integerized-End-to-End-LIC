from compressai.models.google import MeanScaleHyperprior
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder
from gradSTE import floorSTE, roundSTE, wSTE, pchannel_intBit_STE, intBit_STE

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import copy
import quant_utils
import os
import math

def save_model(model, loss, iter, name):
    checkpoint = {}
    checkpoint["state"] = model.state_dict()
    checkpoint["loss"] = loss
    torch.save(checkpoint, os.path.join(name, "iter_{}.pth.tar".format(iter)))

def load_qftmodel(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)["int_model"]
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            if k.split(".", 1)[1] in pretrained_dict:
                pretrained_dict[k] = pretrained_dict[k.split(".", 1)[1]]
        pretrained_dict = {k: v.to(torch.float) for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') 
        st = f.find('iter_', st + 5) + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

def load_t2e2tmodel(model, f):
    # pretrained_dict = torch.load(f, map_location="cpu")
    pretrained_dict = torch.load(f)["state"] ###cuda
    pre_loss = torch.load(f)["loss"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v.to(torch.float) for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed]), pre_loss
    else:
        return 0, pre_loss

class g_a(nn.Module):
    def __init__(self, N, M, ga_left):
        super(g_a, self).__init__()
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            nn.ReLU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            conv(N, M, kernel_size=5, stride=2),
        )
        self.in_scale = 8 #####
        self.ga_left = ga_left


    def forward(self, x, muls, muls_dvds, relus, Bits=8, split=4):
        clp_k = 8
        stas_max = []
        for i in range(len(relus)):
            stas_max.append(round((math.pow(2.0, Bits) - 1) / relus[i] * (2 ** (16 + clp_k))))
        clp = copy.deepcopy(stas_max)
        scl = copy.deepcopy(relus)

        x = torch.clip(roundSTE.apply(x * (2 ** self.in_scale)), 0, 255)
        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.g_a[0].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.g_a[0].bias * (2 ** self.in_scale)) + i) / split), 
                        self.g_a[0].stride, self.g_a[0].padding)
        x = xout * muls[0].reshape(1, -1, 1, 1)
        # x = F.conv2d(torch.clip(roundSTE.apply(x * (2 ** self.in_scale)), 0, 255), 
        #             wSTE.apply(self.g_a[0].weight), 
        #             # roundSTE.apply(self.conv1.weight), 
        #             roundSTE.apply(self.g_a[0].bias * (2 ** self.in_scale)), 
        #             self.g_a[0].stride, 
        #             self.g_a[0].padding)
        # x = x * muls[0].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[0] - 1 + self.in_scale - clp_k))/2 ** (muls_dvds[0] + self.in_scale - clp_k))
        x = torch.clip(x, 0, clp[0])
        scl0_k = 3
        if scl0_k > 0:
            scl0 = floorSTE.apply((torch.tensor(scl[0]) + 2 ** (scl0_k - 1)) / 2 ** scl0_k)
        if (x * scl0).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((roundSTE.apply(x * scl0) + 2 ** (15 + clp_k - scl0_k))/2 ** (16 + clp_k - scl0_k))
        
        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.g_a[2].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.g_a[2].bias) + i) / split), 
                        self.g_a[2].stride, self.g_a[2].padding)
        x = xout * muls[1].reshape(1, -1, 1, 1)
        # x = F.conv2d(x, 
        #             wSTE.apply(self.g_a[2].weight), 
        #             # roundSTE.apply(self.conv2.weight), 
        #             roundSTE.apply(self.g_a[2].bias), 
        #             self.g_a[2].stride, 
        #             self.g_a[2].padding)
        # x = x * muls[1].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[1] - 1 - clp_k))/2 ** (muls_dvds[1] - clp_k))
        x = torch.clip(x, 0, clp[1])
        scl1_k = 3
        if scl1_k > 0:
            scl1 = floorSTE.apply((torch.tensor(scl[1]) + 2 ** (scl1_k - 1)) / 2 ** scl1_k)
        if (x * scl1).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((roundSTE.apply(x * scl1) + 2 ** (15 + clp_k - scl1_k))/2 ** (16 + clp_k - scl1_k))

        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.g_a[4].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.g_a[4].bias) + i) / split), 
                        self.g_a[4].stride, self.g_a[4].padding)
        x = xout * muls[2].reshape(1, -1, 1, 1)
        # x = F.conv2d(x, 
        #             wSTE.apply(self.g_a[4].weight), 
        #             roundSTE.apply(self.g_a[4].bias), 
        #             self.g_a[4].stride, 
        #             self.g_a[4].padding)
        # x = x * muls[2].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[2] - 1 - clp_k))/2 ** (muls_dvds[2] - clp_k))
        x = torch.clip(x, 0, clp[2])
        scl2_k = 3
        if scl2_k > 0:
            scl2 = floorSTE.apply((torch.tensor(scl[2]) + 2 ** (scl2_k - 1)) / 2 ** scl2_k)
        if (x * scl2).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((roundSTE.apply(x * scl2) + 2 ** (15 + clp_k - scl2_k))/2 ** (16 + clp_k - scl2_k))

        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.g_a[6].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.g_a[6].bias) + i) / split), 
                        self.g_a[6].stride, self.g_a[6].padding)
        x = xout * muls[3].reshape(1, -1, 1, 1)
        # x = F.conv2d(x, 
        #             wSTE.apply(self.g_a[6].weight), 
        #             roundSTE.apply(self.g_a[6].bias), 
        #             self.g_a[6].stride, 
        #             self.g_a[6].padding)
        # x = x * muls[3].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[3] - 1 - self.ga_left))/2 ** (muls_dvds[3] - self.ga_left)) #####针对ana最后一层的改进，少除2**4

        return x

class h_a(nn.Module):
    def __init__(self, N, M, ga_left):
        super(h_a, self).__init__()
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(),
            conv(N, N, stride=2, kernel_size=5),
            nn.ReLU(),
            conv(N, N, stride=2, kernel_size=5),
        )
        self.ga_left = ga_left


    def forward(self, x, muls, muls_dvds, relus, Bits=8, split=4):
        clp_k = 10
        stas_max = []
        for i in range(len(relus)):
            stas_max.append(round((math.pow(2.0, Bits) - 1) / relus[i] * (2 ** (16 + clp_k))))
        clp = copy.deepcopy(stas_max)
        scl = copy.deepcopy(relus)

        x = x.clamp(-128, 127)
        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.h_a[0].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.h_a[0].bias) + i) / split), 
                        self.h_a[0].stride, self.h_a[0].padding)
        x = xout * muls[0].reshape(1, -1, 1, 1)
        # x = F.conv2d(x, 
        #             wSTE.apply(self.h_a[0].weight), 
        #             roundSTE.apply(self.h_a[0].bias * (2 ** self.ga_left)), 
        #             self.h_a[0].stride, 
        #             self.h_a[0].padding)
        # x = x * muls[0].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[0] - 1 + self.ga_left - clp_k))/2 ** (muls_dvds[0] + self.ga_left - clp_k))
        x = torch.clip(x, 0, clp[0])
        scl0_k = 4 #####
        if scl0_k > 0:
            scl0 = floorSTE.apply((torch.tensor(scl[0]) + 2 ** (scl0_k - 1)) / 2 ** scl0_k)
        if (x * scl0).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x * scl0 + 2 ** (15 + clp_k - scl0_k))/2 ** (16 + clp_k - scl0_k))

        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.h_a[2].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.h_a[2].bias) + i) / split), 
                        self.h_a[2].stride, self.h_a[2].padding)
        x = xout * muls[1].reshape(1, -1, 1, 1)
        # x = F.conv2d(x, 
        #             wSTE.apply(self.h_a[2].weight), 
        #             roundSTE.apply(self.h_a[2].bias), 
        #             self.h_a[2].stride, 
        #             self.h_a[2].padding)
        # x = x * muls[1].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[1] - 1 - clp_k))/2 ** (muls_dvds[1] - clp_k))
        x = torch.clip(x, 0, clp[1])
        scl1_k = 4 #####
        if scl1_k > 0:
            scl1 = floorSTE.apply((torch.tensor(scl[1]) + 2 ** (scl1_k - 1)) / 2 ** scl1_k)
        if (x * scl1).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x * scl1 + 2 ** (15 + clp_k - scl1_k))/2 ** (16 + clp_k - scl1_k))

        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.h_a[4].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.h_a[4].bias) + i) / split), 
                        self.h_a[4].stride, self.h_a[4].padding)
        x = xout * muls[2].reshape(1, -1, 1, 1)
        # x = F.conv2d(x, 
        #             wSTE.apply(self.h_a[4].weight), 
        #             roundSTE.apply(self.h_a[4].bias), 
        #             self.h_a[4].stride, 
        #             self.h_a[4].padding)
        # x = x * muls[2].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[2] - 1))/2 ** muls_dvds[2])

        return x

class h_s(nn.Module):
    def __init__(self, N, M, hs_left):
        super(h_s, self).__init__()
        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.ReLU(),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.ReLU(),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        self.hs_left = hs_left
                

    def forward(self, x, muls, muls_dvds, relus, Bits=8, split=4):
        clp_k = 10
        stas_max = []
        for i in range(len(relus)):
            stas_max.append(round((math.pow(2.0, Bits) - 1) / relus[i] * (2 ** (16 + clp_k))))
        clp = copy.deepcopy(stas_max)
        scl = copy.deepcopy(relus)

        xout = 0.
        for i in range(split):
            xout += F.conv_transpose2d(x, 
                        floorSTE.apply((wSTE.apply(self.h_s[0].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.h_s[0].bias) + i) / split), 
                        self.h_s[0].stride, self.h_s[0].padding, self.h_s[0].output_padding)
        x = xout * muls[0].reshape(1, -1, 1, 1)
        # x = F.conv_transpose2d(x, 
        #             wSTE.apply(self.h_s[0].weight), 
        #             roundSTE.apply(self.h_s[0].bias), 
        #             self.h_s[0].stride, 
        #             self.h_s[0].padding,
        #             self.h_s[0].output_padding)
        # x = x * muls[0].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[0] - 1 - clp_k))/2 ** (muls_dvds[0] - clp_k)) 
        x = torch.clip(x, 0, clp[0])
        scl0_k = 4 #####
        if scl0_k > 0:
            scl0 = floorSTE.apply((torch.tensor(scl[0]) + 2 ** (scl0_k - 1)) / 2 ** scl0_k)
        if (x * scl0).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x * scl0 + 2 ** (15 + clp_k - scl0_k))/2 ** (16 + clp_k - scl0_k))

        xout = 0.
        for i in range(split):
            xout += F.conv_transpose2d(x, 
                        floorSTE.apply((wSTE.apply(self.h_s[2].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.h_s[2].bias) + i) / split), 
                        self.h_s[2].stride, self.h_s[2].padding, self.h_s[2].output_padding)
        x = xout * muls[1].reshape(1, -1, 1, 1)
        # x = F.conv_transpose2d(x, 
        #             wSTE.apply(self.h_s[2].weight), 
        #             roundSTE.apply(self.h_s[2].bias), 
        #             self.h_s[2].stride, 
        #             self.h_s[2].padding,
        #             self.h_s[2].output_padding)
        # x = x * muls[1].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[1] - 1 - clp_k))/2 ** (muls_dvds[1] - clp_k))
        x = torch.clip(x, 0, clp[1])
        scl1_k = 4 #####
        if scl1_k > 0:
            scl1 = floorSTE.apply((torch.tensor(scl[1]) + 2 ** (scl1_k - 1)) / 2 ** scl1_k)
        if (x * scl1).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x * scl1 + 2 ** (15 + clp_k - scl1_k))/2 ** (16 + clp_k - scl1_k))

        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.h_s[4].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.h_s[4].bias) + i) / split), 
                        self.h_s[4].stride, self.h_s[4].padding)
        x = xout * muls[2].reshape(1, -1, 1, 1)
        # x = F.conv2d(x, 
        #             wSTE.apply(self.h_s[4].weight), 
        #             roundSTE.apply(self.h_s[4].bias), 
        #             self.h_s[4].stride, 
        #             self.h_s[4].padding)
        # x = x * muls[2].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[2] - 1 - self.hs_left))/2 ** (muls_dvds[2] - self.hs_left))

        return x

class g_s(nn.Module):
    def __init__(self, N, M, ga_left):
        super(g_s, self).__init__()
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            nn.ReLU(),
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            deconv(N, 3, kernel_size=5, stride=2),
        )
        self.ga_left = ga_left

        
    def forward(self, x, muls, muls_dvds, relus, Bits=8, split=4):
        clp_k = 9 #####
        stas_max = []
        for i in range(len(relus)):
            stas_max.append(round((math.pow(2.0, Bits) - 1) / relus[i] * (2 ** (16 + clp_k))))
        clp = copy.deepcopy(stas_max)
        scl = copy.deepcopy(relus)

        xout = 0.
        for i in range(split):
            xout += F.conv_transpose2d(x, 
                        floorSTE.apply((wSTE.apply(self.g_s[0].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.g_s[0].bias) + i) / split), 
                        self.g_s[0].stride, self.g_s[0].padding, self.g_s[0].output_padding)
        x = xout * muls[0].reshape(1, -1, 1, 1)
        # x = F.conv_transpose2d(x, 
        #             wSTE.apply(self.g_s[0].weight), 
        #             roundSTE.apply(self.g_s[0].bias), 
        #             self.g_s[0].stride, 
        #             self.g_s[0].padding,
        #             self.g_s[0].output_padding)
        # x = x * muls[0].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[0] - 1 - clp_k))/2 ** (muls_dvds[0]  - clp_k)) #####
        x = torch.clip(x, 0, clp[0])
        scl0_k = 3
        if scl0_k > 0:
            scl0 = floorSTE.apply((torch.tensor(scl[0]) + 2 ** (scl0_k - 1)) / 2 ** scl0_k)
        if (x * scl0).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x * scl0 + 2 ** (15 + clp_k - scl0_k))/2 ** (16 + clp_k - scl0_k))
        
        xout = 0.
        for i in range(split):
            xout += F.conv_transpose2d(x, 
                        floorSTE.apply((wSTE.apply(self.g_s[2].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.g_s[2].bias) + i) / split), 
                        self.g_s[2].stride, self.g_s[2].padding, self.g_s[2].output_padding)
        x = xout * muls[1].reshape(1, -1, 1, 1)
        # x = F.conv_transpose2d(x, 
        #             wSTE.apply(self.g_s[2].weight), 
        #             # roundSTE.apply(self.deconv2.weight), 
        #             roundSTE.apply(self.g_s[2].bias), 
        #             self.g_s[2].stride, 
        #             self.g_s[2].padding,
        #             self.g_s[2].output_padding)
        # x = x * muls[1].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[1] - 1 - clp_k))/2 ** (muls_dvds[1] - clp_k)) #####
        x = torch.clip(x, 0, clp[1])
        scl1_k = 4
        if scl1_k > 0:
            scl1 = floorSTE.apply((torch.tensor(scl[1]) + 2 ** (scl1_k - 1)) / 2 ** scl1_k)
        if (x * scl1).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x * scl1 + 2 ** (15 + clp_k - scl1_k))/2 ** (16 + clp_k - scl1_k))

        xout = 0.
        for i in range(split):
            xout += F.conv_transpose2d(x, 
                        floorSTE.apply((wSTE.apply(self.g_s[4].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.g_s[4].bias) + i) / split), 
                        self.g_s[4].stride, self.g_s[4].padding, self.g_s[4].output_padding)
        x = xout * muls[2].reshape(1, -1, 1, 1)
        # x = F.conv_transpose2d(x, 
        #             wSTE.apply(self.g_s[4].weight), 
        #             roundSTE.apply(self.g_s[4].bias), 
        #             self.g_s[4].stride, 
        #             self.g_s[4].padding,
        #             self.g_s[4].output_padding)
        # x = x * muls[2].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[2] - 1 - clp_k))/2 ** (muls_dvds[2] - clp_k)) #####
        x = torch.clip(x, 0, clp[2])
        scl2_k = 3
        if scl2_k > 0:
            scl2 = floorSTE.apply((torch.tensor(scl[2]) + 2 ** (scl2_k - 1)) / 2 ** scl2_k)
        if (x * scl2).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x * scl2 + 2 ** (15 + clp_k - scl2_k))/2 ** (16 + clp_k - scl2_k))

        xout = 0.
        for i in range(split):
            xout += F.conv_transpose2d(x, 
                        floorSTE.apply((wSTE.apply(self.g_s[6].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.g_s[6].bias) + i) / split), 
                        self.g_s[6].stride, self.g_s[6].padding, self.g_s[6].output_padding)
        x = xout * muls[3].reshape(1, -1, 1, 1)
        # x = F.conv_transpose2d(x, 
        #             wSTE.apply(self.g_s[6].weight), 
        #             roundSTE.apply(self.g_s[6].bias), 
        #             self.g_s[6].stride, 
        #             self.g_s[6].padding,
        #             self.g_s[6].output_padding)
        # x = x * muls[3].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[3] - 9))/2 ** (muls_dvds[3] - 8))
        x /= 255 #####

        return x

class entropy_parameters(nn.Module):
    def __init__(self, N, M, hs_left, en_left):
        super(entropy_parameters, self).__init__()
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.ReLU(),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.ReLU(),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.hs_left = hs_left
        self.en_left = en_left


    def forward(self, x, muls, muls_dvds, relus, Bits=8, split=4):
        clp_k = 9
        stas_max = []
        for i in range(len(relus)):
            stas_max.append(round((math.pow(2.0, Bits) - 1) / relus[i] * (2 ** (16 + clp_k))))
        clp = copy.deepcopy(stas_max)
        scl = copy.deepcopy(relus)

        x = x.clamp(-128, 127)
        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.entropy_parameters[0].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.entropy_parameters[0].bias) + i) / split), 
                        self.entropy_parameters[0].stride, self.entropy_parameters[0].padding)
        x = xout * muls[0].reshape(1, -1, 1, 1)
        # x = F.conv2d(x, 
        #             wSTE.apply(self.entropy_parameters[0].weight), 
        #             roundSTE.apply(self.entropy_parameters[0].bias * (2 ** self.hs_left)), 
        #             self.entropy_parameters[0].stride, 
        #             self.entropy_parameters[0].padding)
        # x = x * muls[0].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[0] - 1 + self.hs_left - clp_k))/2 ** (muls_dvds[0] + self.hs_left - clp_k))
        x = torch.clip(x, 0, clp[0])
        scl0_k = 4 #####
        if scl0_k > 0:
            scl0 = floorSTE.apply((torch.tensor(scl[0]) + 2 ** (scl0_k - 1)) / 2 ** scl0_k)
        if (x * scl0).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x * scl0 + 2 ** (15 + clp_k - scl0_k))/2 ** (16 + clp_k - scl0_k))

        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.entropy_parameters[2].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.entropy_parameters[2].bias) + i) / split), 
                        self.entropy_parameters[2].stride, self.entropy_parameters[2].padding)
        x = xout * muls[1].reshape(1, -1, 1, 1)
        # x = F.conv2d(x, 
        #             wSTE.apply(self.entropy_parameters[2].weight), 
        #             roundSTE.apply(self.entropy_parameters[2].bias), 
        #             self.entropy_parameters[2].stride, 
        #             self.entropy_parameters[2].padding)
        # x = x * muls[1].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[1] - 1 - clp_k))/2 ** (muls_dvds[1] - clp_k))
        x = torch.clip(x, 0, clp[1])
        scl1_k = 4 #####
        if scl1_k > 0:
            scl1 = floorSTE.apply((torch.tensor(scl[1]) + 2 ** (scl1_k - 1)) / 2 ** scl1_k)
        if (x * scl1).abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x * scl1 + 2 ** (15 + clp_k - scl1_k))/2 ** (16 + clp_k - scl1_k))

        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.entropy_parameters[4].weight) + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.entropy_parameters[4].bias) + i) / split), 
                        self.entropy_parameters[4].stride, self.entropy_parameters[4].padding)
        x = xout * muls[2].reshape(1, -1, 1, 1)
        # x = F.conv2d(x, 
        #             wSTE.apply(self.entropy_parameters[4].weight), 
        #             roundSTE.apply(self.entropy_parameters[4].bias), 
        #             self.entropy_parameters[4].stride, 
        #             self.entropy_parameters[4].padding)
        # x = x * muls[2].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[2] - 1 - self.en_left))/2 ** (muls_dvds[2] - self.en_left))

        return x

class context_prediction(nn.Module):
    def __init__(self, N, M, ga_left, hs_left):
        super(context_prediction, self).__init__()
        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )
        self.hs_left = hs_left


    def forward(self, x, muls, muls_dvds, relus, Bits=8, split=4):
        xout = 0.
        for i in range(split):
            xout += F.conv2d(x, 
                        floorSTE.apply((wSTE.apply(self.context_prediction.weight) * self.context_prediction.mask + i) / split), 
                        floorSTE.apply((roundSTE.apply(self.context_prediction.bias) + i) / split), 
                        self.context_prediction.stride, self.context_prediction.padding)
        x = xout * muls[0].reshape(1, -1, 1, 1)
        # x = F.conv2d(x, 
        #             wSTE.apply(self.context_prediction.weight) * self.context_prediction.mask, 
        #             roundSTE.apply(self.context_prediction.bias), 
        #             self.context_prediction.stride, 
        #             self.context_prediction.padding)
        # x = x * muls[0].reshape(1, -1, 1, 1)
        if x.abs().max()/2**31 >= 1:
            print("Overflow!")
        x = floorSTE.apply((x + 2 ** (muls_dvds[0] - 1 - self.hs_left))/2 ** (muls_dvds[0] - self.hs_left))

        return x
        
class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, ga_left=2, hs_left=4, en_left=1, prepath="", preiter=1, device="cuda:0", **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.ga_left = ga_left
        self.hs_left = hs_left
        self.en_left = en_left

        self.g_a = g_a(N, M, self.ga_left)

        self.g_s = g_s(N, M, self.ga_left)

        self.h_a = h_a(N, M, self.ga_left)

        self.h_s = h_s(N, M, self.hs_left)

        self.entropy_parameters = entropy_parameters(N, M, self.hs_left, self.en_left)

        self.context_prediction = context_prediction(N, M, self.ga_left, self.hs_left)

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        load_path = os.path.join(prepath, f"iter_{preiter}")
        self.muls = {}
        for s in ["h_s", "h_a", "g_s", "g_a", "entropy_parameters", "context_prediction"]: 
            self.muls[s] = torch.load(os.path.join(load_path, f"iter_{preiter}_muls_" + s + ".pth.tar"), map_location=device)
        self.muls_dvds = {"h_s":[25, 25, 25], "h_a":[22, 23, 25], 
                "g_s":[23, 24, 24, 28], "g_a":[16, 25, 25, 23],
                "entropy_parameters":[23, 23, 23], "context_prediction":[23]} #####
        self.relus = {}
        for s in ["h_s", "h_a", "g_s", "g_a", "entropy_parameters", "context_prediction"]: 
            self.relus[s] = torch.load(os.path.join(load_path, f"iter_{preiter}_relus_" + s + ".pth.tar"), map_location=device)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x, self.muls["g_a"], self.muls_dvds["g_a"], self.relus["g_a"])
        z = self.h_a(y, self.muls["h_a"], self.muls_dvds["h_a"], self.relus["h_a"])
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat, self.muls["h_s"], self.muls_dvds["h_s"], self.relus["h_s"])

        #####针对ana最后一层的改进，少除2**ga_left
        y = y / 2 ** self.ga_left
        # y = floorSTE.apply((y + 2 ** (self.ga_left - 1)) / (2 ** self.ga_left))
        
        if self.training:
            half = float(0.5)
            noise = torch.empty_like(y).uniform_(-half, half)
            y_hat = y + noise
        else:
            y_hat = floorSTE.apply(y + 0.5)
        # y_hat = self.gaussian_conditional.quantize(
        #     y, "noise" if self.training else "dequantize"
        # )
        ctx_params = self.context_prediction(y_hat,
            self.muls["context_prediction"],
            self.muls_dvds["context_prediction"],
            self.relus["context_prediction"])
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1),
            self.muls["entropy_parameters"],
            self.muls_dvds["entropy_parameters"],
            self.relus["entropy_parameters"])
        gaussian_params /= 2 ** self.en_left
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, self.muls["g_s"], self.muls_dvds["g_s"], self.relus["g_s"])
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }