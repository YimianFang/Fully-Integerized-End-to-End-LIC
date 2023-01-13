import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models_onnxvalOCS import *
import copy
from models_onnxval.gradSTE import wSTE

def save_model(model, iter, name):
    s_dict = {}
    model_dict = model.state_dict()
    for k in model_dict:
        if k == 'Encoder.conv1.bias':
            s_dict[k] = torch.round(model_dict[k] / (2 ** 8)) #.to(torch.int32)
            continue
        if k == 'priorEncoder.conv1.bias':
            s_dict[k] = torch.round(model_dict[k] / (2 ** 4)) #.to(torch.int32)
            continue
        if "deconv" in k and "weight" in k:
            s_dict[k] = torch.flip(model_dict[k].permute(1, 0, 2, 3), [2,3]) #.to(torch.int32)
            continue
        if "weight" in k or "bias" in k:
            s_dict[k] = model_dict[k] #.to(torch.int32)
            continue
        s_dict[k] = model_dict[k]
    torch.save(s_dict, os.path.join(name, "iter_{}.pth.tar".format(iter)))

def load_qftmodel(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)["int_model"]
        # pretrained_dict = torch.load(f) ###cuda
        model_dict = model.state_dict()
        pretrained_dict = {k: v.to(torch.float) for k, v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict:
            if k == 'Encoder.conv1.bias':
                pretrained_dict[k] =torch.round(pretrained_dict[k] * (2 ** 8))
            if k == 'priorEncoder.conv1.bias':
                pretrained_dict[k] =torch.round(pretrained_dict[k] * (2 ** 4))
            if "deconv" in k and "weight" in k:
                pretrained_dict[k] = torch.flip(pretrained_dict[k].permute(1, 0, 2, 3), [2,3])
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

def load_t2e2tmodel(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        # pretrained_dict = torch.load(f) ###cuda
        model_dict = model.state_dict()
        pretrained_dict = {k: v.to(torch.float) for k, v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict:
            if k == 'Encoder.conv1.bias':
                pretrained_dict[k] =torch.round(pretrained_dict[k] * (2 ** 8))
            if k == 'priorEncoder.conv1.bias':
                pretrained_dict[k] =torch.round(pretrained_dict[k] * (2 ** 4))
            if "deconv" in k and "weight" in k:
                pretrained_dict[k] = torch.flip(pretrained_dict[k].permute(1, 0, 2, 3), [2,3])
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

def load_t2e2tmodel_onnxval(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        # pretrained_dict = torch.load(f) ###cuda
        model_dict = model.state_dict()
        pretrained_dict = {k: v.to(torch.float) for k, v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict:
            if k == 'Encoder.conv1.bias':
                pretrained_dict[k] =torch.round(pretrained_dict[k] * (2 ** 8))
            if k == 'priorEncoder.conv1.bias':
                pretrained_dict[k] =torch.round(pretrained_dict[k] * (2 ** 4))
            if "deconv" in k and "weight" in k:
                pretrained_dict[k] = torch.flip(pretrained_dict[k].permute(1, 0, 2, 3), [2,3])
        for k in pretrained_dict:
            if "bias" in k:
                pretrained_dict[k] = torch.round(pretrained_dict[k]).to(torch.int32)
            if "weight" in k:
                pretrained_dict[k] = wSTE.apply(pretrained_dict[k]).to(torch.int32)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_image):
        device = torch.device("cuda:0" if input_image.is_cuda else "cpu")
        feature = self.Encoder(input_image)

        z = self.priorEncoder(feature)
        compressed_z = z

        recon_sigma = self.priorDecoder(compressed_z) 

        compressed_feature_renorm = feature
        recon_image = self.Decoder(compressed_feature_renorm)

        # compressed_feature_renorm[compressed_feature_renorm==8] =7
        # compressed_feature_renorm = torch.floor((compressed_feature_renorm + 2 ** 3) / 2 ** 4)###针对ana最后一层的改进，少除2**4  #####

        return recon_image, compressed_feature_renorm, compressed_z, recon_sigma
        
        ##############################
        ###recon_image是整数，未除以255
        ###compressed_feature_renorm 未除以2**4后的整数y，不可直接用于编码
        ###compressed_z 整数z，可直接用于编码
        ###recon_sigma pd输出，除以(2 ** r1)，未进行exp的整数，exp和编码都通过charts表实现