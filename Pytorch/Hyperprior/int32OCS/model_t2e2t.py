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
from models_t2e2t import *
import copy


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
        pretrained_dict = torch.load(f, map_location="cpu")
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

class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M
        # self.n = 0

    def forward(self, input_image):
        device = torch.device("cuda:0" if input_image.is_cuda else "cpu")
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16,
                                          input_image.size(3) // 16).to(device)
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64,
                                    input_image.size(3) // 64).to(device)
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        mulsE = torch.load("/data1/fym/pre_int32/checkpoints/qftssim_128_eq_2/iter_324001_muls_Encoder.pth.tar", map_location=device)
        relusE = torch.load("/data1/fym/pre_int32/checkpoints/qftssim_128_eq_2/iter_324001_relus_Encoder.pth.tar", map_location=device)
        # mulsE = torch.load("checkpoints/4use/iter_1063_muls_Encoder.pth.tar", map_location=device)
        # relusE = torch.load("checkpoints/4use/iter_1063_relus_Encoder.pth.tar", map_location=device)
        feature = self.Encoder(input_image, mulsE, relusE)
        batch_size = feature.size()[0]

        mulspE = torch.load("/data1/fym/pre_int32/checkpoints/qftssim_128_eq_2/iter_324001_muls_priorEncoder.pth.tar", map_location=device)
        reluspE = torch.load("/data1/fym/pre_int32/checkpoints/qftssim_128_eq_2/iter_324001_relus_priorEncoder.pth.tar", map_location=device)
        # mulspE = torch.load("checkpoints/4use/iter_1063_muls_priorEncoder.pth.tar", map_location=device)
        # reluspE = torch.load("checkpoints/4use/iter_1063_relus_priorEncoder.pth.tar", map_location=device)
        z = self.priorEncoder(feature, mulspE, reluspE)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        mulspD = torch.load("/data1/fym/pre_int32/checkpoints/qftssim_128_eq_2/iter_324001_muls_priorDecoder.pth.tar", map_location=device)
        reluspD = torch.load("/data1/fym/pre_int32/checkpoints/qftssim_128_eq_2/iter_324001_relus_priorDecoder.pth.tar", map_location=device)
        # mulspD = torch.load("checkpoints/4use/iter_1063_muls_priorDecoder.pth.tar", map_location=device)
        # reluspD = torch.load("checkpoints/4use/iter_1063_relus_priorDecoder.pth.tar", map_location=device)
        recon_sigma = self.priorDecoder(compressed_z, mulspD, reluspD) 

        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        mulsD = torch.load("/data1/fym/pre_int32/checkpoints/qftssim_128_eq_2/iter_324001_muls_Decoder.pth.tar", map_location=device)
        relusD = torch.load("/data1/fym/pre_int32/checkpoints/qftssim_128_eq_2/iter_324001_relus_Decoder.pth.tar", map_location=device)
        # mulsD = torch.load("checkpoints/4use/iter_1063_muls_Decoder.pth.tar", map_location=device)
        # relusD = torch.load("checkpoints/4use/iter_1063_relus_Decoder.pth.tar", map_location=device)
        recon_image = self.Decoder(compressed_feature_renorm, mulsD, relusD)
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))
        
        def feature_probs_based_sigma(feature, sigma):
            sigma = sigma.to(feature.device)
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob
        
        delta_dvd = 0 #####
        # compressed_feature_renorm[compressed_feature_renorm==8] = 7
        compressed_feature_renorm_1 = torch.floor((compressed_feature_renorm + 2 ** (3 + delta_dvd)) / 2 ** (4 + delta_dvd))###针对ana最后一层的改进，少除2**4  #####
        # compressed_feature_renorm_1 = torch.round(compressed_feature_renorm / 2 ** (4 + delta_dvd))###针对ana最后一层的改进，少除2**4  #####
        # torch.save(compressed_feature_renorm, "compression/y/y"+str(self.n)+"_code.pth.tar")
        # self.n += 1
        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm_1, recon_sigma)
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        return clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp