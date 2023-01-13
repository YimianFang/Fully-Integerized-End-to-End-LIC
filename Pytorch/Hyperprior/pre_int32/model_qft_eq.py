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
from models_qft_eq import *
import copy
import quant_utils


def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))

def save_checkpoint(model, state, sf_list, loss, iter, name):
    torch.save({"model":model.state_dict(), "state":state, "sf_list":sf_list, "loss":loss}, os.path.join(name, "iter_{}.pth.tar".format(iter)))

def save_checkpoint_intconv(model, state, scale, Bpp, iter, name):
    torch.save({"model": model.state_dict(), "state": state, "scale": scale, "Bpp": Bpp},
               os.path.join(name, "iter_{}.pth.tar".format(iter)))

def save_checkpoint_int32(model_dict, muls, relus, act_target_layer, iter, name):
    torch.save(model_dict, os.path.join(name, "iter_{}.pth.tar".format(iter)))
    for i in act_target_layer:
        torch.save(muls[i], os.path.join(name, "iter_{}_muls_{}.pth.tar".format(str(iter), str(i))))
    for i in act_target_layer:
        torch.save(relus[i], os.path.join(name, "iter_{}_relus_{}.pth.tar".format(str(iter), str(i))))

def save_checkpoint_int32_actft(model_dict, float_dict, state, preLoss, muls, relus, act_target_layer, iter, name):
    torch.save({"int_model":model_dict, "float_dict": float_dict, "float_state": state, "preLoss": preLoss}, os.path.join(name, "iter_{}.pth.tar".format(iter)))
    for i in act_target_layer:
        torch.save(muls[i], os.path.join(name, "iter_{}_muls_{}.pth.tar".format(str(iter), str(i))))
    for i in act_target_layer:
        torch.save(relus[i], os.path.join(name, "iter_{}_relus_{}.pth.tar".format(str(iter), str(i))))

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

def load_model_qft(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict:
            if "deconv" in k and "weight" in k:
                pretrained_dict[k] = pretrained_dict[k].permute(1, 0, 2, 3).flip([2,3])
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

def load_checkpoint(model, f):
    with open(f, 'rb') as f:
        file = torch.load(f, map_location=torch.device('cpu'))
        if "state" in file:
            pretrained_dict = file["model"]
            state = file["state"]
            loss = file["loss"]
            model_dict = model.state_dict()
            pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() if k.replace('module.','') in model_dict}
            sf_list = file["sf_list"]
        else:
            pretrained_dict = file
            loss = 10000
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if not "state" in file:
            state = copy.deepcopy(model.state_dict())
            sf_list = quant_utils.get_sf_CW(model, state)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed]), state, sf_list, loss
    else:
        return 0, state, sf_list, loss

def load_checkpoint_intconv(model, f):
    with open(f, 'rb') as f:
        file = torch.load(f, map_location=torch.device('cpu'))
        if "state" in file:
            pretrained_dict = file["model"]
            state = file["state"]
            Bpp = file["Bpp"]
            model_dict = model.state_dict()
            pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() if k.replace('module.','') in model_dict}
            scale = file["scale"]
        else:
            pretrained_dict = file
            Bpp = 10
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if not "state" in file:
            state = copy.deepcopy(model.state_dict())
            scale = quant_utils.get_sf_CW(model, state)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed]), state, scale, Bpp
    else:
        return 0, state, scale, Bpp

def load_checkpoint_int32(model, f):
    with open(f, 'rb') as f:
        file = torch.load(f, map_location=torch.device('cpu'))
        pretrained_dict = file["model"]
        muls = file["muls"]
        relus = file["relus"]
        model_dict = model.state_dict()
        pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() if k.replace('module.','') in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed]), muls, relus
    else:
        return 0, muls, relus

def load_checkpoint_int32_actft(model, f):
    with open(f, 'rb') as f:
        file = torch.load(f, map_location=torch.device('cpu'))
        int_model = file["int_model"]
        pretrained_dict = file["float_dict"]
        state = file["float_state"]
        preLoss = file["preLoss"]
        model_dict = model.state_dict()
        pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() if k.replace('module.','') in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed]), int_model, state, preLoss
    else:
        return 0, int_model, state, preLoss

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

    def forward(self, input_image):
        device = torch.device("cuda:0" if input_image.is_cuda else "cpu")
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16,
                                          input_image.size(3) // 16).to(device)
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64,
                                    input_image.size(3) // 64).to(device)
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        
        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]

        z = self.priorEncoder(feature)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        recon_sigma = self.priorDecoder(compressed_z)

        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        recon_image = self.Decoder(compressed_feature_renorm)

        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        def feature_probs_based_sigma(feature, sigma):
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

        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        return clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp