"""
    Quantized JAHP, implemented in PyTorch.
"""

from compressai.models.google import MeanScaleHyperprior
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder
from ..quantization_utils.quant_modules import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import copy
import os
import math


class floorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = torch.floor(input).to(torch.int32).to(torch.float32)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        # print("floorSTE_grad: ", grad_output)
        return grad_output, None, None, None


class add_zeros(nn.Module):
    def __init__(self, channel_in, kernel=5, stride=2, padding=2, output_padding=1):
        super(add_zeros, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.channel_in = channel_in
    
    def forward(self, x):
        batch_size = x.size()[0]
        h_in = x.size()[2]
        w_in = x.size()[3]
        h_in_zeros = h_in + (self.stride - 1) * (h_in - 1) + 2 * (self.kernel - self.padding - 1) + self.output_padding
        w_in_zeros = w_in + (self.stride - 1) * (w_in - 1) + 2 * (self.kernel - self.padding - 1) + self.output_padding
        if x.is_cuda:
            feather_in_zeros = torch.zeros([batch_size, self.channel_in, h_in_zeros, w_in_zeros]).cuda()
        else:
            feather_in_zeros = torch.zeros([batch_size, self.channel_in, h_in_zeros, w_in_zeros]).cpu()
        feather_in_zeros[:, :, self.kernel - self.padding - 1:h_in_zeros - self.kernel + self.padding + 1 - self.output_padding:self.stride, self.kernel - self.padding - 1:w_in_zeros - self.kernel + self.padding + 1 - self.output_padding:self.stride] = x
        return feather_in_zeros

class g_a(nn.Module):
    def __init__(self, model, N, M):
        super(g_a, self).__init__()
        self.quant_input = QuantAct()

        self.conv1 = QuantConv2d()
        self.conv1.set_param(model[0])
        self.relu1 = nn.ReLU()
        self.quant_act1 = QuantAct(quant_mode="asymmetric")

        self.conv2 = QuantConv2d()
        self.conv2.set_param(model[2])
        self.relu2 = nn.ReLU()
        self.quant_act2 = QuantAct(quant_mode="asymmetric")

        self.conv3 = QuantConv2d()
        self.conv3.set_param(model[4])
        self.relu3 = nn.ReLU()
        self.quant_act3 = QuantAct(quant_mode="asymmetric")

        self.conv4 = QuantConv2d()
        self.conv4.set_param(model[6])


    def forward(self, x):
        # quantize input
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.relu1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.relu2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv3(x, act_scaling_factor)
        x = self.relu3(x)
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv4(x, act_scaling_factor)

        return x

class h_a(nn.Module):
    def __init__(self, model, N, M):
        super(h_a, self).__init__()
        self.quant_input = QuantAct()

        self.conv1 = QuantConv2d()
        self.conv1.set_param(model[0])
        self.relu1 = nn.ReLU()
        self.quant_act1 = QuantAct(quant_mode="asymmetric")

        self.conv2 = QuantConv2d()
        self.conv2.set_param(model[2])
        self.relu2 = nn.ReLU()
        self.quant_act2 = QuantAct(quant_mode="asymmetric")

        self.conv3 = QuantConv2d()
        self.conv3.set_param(model[4])


    def forward(self, x):
        # quantize input
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.relu1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.relu2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv3(x, act_scaling_factor)

        return x

class h_s(nn.Module):
    def __init__(self, model, N, M):
        super(h_s, self).__init__()
        # add input quantization
        self.quant_input = QuantAct()

        self.addz1 = add_zeros(N)
        self.conv1 = QuantConv2d()
        self.conv1.set_param_TransConv(model[0])
        self.relu1 = nn.ReLU()
        self.quant_act1 = QuantAct(quant_mode="asymmetric")

        self.addz2 = add_zeros(M)
        self.conv2 = QuantConv2d()
        self.conv2.set_param_TransConv(model[2])
        self.relu2 = nn.ReLU()
        self.quant_act2 = QuantAct(quant_mode="asymmetric")

        self.conv3 = QuantConv2d()
        self.conv3.set_param(model[4])
                

    def forward(self, x):
        # quantize input
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.conv1(self.addz1(x), act_scaling_factor)
        x = self.relu1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv2(self.addz2(x), act_scaling_factor)
        x = self.relu2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv3(x, act_scaling_factor)
        
        return x

class g_s(nn.Module):
    def __init__(self, model, N, M):
        super(g_s, self).__init__()
        self.quant_input = QuantAct()

        self.addz1 = add_zeros(M)
        self.conv1 = QuantConv2d()
        self.conv1.set_param_TransConv(model[0])
        self.relu1 = nn.ReLU()
        self.quant_act1 = QuantAct(quant_mode="asymmetric")

        self.addz2 = add_zeros(N)
        self.conv2 = QuantConv2d()
        self.conv2.set_param_TransConv(model[2])
        self.relu2 = nn.ReLU()
        self.quant_act2 = QuantAct(quant_mode="asymmetric")

        self.addz3 = add_zeros(N)
        self.conv3 = QuantConv2d()
        self.conv3.set_param_TransConv(model[4])
        self.relu3 = nn.ReLU()
        self.quant_act3 = QuantAct(quant_mode="asymmetric")

        self.addz4 = add_zeros(N)
        self.conv4 = QuantConv2d()
        self.conv4.set_param_TransConv(model[6])

        
    def forward(self, x):
        # quantize input
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.conv1(self.addz1(x), act_scaling_factor)
        x = self.relu1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv2(self.addz2(x), act_scaling_factor)
        x = self.relu2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv3(self.addz3(x), act_scaling_factor)
        x = self.relu3(x)
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv4(self.addz4(x), act_scaling_factor)
        
        return x
        

class entropy_parameters(nn.Module):
    def __init__(self, model, N, M):
        super(entropy_parameters, self).__init__()
        self.quant_input = QuantAct()

        self.conv1 = QuantConv2d()
        self.conv1.set_param(model[0])
        self.relu1 = nn.ReLU()
        self.quant_act1 = QuantAct(quant_mode="asymmetric")

        self.conv2 = QuantConv2d()
        self.conv2.set_param(model[2])
        self.relu2 = nn.ReLU()
        self.quant_act2 = QuantAct(quant_mode="asymmetric")

        self.conv3 = QuantConv2d()
        self.conv3.set_param(model[4])


    def forward(self, x):
        # quantize input
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.relu1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.relu2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv3(x, act_scaling_factor)

        return x

class context_prediction(nn.Module):
    def __init__(self, model, N, M):
        super(context_prediction, self).__init__()
        self.quant_input = QuantAct()

        self.conv = QuantConv2d()
        self.conv.set_param(model)
        self.mask = Parameter(model.mask.data.clone())


    def forward(self, x):
        # quantize input
        x, act_scaling_factor = self.quant_input(x)
        
        self.conv.weight.data *= self.mask
        x, weight_scaling_factor = self.conv(x, act_scaling_factor)
        
        return x
        
        
class qJAHP(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, model, **kwargs):
        super().__init__(N=model.N, M=model.M, **kwargs)
        self.g_a = g_a(model.g_a, self.N, self.M)
        self.g_s = g_s(model.g_s, self.N, self.M)
        self.h_a = h_a(model.h_a, self.N, self.M)
        self.h_s = h_s(model.h_s, self.N, self.M)
        self.entropy_parameters = entropy_parameters(model.entropy_parameters, self.N, self.M)
        self.context_prediction = context_prediction(model.context_prediction, self.N, self.M)
        self.gaussian_conditional = GaussianConditional(None)


    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

def q_JAHP(model):
    """
    model : nn.Module
        The pretrained floating-point model.
    """
    return qJAHP(model)