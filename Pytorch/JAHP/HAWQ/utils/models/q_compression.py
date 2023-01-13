"""
    Quantized Hyprior, implemented in PyTorch.
"""

import os
import torch.nn as nn
import torch.nn.init as init
from ..quantization_utils.quant_modules import *

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

class Encoder(nn.Module):
    def __init__(self,
                 model,
                 out_channel_N=128,
                 out_channel_M=192):
        super(Encoder, self).__init__()
        # add input quantization
        self.quant_input = QuantAct()

        self.conv1 = QuantConv2d()
        self.conv1.set_param(model.conv1)
        self.relu1 = nn.ReLU()
        self.quant_act1 = QuantAct(quant_mode="asymmetric")

        self.conv2 = QuantConv2d()
        self.conv2.set_param(model.conv2)
        self.relu2 = nn.ReLU()
        self.quant_act2 = QuantAct(quant_mode="asymmetric")

        self.conv3 = QuantConv2d()
        self.conv3.set_param(model.conv3)
        self.relu3 = nn.ReLU()
        self.quant_act3 = QuantAct(quant_mode="asymmetric")

        self.conv4 = QuantConv2d()
        self.conv4.set_param(model.conv4)

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

class priorEncoder(nn.Module):
    def __init__(self,
                 model,
                 out_channel_N=128,
                 out_channel_M=192):
        super(priorEncoder, self).__init__()
        # add input quantization
        self.quant_input = QuantAct()

        self.conv1 = QuantConv2d()
        self.conv1.set_param(model.conv1)
        self.relu1 = nn.ReLU()
        self.quant_act1 = QuantAct(quant_mode="asymmetric")

        self.conv2 = QuantConv2d()
        self.conv2.set_param(model.conv2)
        self.relu2 = nn.ReLU()
        self.quant_act2 = QuantAct(quant_mode="asymmetric")

        self.conv3 = QuantConv2d()
        self.conv3.set_param(model.conv3)

    def forward(self, x):
        # quantize input
        x, act_scaling_factor = self.quant_input(torch.abs(x))

        x, weight_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.relu1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.relu2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv3(x, act_scaling_factor)

        return x

class priorDecoder(nn.Module):
    def __init__(self,
                 model,
                 out_channel_N=128,
                 out_channel_M=192):
        super(priorDecoder, self).__init__()
        # add input quantization
        self.quant_input = QuantAct()

        self.addz1 = add_zeros(out_channel_N)
        self.conv1 = QuantConv2d()
        self.conv1.set_param_TransConv(model.deconv1)
        self.relu1 = nn.ReLU()
        self.quant_act1 = QuantAct(quant_mode="asymmetric")

        self.addz2 = add_zeros(out_channel_N)
        self.conv2 = QuantConv2d()
        self.conv2.set_param_TransConv(model.deconv2)
        self.relu2 = nn.ReLU()
        self.quant_act2 = QuantAct(quant_mode="asymmetric")

        self.addz3 = add_zeros(out_channel_N, kernel=3, stride=1, padding=1, output_padding=0)
        self.conv3 = QuantConv2d()
        self.conv3.set_param_TransConv(model.deconv3)

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
        
        return torch.exp(x)

class Decoder(nn.Module):
    def __init__(self,
                 model,
                 out_channel_N=128,
                 out_channel_M=192):
        super(Decoder, self).__init__()
        # add input quantization
        self.quant_input = QuantAct()

        self.addz1 = add_zeros(out_channel_M)
        self.conv1 = QuantConv2d()
        self.conv1.set_param_TransConv(model.deconv1)
        self.relu1 = nn.ReLU()
        self.quant_act1 = QuantAct(quant_mode="asymmetric")

        self.addz2 = add_zeros(out_channel_N)
        self.conv2 = QuantConv2d()
        self.conv2.set_param_TransConv(model.deconv2)
        self.relu2 = nn.ReLU()
        self.quant_act2 = QuantAct(quant_mode="asymmetric")

        self.addz3 = add_zeros(out_channel_N)
        self.conv3 = QuantConv2d()
        self.conv3.set_param_TransConv(model.deconv3)
        self.relu3 = nn.ReLU()
        self.quant_act3 = QuantAct(quant_mode="asymmetric")

        self.addz4 = add_zeros(out_channel_N)
        self.conv4 = QuantConv2d()
        self.conv4.set_param_TransConv(model.deconv4)

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

class Bitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)

class BitEstimator(nn.Module):
    '''
    Estimate bit
    '''
    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        
    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)

class qImageCompressor(nn.Module):
    def __init__(self, model, out_channel_N=128, out_channel_M=192):
        super(qImageCompressor, self).__init__()
        self.Encoder = Encoder(model.Encoder, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Decoder(model.Decoder, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = priorEncoder(model.priorEncoder, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = priorDecoder(model.priorDecoder, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
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

def q_ImageCompressor(model):
    """
    model : nn.Module
        The pretrained floating-point model.
    """
    return qImageCompressor(model)