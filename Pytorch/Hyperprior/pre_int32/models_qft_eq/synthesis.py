#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from .GDN import GDN
from .analysis import Analysis_net
from .tools import add_zeros2
from quant_utils import *
from torch.quantization import QuantStub, DeQuantStub
import numpy as np
# import scipy.io as scio

class Synthesis_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Synthesis_net, self).__init__()
        # self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.deconv1 = nn.Conv2d(out_channel_M, out_channel_N, 5, stride=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        # self.igdn1 = GDN(out_channel_N, inverse=True)
        self.relu1 = nn.ReLU()
        # self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        # self.igdn2 = GDN(out_channel_N, inverse=True)
        self.relu2 = nn.ReLU()
        # self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        # self.igdn3 = GDN(out_channel_N, inverse=True)
        self.relu3 = nn.ReLU()
        # self.deconv4 = nn.ConvTranspose2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.Conv2d(out_channel_N, 3, 5, stride=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        self.s1 = nn.Parameter(torch.zeros(out_channel_N, 1, 1, 1), requires_grad=False)
        self.sn1 = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.s2 = nn.Parameter(torch.zeros(out_channel_N, 1, 1, 1), requires_grad=False)
        self.sn2 = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.s3 = nn.Parameter(torch.zeros(out_channel_N, 1, 1, 1), requires_grad=False)
        self.sn3 = nn.Parameter(torch.tensor(0.), requires_grad=False)
        # self.ord = 1

    def forward(self, x):
        x = add_zeros2(x, kernel=5, stride=2, padding=2, output_padding=1)
        # x = self.deconv1(x) #####eq1d

        #####eq1
        if not self.training:
            avrg_s = self.s1 / self.sn1
            self.deconv1.weight.data *= avrg_s
            self.deconv1.bias.data *= avrg_s.reshape(-1)
            x = self.deconv1(x)
            self.deconv1.weight.data /= avrg_s
            self.deconv1.bias.data /= avrg_s.reshape(-1)

        #####eq1
        if self.training:
            x = self.deconv1(x)
            ####fix####
            b = x.size()[0]
            in_max = torch.max(torch.max(x.abs(), dim=-1)[0], dim=-1)[0]
            self.s1 += torch.sum((2 ** 4 / in_max), dim=0).detach().reshape(-1, 1, 1, 1)
            self.sn1 += b
            ####fix####
            avrg_s = self.s1 / self.sn1
            x = (x.permute(1, 0, 2, 3) * avrg_s).permute(1, 0, 2, 3)

        # self.relu1.scl_test.data = torch.Tensor([0.9]).cuda()
        x = self.relu1(x)

        x = add_zeros2(x, kernel=5, stride=2, padding=2, output_padding=1)
        self.deconv2.weight.data = (self.deconv2.weight.data.permute(1, 0, 2, 3) / avrg_s).permute(1, 0, 2, 3) #####eq1
        x = self.deconv2(x) #####eq2d
        self.deconv2.weight.data = (self.deconv2.weight.data.permute(1, 0, 2, 3) * avrg_s).permute(1, 0, 2, 3) #####eq1
        
        # #####eq2
        # if not self.training:
        #     avrg_s = self.s2 / self.sn2
        #     self.deconv2.weight.data *= avrg_s
        #     self.deconv2.bias.data *= avrg_s.reshape(-1)
        #     x = self.deconv2(x)
        #     self.deconv2.weight.data /= avrg_s
        #     self.deconv2.bias.data /= avrg_s.reshape(-1)

        # #####eq2
        # if self.training:
        #     x = self.deconv2(x)
        #     ####fix####
        #     b = x.size()[0]
        #     in_max = torch.max(torch.max(x.abs(), dim=-1)[0], dim=-1)[0]
        #     self.s2 += torch.sum((2 ** 5 / in_max), dim=0).detach().reshape(-1, 1, 1, 1)
        #     self.sn2 += b
        #     ####fix####
        #     avrg_s = self.s2 / self.sn2
        #     x = (x.permute(1, 0, 2, 3) * avrg_s).permute(1, 0, 2, 3)

        # torch.save(x, "c2out_eq/pic"+str(self.ord)+".pth.tar")
        # self.ord += 1
        x = self.relu2(x)

        x = add_zeros2(x, kernel=5, stride=2, padding=2, output_padding=1)
        # self.deconv3.weight.data = (self.deconv3.weight.data.permute(1, 0, 2, 3) / avrg_s).permute(1, 0, 2, 3) #####eq2
        x = self.deconv3(x) #####eq3d
        # self.deconv3.weight.data = (self.deconv3.weight.data.permute(1, 0, 2, 3) * avrg_s).permute(1, 0, 2, 3) #####eq2


        # #####eq3
        # if not self.training:
        #     avrg_s = self.s3 / self.sn3
        #     self.deconv3.weight.data *= avrg_s
        #     self.deconv3.bias.data *= avrg_s.reshape(-1)
        #     x = self.deconv3(x)
        #     self.deconv3.weight.data /= avrg_s
        #     self.deconv3.bias.data /= avrg_s.reshape(-1)

        # #####eq3
        # if self.training:
        #     x = self.deconv3(x)
        #     ####fix####
        #     b = x.size()[0]
        #     in_max = torch.max(torch.max(x.abs(), dim=-1)[0], dim=-1)[0]
        #     self.s3 += torch.sum((2 ** 4 / in_max), dim=0).detach().reshape(-1, 1, 1, 1)
        #     self.sn3 += b
        #     ####fix####
        #     avrg_s = self.s3 / self.sn3
        #     x = (x.permute(1, 0, 2, 3) * avrg_s).permute(1, 0, 2, 3)

        x = self.relu3(x)

        x = add_zeros2(x, kernel=5, stride=2, padding=2, output_padding=1)
        # self.deconv4.weight.data = (self.deconv4.weight.data.permute(1, 0, 2, 3) / avrg_s).permute(1, 0, 2, 3) #####eq3
        x = self.deconv4(x)
        # self.deconv4.weight.data = (self.deconv4.weight.data.permute(1, 0, 2, 3) * avrg_s).permute(1, 0, 2, 3) #####eq3

        return x


# synthesis_one_pass = tf.make_template('synthesis_one_pass', synthesis_net)

def build_model():
    input_image = torch.zeros([7,3,256,256])
    analysis_net = Analysis_net()
    synthesis_net = Synthesis_net()
    feature = analysis_net(input_image)
    recon_image = synthesis_net(feature)

    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("recon_image : ", recon_image.size())

# def main(_):
#   build_model()


if __name__ == '__main__':
    build_model()
