from compressai.models.google import MeanScaleHyperprior
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder

import torch
import torch.nn as nn
import torch.functional as F
import warnings
import copy
import quant_utils
import os

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
        st = f.find('iter_') 
        st = f.find('iter_', st + 5) + 5
        ed = f.find('.pth', st)
        return int(f[st:ed]), int_model, state, preLoss
    else:
        return 0, int_model, state, preLoss

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

    def __init__(self, eqlayer=0, r=4, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            nn.ReLU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            nn.ReLU(),
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(),
            conv(N, N, stride=2, kernel_size=5),
            nn.ReLU(),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.ReLU(),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.ReLU(),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.ReLU(),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.ReLU(),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.eqlayer = eqlayer
        self.r = r
        self.s = nn.Parameter(torch.zeros(self.g_s[self.eqlayer].weight.shape[1], 1, 1, 1), requires_grad=False)
        self.sn = nn.Parameter(torch.tensor(0.), requires_grad=False)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, stat=False):
        _, _, H, W = x.shape
        if not stat:
            y = self.g_a(x)
        else:
            for i in range(len(self.g_a)):
                x = self.g_a[i](x)
                if i % 2 == 0:
                    torch.save(x, f"g_a{i}.pth.tar")
            y = x
        
        if not stat:
            z = self.h_a(y)
        else:
            yy = y
            for i in range(len(self.h_a)):
                yy = self.h_a[i](yy)
                if i % 2 == 0:
                    torch.save(yy, f"h_a{i}.pth.tar")
            z = yy

        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        if not stat:
            params = self.h_s(z_hat)
        else:
            for i in range(len(self.h_s)):
                z_hat = self.h_s[i](z_hat)
                if i % 2 == 0:
                    torch.save(z_hat, f"h_s{i}.pth.tar")
            params = z_hat

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        if stat:
            torch.save(ctx_params, "context.pth.tar")
        
        if not stat:
            gaussian_params = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
        else:
            xx = torch.cat((params, ctx_params), dim=1)
            for i in range(len(self.entropy_parameters)):
                xx = self.entropy_parameters[i](xx)
                if i % 2 == 0:
                    torch.save(xx, f"entropy{i}.pth.tar")
            gaussian_params = xx
        
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        if not stat:
            for i in range(len(self.g_s)):
                if i == self.eqlayer:
                    if self.training:
                        y_hat = self.g_s[i](y_hat)
                        b = y_hat.size()[0]
                        in_max = torch.max(torch.max(y_hat.abs(), dim=-1)[0], dim=-1)[0]
                        self.s += torch.sum((2 ** self.r / in_max), dim=0).detach().reshape(-1, 1, 1, 1)
                        self.sn += b
                        avrg_s = self.s / self.sn
                        y_hat = (y_hat.permute(1, 0, 2, 3) * avrg_s).permute(1, 0, 2, 3)
                    else:
                        avrg_s = self.s / self.sn
                        self.g_s[i].weight.data *= avrg_s.reshape(1, -1, 1, 1)
                        self.g_s[i].bias.data *= avrg_s.reshape(-1)
                        y_hat = self.g_s[i](y_hat)
                        self.g_s[i].weight.data /= avrg_s.reshape(1, -1, 1, 1)
                        self.g_s[i].bias.data /= avrg_s.reshape(-1)
                elif i == self.eqlayer + 2:
                    avrg_s = self.s / self.sn
                    self.g_s[i].weight.data = self.g_s[i].weight.data / avrg_s
                    y_hat = self.g_s[i](y_hat)
                    self.g_s[i].weight.data = self.g_s[i].weight.data * avrg_s
                else:
                    y_hat = self.g_s[i](y_hat)
            x_hat = y_hat
        else:
            for i in range(len(self.g_s)):
                y_hat = self.g_s[i](y_hat)
                if i % 2 == 0:
                    torch.save(y_hat, f"g_s{i}.pth.tar")
            x_hat = y_hat

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv