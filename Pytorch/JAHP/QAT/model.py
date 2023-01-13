from compressai.models.google import MeanScaleHyperprior
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Any
from torch import Tensor

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

class MaskedConv2d(nn.Module):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2, mask_type: str = "A"):
        super(MaskedConv2d, self).__init__()

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')
        
        self.register_buffer("mask", torch.ones(channel_out, channel_in, kernel_size, kernel_size))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        # TODO(begaintj): weight assigment is not supported by torchscript
        # self.weight.data *= self.mask

        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        
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

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            nn.ReLU(),
            # GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            # GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            # GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            add_zeros(M),
            torch.quantization.QuantStub(),
            nn.Conv2d(M, N, kernel_size=5),
            # deconv(M, N, kernel_size=5, stride=2),
            nn.ReLU(),
            # GDN(N),
            torch.quantization.DeQuantStub(),
            add_zeros(N),
            torch.quantization.QuantStub(),
            nn.Conv2d(N, N, kernel_size=5),
            # deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            # GDN(N),
            torch.quantization.DeQuantStub(),
            add_zeros(N),
            torch.quantization.QuantStub(),
            nn.Conv2d(N, N, kernel_size=5),
            # deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(),
            # GDN(N),
            torch.quantization.DeQuantStub(),
            add_zeros(N),
            torch.quantization.QuantStub(),
            nn.Conv2d(N, 3, kernel_size=5),
            # deconv(N, 3, kernel_size=5, stride=2),
            torch.quantization.DeQuantStub(),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(),
            # nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.ReLU(),
            # nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            add_zeros(N),
            torch.quantization.QuantStub(),
            nn.Conv2d(N, M, kernel_size=5),
            # deconv(N, M, stride=2, kernel_size=5),
            nn.ReLU(),
            # nn.LeakyReLU(inplace=True),
            torch.quantization.DeQuantStub(),
            add_zeros(M),
            torch.quantization.QuantStub(),
            nn.Conv2d(M, M * 3 // 2, kernel_size=5),
            # deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.ReLU(),
            # nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
            torch.quantization.DeQuantStub(),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.ReLU(),
            # nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.ReLU(),
            # nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.quant1 = torch.quantization.QuantStub()
        self.dequant1 = torch.quantization.DeQuantStub()
        self.quant2 = torch.quantization.QuantStub()
        self.dequant2 = torch.quantization.DeQuantStub()
        self.quant3 = torch.quantization.QuantStub()
        self.dequant3 = torch.quantization.DeQuantStub()


    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.dequant1(self.g_a(self.quant1(x)))

        z = self.dequant2(self.h_a(self.quant2(y)))

        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        ctx_params = self.context_prediction(y_hat)

        gaussian_params = self.dequant3(self.entropy_parameters(
            self.quant3(torch.cat((params, ctx_params), dim=1))
        ))

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        x_hat = self.g_s(y_hat)

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