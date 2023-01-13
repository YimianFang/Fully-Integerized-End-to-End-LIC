
import torch
from collections import OrderedDict
import torch.nn as nn

def symmetric_linear_quantization_params(num_bits, saturation_val):
    # Leave one bit for sign
    n = 2 ** (num_bits - 1) - 1
    scale = n / saturation_val
    if isinstance(scale, torch.Tensor):
        zero_point = torch.zeros_like(scale)
    else:
        zero_point = 0.0
    return scale, zero_point


def  asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    n = 2 ** num_bits - 1
    scale = n / (saturation_max - saturation_min)
    zero_point = scale * saturation_min
    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2 ** (num_bits - 1)
    return scale, zero_point


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_quantize_clamp(input, scale, zero_point, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale, zero_point, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale
    
def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return -n, n - 1
    return 0, 2 ** num_bits - 1

class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None
        
class LearnedClippedLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, dequantize, inplace):
        ctx.save_for_backward(input, clip_val)
        if inplace:
            ctx.mark_dirty(input)
        scale, zero_point = asymmetric_linear_quantization_params(num_bits, 0, clip_val.data[0], signed=False)
        output = clamp(input, 0, clip_val.data[0], inplace)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.le(0)] = 0
        grad_input[input.ge(clip_val.data[0])] = 0

        grad_alpha = grad_output.clone()
        grad_alpha[input.lt(clip_val.data[0])] = 0
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        # Straight-through estimator for the scale factor calculation
        return grad_input, grad_alpha, None, None, None


class ClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, clip_val, dequantize=True, inplace=False):
        super(ClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = clip_val
        self.scale, self.zero_point = asymmetric_linear_quantization_params(num_bits, 0, clip_val, signed=False)
        self.dequantize = dequantize
        self.inplace = inplace 

    def forward(self, input):
        input = clamp(input, 0, self.clip_val, self.inplace)
        # print((input > 0).sum())
        input = LinearQuantizeSTE.apply(input, self.scale, self.zero_point, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val,
                                                           inplace_str)

class ClippedLinearQuantization_t(nn.Module):
    def __init__(self, num_bits, dequantize=True, inplace=False, device='cpu'):
        super(ClippedLinearQuantization_t, self).__init__()
        self.device = device
        self.num_bits = num_bits
        self.dequantize = dequantize
        self.inplace = inplace
        self.statis_max = nn.Parameter(torch.Tensor([0]).to(self.device))
        self.statisN = nn.Parameter(torch.Tensor([0]).to(self.device))
        self.scl_train = nn.Parameter(torch.Tensor([1.]).to(self.device))
        self.scl_test = nn.Parameter(torch.Tensor([1.]).to(self.device))

    def forward(self, input):
        if self.training:
            if input.max()<=0:
                input_max = 1e-4
            else:
                input_max = input[input.gt(1e-4)].max().float()
            self.statis_max.data = self.statis_max.data + input_max
            self.statisN.data = self.statisN.data + 1
        if self.training:
            clip_val = (self.statis_max.data / self.statisN.data * self.scl_train).item()
        else:
            clip_val = (self.statis_max.data / self.statisN.data * self.scl_test).item()
        scale, zero_point = asymmetric_linear_quantization_params(self.num_bits, 0, clip_val, signed=False)
        input = clamp(input, 0, clip_val, self.inplace)
        # print(input.gt(1e-4).sum())
        input = LinearQuantizeSTE.apply(input, scale, zero_point, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, inplace_str={2})'.format(self.__class__.__name__, self.num_bits, inplace_str)

class StatisticReLU(nn.Module):
    def __init__(self,clip_val=6,inplace=False):
        super(StatisticReLU, self).__init__()
        self.inplace = inplace 
        self.statis1=nn.Parameter(torch.Tensor([0]),requires_grad=False)
        self.statis2=nn.Parameter(torch.Tensor([0]),requires_grad=False)
        self.statisN=nn.Parameter(torch.Tensor([0]),requires_grad=False)
        self.paraN=nn.Parameter(torch.Tensor([0]),requires_grad=False)
        # self.paraN=0
        self.clampv=clip_val/3

    def forward(self, input):
        input=nn.functional.relu(input, inplace=self.inplace)
        temp=input.gt(1e-4)
        self.statisN.data=self.statisN+temp.sum().float()
        temp2=input[temp]
        self.statis1.data=self.statis1+temp2.sum().float()
        self.statis2.data=self.statis2+temp2.pow(2).sum().float()
        self.paraN.data=input[0].gt(-1).sum()
        # self.paraN=input[0].numel()
        # torch.cuda.empty_cache()
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(paraN={1}{2})'.format(self.__class__.__name__, float(self.paraN.cpu().detach().numpy()),
                                                           inplace_str)#[float(self.paraN.cpu().detach().numpy())],

class StatisticReLU_max(nn.Module):
    def __init__(self, inplace=False):
        super(StatisticReLU_max, self).__init__()
        self.inplace = inplace
        self.statis_max = nn.Parameter(torch.Tensor([0]),requires_grad=False)
        self.statisN = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.paraN = nn.Parameter(torch.Tensor([0]), requires_grad=False)

    def forward(self, input):
        input=nn.functional.relu(input, inplace=self.inplace)
        # print(input.gt(1e-4).sum())
        input_max = input[input.gt(1e-4)].max().float()
        self.statis_max.data = self.statis_max.data + input_max
        self.statisN.data = self.statisN.data + 1
        self.paraN.data = input[0].gt(-1).sum()
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(paraN={1}{2})'.format(self.__class__.__name__, float(self.paraN.cpu().detach().numpy()),
                                                           inplace_str)#[float(self.paraN.cpu().detach().numpy())],

class LearnedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, init_act_clip_val, dequantize=True, inplace=False):
        super(LearnedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]))
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input):
        input = LearnedClippedLinearQuantizeSTE.apply(input, self.clip_val, self.num_bits,
                                                      self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val,
                                                           inplace_str)
def relu_replace_fn1(module, name, bits_acts, clip_val=10):
    if bits_acts is None:
        return module
    return ClippedLinearQuantization(bits_acts, clip_val, dequantize=True, inplace=module.inplace)    

def relu_replace_max(module, bits_acts, clip_val=10, inplace = False, dequantize = True):
    if bits_acts is None:
        return module
    return ClippedLinearQuantization(bits_acts, clip_val, dequantize = dequantize, inplace = inplace)

def relu_replace_train(module, bits_acts, inplace = False, dequantize = True, device = 'cpu'):
    if bits_acts is None:
        return module
    return ClippedLinearQuantization_t(bits_acts, dequantize = dequantize, inplace = inplace, device = device)

def relu_replace_fn2(module, name, bits_acts):
    act_clip_init_val=8.0
    if bits_acts is None:
        return module
    return LearnedClippedLinearQuantization(bits_acts, act_clip_init_val, dequantize=True,
                                            inplace=module.inplace)
                                            
def relu_replace_fn0(module, name, clip_val=6):
    return StatisticReLU( clip_val, inplace=module.inplace)

def relu_log_max(inplace = False):
    return StatisticReLU_max(inplace=inplace)
    
    