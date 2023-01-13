from turtle import shape
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import time
import threading


def add_zeros(feature_in, kernel=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)):
    batch_size = feature_in.shape[0]
    channel_in = feature_in.shape[1]
    h_in = feature_in.shape[2]
    w_in = feature_in.shape[3]
    h_in_zeros = h_in + (stride[0] - 1) * (h_in - 1) + 2 * (kernel[0] - padding[0] - 1) + output_padding[0]
    w_in_zeros = w_in + (stride[1] - 1) * (w_in - 1) + 2 * (kernel[1] - padding[1] - 1) + output_padding[1]
    if feature_in.is_cuda:
        feature_in_zeros = torch.zeros([batch_size, channel_in, h_in_zeros, w_in_zeros], dtype=feature_in.dtype).cuda()
    else:
        feature_in_zeros = torch.zeros([batch_size, channel_in, h_in_zeros, w_in_zeros], dtype=feature_in.dtype).cpu()
    feature_in_zeros[:, :, kernel[0] - padding[0] - 1:h_in_zeros - kernel[0] + padding[0] + 1 - output_padding[0]:stride[0], kernel[1] - padding[1] - 1:w_in_zeros - kernel[1] + padding[1] + 1 - output_padding[1]:stride[1]] = feature_in
    return feature_in_zeros


def blockshaped(tensor, nrows, ncols):
    """
    Return an Tensor of shape (nrows, ncols, n, m) where
    n * nrows, m * ncols = tensor.shape.
    This should be a view of the original tensor.
    """
    h, w = tensor.shape
    n, m = h // nrows, w // ncols
    return tensor.reshape(nrows, n, ncols, m).swapaxes(1, 2)


def do_matmul(a, b, out):
    out[:] = torch.matmul(a, b)


def par_matmul(a, b, nblocks, mblocks, do_matmul=do_matmul):
    """
    Return the matrix product a * b.
    The product is split into nblocks * mblocks partitions that are performed
    in parallel threads.
    """
    n_jobs = nblocks * mblocks
    print('running {} jobs in parallel'.format(n_jobs))

    out = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype)

    out_blocks = blockshaped(out, nblocks, mblocks)
    a_blocks = blockshaped(a, nblocks, 1)
    b_blocks = blockshaped(b, 1, mblocks)

    threads = []
    for i in range(nblocks):
        for j in range(mblocks):
            th = threading.Thread(target=do_matmul, 
                                  args=(a_blocks[i, 0, :, :], 
                                        b_blocks[0, j, :, :], 
                                        out_blocks[i, j, :, :]))
            th.start()
            threads.append(th)

    for th in threads:
        th.join()

    return out


def simu_conv(input, weight, bias, stride_h=1, stride_w=1, padding_h=1, padding_w=1, type="int32"): #input已padding
    assert len(input.shape) == 4, 'The shape of input should be 4-dimensional'
    assert len(weight.shape) == 4, 'The shape of weight should be 4-dimensional'
    assert len(bias.shape) == 1, 'The shape of bias should be 1-dimensional'
    assert input.shape[1] == weight.shape[1], 'The in_channel of weight should equal to that of input'
    if type == "int32":
        type = torch.int32
    elif type == "float32":
        type = torch.float32
    pdd = nn.ConstantPad2d((padding_w, padding_w, padding_h, padding_h), 0)
    input = pdd(input)
    batch = input.shape[0]
    in_h = input.shape[-2]
    in_w = input.shape[-1]
    in_c = weight.shape[1]
    W_h = weight.shape[-2]
    W_w = weight.shape[-1]
    out_c = weight.shape[0]
    out_h = (in_h - W_h) // stride_h + 1
    out_w = (in_w - W_w) // stride_w + 1
    W_re = weight.reshape([out_c, in_c * W_h * W_w])
    delta_kh = torch.arange(0, in_w * W_h, in_w).reshape(1, W_h, 1).repeat(in_c, 1, W_w)
    delta_kw = torch.arange(0, W_w).reshape(1, 1, W_w).repeat(in_c, W_h, 1)
    delta_kc = torch.arange(0, in_h * in_w * in_c, in_h * in_w).reshape(in_c, 1, 1).repeat(1, W_h, W_w)
    delta_k = (delta_kh + delta_kw + delta_kc).reshape(1, 1, -1)
    delta_inh = torch.arange(0, stride_h * in_w * out_h, stride_h * in_w).reshape(out_h, 1, 1).repeat(1, out_w, in_c * W_h * W_w)
    delta_inw = torch.arange(0, stride_w * out_w, stride_w).reshape(1, out_w, 1).repeat(out_h, 1, in_c * W_h * W_w)
    in_re = delta_inh + delta_inw + delta_k
    in_re = input.reshape(batch, -1)[:, in_re]
    bias = bias.reshape(1, -1, 1, 1)
    assert in_re.dtype == type, 'The dtype of input does not meet the requirement'
    print(in_re.shape)
    mt1 = time.time()
    output = torch.matmul(in_re, W_re.T)
    mt2 = time.time()
    output = output.permute([0, 3, 1, 2])
    output += bias
    return output, mt2 - mt1


def simu_transconv(input, weight, bias, stride_h=1, stride_w=1, padding_h=1, padding_w=1, output_padding_h=0, output_padding_w=0, type="int32"): #input已padding
    assert len(input.shape) == 4, 'The shape of input should be 4-dimensional'
    assert len(weight.shape) == 4, 'The shape of weight should be 4-dimensional'
    assert len(bias.shape) == 1, 'The shape of bias should be 1-dimensional'
    assert input.shape[1] == weight.shape[1], 'The in_channel of weight should equal to that of input'
    if type == "int32":
        type = torch.int32
    elif type == "float32":
        type = torch.float32
    input = add_zeros(input,
                                    kernel = (weight.shape[-2], weight.shape[-1]), 
                                    stride = (stride_h, stride_w),
                                    padding=(padding_h, padding_w),
                                    output_padding=(output_padding_h, output_padding_w))
    batch = input.shape[0]
    in_h = input.shape[-2]
    in_w = input.shape[-1]
    in_c = weight.shape[1]
    W_h = weight.shape[-2]
    W_w = weight.shape[-1]
    out_c = weight.shape[0]
    out_h = in_h - W_h + 1
    out_w = in_w - W_w + 1
    W_re = weight.reshape([out_c, in_c * W_h * W_w])
    delta_kh = torch.arange(0, in_w * W_h, in_w).reshape(1, W_h, 1).repeat(in_c, 1, W_w)
    delta_kw = torch.arange(0, W_w).reshape(1, 1, W_w).repeat(in_c, W_h, 1)
    delta_kc = torch.arange(0, in_h * in_w * in_c, in_h * in_w).reshape(in_c, 1, 1).repeat(1, W_h, W_w)
    delta_k = (delta_kh + delta_kw + delta_kc).reshape(1, 1, -1)
    delta_inh = torch.arange(0, in_w * out_h, in_w).reshape(out_h, 1, 1).repeat(1, out_w, in_c * W_h * W_w)
    delta_inw = torch.arange(0, out_w).reshape(1, out_w, 1).repeat(out_h, 1, in_c * W_h * W_w)
    in_re = delta_inh + delta_inw + delta_k
    in_re = input.reshape(batch, -1)[:, in_re]
    bias = bias.reshape(1, -1, 1, 1)
    assert in_re.dtype == type, 'The dtype of input does not meet the requirement'
    print(in_re.shape)
    mt1 = time.time()
    output = torch.matmul(in_re, W_re.T)
    # output = par_matmul(in_re.reshape(-1, in_c * W_h * W_w), W_re.T, 32, 32).reshape(batch, out_h, out_w, out_c)
    mt2 = time.time()
    output = output.permute([0, 3, 1, 2])
    output += bias
    return output, mt2 - mt1

device = "cpu"
input = torch.randint(-10, 11, (1, 3, 768, 512), dtype=torch.int32).to(device)
weight = torch.randint(-5, 6, (192, 3, 5, 5), dtype=torch.int32).to(device)
bias = torch.randint(0, 192, (192,), dtype=torch.int32).to(device)
trans = True

if trans:
    # output1 = F.conv_transpose2d(input.to(torch.float), torch.flip(weight.permute(1, 0, 2, 3), [2,3]).to(torch.float), bias.to(torch.float), 
    #                                                 stride=2, padding=2, output_padding=1)
    T1 = time.time()
    output1, fltt = simu_transconv(input.to(torch.float), weight.to(torch.float), bias.to(torch.float), 
                                                    stride_h=2, stride_w=2, 
                                                    padding_h=2, padding_w=2, 
                                                    output_padding_h=1, output_padding_w=1, type="float32")
    T2 = time.time()
    output2, intt = simu_transconv(input, weight, bias, stride_h=2, stride_w=2, 
                                                        padding_h=2, padding_w=2, output_padding_h=1, output_padding_w=1)
    T3 = time.time()                                                    
else:
    T1 = time.time()
    output1, fltt = simu_conv(input.to(torch.float), weight.to(torch.float), bias.to(torch.float), 
                                                    stride_h=2, stride_w=2, 
                                                    padding_h=2, padding_w=2, type="float32")
    T2 = time.time()
    output2, intt = simu_conv(input, weight, bias, stride_h=2, stride_w=2, padding_h=2, padding_w=2)
    T3 = time.time()  

print(abs(output1 - output2).max())
print(fltt)
print(intt)
print(T2 - T1)
print(T3 - T2)

# a = torch.randint(-100, 101, (1, 540, 540, 75))
# b = torch.randint(-100, 101, (75, 128))
# T1 = time.time()
# # output = torch.matmul(a.to(torch.float), b.to(torch.float))
# output = a.to(torch.float) * b.to(torch.float)
# T2 = time.time()
# output = torch.matmul(a.to(torch.int32), b.to(torch.int32))
# T3 = time.time() 
# print(T2 - T1)
# print(T3 - T2)

# import numpy as np
# a = torch.randint(-100, 101, (1, 540, 540, 75)).numpy()
# b = torch.randint(-100, 101, (75, 128)).numpy()
# T1 = time.time()
# output = np.dot(a.astype(np.float32), b.astype(np.float32))
# T2 = time.time()
# output = np.dot(a.astype(np.int32), b.astype(np.int32))
# T3 = time.time() 
# print(T2 - T1)
# print(T3 - T2)