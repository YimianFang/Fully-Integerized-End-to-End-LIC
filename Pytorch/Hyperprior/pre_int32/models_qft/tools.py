import torch
import torch.nn as nn
import os


def add_zeros(feather_in, kernel=5, stride=2, padding=2, output_padding=1):
    batch_size = feather_in.size()[0]
    channel_in = feather_in.size()[1]
    h_in = feather_in.size()[2]
    w_in = feather_in.size()[3]
    h_in_zeros = h_in + (stride - 1) * (h_in - 1) + 2 * (kernel - padding - 1) + output_padding
    w_in_zeros = w_in + (stride - 1) * (w_in - 1) + 2 * (kernel - padding - 1) + output_padding
    feather_in_zeros = torch.zeros([batch_size, channel_in, h_in_zeros, w_in_zeros])
    for i in range(h_in):
        for j in range(w_in):
            feather_in_zeros[:, :, output_padding + kernel - padding - 1 + i * stride, output_padding + kernel - padding - 1 + j * stride] \
                = feather_in[:, :, i, j]
    return feather_in_zeros


def add_zeros2(feather_in, kernel=5, stride=2, padding=2, output_padding=1):
    batch_size = feather_in.size()[0]
    channel_in = feather_in.size()[1]
    h_in = feather_in.size()[2]
    w_in = feather_in.size()[3]
    h_in_zeros = h_in + (stride - 1) * (h_in - 1) + 2 * (kernel - padding - 1) + output_padding
    w_in_zeros = w_in + (stride - 1) * (w_in - 1) + 2 * (kernel - padding - 1) + output_padding
    if feather_in.is_cuda:
        feather_in_zeros = torch.zeros([batch_size, channel_in, h_in_zeros, w_in_zeros]).cuda()
    else:
        feather_in_zeros = torch.zeros([batch_size, channel_in, h_in_zeros, w_in_zeros]).cpu()
    feather_in_zeros[:, :, kernel - padding - 1:h_in_zeros - kernel + padding + 1 - output_padding:stride, kernel - padding - 1:w_in_zeros - kernel + padding + 1 - output_padding:stride] = feather_in
    return feather_in_zeros


def add_zeros_pool(x, stride):
    pool = nn.MaxPool2d(stride, stride=stride, padding=stride-1)
    unpool = nn.MaxUnpool2d(1, stride=stride)
    h_indx = x.size()[2] + (stride - 1) * (x.size()[2] - 1)
    w_indx = x.size()[3] + (stride - 1) * (x.size()[3] - 1)
    indices = torch.arange(0, h_indx * w_indx).view(h_indx, w_indx)\
        .expand(x.size()[0], x.size()[1], h_indx, w_indx).cpu()
    indices_d = pool(indices.float())
    x = unpool(x, indices_d.to(torch.int64))
    return x


def add_zeros_pool2(x, stride):
    unpool = nn.MaxUnpool2d(1, stride=stride)
    h_indx = x.size()[2] + (stride - 1) * (x.size()[2] - 1)
    w_indx = x.size()[3] + (stride - 1) * (x.size()[3] - 1)
    indices = torch.arange(0, h_indx * w_indx).view(h_indx, w_indx)\
        .expand(x.size()[0], x.size()[1], h_indx, w_indx).cpu()
    indices_d = indices[:, :, 0::stride, 0::stride]
    x = unpool(x, indices_d)
    return x


def diff_print(model_name, fig_order, name, x):
    path = "convout/"+model_name+"/"+str(fig_order)
    if not os.path.exists(path):
        os.makedirs(path)
    noq = torch.load(path+"/"+name+".pt")
    torch.save(x, path + "/q_" + name + ".pt")
    print("diff of " + name + ":", abs(noq - x).mean())


def convout_save(model_name, fig_order, convout_name, x):
    # noq_convout save
    path = "convout/"+model_name+"/"+str(fig_order)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(x, path + "/" + convout_name + ".pt")


def simu_conv_int32(input, weight, bias, stride_h, stride_w, padding_h = 0, padding_w = 0): #input已padding
    assert len(input.shape) == 4, 'The shape of input should be 4-dimensional'
    assert len(weight.shape) == 4, 'The shape of weight should be 4-dimensional'
    assert len(bias.shape) == 1, 'The shape of bias should be 1-dimensional'
    assert input.shape[1] == weight.shape[1], 'The in_channel of weight should equal to that of input'
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
    in_re = torch.zeros([batch, out_h, out_w, in_c * W_h * W_w]).to(torch.int32)
    W_re = weight.reshape([out_c, in_c * W_h * W_w])
    for hi in range(out_h):
        for wi in range(out_w):
            in_re[:, hi, wi, :] = \
                input[:, :, hi * stride_h:hi * stride_h + W_h, wi * stride_w:wi * stride_w + W_w].reshape(batch, -1)
    output = torch.matmul(in_re, W_re.T).permute([0, 3, 1, 2])
    bias = bias.reshape(-1, 1, 1, 1)
    output = output.permute(1, 0, 2, 3)
    output += bias
    output = output.permute(1, 0, 2, 3)
    return output
