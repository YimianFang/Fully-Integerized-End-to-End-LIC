import numpy as np
from numpy.testing import assert_array_equal
import threading
import time
# import torch
# import torch.nn.functional as F


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (nrows, ncols, n, m) where
    n * nrows, m * ncols = arr.shape.
    This should be a view of the original array.
    """
    h, w = arr.shape
    n, m = h // nrows, w // ncols
    return arr.reshape(nrows, n, ncols, m).swapaxes(1, 2)


def do_dot(a, b, out):
    #np.dot(a, b, out)  # does not work. maybe because out is not C-contiguous?
    out[:] = np.dot(a, b)  ##### less efficient because the output is stored in a temporary array?
    # for i in range(a.shape[0]):
    #     for j in range(b.shape[1]):
    #         # out[i, j] = sum(a[i, :] * b[:, j])
    #         out[i, j] = sum(i * j for i, j in zip(a[i, :], b[:, j]))


def pardot(a, b, nblocks, mblocks, dot_func=do_dot):
    """
    Return the matrix product a * b.
    The product is split into nblocks * mblocks partitions that are performed
    in parallel threads.
    """
    n_jobs = nblocks * mblocks
    print('running {} jobs in parallel'.format(n_jobs))

    out = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)

    out_blocks = blockshaped(out, nblocks, mblocks)
    a_blocks = blockshaped(a, nblocks, 1)
    b_blocks = blockshaped(b, 1, mblocks)

    threads = []
    for i in range(nblocks):
        for j in range(mblocks):
            th = threading.Thread(target=dot_func, 
                                  args=(a_blocks[i, 0, :, :], 
                                        b_blocks[0, j, :, :], 
                                        out_blocks[i, j, :, :]))
            th.start()
            threads.append(th)

    for th in threads:
        th.join()

    return out


# if __name__ == '__main__':
#     c = 32
#     r = 32
#     a = np.ones((64, 32), dtype=np.int32)
#     b = np.arange(32*32, dtype=np.int32).reshape(32, 32)
#     assert_array_equal(pardot(a, b, r, c), np.dot(a, b))

#     a = np.random.randn(128, 64).astype(np.int32)
#     b = np.random.randn(64, 192).astype(np.int32)
#     a_f = np.random.randn(128, 64).astype(np.float32)
#     b_f = np.random.randn(64, 192).astype(np.float32)

#     start = time()
#     rslt1 = pardot(a, b, r, c)
#     time_par = time() - start
#     print('int_pardot: {:.2f} seconds taken'.format(time_par))

#     start = time()
#     rslt2 = pardot(a_f, b_f, r, c)
#     time_par = time() - start
#     print('flt_pardot: {:.2f} seconds taken'.format(time_par))

#     start = time()
#     np.dot(a, b)
#     time_dot = time() - start
#     print('int_np.dot: {:.2f} seconds taken'.format(time_dot))

#     start = time()
#     np.dot(a_f, b_f)
#     time_dot = time() - start
#     print('flt_np.dot: {:.2f} seconds taken'.format(time_dot))


def add_zeros(feature_in, kernel=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)):
    batch_size = feature_in.shape[0]
    channel_in = feature_in.shape[1]
    h_in = feature_in.shape[2]
    w_in = feature_in.shape[3]
    h_in_zeros = h_in + (stride[0] - 1) * (h_in - 1) + 2 * (kernel[0] - padding[0] - 1) + output_padding[0]
    w_in_zeros = w_in + (stride[1] - 1) * (w_in - 1) + 2 * (kernel[1] - padding[1] - 1) + output_padding[1]
    feature_in_zeros = np.zeros([batch_size, channel_in, h_in_zeros, w_in_zeros], dtype=feature_in.dtype)
    feature_in_zeros[:, :, kernel[0] - padding[0] - 1:h_in_zeros - kernel[0] + padding[0] + 1 - output_padding[0]:stride[0], kernel[1] - padding[1] - 1:w_in_zeros - kernel[1] + padding[1] + 1 - output_padding[1]:stride[1]] = feature_in
    return feature_in_zeros


def simu_conv(input, weight, bias, stride_h=1, stride_w=1, padding_h=1, padding_w=1, type="int32", nblocks=8, mblocks=4): #input已padding
    assert len(input.shape) == 4, 'The shape of input should be 4-dimensional'
    assert len(weight.shape) == 4, 'The shape of weight should be 4-dimensional'
    assert len(bias.shape) == 1, 'The shape of bias should be 1-dimensional'
    assert input.shape[1] == weight.shape[1], 'The in_channel of weight should equal to that of input'
    if type == "int32":
        type = np.int32
    elif type == "float32":
        type = np.float32
    input = np.pad(input, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)), "constant")
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
    delta_kh = np.tile(np.arange(0, in_w * W_h, in_w).reshape(1, W_h, 1), (in_c, 1, W_w))
    delta_kw = np.tile(np.arange(0, W_w).reshape(1, 1, W_w), (in_c, W_h, 1))
    delta_kc = np.tile(np.arange(0, in_h * in_w * in_c, in_h * in_w).reshape(in_c, 1, 1), (1, W_h, W_w))
    delta_k = (delta_kh + delta_kw + delta_kc).reshape(1, 1, -1)
    delta_inh = np.tile(np.arange(0, stride_h * in_w * out_h, stride_h * in_w).reshape(out_h, 1, 1), (1, out_w, in_c * W_h * W_w))
    delta_inw = np.tile(np.arange(0, stride_w * out_w, stride_w).reshape(1, out_w, 1), (out_h, 1, in_c * W_h * W_w))
    in_re = delta_inh + delta_inw + delta_k
    in_re = input.reshape(batch, -1)[:, in_re]
    bias = bias.reshape(1, -1, 1, 1)
    assert in_re.dtype == type, 'The dtype of input does not meet the requirement'
    print(in_re.shape)
    mt1 = time.time()
    # output = np.dot(in_re, W_re.T)
    output = pardot(in_re.reshape(-1, in_c * W_h * W_w), W_re.T, nblocks, mblocks).reshape(batch, out_h, out_w, out_c)
    mt2 = time.time()
    output = output.transpose([0, 3, 1, 2])
    output += bias
    return output, mt2 - mt1


def simu_transconv(input, weight, bias, stride_h=1, stride_w=1, padding_h=1, padding_w=1, output_padding_h=0, output_padding_w=0, type="int32", nblocks=32, mblocks=8): #input已padding
    assert len(input.shape) == 4, 'The shape of input should be 4-dimensional'
    assert len(weight.shape) == 4, 'The shape of weight should be 4-dimensional'
    assert len(bias.shape) == 1, 'The shape of bias should be 1-dimensional'
    assert input.shape[1] == weight.shape[1], 'The in_channel of weight should equal to that of input'
    if type == "int32":
        type = np.int32
    elif type == "float32":
        type = np.float32
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
    delta_kh = np.tile(np.arange(0, in_w * W_h, in_w).reshape(1, W_h, 1), (in_c, 1, W_w))
    delta_kw = np.tile(np.arange(0, W_w).reshape(1, 1, W_w), (in_c, W_h, 1))
    delta_kc = np.tile(np.arange(0, in_h * in_w * in_c, in_h * in_w).reshape(in_c, 1, 1), (1, W_h, W_w))
    delta_k = (delta_kh + delta_kw + delta_kc).reshape(1, 1, -1)
    delta_inh = np.tile(np.arange(0, in_w * out_h, in_w).reshape(out_h, 1, 1), (1, out_w, in_c * W_h * W_w))
    delta_inw = np.tile(np.arange(0, out_w).reshape(1, out_w, 1), (out_h, 1, in_c * W_h * W_w))
    in_re = delta_inh + delta_inw + delta_k
    in_re = input.reshape(batch, -1)[:, in_re]
    bias = bias.reshape(1, -1, 1, 1)
    assert in_re.dtype == type, 'The dtype of input does not meet the requirement'
    print(in_re.shape)
    mt1 = time.time()
    # output = np.dot(in_re, W_re.T)
    output = pardot(in_re.reshape(-1, in_c * W_h * W_w), W_re.T, nblocks, mblocks).reshape(batch, out_h, out_w, out_c)
    mt2 = time.time()
    output = output.transpose([0, 3, 1, 2])
    output += bias
    return output, mt2 - mt1


input = np.random.randint(-10, 11, (1, 3, 768, 512), dtype=np.int32)
weight = np.random.randint(-5, 6, (192, 3, 5, 5), dtype=np.int32)
bias = np.random.randint(0, 20, (192,), dtype=np.int32)
nblocks=8 # 8
mblocks=4 # 2
trans = False # False

if trans:
    # outputF = F.conv_transpose2d(torch.tensor(input).to(torch.float), 
    #                                                     torch.flip(torch.tensor(weight).permute(1, 0, 2, 3), [2,3]).to(torch.float), 
    #                                                     torch.tensor(bias).to(torch.float), 
    #                                                     stride=2, padding=2, output_padding=1)
    T1 = time.time()
    output1, fltt = simu_transconv(input.astype(np.float32), weight.astype(np.float32), bias.astype(np.float32), 
                                                    stride_h=2, stride_w=2, 
                                                    padding_h=2, padding_w=2, 
                                                    output_padding_h=1, output_padding_w=1, type="float32", 
                                                    nblocks=nblocks, mblocks=mblocks)
    T2 = time.time()
    output2, intt = simu_transconv(input, weight, bias, stride_h=2, stride_w=2, 
                                                        padding_h=2, padding_w=2, output_padding_h=1, output_padding_w=1,
                                                        nblocks=nblocks, mblocks=mblocks)
    T3 = time.time()                                                    
else:
    # outputF = F.conv2d(torch.tensor(input).to(torch.float), 
    #                                     torch.tensor(weight).to(torch.float), 
    #                                     torch.tensor(bias).to(torch.float), 
    #                                     stride=2, padding=2)
    T1 = time.time()
    output1, fltt = simu_conv(input.astype(np.float32), weight.astype(np.float32), bias.astype(np.float32), 
                                                stride_h=2, stride_w=2, 
                                                padding_h=2, padding_w=2, type="float32", 
                                                nblocks=nblocks, mblocks=mblocks)
    T2 = time.time()
    output2, intt = simu_conv(input, weight, bias, stride_h=2, stride_w=2, padding_h=2, padding_w=2, 
                                                nblocks=nblocks, mblocks=mblocks)
    T3 = time.time()  

# print(abs(output1 - outputF.numpy()).max())
# print(abs(output2 - outputF.numpy()).max())
print(fltt)
print(intt)
print(T2 - T1)
print(T3 - T2)
