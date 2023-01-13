import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm import relay
from tvm.relay import testing
import tvm.testing
from tvm.contrib import utils
import math

m = np.load("npy/model_t2e2t.npy",allow_pickle=True).item()

muls = np.load("npy/muls_E.npy", allow_pickle=True).tolist()

in_scale = 8
Bits = 8
clp_k = 7
odr = 22
n = 2 ** Bits - 1
relus = np.load("npy/relus_E.npy", allow_pickle=True).tolist()
stas_max = []
for i in range(len(relus)):
    relus[i] /= (2 ** 16)
    stas_max.append(round((math.pow(2.0, Bits) - 1) / relus[i] * (2 ** clp_k)))
relus = np.load("npy/relus_E.npy", allow_pickle=True).tolist()
fx = tvm.relay.Constant(tvm.nd.array(np.array([2 ** (16 + clp_k)], dtype="int32")))
fx_s = tvm.relay.Constant(tvm.nd.array(np.array([16 + clp_k], dtype="int32")))
d2 = tvm.relay.Constant(tvm.nd.array(np.array([2], dtype="int32")))

batch_size = 1
out_channel_N=128
out_channel_M=192

from PIL import Image
odr = 1
in_scale = 8
pic_path = "1920x1080/" + str(odr) + ".jpeg"
image = np.transpose(np.array(Image.open(pic_path).convert('RGB')).astype('float')/255, (2,0,1))
input_image = np.round(image[np.newaxis, :] * (2 ** in_scale)).astype('int32')

data_shape = tuple(input_image.shape)
data = relay.var("data", relay.TensorType(data_shape, "int32"))

strides0 = (2, 2)
padding0 = (2, 2)
kernel_size0 = 5
kernel_sizes0 = (kernel_size0, kernel_size0)
in_channels0 = 3
out_channels0 = out_channel_N
w0 = m['Encoder.conv1.weight']
weight0 = relay.const(w0)
b0 = m['Encoder.conv1.bias'] * (2 ** in_scale)
bias0 = tvm.relay.Constant(tvm.nd.array(b0))
muls0 = tvm.relay.Constant(tvm.nd.array(muls[0].reshape(-1, 1, 1, 1)))
dr0 = tvm.relay.Constant(tvm.nd.array(np.array([2 ** (19 + in_scale - clp_k)], dtype="int32")))
sr0 = tvm.relay.Constant(tvm.nd.array(np.array([19 + in_scale - clp_k], dtype="int32")))
clp0 = stas_max[0]
scl0 = tvm.relay.Constant(tvm.nd.array(np.array([relus[0]], dtype="int32")))

syn = relay.nn.conv2d(
    data, weight0,
    strides=strides0,
    padding=padding0,
    kernel_size=kernel_sizes0,
    channels=out_channels0)
syn = relay.nn.bias_add(syn, bias0)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = syn * muls0
syn = relay.right_shift((syn + dr0 / d2), sr0)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = relay.clip(syn, 0, clp0)
syn = relay.right_shift(syn * scl0 + fx / d2, fx_s)

strides1 = (2, 2)
padding1 = (2, 2)
kernel_size1 = 5
kernel_sizes1 = (kernel_size1, kernel_size1)
in_channels1 = out_channel_N
out_channels1 = out_channel_N
w1 =m['Encoder.conv2.weight']
weight1 = relay.const(w1)
b1 = m['Encoder.conv2.bias']
bias1 = tvm.relay.Constant(tvm.nd.array(b1))
muls1 = tvm.relay.Constant(tvm.nd.array(muls[1].reshape(-1, 1, 1, 1)))
dr1 = tvm.relay.Constant(tvm.nd.array(np.array([2 ** (23 - clp_k)], dtype="int32")))
sr1 = tvm.relay.Constant(tvm.nd.array(np.array([23 - clp_k], dtype="int32")))
clp1 = stas_max[1]
scl1 = tvm.relay.Constant(tvm.nd.array(np.array([relus[1]], dtype="int32")))

syn = relay.nn.conv2d(
    syn, weight1,
    strides=strides1,
    padding=padding1,
    kernel_size=kernel_sizes1,
    channels=out_channels1)
syn = relay.nn.bias_add(syn, bias1)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = syn * muls1
syn = relay.right_shift((syn + dr1 / d2), sr1)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = relay.clip(syn, 0, clp1)
syn = relay.right_shift(syn * scl1 + fx / d2, fx_s)

strides2 = (2, 2)
padding2 = (2, 2)
kernel_size2 = 5
kernel_sizes2 = (kernel_size2, kernel_size2)
in_channels2 = out_channel_N
out_channels2 = out_channel_N
w2 = m['Encoder.conv3.weight']
weight2 = relay.const(w2)
b2 = m['Encoder.conv3.bias']
bias2 = tvm.relay.Constant(tvm.nd.array(b2))
muls2 = tvm.relay.Constant(tvm.nd.array(muls[2].reshape(-1, 1, 1, 1)))
dr2 = tvm.relay.Constant(tvm.nd.array(np.array([2 ** (23 - clp_k)], dtype="int32")))
sr2 = tvm.relay.Constant(tvm.nd.array(np.array([23 - clp_k], dtype="int32")))
clp2 = stas_max[2]
scl2 = tvm.relay.Constant(tvm.nd.array(np.array([relus[2]], dtype="int32")))

syn = relay.nn.conv2d(
    syn, weight2,
    strides=strides2,
    padding=padding2,
    kernel_size=kernel_sizes2,
    channels=out_channels2)
syn = relay.nn.bias_add(syn, bias2)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = syn * muls2
syn = relay.right_shift((syn + dr2 / d2), sr2)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = relay.clip(syn, 0, clp2)
syn = relay.right_shift(syn * scl2 + fx / d2, fx_s)

strides3 = (2, 2)
padding3 = (2, 2)
kernel_size3 = 5
kernel_sizes3 = (kernel_size3, kernel_size3)
in_channels3 = out_channel_N
out_channels3 = out_channel_M
w3 = m['Encoder.conv4.weight']
weight3 = relay.const(w3)
b3 = m['Encoder.conv4.bias']
bias3 = tvm.relay.Constant(tvm.nd.array(b3))
muls3 = tvm.relay.Constant(tvm.nd.array(muls[3].reshape(-1, 1, 1, 1)))
dr3 = tvm.relay.Constant(tvm.nd.array(np.array([2 ** 19], dtype="int32")))
sr3 = tvm.relay.Constant(tvm.nd.array(np.array([19], dtype="int32")))

syn = relay.nn.conv2d(
    syn, weight3,
    strides=strides3,
    padding=padding3,
    kernel_size=kernel_sizes3,
    channels=out_channels3)
syn = relay.nn.bias_add(syn, bias3)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = syn * muls3
syn = relay.right_shift((syn + dr3 / d2), sr3)
syn = relay.transpose(syn, [1, 0, 2, 3])

muls = np.load("npy/muls_D.npy", allow_pickle=True).tolist()

Bits = 8
clp_k = 6
n = 2 ** Bits - 1
relus = np.load("npy/relus_D.npy", allow_pickle=True).tolist()
stas_max = []
for i in range(len(relus)):
    relus[i] /= (2 ** 16)
    stas_max.append(round((math.pow(2.0, Bits) - 1) / relus[i] * (2 ** clp_k)))
relus = np.load("npy/relus_D.npy", allow_pickle=True).tolist()
fx = tvm.relay.Constant(tvm.nd.array(np.array([2 ** (16 + clp_k)], dtype="int32")))
fx_s = tvm.relay.Constant(tvm.nd.array(np.array([16 + clp_k], dtype="int32")))
d2 = tvm.relay.Constant(tvm.nd.array(np.array([2], dtype="int32")))

batch_size = 1
out_channel_N=128
out_channel_M=192

strides0 = (2, 2)
padding0 = (2, 2)
output_padding0 = (1, 1)
kernel_size0 = 5
kernel_sizes0 = (kernel_size0, kernel_size0)
in_channels0 = out_channel_M
out_channels0 = out_channel_N
w0 =m['Decoder.deconv1.weight']
w0 = np.flip(np.transpose(w0, (1, 0, 2, 3)), (2,3))
weight0 = relay.const(w0)
b0 = m['Decoder.deconv1.bias']
bias0 = tvm.relay.Constant(tvm.nd.array(b0))
muls0 = tvm.relay.Constant(tvm.nd.array(muls[0].reshape(-1, 1, 1, 1)))
dr0 = tvm.relay.Constant(tvm.nd.array(np.array([2 ** (19 - clp_k)], dtype="int32")))
sr0 = tvm.relay.Constant(tvm.nd.array(np.array([19 - clp_k], dtype="int32")))
clp0 = stas_max[0]
scl0 = tvm.relay.Constant(tvm.nd.array(np.array([relus[0]], dtype="int32")))

syn = relay.nn.conv2d_transpose(
    syn, weight0,
    strides=strides0,
    padding=padding0,
    output_padding=output_padding0,
    kernel_size=kernel_sizes0,
    channels=out_channels0)
syn = relay.nn.bias_add(syn, bias0)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = syn * muls0
syn = relay.right_shift((syn + dr0 / d2), sr0)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = relay.clip(syn, 0, clp0)
syn = relay.right_shift(syn * scl0 + fx / d2, fx_s)

strides1 = (2, 2)
padding1 = (2, 2)
output_padding1 = (1, 1)
kernel_size1 = 5
kernel_sizes1 = (kernel_size1, kernel_size1)
in_channels1 = out_channel_N
out_channels1 = out_channel_N
w1 = m['Decoder.deconv2.weight']
w1 = np.flip(np.transpose(w1, (1, 0, 2, 3)),(2,3))
weight1 = relay.const(w1)
b1 = m['Decoder.deconv2.bias']
bias1 = tvm.relay.Constant(tvm.nd.array(b1))
muls1 = tvm.relay.Constant(tvm.nd.array(muls[1].reshape(-1, 1, 1, 1)))
dr1 = tvm.relay.Constant(tvm.nd.array(np.array([2 ** (19 - clp_k)], dtype="int32")))
sr1 = tvm.relay.Constant(tvm.nd.array(np.array([19 - clp_k], dtype="int32")))
clp1 = stas_max[1]
scl1 = tvm.relay.Constant(tvm.nd.array(np.array([relus[1]], dtype="int32")))

syn = relay.nn.conv2d_transpose(
    syn, weight1,
    strides=strides1,
    padding=padding1,
    output_padding=output_padding1,
    kernel_size=kernel_sizes1,
    channels=out_channels1)
syn = relay.nn.bias_add(syn, bias1)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = syn * muls1
syn = relay.right_shift((syn + dr1 / d2), sr1)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = relay.clip(syn, 0, clp1)
syn = relay.right_shift(syn * scl1 + fx / d2, fx_s)

strides2 = (2, 2)
padding2 = (2, 2)
output_padding2 = (1, 1)
kernel_size2 = 5
kernel_sizes2 = (kernel_size2, kernel_size2)
in_channels2 = out_channel_N
out_channels2 = out_channel_N
w2 = m['Decoder.deconv3.weight']
w2 = np.flip(np.transpose(w2, (1, 0, 2, 3)),(2,3))
weight2 = relay.const(w2)
b2 = m['Decoder.deconv3.bias']
bias2 = tvm.relay.Constant(tvm.nd.array(b2))
muls2 = tvm.relay.Constant(tvm.nd.array(muls[2].reshape(-1, 1, 1, 1)))
dr2 = tvm.relay.Constant(tvm.nd.array(np.array([2 ** (19 - clp_k)], dtype="int32")))
sr2 = tvm.relay.Constant(tvm.nd.array(np.array([19 - clp_k], dtype="int32")))
clp2 = stas_max[2]
scl2 = tvm.relay.Constant(tvm.nd.array(np.array([relus[2]], dtype="int32")))

syn = relay.nn.conv2d_transpose(
    syn, weight2,
    strides=strides2,
    padding=padding2,
    output_padding=output_padding2,
    kernel_size=kernel_sizes2,
    channels=out_channels2)
syn = relay.nn.bias_add(syn, bias2)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = syn * muls2
syn = relay.right_shift((syn + dr2 / d2), sr2)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = relay.clip(syn, 0, clp2)
syn = relay.right_shift(syn * scl2 + fx / d2, fx_s)

strides3 = (2, 2)
padding3 = (2, 2)
output_padding3 = (1, 1)
kernel_size3 = 5
kernel_sizes3 = (kernel_size3, kernel_size3)
in_channels3 = out_channel_N
out_channels3 = 3
w3 = m['Decoder.deconv4.weight']
w3 = np.flip(np.transpose(w3, (1, 0, 2, 3)),(2,3))
weight3 = relay.const(w3)
b3 = m['Decoder.deconv4.bias']
bias3 = tvm.relay.Constant(tvm.nd.array(b3))
muls3 = tvm.relay.Constant(tvm.nd.array(muls[3].reshape(-1, 1, 1, 1)))

syn = relay.nn.conv2d_transpose(
    syn, weight3,
    strides=strides3,
    padding=padding3,
    output_padding=output_padding3,
    kernel_size=kernel_sizes3,
    channels=out_channels3)
syn = relay.nn.bias_add(syn, bias3)
syn = relay.transpose(syn, [1, 0, 2, 3])
syn = syn * muls3
syn = relay.transpose(syn, [1, 0, 2, 3])

# dr3 = tvm.relay.Constant(tvm.nd.array(np.array([2 ** 23], dtype="float32")))
m = tvm.relay.Constant(tvm.nd.array(np.array([2 ** 7], dtype="int32")))
dr3 = tvm.relay.Constant(tvm.nd.array(np.array([2 ** 23], dtype="int32")))
sr3 = tvm.relay.Constant(tvm.nd.array(np.array([23], dtype="int32")))

syn = relay.right_shift(syn * m + dr3 / d2, sr3)

fr = relay.analysis.free_vars(syn)
syn = relay.Function(fr, syn)
net, params = testing.create_workload(syn)
# target = "llvm -mtriple=aarch64-linux-gnu"
print("Ready to build")
target = tvm.target.cuda(model='tx2') ###
lib = relay.build(net, target, params=params)
lib.export_library("arm_cuda_t2e2tm_1080.so")

from PIL import Image
odr = 1
in_scale = 8
pic_path = "1920x1080/" + str(odr) + ".jpeg"
image1 = np.transpose(np.array(Image.open(pic_path).convert('RGB')).astype('float')/255, (2,0,1))
input_data = np.round(image1[np.newaxis, :] * (2 ** in_scale)).astype('int32')

import time
dev = tvm.cuda() ###
module = tvm.contrib.graph_executor.GraphModule(lib['default'](dev))
module.set_input("data", tvm.nd.array(input_data, dev))
print("Warm running")
T1 = time.time()
module.run()
out = module.get_output(0).numpy()
T2 = time.time()
unop0 = T2 - T1

from PIL import Image
odr = 2
in_scale = 8
pic_path = "1920x1080/" + str(odr) + ".jpeg"
image1 = np.transpose(np.array(Image.open(pic_path).convert('RGB')).astype('float')/255, (2,0,1))
input_data = np.round(image1[np.newaxis, :] * (2 ** in_scale)).astype('int32')
module.set_input("data", tvm.nd.array(input_data, dev))
print("Ready to run")
T1 = time.time()
module.run()
out = module.get_output(0).numpy()
T2 = time.time()
unop = T2 - T1
out_llvm = out
print(unop0)
print(unop)
# np.save("arm_t2e2tm_out.npy", out_llvm)

from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
net, config = partition_for_tensorrt(net, params)

