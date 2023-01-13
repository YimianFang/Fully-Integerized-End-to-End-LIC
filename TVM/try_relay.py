import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm import relay
from tvm.relay import testing
import tvm.testing
from tvm.contrib import utils
import math

data = relay.var("data", relay.TensorType([1], "int32"))
# m = tvm.relay.Constant(tvm.nd.array(np.array([2 ** 23], dtype="int32")))
# e = tvm.relay.Constant(tvm.nd.array(np.array([32], dtype="int32")))

syn = relay.fixed_point_multiply(data, 2 ** 30, 10)

fr = relay.analysis.free_vars(syn)
syn = relay.Function(fr, syn)
net, params = testing.create_workload(syn)
# target = "llvm -mtriple=aarch64-linux-gnu"
print("Ready to build")
target = "llvm -mtriple=aarch64-linux-gnu"
lib = relay.build(net, target, params=params)

dev = tvm.cpu() ###
module = tvm.contrib.graph_executor.GraphModule(lib['default'](dev))

print("Ready to run")
module.set_input("data", tvm.nd.array(np.round(2 ** 11).astype('int32'), dev))
module.run()
out = module.get_output(0) #####1
out_llvm = out.numpy() #####1
print(out_llvm)

