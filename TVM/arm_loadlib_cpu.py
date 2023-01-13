import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm import relay
from tvm.relay import testing
import tvm.testing
from tvm.contrib import utils
from PIL import Image
import time

dev = tvm.cpu()
T_load_begin = time.perf_counter()
print("load begin")
lib = tvm.runtime.load_module('arm_cpu_t2e2tm.so')
T_load_end = time.perf_counter()
print("load finished: ", T_load_end - T_load_begin, "s")
module = tvm.contrib.graph_executor.GraphModule(lib['default'](dev))

odr = 20
in_scale = 8
while (odr <= 24):
    pic_path = "val/kodim" + str(odr) + ".png"
    image = np.transpose(np.array(Image.open(pic_path).convert('RGB')).astype('float')/255, (2,0,1))
    input_data = np.round(image[np.newaxis, :] * (2 ** in_scale)).astype('int32')
    
    module.set_input("data", tvm.nd.array(input_data, dev))
    # T_run_begin = time.perf_counter()
    T_run_begin = time.process_time()
    print("Ready to run pic ", odr)
    module.run()
    # T_run_end = time.perf_counter()
    T_run_end = time.process_time()
    print("over: ", (T_run_end - T_run_begin)*1000, "ms")

    out = module.get_output(0)
    out_llvm = out.numpy()
    np.save("out/arm_t2e2tcpu_" + str(odr) + ".npy", out_llvm)

    odr += 1
