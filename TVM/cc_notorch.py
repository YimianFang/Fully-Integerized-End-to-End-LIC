import tvm
import numpy as np
from tvm.contrib import graph_executor as runtime

dev = tvm.cpu(0)
loaded_lib = tvm.runtime.load_module('main_lib_remote.so')
gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

from PIL import Image
odr = 24
in_scale = 8
pic_path = "val/kodim" + str(odr) + ".png"
image = np.transpose(np.array(Image.open(pic_path).convert('RGB')).astype('float')/255, (2,0,1))
input_data = np.round(image[np.newaxis, :] * (2 ** in_scale)).astype('int32')

module.set_input("data", tvm.nd.array(input_data, dev))
module.run()
out = module.get_output(0)
out_llvm = out.numpy()

import matplotlib.pyplot as plt
input_image = np.transpose(np.array(Image.open(pic_path).convert('RGB')).astype('float')/255, (2,0,1))

mse_loss = np.mean(np.power(out_llvm/128 - input_image, 2))
psnr = 10 * (np.log(1. / mse_loss) / np.log(10))
print(psnr)

clipped_recon_image = np.clip(out_llvm/128, 0, 1)
plt.axis('off')
fig = plt.gcf()
fig.set_size_inches(10, 10)  # dpi = 300, output = 700*700 pixels
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.imshow(clipped_recon_image.squeeze().transpose(1, 2, 0))
plt.savefig("savepic/pic" + str(batch_idx) + ".png", format='png', transparent=True, dpi=300, pad_inches=0)
