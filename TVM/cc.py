import torch
import math
from torchvision import transforms
import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm import relay
from tvm.relay import testing
import tvm.testing
from tvm import rpc
from tvm.contrib import utils

dev = tvm.cpu(0)
loaded_lib = tvm.runtime.load_module('lib_acl.so')
gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

from PIL import Image
odr = 24
in_scale = 8
pic_path = "val/kodim" + str(odr) + ".png"
image = Image.open(pic_path).convert('RGB')
transform = transforms.Compose([
        transforms.ToTensor(),
    ])
input_image = transform(image).unsqueeze(0)
input_image = torch.round(input_image * (2 ** in_scale)).to(torch.int32)
input_data = input_image.numpy()

module.set_input("data", tvm.nd.array(input_data, dev))
module.run()
out = module.get_output(0)
out_llvm = out.numpy()

import matplotlib.pyplot as plt
image = Image.open(pic_path).convert('RGB')
transform = transforms.Compose([
        transforms.ToTensor(),
    ])
input_image = transform(image)

mse_loss = torch.mean((torch.tensor(out_llvm/128) - input_image).pow(2))
psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
print(psnr)

clipped_recon_image = torch.clamp(torch.tensor(out_llvm/128), 0, 1)
plt.axis('off')
fig = plt.gcf()
fig.set_size_inches(10, 10)  # dpi = 300, output = 700*700 pixels
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.imshow(clipped_recon_image.squeeze().permute(1, 2, 0).cpu())
plt.savefig("savepic/pic" + str(batch_idx) + ".png", format='png', transparent=True, dpi=300, pad_inches=0)
