"""
改变保存的整数模型目标层的muls系数的定点参数m
"""

import os
import torch
import torch.nn as nn
import copy
import math
import numpy as np
from Quant_act import ClippedLinearQuantization_t, relu_replace_train
from model_qft import ImageCompressor

def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False

def print_container_max(container, statmax, prefix=''):
    # Iterate through model, insert quantization functions as appropriate
    for name, module in container.named_children():
        full_name = prefix + name
        if type(module) == ClippedLinearQuantization_t:
            maxcuda = module.statis_max / module.statisN
            statmax[full_name.replace("module.", "").split('.')[0]] = statmax[full_name.replace("module.", "").split('.')[0]] + [float(maxcuda.cpu().detach().numpy())]
        if has_children(module):
            statmax = print_container_max(module, statmax, full_name + '.')
    return statmax

def w_scale(model):
    global act_target_layer, w_Bits
    model_dict = model.state_dict()
    target_layer = act_target_layer
    scale = {}
    for l in target_layer:
        l_scale = []
        for k, v in model.named_parameters():
            if l == k.replace("module.", "").split('.')[0] and "weight" in k:
                round_weight = torch.zeros_like(v)
                sub_scale = torch.zeros_like(torch.Tensor(v.shape[0]))
                for i in range(v.shape[0]):
                    sub_scale[i] = 1 / v[i].abs().max() * math.pow(2.0, w_Bits - 1)
                    round_weight[i] = torch.round(v[i] * sub_scale[i])
                model_dict[k] = round_weight
                l_scale.append(sub_scale)
        scale[l] = l_scale

    for l in target_layer:
        kk = 0
        for k, v in model.named_parameters():
            if l == k.replace("module.", "").split('.')[0] and "bias" in k:
                round_bias = torch.zeros_like(v)
                for i in range(v.shape[0]):
                    round_bias[i] = torch.round(v[i] * scale[l][kk][i])
                model_dict[k] = round_bias
                kk += 1

    model.load_state_dict(model_dict)
    return scale

def model2int32(model, float_state_dict, act_target_layer, scale, relu_scale): #model -- 权重量化后的网络,权重偏置为整数 float_state_dict -- 权重量化微调后、已经量化过的的浮点数网络
    global muls_m
    muls = {}
    relus = {}
    model_dict = model.state_dict()
    modelint32 = copy.deepcopy(model_dict)
    for l in act_target_layer:
        kk = 0
        sub_muls = []
        sub_relus = []
        prn = ''
        for n, m in model.named_modules():
            if "conv" in n and n.replace("module.", "").split('.')[0] == l:
                modelint32[n+".weight"] = model_dict[n+".weight"].to(torch.int32)
                if 'relu' in prn and prn.replace("module.", "").split('.')[0] == l:
                    modelint32[n+".bias"] = torch.round(float_state_dict[n+".bias"] * scale[l][kk] * relu_scale[l][kk - 1]).to(torch.int32)
                    sub_muls.append(torch.round((2 ** muls_m[l][kk]) / (scale[l][kk] * relu_scale[l][kk - 1])).to(torch.int32))
                    sub_relus.append(round(relu_scale[l][kk - 1] * (2 ** 16)))
                    kk += 1
                else:
                    modelint32[n+".bias"] = torch.round(float_state_dict[n+".bias"] * scale[l][kk]).to(torch.int32)
                    sub_muls.append(torch.round((2 ** muls_m[l][kk]) / scale[l][kk]).to(torch.int32))
                    kk += 1
            prn = n
        muls[l] = sub_muls
        relus[l] = list(np.array(sub_relus).astype(np.int32))
    # print(muls)
    # print(relus)
    return modelint32, muls, relus

def save_intmodel(model, save_path, global_step, state, preLoss):
    global act_target_layer
    act_Bits = 8
    s_model = copy.deepcopy(model).cpu()
    float_state_dict = copy.deepcopy(model.state_dict())
    statmax = {}
    for l in act_target_layer:
        statmax[l] = []
    statmax = print_container_max(s_model, statmax)
    # print(statmax)
    relu_scale = {}
    for l in act_target_layer:
        sub_relu_scale = []
        for i in statmax[l]:
            sub_relu_scale.append(((2 ** act_Bits) - 1) / i)
        relu_scale[l] = sub_relu_scale
    # print(relu_scale)
    scale = w_scale(s_model)
    # print(scale)
    int_model_dict, muls, relus = model2int32(s_model, float_state_dict, act_target_layer, scale, relu_scale)
    int_model_dict_new = {}
    float_dict_new = {}
    for k, v in float_state_dict.items():
        int_model_dict_new[k.replace("module.", "")] = int_model_dict[k]
        if "statis" in k:
            int_model_dict_new.pop(k.replace("module.", ""))
        float_dict_new[k.replace("module.", "")] = v
    save_checkpoint_int32_actft(int_model_dict_new, float_dict_new, state, preLoss, muls, relus, act_target_layer, global_step, save_path)

def save_checkpoint_int32_actft(model_dict, float_dict, state, preLoss, muls, relus, act_target_layer, iter, name):
    torch.save({"int_model":model_dict, "float_dict": float_dict, "float_state": state, "preLoss": preLoss}, os.path.join(name, "iter_{}.pth.tar".format(iter)))
    for i in act_target_layer:
        torch.save(muls[i], os.path.join(name, "iter_{}_muls_{}.pth.tar".format(str(iter), str(i))))
    for i in act_target_layer:
        torch.save(relus[i], os.path.join(name, "iter_{}_relus_{}.pth.tar".format(str(iter), str(i))))

def load_checkpoint_int32_actft(model, f):
    with open(f, 'rb') as f:
        file = torch.load(f, map_location=torch.device('cpu'))
        int_model = file["int_model"]
        pretrained_dict = file["float_dict"]
        state = file["float_state"]
        preLoss = file["preLoss"]
        model_dict = model.state_dict()
        pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() if k.replace('module.','') in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed]), int_model, state, preLoss
    else:
        return 0, int_model, state, preLoss

def targetlayer_func(container, target_layer, target_layer_list, prefix=''):
    for name, module in container.named_children():
        full_name = prefix + name
        if type(module) == nn.ReLU:
            for i in target_layer:
                if i == full_name.replace("module.", "").split('.')[0]:
                    target_layer_list[i].append(full_name)
                    break
        if has_children(module):
            # For container we call recursively
            target_layer_list = targetlayer_func(module, target_layer, target_layer_list, full_name + '.')
    return target_layer_list

def act_t_replace(container, target_layer_list, prefix=''):
    global act_Bits, dequantize
    device = torch.device('cuda' if if_cuda else 'cpu')
    for name, module in container.named_children():
        full_name = prefix + name
        if type(module)==nn.ReLU and full_name in target_layer_list:
            new_module = relu_replace_train(module, act_Bits, dequantize = dequantize, device = device)
            setattr(container, name, new_module)
        if has_children(module):
            # For container we call recursively
            act_t_replace(module, target_layer_list, full_name + '.')

act_target_layer = ["priorDecoder", "priorEncoder", "Decoder", "Encoder"]
muls_m = {"priorDecoder":[18, 21, 21], "priorEncoder":[16, 23, 23], "Decoder":[19, 19, 19, 23], "Encoder":[19, 23, 23, 19]} # "Decoder":[19, 19, 19, 23], "Encoder":[19, 23, 23, 19]
load_path = "checkpoints/qft_2048_eq_4/iter_52533.pth.tar"
save_path = "checkpoints/qft_2048_eq_4_2/"
if not os.path.isdir(save_path):
    os.mkdir(save_path)

w_Bits = 10 #####
act_Bits = 8
if_cuda = True
dequantize = True
model = ImageCompressor(128, 192)
target_layer_list = {}
for l in act_target_layer:
    target_layer_list[l] = []
target_layer_list = targetlayer_func(model, act_target_layer, target_layer_list)
# print(target_layer_list)

for l in act_target_layer:
    act_t_replace(model, target_layer_list[l])

global_step, _, state_dict, preLoss = load_checkpoint_int32_actft(model, load_path)
save_intmodel(model, save_path, global_step, state_dict, preLoss)
