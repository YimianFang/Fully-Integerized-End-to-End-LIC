import argparse
import math
import random
import shutil
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.layers import GDN, MaskedConv2d

from JAHPmodel import JointAutoregressiveHierarchicalPriors, load_checkpoint_int32_actft, save_checkpoint_int32_actft
from Quant_act import relu_replace_train, ClippedLinearQuantization_t
import quant_utils
import os, copy
from pytorch_msssim import ms_ssim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False

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
    device = torch.device('cuda' if use_cuda else 'cpu')
    for name, module in container.named_children():
        full_name = prefix + name
        if type(module)==nn.ReLU and full_name in target_layer_list:
            new_module = relu_replace_train(module, act_Bits, dequantize = dequantize, device = device)
            setattr(container, name, new_module)
        if has_children(module):
            # For container we call recursively
            act_t_replace(module, target_layer_list, full_name + '.')

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
                if l != "g_s" and k != "h_s.0.weight" and k != "h_s.2.weight":
                    round_weight = torch.zeros_like(v)
                    sub_scale = torch.zeros_like(torch.Tensor(v.shape[0]))
                    for i in range(v.shape[0]):
                        sub_scale[i] = 1 / v[i].abs().max() * math.pow(2.0, w_Bits - 1)
                        round_weight[i] = torch.round(v[i] * sub_scale[i])
                else:
                    vT = v.permute(1, 0, 2, 3).flip([2,3])
                    round_weight = torch.zeros_like(vT)
                    sub_scale = torch.zeros_like(torch.Tensor(vT.shape[0]))
                    for i in range(vT.shape[0]):
                        sub_scale[i] = 1 / vT[i].abs().max() * math.pow(2.0, w_Bits - 1)
                        round_weight[i] = torch.round(vT[i] * sub_scale[i])
                    round_weight = round_weight.permute(1, 0, 2, 3).flip([2,3])
                
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
        prm = None
        for n, m in model.named_modules():
            if (type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d or type(m) == MaskedConv2d) and n.replace("module.", "").split('.')[0] == l:
                modelint32[n+".weight"] = model_dict[n+".weight"].to(torch.int32)
                if type(prm) == ClippedLinearQuantization_t and prn.replace("module.", "").split('.')[0] == l:
                    modelint32[n+".bias"] = torch.round(float_state_dict[n+".bias"] * scale[l][kk] * relu_scale[l][kk - 1]).to(torch.int32)
                    sub_muls.append(torch.round((2 ** muls_m[l][kk]) / (scale[l][kk] * relu_scale[l][kk - 1])).to(torch.int32))
                    sub_relus.append(round(relu_scale[l][kk - 1] * (2 ** 16)))
                    kk += 1
                else:
                    modelint32[n+".bias"] = torch.round(float_state_dict[n+".bias"] * scale[l][kk]).to(torch.int32)
                    sub_muls.append(torch.round((2 ** muls_m[l][kk]) / scale[l][kk]).to(torch.int32))
                    kk += 1
            prm = m
            prn = n
        muls[l] = sub_muls
        relus[l] = list(np.array(sub_relus).astype(np.int32))
    # print(muls)
    # print(relus)
    return modelint32, muls, relus

def save_intmodel(model, save_path, global_step, state, preLoss):
    global act_target_layer, act_Bits
    s_model = copy.deepcopy(model).cpu()
    float_state_dict = copy.deepcopy(s_model.state_dict())
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

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["psnr"] = 10 * (torch.log(1 * 1 / out["mse_loss"]) / np.log(10))
        msssim = ms_ssim(output["x_hat"], target, data_range=1.0, size_average=True)
        out["msssim"] = msssim
        out["msssimDB"] = -10 * (torch.log(1-msssim) / np.log(10))
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, learning_rate, aux_learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, save, lmbda, sf_list, state
):
    global best_loss, global_step
    model.train()
    device = next(model.parameters()).device

    quant_utils.changemodelbit_CW(w_Bits, model, sf_list, state)
    for i, d in enumerate(train_dataloader):
        global_step += 1
        d = d.to(device)

        model.train()
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        quant_utils.quantback(model, state) ###
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        sf_list = quant_utils.get_sf_CW(model, state) ###
        quant_utils.changemodelbit_CW(w_Bits, model, sf_list, state)

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 5 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tPSNR: {out_criterion["psnr"].item():.3f} |'
                f'\tmsssimDB: {out_criterion["msssimDB"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

        if i % 1 == 0:
            loss, psnr, bpp, msssimDB = test_epoch(epoch, test_dataloader, model, criterion)
            lr_scheduler.step(loss)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if save and is_best:
                model_path = os.path.join(model_dir, "iter_" + str(global_step))
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                save_intmodel(model, model_path, global_step, state, best_loss)
                f = open(model_dir + "/log.txt", "a")
                f.write("\nglobal_step:{:}, Loss:{:.6f}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM-DB:{:.6f}".format(global_step, loss, bpp, psnr, msssimDB))
                f.close()
                print("Model Saved!")
        


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    psnr = AverageMeter()
    msssimDB = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            psnr.update(out_criterion["psnr"])
            msssimDB.update(out_criterion["msssimDB"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tPSNR: {psnr.avg:.3f} |"
        f'\tmsssimDB: {msssimDB.avg:.3f} |'
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}"
    )

    return loss.avg, psnr.avg, bpp_loss.avg, msssimDB.avg


def main():
    global use_cuda, act_target_layer, muls_m, w_Bits, act_Bits, dequantize
    epochs = 10
    weight_decay = 0.1
    clip_max_norm = 1.0
    data_dir = "/data0/fym"
    image_size = 256
    batch_size = 8
    act_target_layer = ["h_s", "h_a", "g_s", "g_a", "entropy_parameters", "context_prediction"]
    muls_m = {"h_s":[23, 23, 23], "h_a":[18, 23, 24], 
                "g_s":[23, 23, 25, 25], "g_a":[21, 25, 25, 23],
                "entropy_parameters":[23, 23, 23], "context_prediction":[22]} #####
    save = True
    dequantize = True
    use_cuda = True
    device = "cuda:0" #####
    num_workers = 4 #####
    w_Bits = 10 #####
    act_Bits = 8 #####
    train_lambda = 0.0130 #####
    lr = 0.00001 #####
    aux_lr = 0.0001 #####
    optim_Adam = True #####
    save_path = "pre_int32/checkpoints/mse_" + str(train_lambda) + "/" #####
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    loadFP = True ##### False -- "iter_1/iter_1.pth.tar" True -- "checkpoint_best_loss.pth.tar"
    dir = "FP" if loadFP else "pre_int32"
    load_path = os.path.join(dir, "checkpoints/mse_" + str(train_lambda))
    # model_name = "iter_1/iter_1.pth.tar" #####
    model_name = "checkpoint_best_loss.pth.tar" #####
    checkpoint = os.path.join(load_path, model_name)

    train_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolder(data_dir, split="patch", transform=train_transforms)
    test_dataset = ImageFolder(data_dir, split="val", transform=test_transforms)

    device = device if use_cuda and torch.cuda.is_available() else "cpu" #####

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=(use_cuda and torch.cuda.is_available()),
    )

    global test_dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(use_cuda and torch.cuda.is_available()),
    )

    if train_lambda < 0.0250:
        N, M = 192, 192
    else:
        N, M = 192, 320
    net = JointAutoregressiveHierarchicalPriors(N, M)
    net = net.to(device)

    # if use_cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    global global_step, best_loss, lr_scheduler
    global_step = 0
    last_epoch = 1
    if loadFP:  # load from previous checkpoint
        print("Loading", checkpoint)
        checkpoint = torch.load(checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"] * 1.2
        net.load_state_dict(checkpoint["state_dict"])
        optimizer, aux_optimizer = configure_optimizers(net, lr, aux_lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        criterion = RateDistortionLoss(lmbda=train_lambda)
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        state_dict = copy.deepcopy(net.state_dict())
        target_layer_list = {}
        for l in act_target_layer:
            target_layer_list[l] = []
        target_layer_list = targetlayer_func(net, act_target_layer, target_layer_list)
        for l in act_target_layer:
            act_t_replace(net, target_layer_list[l])
    else:
        target_layer_list = {}
        for l in act_target_layer:
            target_layer_list[l] = []
        target_layer_list = targetlayer_func(net, act_target_layer, target_layer_list)
        for l in act_target_layer:
            act_t_replace(net, target_layer_list[l])
        global_step, _, state_dict, best_loss = load_checkpoint_int32_actft(net, checkpoint)
        optimizer, aux_optimizer = configure_optimizers(net, lr, aux_lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        criterion = RateDistortionLoss(lmbda=train_lambda)

    sf_list = quant_utils.get_sf_CW(net, state_dict)
    for k, v in state_dict.items():
        if use_cuda:
            state_dict[k] = Variable(v.float().cuda(), requires_grad=True) #cuda()
        else:
            state_dict[k] = Variable(v.float().cpu(), requires_grad=True)
    
    if optim_Adam:
        optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, net.parameters())},
                                {'params': (state_dict[k] for k in state_dict.keys())}], lr,
                                weight_decay=weight_decay) #####
    else:
        optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, net.parameters())},
                                {'params': (state_dict[k] for k in state_dict.keys())}], lr,
                                weight_decay=weight_decay) #####

    global model_dir
    model_dir = "pre_int32/checkpoints/mse_" + str(train_lambda)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    f = open(model_dir + "/log.txt", "a")
    f.write("\n\nbegin to log the best model with Loss:{:.6f} and lr:{:.2e}".format(best_loss, optimizer.param_groups[0]['lr']))
    f.close()

    if not loadFP:
        loss, psnr, bpp, msssimDB = test_epoch(last_epoch, test_dataloader, net, criterion)
    for epoch in range(last_epoch, epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            clip_max_norm,
            save,
            train_lambda,
            sf_list,
            state_dict
        )


if __name__ == "__main__":
    main()