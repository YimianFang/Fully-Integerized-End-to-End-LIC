"""
Waseda: 
Weight Quantization Only

Step1: Train
"""
import os, copy, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
import time
import quant_utils
from typing import List
from Quant_act import relu_replace_train, ClippedLinearQuantization_t
import argparse
from JAHPmodel import JointAutoregressiveHierarchicalPriors, load_pretrained_Waseda, save_checkpoint
from pytorch_msssim import ms_ssim
from compressai.datasets import ImageFolder

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
        out["loss"] = self.lmbda * (1 - msssim) + out["bpp_loss"]

        return out

def configure_optimizers(net, learning_rate, aux_learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    global state_dict

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

    optimizer = optim.Adam([{'params': (params_dict[n] for n in sorted(parameters))},
                        {'params': (state_dict[k] for k in state_dict.keys())}],
                        lr=learning_rate,
                        weight_decay=weight_decay)
    # optimizer = optim.Adam(
    #     (params_dict[n] for n in sorted(parameters)),
    #     lr=learning_rate,
    # )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_learning_rate,
    )
    return optimizer, aux_optimizer

def consine_learning_rate(optimizer, epoch, init_lr=0.1, T_max=100):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 1e-6 + init_lr * (1 + math.cos(math.pi * epoch / T_max)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("lr of epoch", epoch, ":", optimizer.param_groups[0]['lr'])
    return lr


def train(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, sf_list, state
):
    global preLoss, global_step
    device = next(model.parameters()).device
    
    # switch to train mode
    model.train()
    quant_utils.changemodelbit_Waseda_CW(w_Bits, model, sf_list, state)
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
        def Waseda_GradientClip(optimizer):
            k = 0
            for param in optimizer.param_groups[0]["params"]:
                if param.grad is not None and len(param.data.shape) == 4 and param.data.shape[0] != 1:
                    wk_epsilon = (param.data.reshape(param.data.shape[0], -1).abs().max(-1)[0] - epsilon).reshape(-1, 1, 1, 1)
                    idx = (param.data <= wk_epsilon) & (param.data >= -wk_epsilon)
                    param.grad.data *= idx
                    optimizer.param_groups[1]["params"][k] = param.grad.data.clone()
                k += 1
        Waseda_GradientClip(optimizer)
        optimizer.step()
        sf_list = quant_utils.get_sf_CW(model, state) ###
        quant_utils.changemodelbit_Waseda_CW(w_Bits, model, sf_list, state) ###
        
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if global_step % print_freq == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.4f} |'
                f'\tPSNR: {out_criterion["psnr"].item():.4f} |'
                f'\tmsssimDB: {out_criterion["msssimDB"].item():.4f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.4f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f} |'
                f"\tAux loss: {aux_loss.item():.4f}"
            )
        
        if global_step % save_freq == 0:
            loss = validate(model, state, sf_list, global_step)


def validate(model, state, sf_list, global_step):
    global train_lambda, preLoss, folder
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
        f"\tLoss: {loss.avg:.4f} |"
        f"\tPSNR: {psnr.avg:.4f} |"
        f'\tmsssimDB: {msssimDB.avg:.4f} |'
        f"\tMSE loss: {mse_loss.avg:.4f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.4f}"
    )

    if loss.avg < preLoss:
        preLoss = loss.avg
        print("new record!")
        save_path = folder
        save_checkpoint(model, state, sf_list, preLoss, global_step, save_path)
        f = open(folder + "/log.txt", "a")
        f.write("\nglobal_step:{:}, Loss:{:.6e}, Bpp:{:.6f}, PSNR:{:.6f}, MSE:{:.6f}, MS-SSIM-DB:{:.6f}".format(global_step, loss.avg, bpp_loss.avg, psnr.avg, mse_loss.avg, msssimDB.avg))
        f.close()

    return loss.avg


parser = argparse.ArgumentParser(description='Waseda: Step2')
parser.add_argument('-p', '--pretrained', 
                    help='path to pretrained model in Step 1')
parser.add_argument('-l', '--train-lambda', default=31.73, type=float,
                    help='train_lambda')
parser.add_argument('-e', '--epsilon', default=1e-4, type=float,
                    help='epsilon')
parser.add_argument('-lr', '--learning_rate', default=1e-5, type=float,
                    help='learning_rate')
parser.add_argument("--aux-learning-rate", default=1e-4, type=float,
                    help="Auxiliary loss learning rate (default: %(default)s)")
parser.add_argument("-n", "--num-workers", type=int, default=4,
                    help="Dataloaders threads (default: %(default)s)")
args = parser.parse_args()


w_Bits = 8
act_Bits = 8
num_workers = args.num_workers
lr = args.learning_rate
aux_lr = args.aux_learning_rate
epsilon = np.float64(args.epsilon)
weight_decay = 0.1
epochs = 3
print_freq = 5
save_freq = 1
out_channel_N = 128
out_channel_M = 192
train_lambda = args.train_lambda
preLoss = 1.7
data_dir = "/data0/fym"
image_size = 256
batch_size = 8
dequantize = True
use_cuda = True
# f = "/data/fym/compression/checkpoints/l_256/iter_215000.pth.tar" #####
# f = "checkpoints/Waseda_2048/iter_51165.pth.tar" #####
folder = "Waseda/checkpoints/msssim_" + str(train_lambda)
if not os.path.isdir(folder):
    os.mkdir(folder)
train_transforms = transforms.Compose(
    [transforms.ToTensor()]
)

test_transforms = transforms.Compose(
    [transforms.ToTensor()]
)

train_dataset = ImageFolder(data_dir, split="patch", transform=train_transforms)
test_dataset = ImageFolder(data_dir, split="val", transform=test_transforms)

device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu" #####

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=(use_cuda and torch.cuda.is_available()),
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=(use_cuda and torch.cuda.is_available()),
)
if_cuda = torch.cuda.is_available() and use_cuda

if train_lambda < 31.73:
    N, M = 192, 192
else:
    N, M = 192, 320
net = JointAutoregressiveHierarchicalPriors(N, M)
net = net.to(device)
global_step = 0
last_epoch = 1
global_step, state_dict, sf_list, preLoss = load_pretrained_Waseda(net, args.pretrained)
    
file = open(folder + "/log.txt", 'a')
file.write("\n\nbegin to train with Lambda:{} | Waseda 2".format(train_lambda))
file.write("\n---Global_step:{:}, PreLoss:{:.4e} and lr:{:.2e}---".format(global_step, preLoss, lr))
file.close()

for k, v in state_dict.items():
    if if_cuda:
        state_dict[k] = Variable(v.float().cuda(), requires_grad=True) #cuda()
    else:
        state_dict[k] = Variable(v.float().cpu(), requires_grad=True)

def Weight_Clipping(model, epsilon):
    model_dict = model.state_dict()
    modified_dict = copy.deepcopy(model_dict)
    for k, v in model_dict.items():
        if "weight" in k:
            k = k.replace("module.", "")
            l = k.split('.')[0]
            if l != "g_s" and k != "h_s.0.weight" and k != "h_s.2.weight":
                w_clp = copy.deepcopy(v)
                c_in = w_clp.shape[1]
                h = w_clp.shape[2]
                w = w_clp.shape[3]
                wk_epsilon = w_clp.reshape(w_clp.shape[0], -1).abs().max(-1)[0] - epsilon
                wk_clp = wk_epsilon.reshape(-1, 1, 1, 1).repeat(1, c_in, h, w)
                modified_dict[k] = w_clp.clamp(-wk_clp, wk_clp)
            else:
                w_clp = copy.deepcopy(v).permute(1, 0, 2, 3).flip([2,3])
                c_in = w_clp.shape[1]
                h = w_clp.shape[2]
                w = w_clp.shape[3]
                wk_epsilon = w_clp.reshape(w_clp.shape[0], -1).abs().max(-1)[0] - epsilon
                wk_clp = wk_epsilon.reshape(-1, 1, 1, 1).repeat(1, c_in, h, w)
                modified_dict[k] = w_clp.clamp(-wk_clp, wk_clp).permute(1, 0, 2, 3).flip([2,3])
    model_dict.update(modified_dict)
    model.load_state_dict(model_dict)

Weight_Clipping(net, epsilon)

if if_cuda:
    model = torch.nn.DataParallel(net, list(range(1))).cuda() ###cuda

optimizer, aux_optimizer = configure_optimizers(model, lr, aux_lr)
criterion = RateDistortionLoss(lmbda=train_lambda)
# optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters())},
#                         {'params': (state_dict[k] for k in state_dict.keys())}], lr,
#                         weight_decay=weight_decay)

for epoch in range(epochs):
    # train for one epoch
    train(
        net,
        criterion,
        train_dataloader,
        optimizer,
        aux_optimizer,
        epoch,
        sf_list, 
        state_dict
    )
