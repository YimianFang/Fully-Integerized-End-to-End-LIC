import torch
import os, math
import numpy as np
from simu_JAHPquantint.q_JAHP import qJAHP
from simu_JAHPquantint.quant_modules import freeze_model
from JAHPfltmodel import *
from bit_config import *
from torchvision.utils import save_image
from compressai.datasets import ImageFolder
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim

# qnet = torch.load("checkpoints/kodak/test/quantized_checkpoint.pth.tar")
# fltnet = torch.load("checkpoints/kodak/test/model_best.pth.tar")

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

def load_qnet(path):
    if train_lambda < 31.73:
        N, M = 192, 192
    else:
        N, M = 192, 320
    model = JointAutoregressiveHierarchicalPriors(N, M)
    predict = torch.load(path)["state_dict"]
    epoch = torch.load(path)['epoch']
    qnet = qJAHP(model, ga_left=5, hs_left=1, en_left=5) # ga_left=5, 10
    qnet_dict = qnet.state_dict()
    qnet_dict.update(predict)
    qnet.load_state_dict(qnet_dict)
    return qnet, epoch


def validate(epoch, test_dataloader, model, criterion):
    global train_lambda
    # switch to evaluate mode
    freeze_model(model)
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
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg, psnr.avg, bpp_loss.avg, msssimDB.avg

train_lambda = 31.73 #####
w_bit = 10 #####
criterion = RateDistortionLoss(train_lambda)
bit_config = bit_config_dict["bit_config_JAHP_w" + str(w_bit)]
qnet, epoch = load_qnet("HAWQ/checkpoints/msssim_"+str(train_lambda)+"_w"+str(w_bit)+"/checkpoint_best_loss.pth.tar")
qnet = qnet.cuda()
data_path = '/data0/fym'
test_transforms = transforms.Compose([transforms.ToTensor()])
test_dataset = ImageFolder(data_path, split="val", transform=test_transforms)
test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )
name_counter = 0
for name, m in qnet.named_modules():
    if name in bit_config.keys():
        name_counter += 1
        # setattr(m, 'quant_mode', 'symmetric')
        setattr(m, 'bias_bit', 32)
        setattr(m, 'quantize_bias', True)
        setattr(m, 'per_channel', True)
        setattr(m, 'act_percentile', 0)
        setattr(m, 'act_range_momentum', -1)
        setattr(m, 'weight_percentile', 0)
        setattr(m, 'fix_flag', False)
        setattr(m, 'fix_BN', False)
        setattr(m, 'fix_BN_threshold', None)
        setattr(m, 'training_BN_mode', False)
        setattr(m, 'checkpoint_iter_threshold', -1)
        setattr(m, 'fixed_point_quantization', False)

        if type(bit_config[name]) is tuple:
            bitwidth = bit_config[name][0]
        else:
            bitwidth = bit_config[name]

        if hasattr(m, 'activation_bit'):
            setattr(m, 'activation_bit', bitwidth)
            if bitwidth == 4:
                setattr(m, 'quant_mode', 'asymmetric')
        else:
            setattr(m, 'weight_bit', bitwidth)
loss, psnr, bpp, msssimDB = validate(epoch, test_dataloader, qnet, criterion)