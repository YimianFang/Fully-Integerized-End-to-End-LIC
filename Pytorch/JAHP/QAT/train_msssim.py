import argparse
import math
import random
import shutil
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder

from model import JointAutoregressiveHierarchicalPriors
import os
from pytorch_msssim import ms_ssim

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


def configure_optimizers(net, args):
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
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, save, lmbda
):
    global best_loss, global_step
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        global_step += 1
        d = d.to(device)
        
        model.train()
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        ##### introduce mask
        optimizer.param_groups[0]["params"][1].data *= mask   
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if global_step % prnt_freq == 0:
            print(
                f"Train global step {global_step}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tPSNR: {out_criterion["psnr"].item():.3f} |'
                f'\tmsssimDB: {out_criterion["msssimDB"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

        if global_step > freeze_step:
            # Freeze quantizer parameters
            model.apply(torch.quantization.disable_observer)

        if global_step % save_freq == 0:
            quantized_model = torch.quantization.convert(model.cpu())
            loss, psnr, bpp, msssimDB = test_epoch(epoch, test_dataloader, quantized_model, criterion)
            model.cuda()
            lr_scheduler.step(loss)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "state_dict": model.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "mask": mask
                    },
                    is_best,
                    model_dir,
                    psnr, msssimDB, bpp
                )
        


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
        f"Test global step {global_step}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tPSNR: {psnr.avg:.3f} |"
        f'\tmsssimDB: {msssimDB.avg:.3f} |'
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg, psnr.avg, bpp_loss.avg, msssimDB.avg


def save_checkpoint(state, is_best, save_path, psnr, msssimDB, bpp):
    filename = f"iter_p{psnr:.4f}_m{msssimDB:.4f}_b{bpp:.4f}.pth.tar"
    save_file = os.path.join(save_path, filename)
    if is_best:
        torch.save(state, save_file)
        best_file = os.path.join(save_path, "checkpoint_best_loss.pth.tar")
        f = open(save_path + "/log.txt", "a")
        f.write("\nBest Loss Model: Global Step: {:d} | PSNR: {:.4f} | MS-SSIMDB: {:.4f} | Bpp: {:.4f}".format(global_step, psnr, msssimDB, bpp))
        f.close()
        shutil.copyfile(save_file, best_file)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", default="/data0/fym", type=str, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--prnt-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("-f", "--freeze-step", type=int, default=300000)
    parser.add_argument("--pretrain", type=str, help="Path to a float point pretrainted checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to a QAT checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    global freeze_step, prnt_freq, save_freq
    freeze_step = args.freeze_step
    prnt_freq = args.prnt_freq
    save_freq = args.save_freq

    train_transforms = transforms.Compose(
        [transforms.ToTensor()] # transforms.RandomCrop(args.patch_size), 
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="patch", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="val", transform=test_transforms)

    device = args.device if args.cuda and torch.cuda.is_available() else "cpu" #####

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(args.cuda and torch.cuda.is_available()),
    )

    global test_dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(args.cuda and torch.cuda.is_available()),
    )

    if args.lmbda < 31.73:
        N, M = 192, 192
    else:
        N, M = 192, 320
    net = JointAutoregressiveHierarchicalPriors(N, M)
    net = net.to(device)

    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    global lr_scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    global model_dir
    model_dir = "QAT/checkpoints/msssim_" + str(args.lmbda)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    global best_loss, global_step, mask
    best_loss = float("inf")
    last_epoch, global_step = 0, 0
    if args.pretrain:  # load from previous checkpoint
        print("Loading", args.pretrain)
        checkpoint = torch.load(args.pretrain, map_location=device)
        pretrain_dict = checkpoint["state_dict"]
        toquant_dict = {}
        for k, v in pretrain_dict.items():
            if "g_s" in k or "h_s.0" in k or "h_s.2" in k:
                preord = k.split(".")[1]
                ord = str(int(preord) // 2 * 5 + 2)
                if "weight" in k:
                    toquant_dict[k.replace(preord, ord)] = v.permute(1, 0, 2, 3).flip([2,3])
                else:
                    toquant_dict[k.replace(preord, ord)] = v
            elif "h_s.4" in k:
                preord = "4"
                ord = str(9)
                toquant_dict[k.replace(preord, ord)] = v
            elif "context" in k and ("weight" in k or "bias" in k):
                toquant_dict[k.replace("context_prediction", "context_prediction.conv")] = v
            elif "context" in k and "mask" in k:
                toquant_dict[k] = v
                mask = v
            else:
                toquant_dict[k] = v
        net.load_state_dict(toquant_dict)
        f = open(model_dir + "/log.txt", "a")
        f.write("\n\nbegin to log the PRETRAINED model with Loss:{:.6f} and lr:{:.2e}".format(best_loss, optimizer.param_groups[0]['lr']))
        f.close()
    elif args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_loss = checkpoint["loss"]
        mask = checkpoint["mask"]
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        f = open(model_dir + "/log.txt", "a")
        f.write("\n\nbegin to log the QAT model with Loss:{:.6f} and lr:{:.2e}".format(best_loss, optimizer.param_groups[0]['lr']))
        f.close()

    if args.checkpoint:
        loss, psnr, bpp, msssimDB = test_epoch(last_epoch - 1, test_dataloader, net, criterion)

    modules_to_fuse = [["g_a.0", "g_a.1"], ["g_a.2", "g_a.3"], ["g_a.4", "g_a.5"],
                        ["g_s.2", "g_s.3"], ["g_s.7", "g_s.8"], ["g_s.12", "g_s.13"],
                        ["h_a.0", "h_a.1"], ["h_a.2", "h_a.3"], 
                        ["h_s.2", "h_s.3"], ["h_s.7", "h_s.8"], 
                        ["entropy_parameters.0", "entropy_parameters.1"],
                        ["entropy_parameters.2", "entropy_parameters.3"],]
    torch.quantization.fuse_modules(net, modules_to_fuse, inplace=True)
    net.train()
    net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(net, inplace=True)
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            args.save,
            args.lmbda
        )
        loss, psnr, bpp, msssimDB = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "mask": mask
                },
                is_best,
                model_dir,
                psnr, msssimDB, bpp
            )


if __name__ == "__main__":
    main(sys.argv[1:])