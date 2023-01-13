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

from JAHPmodel_mse import JointAutoregressiveHierarchicalPriors, load_qftmodel, load_t2e2tmodel, save_model
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
        out["bpp_y"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["bpp_z"] = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)

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
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if global_step % 100 == 0:
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

        if global_step % 500 == 0:
            loss, psnr, bpp, msssimDB = test_epoch(epoch, test_dataloader, model, criterion)
            lr_scheduler.step(loss)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if save and is_best:
                save_model(model, loss, global_step, model_dir)
                f = open(model_dir + "/log.txt", "a")
                f.write("\nBest Loss Model: Step: {} | PSNR: {:.4f} | MS-SSIMDB: {:.4f} | Bpp: {:.4f}".format(global_step, psnr, msssimDB, bpp))
                f.close()
        


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
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            psnr.update(out_criterion["psnr"])
            msssimDB.update(out_criterion["msssimDB"])
            mse_loss.update(out_criterion["mse_loss"])

            # pic_loss = out_criterion["loss"]
            # pic_psnr = out_criterion["psnr"]
            # pic_msssimDB = out_criterion["msssimDB"]
            # pic_mse_loss = out_criterion["mse_loss"]
            # pic_bpp_loss = out_criterion["bpp_loss"]
            # pic_bpp_y = out_criterion["bpp_y"]
            # pic_bpp_z = out_criterion["bpp_z"]
            # pic_aux_loss = model.aux_loss()
            # print(
            #     f"Test epoch {epoch}: Pic{i:02d} losses:"
            #     f"\tLoss: {pic_loss:.4f} |"
            #     f"\tPSNR: {pic_psnr:.4f} |"
            #     f'\tmsssimDB: {pic_msssimDB:.4f} |'
            #     f"\tMSE loss: {pic_mse_loss:.4f} |"
            #     f"\tBpp y: {pic_bpp_y:.4f} |"
            #     f"\tBpp z: {pic_bpp_z:.4f} |"
            #     f"\tBpp loss: {pic_bpp_loss:.4f} |"
            #     f"\tAux loss: {pic_aux_loss:.4f}"
            # )


    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.4f} |"
        f"\tPSNR: {psnr.avg:.4f} |"
        f'\tmsssimDB: {msssimDB.avg:.4f} |'
        f"\tMSE loss: {mse_loss.avg:.4f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.4f}\n"
    )

    return loss.avg, psnr.avg, bpp_loss.avg, msssimDB.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
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
        default=64,
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
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--eq", action="store_true", default=False, help="load a equalized pretrain pre_int32 model")
    parser.add_argument("--preiter", type=int, help="Iter of a pretrain pre_int32 model")
    parser.add_argument("--ga-left", type=int, default=2)
    parser.add_argument("--hs-left", type=int, default=1)
    parser.add_argument("--en-left", type=int, default=2)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

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

    if args.lmbda < 0.0250:
        N, M = 192, 192
    else:
        N, M = 192, 320
    if args.eq:
        prepath = "pre_int32/checkpoints/eq_mse_" + str(args.lmbda) #####
    else:
        prepath = "pre_int32/checkpoints/mse_" + str(args.lmbda) #####
    net = JointAutoregressiveHierarchicalPriors(N, M,
                                                ga_left=args.ga_left, 
                                                hs_left=args.hs_left, 
                                                en_left=args.en_left,
                                                prepath=prepath, 
                                                preiter=args.preiter,
                                                device=device) #####
    net = net.to(device)

    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    global lr_scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    global best_loss, global_step, model_dir
    best_loss = float("inf")
    last_epoch = 1
    if args.checkpoint:  # load from previous checkpoint
        print("Loading checkpoint", args.checkpoint)
        global_step, best_loss = load_t2e2tmodel(net, args.checkpoint)

    model_dir = "OCS/checkpoints/mse_" + str(args.lmbda)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    f = open(model_dir + "/log.txt", "a")
    f.write("\n\nbegin to log the best model with Loss:{:.6f} and lr:{:.2e}".format(best_loss, optimizer.param_groups[0]['lr']))
    f.write("\nga_left: {} | hs_left: {} | en_left: {}".format(args.ga_left, args.hs_left, args.en_left))
    f.close()
    if args.checkpoint:
        loss, psnr, bpp, msssimDB = test_epoch(last_epoch, test_dataloader, net, criterion)
    # loss, psnr, bpp, msssimDB = test_epoch(last_epoch, test_dataloader, net, criterion) ##### for test
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

        if args.save and is_best:
            save_model(net, global_step, model_dir)
            f = open(model_dir + "/log.txt", "a")
            f.write("\nBest Loss Model: Step: {} | PSNR: {:.4f} | MS-SSIMDB: {:.4f} | Bpp: {:.4f}".format(global_step, psnr, msssimDB, bpp))
            f.close()


if __name__ == "__main__":
    main(sys.argv[1:])