import argparse
import os
import random
import shutil
import time
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim

from bit_config import *
from utils.models.q_JAHP import *
from compressai.datasets import ImageFolder
from JAHPfltmodel import JointAutoregressiveHierarchicalPriors, load_model
from pytorch_msssim import ms_ssim

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/data0/fym',
                    help='path to dataset')
parser.add_argument('--train-lambda', default=31.73, type=float,
                    help='train_lambda')
parser.add_argument('--w-bit', type=int,
                    help='w_bit')
parser.add_argument('-a', '--arch', metavar='ARCH', default='JAHP',
                    help='model architecture')
parser.add_argument('--teacher-arch',
                    type=str,
                    default='resnet101',
                    help='teacher network used to do distillation')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument("--aux-learning-rate", default=1e-5, type=float,
                    help="Auxiliary loss learning rate (default: %(default)s)",)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained floating-point checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--act-range-momentum',
                    type=float,
                    default=-1,
                    help='momentum of the activation range moving average, '
                         '-1 stands for using minimum of min and maximum of max')
parser.add_argument('--quant-mode',
                    type=str,
                    default='symmetric',
                    choices=['asymmetric', 'symmetric'],
                    help='quantization mode')
parser.add_argument('--save-path',
                    type=str,
                    default='HAWQ/checkpoints/',
                    help='path to save the quantized model')
parser.add_argument('--data-percentage',
                    type=float,
                    default=1,
                    help='data percentage of training data')
parser.add_argument('--fix-BN',
                    action='store_true',
                    help='whether to fix BN statistics and fold BN during training')
parser.add_argument('--fix-BN-threshold',
                    type=int,
                    default=None,
                    help='when to start training with fixed and folded BN,'
                         'after the threshold iteration, the original fix-BN will be overwritten to be True')
parser.add_argument('--checkpoint-iter',
                    type=int,
                    default=-1,
                    help='the iteration that we save all the featuremap for analysis')
parser.add_argument('--evaluate-times',
                    type=int,
                    default=-1,
                    help='The number of evaluations during one epoch')
parser.add_argument('--quant-scheme',
                    type=str,
                    default='uniform4',
                    help='quantization bit configuration')
parser.add_argument('--resume-quantize',
                    action='store_true',
                    help='if True map the checkpoint to a quantized model,'
                         'otherwise map the checkpoint to an ordinary model and then quantize')
parser.add_argument('--act-percentile',
                    type=float,
                    default=0,
                    help='the percentage used for activation percentile'
                         '(0 means no percentile, 99.9 means cut off 0.1%)')
parser.add_argument('--weight-percentile',
                    type=float,
                    default=0,
                    help='the percentage used for weight percentile'
                         '(0 means no percentile, 99.9 means cut off 0.1%)')
parser.add_argument('--channel-wise',
                    action='store_false',
                    help='whether to use channel-wise quantizaiton or not')
parser.add_argument('--bias-bit',
                    type=int,
                    default=32,
                    help='quantizaiton bit-width for bias')
parser.add_argument('--distill-method',
                    type=str,
                    default='None',
                    help='you can choose None or KD_naive')
parser.add_argument('--distill-alpha',
                    type=float,
                    default=0.95,
                    help='how large is the ratio of normal loss and teacher loss')
parser.add_argument('--temperature',
                    type=float,
                    default=6,
                    help='how large is the temperature factor for distillation')
parser.add_argument('--fixed-point-quantization',
                    action='store_true',
                    help='whether to skip deployment-oriented operations and '
                         'use fixed-point rather than integer-only quantization')
parser.add_argument("--clip_max_norm",
                    default=1.0,
                    type=float,
                    help="gradient clipping max norm (default: %(default)s")

best_loss = 20
args = parser.parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

hook_counter = args.checkpoint_iter
hook_keys = []
hook_keys_counter = 0

dir = "msssim_" + str(args.train_lambda) + "_w" + str(args.w_bit)
log_path = os.path.join(args.save_path, dir)
if not os.path.isdir(log_path):
    os.mkdir(log_path)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filename=os.path.join(log_path, 'log.log'))
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

logging.info(args)

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
        lr=args.lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def load_qnet(path):
    if args.train_lambda < 31.73:
        N, M = 192, 192
    else:
        N, M = 192, 320
    model = JointAutoregressiveHierarchicalPriors(N, M)
    predict = torch.load(path)["state_dict"]
    qnet = qJAHP(model)
    qnet_dict = qnet.state_dict()
    qnet_dict.update(predict)
    qnet.load_state_dict(qnet_dict)
    return qnet

def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    args.gpu = gpu

    if args.gpu is not None:
        logging.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.train_lambda < 31.73:
        N, M = 192, 192
    else:
        N, M = 192, 320
    model = JointAutoregressiveHierarchicalPriors(N, M)

    optimizer, aux_optimizer = configure_optimizers(model, args)
    global lr_scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.train_lambda)

    if args.pretrained != '' and not args.resume:
        logging.info("=> using pre-trained floating-point model '{}'".format(args.arch))
        checkpoint = torch.load(args.pretrained)
        last_epoch = 0
        best_loss = float("inf")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        optimizer.param_groups[0]['lr'] = args.lr
        aux_optimizer.param_groups[0]['lr'] = args.aux_learning_rate
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    else:
        logging.info("=> creating floating-point model '{}'".format(args.arch))

    ##### qJAHP
    model = q_JAHP(model)

    if args.resume and not args.resume_quantize:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            model = load_qnet(args.resume)
            logging.info(model)
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    bit_config = bit_config_dict["bit_config_" + args.arch + "_w" + str(args.w_bit)]
    name_counter = 0

    for name, m in model.named_modules():
        if name in bit_config.keys():
            name_counter += 1
            # setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', args.bias_bit)
            setattr(m, 'quantize_bias', (args.bias_bit != 0))
            setattr(m, 'per_channel', args.channel_wise)
            setattr(m, 'act_percentile', args.act_percentile)
            setattr(m, 'act_range_momentum', args.act_range_momentum)
            setattr(m, 'weight_percentile', args.weight_percentile)
            setattr(m, 'fix_flag', False)
            setattr(m, 'fix_BN', args.fix_BN)
            setattr(m, 'fix_BN_threshold', args.fix_BN_threshold)
            setattr(m, 'training_BN_mode', args.fix_BN)
            setattr(m, 'checkpoint_iter_threshold', args.checkpoint_iter)
            setattr(m, 'save_path', args.save_path)
            setattr(m, 'fixed_point_quantization', args.fixed_point_quantization)

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

    logging.info("match all modules defined in bit_config: {}".format(len(bit_config.keys()) == name_counter))
    logging.info(model)
    logging.info("train_lambda: {}".format(args.train_lambda))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            if args.distill_method != 'None':
                teacher.cuda(args.gpu)
                teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            if args.distill_method != 'None':
                teacher.cuda()
                teacher = torch.nn.parallel.DistributedDataParallel(teacher)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if args.distill_method != 'None':
            teacher = teacher.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            # teacher is not alexnet or vgg
            if args.distill_method != 'None':
                teacher = torch.nn.DataParallel(teacher).cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            if args.distill_method != 'None':
                teacher = torch.nn.DataParallel(teacher).cuda()

    # # define optimizer
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # optionally resume optimizer and meta information from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            if args.gpu is not None:
                # best_loss may be from a checkpoint from a different GPU
                best_loss = best_loss.to(args.gpu)
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            logging.info("=> loaded optimizer and meta information from checkpoint '{}' (epoch {})".
                         format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.data, split="patch", transform=train_transforms)
    test_dataset = ImageFolder(args.data, split="val", transform=test_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataset_length = int(len(train_dataset) * args.data_percentage)
    if args.data_percentage == 1:
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=True,
            pin_memory=(args.gpu is not None and torch.cuda.is_available()),
        )
    else:
        partial_train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                                 [dataset_length, len(train_dataset) - dataset_length])
        train_loader = torch.utils.data.DataLoader(
            partial_train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # evaluate on validation set
    global test_dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=(args.gpu is not None and torch.cuda.is_available()),
    )

    if args.evaluate:
        loss, psnr, bpp, msssimDB = test_epoch(last_epoch, test_dataloader, model, criterion)
        return

    for epoch in range(last_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # train for one epoch
        train_one_epoch(
            model,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            True,
            args.train_lambda
        )

        loss, psnr, bpp, msssimDB = test_epoch(epoch, test_dataloader, model, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                "msssim_" + str(args.train_lambda) + "_w" + str(args.w_bit),
                psnr, msssimDB, bpp
            )


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, save, lmbda
):
    global best_loss
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
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

        if i % args.print_freq == 0:
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

        if i % args.print_freq == 0:
            loss, psnr, bpp, msssimDB = test_epoch(epoch, test_dataloader, model, criterion)
            lr_scheduler.step(loss)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    "msssim_" + str(lmbda) + "_w" + str(args.w_bit),
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
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tPSNR: {psnr.avg:.3f} |"
        f'\tmsssimDB: {msssimDB.avg:.3f} |'
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg, psnr.avg, bpp_loss.avg, msssimDB.avg


def save_checkpoint(state, is_best, dir, psnr, msssimDB, bpp):
    filename = f"iter_p{psnr:.4f}_m{msssimDB:.4f}_b{bpp:.4f}.pth.tar"
    save_path = os.path.join(args.save_path, dir)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, filename)
    torch.save(state, save_file)
    if is_best:
        best_file = os.path.join(save_path, "checkpoint_best_loss.pth.tar")
        logging.info("\nBest Loss Model: PSNR: {:.4f} | MS-SSIMDB: {:.4f} | Bpp: {:.4f}".format(psnr, msssimDB, bpp))
        shutil.copyfile(save_file, best_file)


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    print('lr = ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
