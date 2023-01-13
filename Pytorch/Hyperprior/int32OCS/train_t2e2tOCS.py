import os
import argparse
from model_t2e2tOCS import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from datasets import Datasets, TestKodakDataset
from tensorboardX import SummaryWriter
from Meter import AverageMeter


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.enabled = True
# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 8192
print_freq = 100
cal_step = 50
warmup_step = 0#  // gpu_num
batch_size = 4
tot_epoch = 1000000
tot_step = 2500000
decay_interval = 2200000
decay_interval2 = 4000000
lr_decay = 0.1
lr_decay2 = 0.05
image_size = 256
logger = logging.getLogger("ImageCompression")
tb_logger = None
global_step = 0
save_model_freq = 50000
test_step = 10000
out_channel_N = 192
out_channel_M = 320
parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')

parser.add_argument('-n', '--name', default='', help='experiment name')
parser.add_argument('-p', '--pretrain', default='', help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument('--config', dest='config', required=False, help='hyperparameter in json format')
parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--train', dest='train', required=True, help='the path of training dataset')
parser.add_argument('--val', dest='val', required=True, help='the path of validation dataset')


def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, base_lr, cur_lr, lr_decay, decay_interval, lr_decay2, decay_interval2, train_lambda, batch_size, print_freq, \
        out_channel_M, out_channel_N, save_model_freq, test_step
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
        if train_lambda < 4096:
            out_channel_N = 128
            out_channel_M = 192
        else:
            out_channel_N = 192
            out_channel_M = 320
    if 'batch_size' in config:
        batch_size = config['batch_size']
    if "print_freq" in config:
        print_freq = config['print_freq']
    if "test_step" in config:
        test_step = config['test_step']
    if "save_model_freq" in config:
        save_model_freq = config['save_model_freq']
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']
        if 'decay2' in config['lr']:
            lr_decay2 = config['lr']['decay2']
        if 'decay_interval2' in config['lr']:
            decay_interval2 = config['lr']['decay_interval2']
    if 'out_channel_N' in config:
        out_channel_N = config['out_channel_N']
    if 'out_channel_M' in config:
        out_channel_M = config['out_channel_M']


def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    global warmup_step
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:#  // gpu_num:
        lr = base_lr
    elif global_step < decay_interval2:
        # lr = base_lr * (lr_decay ** (global_step // decay_interval))
        lr = base_lr * lr_decay
    else:
        lr = base_lr * lr_decay2
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch, global_step):
    logger.info("Epoch {} begin".format(epoch))
    time1= time.time()
    testKodak(global_step)
    time2= time.time()
    print(time2-time1)
    net.train()
    global optimizer
    elapsed, losses, psnrs, bpps, bpp_features, bpp_zs, mse_losses = [AverageMeter(print_freq) for _ in range(7)]
    # model_time = 0
    # compute_time = 0
    # log_time = 0
    for batch_idx, input in enumerate(train_loader):
        adjust_learning_rate(optimizer, global_step)
        if global_step > tot_step:
            save_model(model, global_step, save_path)
            exit()
        start_time = time.time()
        global_step += 1
        # print("debug", torch.max(input), torch.min(input))
        clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(input)
        # print("debug", clipped_recon_image.shape, " ", mse_loss.shape, " ", bpp.shape)
        # print("debug", mse_loss, " ", bpp_feature, " ", bpp_z, " ", bpp)
        msssim = ms_ssim(clipped_recon_image, input.cuda(), data_range=1.0, size_average=True)
        distribution_loss = bpp
        distortion = mse_loss ###index
        rd_loss = train_lambda * distortion + distribution_loss
        optimizer.zero_grad()
        rd_loss.backward()
        # for p in net.named_parameters():
        #     print(p[1].grad, p[0])

        # def clip_gradient(optimizer, grad_clip):
        #     for group in optimizer.param_groups:
        #         for param in group["params"]:
        #             if param.grad is not None:
        #                 param.grad.data.clamp_(-grad_clip, grad_clip)
        # clip_gradient(optimizer, 5)
        optimizer.step()
        # print(net.module.Encoder.conv1.weight.grad)
        # model_time += (time.time()-start_time)
        if (global_step % cal_step) == 0:
            # t0 = time.time()
            if mse_loss.item() > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                psnrs.update(psnr.item())
            else:
                psnrs.update(100)
            # t1 = time.time()
            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item())
            bpps.update(bpp.item())
            bpp_features.update(bpp_feature.item())
            bpp_zs.update(bpp_z.item())
            mse_losses.update(mse_loss.item())

        if (global_step % print_freq) == 0:
            # begin = time.time()
            tb_logger.add_scalar('lr', cur_lr, global_step)
            tb_logger.add_scalar('rd_loss', losses.avg, global_step)
            tb_logger.add_scalar('psnr', psnrs.avg, global_step)
            tb_logger.add_scalar('bpp', bpps.avg, global_step)
            tb_logger.add_scalar('bpp_feature', bpp_features.avg, global_step)
            tb_logger.add_scalar('bpp_z', bpp_zs.avg, global_step)
            process = global_step / tot_step * 100.0
            log = (' | '.join([
                f'Step [{global_step}/{tot_step}={process:.2f}%]',
                f'Epoch {epoch}',
                f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                f'Lr {cur_lr}',
                f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                f'Bpp_feature {bpp_features.val:.5f} ({bpp_features.avg:.5f})',
                f'Bpp_z {bpp_zs.val:.5f} ({bpp_zs.avg:.5f})',
                f'MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})',
            ]))
            logger.info(log)

        if (global_step % save_model_freq) == 0:
            testKodak(global_step, False)
            save_model(model, global_step, save_path)
            net.train()
        if (global_step % test_step) == 0:
            testKodak(global_step)
            net.train()

    return global_step


def testKodak(step, if_prnt=True):
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMselss = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            # input = torch.load("imout.pth.tar")
            # time1 = time.time()
            clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(input)
            # time2 = time.time()
            # print(time2 - time1)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            sumMselss += mse_loss
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input, data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            if if_prnt:
                logger.info("Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(bpp, psnr, msssim, msssimDB))
            cnt += 1

        if if_prnt:
            logger.info("Test on Kodak dataset: model-{}".format(step))
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMselss /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        distribution_loss = sumBpp
        distortion = sumMselss
        rd_loss = train_lambda * distortion + distribution_loss
        if if_prnt:
            logger.info("Dataset Average result---Loss:{:.6f}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(rd_loss, sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
            if tb_logger !=None:
                logger.info("Add tensorboard---Step:{}".format(step))
                tb_logger.add_scalar("BPP_Test", sumBpp, step)
                tb_logger.add_scalar("PSNR_Test", sumPsnr, step)
                tb_logger.add_scalar("MS-SSIM_Test", sumMsssim, step)
                tb_logger.add_scalar("MS-SSIM_DB_Test", sumMsssimDB, step)
            else:
                logger.info("No need to add tensorboard")


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(seed=args.seed)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    tb_logger = None
    save_path = os.path.join('checkpoints', args.name)
    if args.name != '':
        os.makedirs(save_path, exist_ok=True)
        filehandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("image compression training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)
    logger.info("out_channel_N:{}, out_channel_M:{}".format(out_channel_N, out_channel_M))
    model = ImageCompressor(out_channel_N, out_channel_M)
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        # global_step = load_qftmodel(model, args.pretrain) #####
        global_step = load_t2e2tmodel(model, args.pretrain) #####
    net = model.cuda() ###cuda
    # net = model.cpu() ###cpu
    net = torch.nn.DataParallel(net, list(range(1))) ###cuda
    parameters = net.parameters()
    global test_loader
    test_dataset = TestKodakDataset(data_dir=args.val)
    # test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1) ###cpu
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1) ###cuda
    if args.test:
        testKodak(global_step)
        exit(-1)
    # prior_group = []
    # other_group = []
    # for pn, pv in net.named_parameters():
    #     if "prior" in pn:
    #         prior_group += [pv]
    #     else:
    #         other_group += [pv]

    # for k,v in model.named_parameters():
    #     if (not "Decoder" in k) or "prior" in k:
    #         v.requires_grad = False
    
    # optimizer = optim.Adam(parameters, lr=base_lr)
    optimizer = optim.SGD(parameters, lr=base_lr, momentum=0.9)
    # save_model(model, 0)
    global train_loader
    tb_logger = SummaryWriter(os.path.join(save_path, 'events'))
    train_data_dir = args.train
    train_dataset = Datasets(train_data_dir, image_size)
    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=batch_size,
    #                           shuffle=True) ###cpu
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=2) ###cuda
    steps_epoch = global_step // (len(train_dataset) // (batch_size))
    # save_model(model, global_step, save_path)
    for epoch in range(steps_epoch, tot_epoch):
        adjust_learning_rate(optimizer, global_step)
        if global_step > tot_step:
            save_model(model, global_step, save_path)
            break
        global_step = train(epoch, global_step)
        save_model(model, global_step, save_path)