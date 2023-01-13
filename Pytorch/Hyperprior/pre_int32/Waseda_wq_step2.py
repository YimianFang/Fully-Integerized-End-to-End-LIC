"""
Waseda: 
Weight Quantization Only

Step2: Weight Clipping and Fine Tuning
"""
import copy
from model_W import *
from datasets import Datasets, TestKodakDataset
from torch.utils.data import DataLoader
import time
import quant_utils
from typing import List
from Quant_act import relu_replace_train, ClippedLinearQuantization_t

class AverageMeter:
    def __init__(self, length: int, name: str = None):
        assert length > 0
        self.name = name
        self.count = 0
        self.sum = 0.0
        self.current: int = -1
        self.history: List[float] = [None] * length

    @property
    def val(self) -> float:
        return self.history[self.current]

    @property
    def avg(self) -> float:
        return self.sum / self.count

    def update(self, val: float):
        self.current = (self.current + 1) % len(self.history)
        self.sum += val

        old = self.history[self.current]
        if old is None:
            self.count += 1
        else:
            self.sum -= old
        self.history[self.current] = val

def consine_learning_rate(optimizer, epoch, init_lr=0.1, T_max=100):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 1e-6 + init_lr * (1 + math.cos(math.pi * epoch / T_max)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("lr of epoch", epoch, ":", optimizer.param_groups[0]['lr'])
    return lr

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
    device = torch.device('cuda' if if_cuda else 'cpu')
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

def train(train_loader, model, sf_list, state, test_loader=None):
    global epoch, lr, print_freq, train_freq, train_lambda, optimizer, global_step, save_freq, act_Bits, w_Bits
    losses, psnrs, bpps = [AverageMeter(train_freq) for _ in range(3)]

    # switch to train mode
    model.train()
    # quant_utils.changemodelbit_Waseda_CW(w_Bits, model, sf_list, state)
    # quant_utils.changemodelbit_CW(w_Bits, model, sf_list, state)
    for i, input in enumerate(train_loader):
        # measure data loading time

        clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = model(input)
        distribution_loss = bpp
        distortion = mse_loss
        rd_loss = train_lambda * distortion + distribution_loss
        optimizer.zero_grad()
        rd_loss.backward()
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
        # quant_utils.changemodelbit_CW(w_Bits, model, sf_list, state)

        # t0 = time.time()
        if mse_loss.item() > 0:
            psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
            psnrs.update(psnr.item())
        else:
            psnrs.update(100)
        # t1 = time.time()
        losses.update(rd_loss.item())
        bpps.update(bpp.item())

        if i % print_freq == 0:
            print('epoch {epoch} | step {i} | lr {lr:.6f} | losses {lossesv:.6f} ({lossesa:.6f}) | psnrs {psnrsv:.6f} ({psnrsa:.6f}) | bpps {bppsv:.6f} ({bppsa:.6f})'
                  .format(epoch = epoch, i = global_step, lr = lr,
                          lossesv = losses.val, lossesa = losses.avg,
                          psnrsv = psnrs.val, psnrsa = psnrs.avg,
                          bppsv = bpps.val, bppsa = bpps.avg))

        if global_step % save_freq == 0:
            validate(test_loader, model, state, sf_list, global_step)
        global_step += 1


def validate(test_loader, model, state, sf_list, global_step):
    global test_freq, train_lambda, preLoss, act_target_layer, folder
    with torch.no_grad():
        model.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        sumLoss = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp= model(input)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            distribution_loss = bpp
            distortion = mse_loss
            Loss = train_lambda * distortion + distribution_loss
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input, data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            sumLoss += Loss
            # if global_step % test_freq == 0:
            #     print("No{:} | Loss:{:.6f}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(batch_idx, Loss, bpp, psnr, msssim, msssimDB))
            cnt += 1

        sumLoss /= cnt
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt

        print(
            "TEST | global_step:{:}, Loss:{:.6f}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(
                global_step,
                sumLoss,
                sumBpp,
                sumPsnr,
                sumMsssim,
                sumMsssimDB))

        if sumLoss < preLoss:
            preLoss = sumLoss
            print("new record!")
            save_path = folder
            save_checkpoint(model, state, sf_list, preLoss, global_step, save_path)
            f = open(folder + "/log.txt", "a")
            f.write("\nglobal_step:{:}, Loss:{:.6f}, Bpp:{:.6e}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(global_step, sumLoss, sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
            f.close()

    return sumLoss

import argparse

parser = argparse.ArgumentParser(description='Waseda: Step2')
parser.add_argument('-p', '--pretrained', 
                    help='path to pretrained model in Step 1')
parser.add_argument('-l', '--train_lambda', default=2048, type=int,
                    help='train_lambda')
parser.add_argument('-e', '--epsilon', default=1e-4, type=float,
                    help='epsilon')
parser.add_argument('-lr', '--learning_rate', default=1e-6, type=float,
                    help='learning_rate')
args = parser.parse_args()

w_Bits = 8
act_Bits = 8
lr = args.learning_rate
weight_decay = 0.1
epochs = 3
print_freq = 5
train_freq = 10
test_freq = 20
save_freq = 1
out_channel_N = 128
out_channel_M = 192
train_lambda = args.train_lambda
preLoss = 1.7
train_data_dir = "/data/fym/int32/patch"
val_data_dir = "val"
image_size = 256
batch_size = 4
act_target_layer = ["priorDecoder", "priorEncoder", "Decoder", "Encoder"]
dequantize = True
use_cuda = True
# f = "checkpoints/Waseda_2048/iter_51004.pth.tar" #####
folder = "checkpoints/Waseda_" + str(train_lambda) + "_WCFT"
epsilon = np.float64(args.epsilon)
if not os.path.isdir(folder):
    os.mkdir(folder)
train_dataset = Datasets(train_data_dir, image_size)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=0) ###worker 1
test_dataset = TestKodakDataset(data_dir = val_data_dir)
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=0) ###worker 1
if_cuda = torch.cuda.is_available() and use_cuda

model = ImageCompressor(out_channel_N, out_channel_M)
global_step, state_dict, sf_list, preLoss = load_pretrained_Waseda(model, args.pretrained)
file = open(folder + "/log.txt", 'a')
file.write("\n\nbegin to train with Lambda:{}".format(train_lambda))
file.write("\n---Global_step:{:}, PreLoss:{:.4e} and lr:{:.2e}---".format(global_step, preLoss, lr))
file.close()

for k, v in state_dict.items():
    if if_cuda:
        state_dict[k] = Variable(v.float().cuda(), requires_grad=True) #cuda()
    else:
        state_dict[k] = Variable(v.float().cpu(), requires_grad=True)

### no activation quantization
# target_layer_list = {}
# for l in act_target_layer:
#     target_layer_list[l] = []
# target_layer_list = targetlayer_func(model, act_target_layer, target_layer_list)
# # print(target_layer_list)

# for l in act_target_layer:
#     act_t_replace(model, target_layer_list[l])

# model,sf_list,n_dict=quant_utils.quant_model_bit(model,Bits)

if if_cuda:
    model = torch.nn.DataParallel(model, list(range(1))).cuda() ###cuda

optimizer = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, model.parameters())},
                             {'params': (state_dict[k] for k in state_dict.keys())}], lr,
                            weight_decay=weight_decay)

def Weight_Clipping(model, epsilon):
    model_dict = model.state_dict()
    modified_dict = copy.deepcopy(model_dict)
    for k, v in model_dict.items():
        if "weight" in k:
            w_clp = copy.deepcopy(v)
            c_in = w_clp.shape[1]
            h = w_clp.shape[2]
            w = w_clp.shape[3]
            wk_epsilon = w_clp.reshape(w_clp.shape[0], -1).abs().max(-1)[0] - epsilon
            wk_clp = wk_epsilon.reshape(-1, 1, 1, 1).repeat(1, c_in, h, w)
            modified_dict[k] = w_clp.clamp(-wk_clp, wk_clp)
    model_dict.update(modified_dict)
    model.load_state_dict(model_dict)

Weight_Clipping(model, epsilon)

for epoch in range(epochs):
    # lr = consine_learning_rate(optimizer, epoch, lr, epochs)

    # train for one epoch
    train(train_loader, model, sf_list, state_dict, test_loader)
