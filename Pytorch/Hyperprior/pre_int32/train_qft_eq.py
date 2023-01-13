import copy
from model_qft_eq import *
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
            maxcuda = module.statis_max / module.statisN * module.scl_test
            statmax[full_name.replace("module.", "").split('.')[0]] = statmax[full_name.replace("module.", "").split('.')[0]] + [float(maxcuda.cpu().detach().numpy())]
        if has_children(module):
            statmax = print_container_max(module, statmax, full_name + '.')
    return statmax

def train(train_loader, model, sf_list, state, test_loader=None):
    global epoch, lr, print_freq, train_freq, train_lambda, optimizer, global_step, save_freq, act_Bits, w_Bits, save_path, load_path, model_name
    losses, psnrs, bpps = [AverageMeter(train_freq) for _ in range(3)]
    # validate(test_loader, model, state, sf_list, global_step)

    # switch to train mode
    model.train()
    # quant_utils.changemodelbit_Waseda_CW(w_Bits, model, sf_list, state)
    quant_utils.changemodelbit_CW(w_Bits, model, sf_list, state)
    for i, input in enumerate(train_loader):
        # measure data loading time

        clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = model(input)
        msssim = ms_ssim(clipped_recon_image, input.cuda(), data_range=1.0, size_average=True)
        distribution_loss = bpp
        # distortion = 1 - msssim #####index
        distortion = mse_loss #####index
        rd_loss = train_lambda * distortion + distribution_loss
        optimizer.zero_grad()
        rd_loss.backward()
        quant_utils.quantback(model, state) ###
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 5)
        optimizer.step()
        sf_list = quant_utils.get_sf_CW(model, state) ###
        # quant_utils.changemodelbit_Waseda_CW(w_Bits, model, sf_list, state) ###
        quant_utils.changemodelbit_CW(w_Bits, model, sf_list, state)

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
            print('epoch {epoch} | step {i} | lr {lr:.8f} | losses {lossesv:.6f} ({lossesa:.6f}) | psnrs {psnrsv:.6f} ({psnrsa:.6f}) | bpps {bppsv:.6f} ({bppsa:.6f})'
                  .format(epoch = epoch, i = global_step, lr = lr,
                          lossesv = losses.val, lossesa = losses.avg,
                          psnrsv = psnrs.val, psnrsa = psnrs.avg,
                          bppsv = bpps.val, bppsa = bpps.avg))

        if global_step % save_freq == 0:
            validate(test_loader, model, state, sf_list, global_step)
            model.train()
        global_step += 1


def validate(test_loader, model, state, sf_list, global_step):
    global test_freq, train_lambda, preLoss, act_target_layer
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
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input, data_range=1.0, size_average=True)
            distribution_loss = bpp
            # distortion = 1 - msssim #####index
            distortion = mse_loss #####index
            Loss = train_lambda * distortion + distribution_loss
            sumBpp += bpp
            sumPsnr += psnr
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
            save_intmodel(model, save_path, global_step, state, sumLoss)
            # save_checkpoint(model, state, sf_list, preLoss, global_step, save_path)
            f = open(save_path + "log.txt", "a")
            f.write("\nglobal_step:{:}, Loss:{:.6f}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(global_step, sumLoss, sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
            f.close()

    return sumLoss

def w_scale(model):
    global act_target_layer, w_Bits
    model_dict = copy.deepcopy(model.state_dict())
    target_layer = act_target_layer
    scale = {}

    #####eq
    for k in model_dict:
        if "deconv" in k and "weight" in k:
            ord = k.replace("module.", "").split('.')[1][-1]
            sn_name = k.rsplit(".", 2)[0] + ".sn" + ord
            s_name = k.rsplit(".", 2)[0] + ".s" + ord
            nxtw_name= k.replace(ord, str(int(ord)+1))
            if sn_name in model_dict and model_dict[sn_name] > 0:
                avrg_s = model_dict[s_name] / model_dict[sn_name]
                model_dict[k] *= avrg_s
                model_dict[k.replace("weight", "bias")] *= avrg_s.reshape(-1)
                model_dict[nxtw_name] = (model_dict[nxtw_name].permute(1, 0, 2, 3) / avrg_s).permute(1, 0, 2, 3)

    for l in target_layer:
        l_scale = []
        for k in model_dict:
            if l == k.replace("module.", "").split('.')[0] and "weight" in k:
                v = model_dict[k]
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
        for k in model_dict:
            if l == k.replace("module.", "").split('.')[0] and "bias" in k:
                v = model_dict[k]
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
    global act_target_layer, act_Bits, out_channel_N, out_channel_M
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

weight_decay = 0.1
epochs = 3
print_freq = 5
train_freq = 10
test_freq = 20
save_freq = 1
out_channel_N = 128
out_channel_M = 192
train_data_dir = "/data0/fym/patch"
val_data_dir = "/data0/fym/val"
image_size = 256
batch_size = 8
act_target_layer = ["priorDecoder", "priorEncoder", "Decoder", "Encoder"]
muls_m = {"priorDecoder":[18, 21, 21], "priorEncoder":[16, 23, 23], "Decoder":[19, 19, 19, 23], "Encoder":[17, 23, 23, 19]}
dequantize = True
use_cuda = True
w_Bits = 10 #####
act_Bits = 8 #####
train_lambda = 2048 #####
lr = 0.00001 #####
save_path = "checkpoints/qft_2048_eq/" #####
if not os.path.isdir(save_path):
    os.mkdir(save_path)
load_path = "/data1/fym/compression/checkpoints/l_2048/" #####
# load_path = "checkpoints/qft_1024_w12/" #####
model_name = "iter_51000.pth.tar" #####
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
target_layer_list = {}
for l in act_target_layer:
    target_layer_list[l] = []
target_layer_list = targetlayer_func(model, act_target_layer, target_layer_list)
# print(target_layer_list)

for l in act_target_layer:
    act_t_replace(model, target_layer_list[l])
    
premodel_path = load_path + model_name
####################################################################
global_step = load_model_qft(model, premodel_path)
state_dict = copy.deepcopy(model.state_dict())
preLoss = 2.4
#####################################################################
# global_step, _, state_dict, preLoss = load_checkpoint_int32_actft(model, premodel_path)
#####################################################################
f = open(save_path + "log.txt", "a")
f.write("\n\nbegin to train with global_step:{:}, preLoss:{:.6f} and lr:{:}".format(global_step, preLoss, lr))
f.close()

sf_list = quant_utils.get_sf_CW(model, state_dict)
for k, v in state_dict.items():
    if if_cuda:
        state_dict[k] = Variable(v.float().cuda(), requires_grad=True) #cuda()
    else:
        state_dict[k] = Variable(v.float().cpu(), requires_grad=True)

# model,sf_list,n_dict=quant_utils.quant_model_bit(model,Bits)

if if_cuda:
    model = torch.nn.DataParallel(model, list(range(1))).cuda() ###cuda

optimizer = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, model.parameters())},
                             {'params': (state_dict[k] for k in state_dict.keys())}], lr,
                            weight_decay=weight_decay)
best_psnr = 0

# validate(test_loader, model, state_dict, sf_list, global_step) #####
for epoch in range(epochs):
    # lr = consine_learning_rate(optimizer, epoch, lr, epochs)

    # train for one epoch
    train(train_loader, model, sf_list, state_dict, test_loader)
