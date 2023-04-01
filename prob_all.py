import torch
import torchvision
from torchvision import transforms
import numpy as np

import time
import argparse
from cw_attack import Attack, WCW, binary_search_c, binary_search_dist, PGD
import torch
import torch.nn as nn
import torch.optim as optim
from utils import str2bool, attack_wcw, get_dataset, get_model, prepare_model
from utils import TPMTrain, MTrain, TCEval, TMEachEval, PGD_Eval, CEval, MEachEval, UpdateBN
from utils import copy_model

from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from scipy.special import softmax
from qmodules import QSConv2d, QNConv2d, QSLinear, QNLinear
from modules import SModel, SReLU, NModel
from scipy import stats
import os

def perform_attack(model_group, attack_dist, use_tqdm=False):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.clear_noise()
    crr_acc = CEval(model_group)
    model.normalize()
    attack_runs = 100
    steps = attack_runs
    step_size = attack_dist / steps
    attacker = PGD(model, attack_dist, step_size=step_size, steps=steps * 2)
    attacker.set_f("act")
    attacker(testloader, use_tqdm)
    # attacker.save_noise(f"lol_{header}_{args.attack_dist:.4f}.pt")
    this_accuracy = CEval(model_group)
    this_max = attacker.noise_max().item()
    this_l2 = attacker.noise_l2().item()
    end_time = time.time()
    print(f"acc: {crr_acc:.4f} vs {this_accuracy:.5f}, max: {this_max:.4f}")
    for m in model.modules():
        if isinstance(m, QSLinear) or isinstance(m, QNLinear):
            print(m.noise.max().item())
    # model.clear_noise()
    model.de_normalize()

def manage_data(m, index, target):
    m.noise[index].data *= 0 
    m.noise[index].data += target

def Eval_Gap_Linear(model_group, mag, n_steps=100, select_name = "", start=None, end=None, use_tqdm=True):
    def calculate_pbar_size(model, select_name):
        bar_size = 0
        for n, m in model.named_modules():
            if (isinstance(m, QSLinear) or isinstance(m, QNLinear)) and select_name in n:
                bar_size += m.noise.shape.numel()
        if start is not None and end is not None:
            bar_size = end - start
        return bar_size
    
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.normalize()
    ori_acc = CEval(model_group)
    result_list = []

    with tqdm(total = calculate_pbar_size(model, select_name)) as pbar:
        crr_index = 0
        for n, m in model.named_modules():
            if (isinstance(m, QSLinear) or isinstance(m, QNLinear)) and select_name in n:
                m_shape = m.noise.shape
                for i in range(m_shape[0]):
                    for j in range(m_shape[1]):
                        if start is not None:
                            if crr_index < start:
                                crr_index += 1
                                continue
                        ori_data = m.noise[i,j].item()
                        this_list = []
                        step = mag / n_steps
                        for diff in list(np.arange(-mag,mag,step)) + [0]:
                            target = diff + ori_data
                            manage_data(m, (i,j), target)
                            acc = CEval(model_group)
                            if acc <= ori_acc:
                                this_list.append(target)
                                break
                        for diff in list(np.arange(mag,-mag,-step)) + [0]:
                            target = diff + ori_data
                            manage_data(m, (i,j), target)
                            acc = CEval(model_group)
                            if acc <= ori_acc:
                                this_list.append(target)
                                break

                        this_min = -np.inf if np.min(this_list) - ori_data <= -mag else np.min(this_list)
                        this_max = np.inf if np.max(this_list) - ori_data >= mag else np.max(this_list)
                        result_list.append([ori_data, this_min, this_max])
                        manage_data(m, (i,j), ori_data)
                        if use_tqdm:
                            pbar.update(1)
                            pbar.set_description(f"{this_min:.4f}, {this_max:.4f}")
                        crr_index += 1
                        if end is not None:
                            if crr_index == end:
                                model.de_normalize()
                                return result_list
    model.de_normalize()
    return result_list

def Eval_Gap_Conv(model_group, mag, n_steps=100, select_name = "", start=None, end=None, use_tqdm=True):
    def calculate_pbar_size(model, select_name):
        bar_size = 0
        for n, m in model.named_modules():
            if (isinstance(m, QSConv2d) or isinstance(m, QNConv2d)) and select_name in n:
                bar_size += m.noise.shape.numel()
        if start is not None and end is not None:
            bar_size = end - start
        return bar_size
    
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.normalize()
    ori_acc = CEval(model_group)
    result_list = []

    with tqdm(total = calculate_pbar_size(model, select_name)) as pbar:
        crr_index = 0
        for n, m in model.named_modules():
            if (isinstance(m, QSConv2d) or isinstance(m, QNConv2d)) and select_name in n:
                m_shape = m.noise.shape
                for i in range(m_shape[0]):
                    for j in range(m_shape[1]):
                        for k in range(m_shape[2]):
                            for l in range(m_shape[3]):
                                if start is not None:
                                    if crr_index < start:
                                        crr_index += 1
                                        continue
                                ori_data = m.noise[i,j,k,l].item()
                                this_list = []
                                step = mag / n_steps
                                for diff in list(np.arange(-mag,mag,step)) + [0]:
                                    target = diff + ori_data
                                    manage_data(m, (i,j,k,l), target)
                                    acc = CEval(model_group)
                                    if acc <= ori_acc:
                                        this_list.append(target)
                                        break
                                for diff in list(np.arange(mag,-mag,-step)) + [0]:
                                    target = diff + ori_data
                                    manage_data(m, (i,j,k,l), target)
                                    acc = CEval(model_group)
                                    if acc <= ori_acc:
                                        this_list.append(target)
                                        break

                                this_min = -np.inf if np.min(this_list) - ori_data <= -mag else np.min(this_list)
                                this_max = np.inf if np.max(this_list) - ori_data >= mag else np.max(this_list)
                                result_list.append([ori_data, this_min, this_max])
                                manage_data(m, (i,j,k,l), ori_data)
                                if use_tqdm:
                                    pbar.update(1)
                                    pbar.set_description(f"{this_min:.4f}, {this_max:.4f}")
                                crr_index += 1
                                if end is not None:
                                    if crr_index == end:
                                        model.de_normalize()
                                        return result_list
    model.de_normalize()
    return result_list

def save_noise(model, filename):
    noise_list = []
    for m in model.modules():
        if isinstance(m, QSConv2d) or isinstance(m, QNConv2d) or isinstance(m, QSLinear) or isinstance(m, QNLinear):
            noise_list.append(m.noise.data)
    torch.save(noise_list, filename)

def load_noise(model, filename, device):
    noise_list = torch.load(filename)
    i = 0
    for m in model.modules():
        if isinstance(m, QSConv2d) or isinstance(m, QNConv2d) or isinstance(m, QSLinear) or isinstance(m, QNLinear):
            m.noise.data = noise_list[i].data.to(device)
            i += 1
    

def calculate_prob(result_list, sigma):
    dist = stats.norm(loc=0, scale = sigma)
    prob = []
    for center, start, end in result_list:
        s = dist.cdf(start)
        e = dist.cdf(end)
        prob.append(e - s)
#     print(prob)
    print(np.prod(prob))

parser = argparse.ArgumentParser()
parser.add_argument('--train_epoch', action='store', type=int, default=20,
        help='# of epochs of training')
parser.add_argument('--noise_epoch', action='store', type=int, default=100,
        help='# of epochs of noise validations')
parser.add_argument('--train_var', action='store', type=float, default=0.1,
        help='device variation [std] when training')
parser.add_argument('--dev_var', action='store', type=float, default=0.3,
        help='device variation [std] before write and verify')
parser.add_argument('--write_var', action='store', type=float, default=0.03,
        help='device variation [std] after write and verify')
parser.add_argument('--rate_zero', action='store', type=float, default=0.03,
        help='pepper rate, rate of noise being zero')
parser.add_argument('--rate_max', action='store', type=float, default=0.03,
        help='salt rate, rate of noise being one')
parser.add_argument('--noise_type', action='store', default="Gaussian",
        help='type of noise used')
parser.add_argument('--mask_p', action='store', type=float, default=0.01,
        help='portion of the mask')
parser.add_argument('--device', action='store', default="cuda:0",
        help='device used')
parser.add_argument('--verbose', action='store', type=str2bool, default=False,
        help='see training process')
parser.add_argument('--model', action='store', default="QLeNet", choices=["MLP3", "MLP3_2", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG", "Adv", "QVGGIN", "QResIN"],
        help='model to use')
parser.add_argument('--alpha', action='store', type=float, default=1e6,
        help='weight used in saliency - substract')
parser.add_argument('--header', action='store', default=None,
        help='use which saved state dict')
parser.add_argument('--pretrained', action='store',type=str2bool, default=True,
        help='if to use pretrained model')
parser.add_argument('--use_mask', action='store',type=str2bool, default=True,
        help='if to do the masking experiment')
parser.add_argument('--model_path', action='store', default="./pretrained",
        help='where you put the pretrained model')
parser.add_argument('--save_file', action='store',type=str2bool, default=True,
        help='if to save the files')
parser.add_argument('--calc_S', action='store',type=str2bool, default=True,
        help='if calculated S grad if not necessary')
parser.add_argument('--div', action='store', type=int, default=1,
        help='division points for second')
parser.add_argument('--layerwise', action='store',type=str2bool, default=False,
        help='if do it layer by layer')
parser.add_argument('--attack_c', action='store',type=float, default=1e-4,
        help='c value for attack')
parser.add_argument('--attack_runs', action='store',type=int, default=10,
        help='# of runs for attack')
parser.add_argument('--attack_lr', action='store',type=float, default=1e-4,
        help='learning rate for attack')
parser.add_argument('--attack_method', action='store', default="l2", choices=["max", "l2", "linf", "loss"],
        help='method used for attack')
parser.add_argument('--load_atk', action='store',type=str2bool, default=True,
        help='if we should load the attack')
parser.add_argument('--load_direction', action='store',type=str2bool, default=False,
        help='if we should load the noise directions')
parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
        help='whether to use tqdm')
parser.add_argument('--start_index', action='store',type=int, default=None,
        help='weight to start')
parser.add_argument('--end_index', action='store',type=int, default=None,
        help='weight to end')
args = parser.parse_args()

print(args)

BS = 16384
NW = 4
header = args.header
parent_path = args.model_path

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

BS = 4096
NW = 0

trainloader, secondloader, testloader = get_dataset(args, BS, NW)
model = get_model(args)
model, optimizer, w_optimizer, scheduler = prepare_model(model, device, args)
model.from_first_back_second()
state_dict = torch.load(os.path.join(parent_path, f"saved_B_{header}.pt"), map_location=device)
model.load_state_dict(state_dict, strict=True)
model, optimizer, w_optimizer, scheduler = prepare_model(model, device, args)

criteriaF = torch.nn.CrossEntropyLoss()
model_group = model, criteriaF, optimizer, scheduler, device, trainloader, testloader

model.normalize()
crr_acc = CEval(model_group)
model.clear_noise()
print(f"model accuracy: {crr_acc:.4f}")

distance = 0.06
perform_attack(model_group, distance, args.use_tqdm)
save_noise(model, f"noise_attack_{distance:.4f}_{args.model}_{header}.pt")
# exit()
# load_noise(model, f"noise_attack_{distance:.4f}_{args.model}_{header}.pt", device)

large_test_loader = []
for data, label in testloader:
    data, label = data.to(device), label.to(device)
    large_test_loader.append((data, label))

model_group = model, criteriaF, optimizer, scheduler, device, trainloader, large_test_loader

result_list_conv = Eval_Gap_Conv(model_group, mag=distance*2, n_steps=100, start=args.start_index, end = args.end_index, use_tqdm=args.use_tqdm)
result_list_linear = Eval_Gap_Linear(model_group, mag=distance*2, n_steps=100, start=args.start_index, end = args.end_index, use_tqdm=args.use_tqdm)

result_list = result_list_linear + result_list_conv

torch.save(result_list, f"very_large_prob_res_list_{args.model}_{args.start_index}_{args.end_index}_{header}.pt")

calculate_prob(result_list, sigma = 0.03)





