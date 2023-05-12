from cProfile import label
from email.policy import strict
from sklearn.utils import shuffle
import torch
from torch.nn.modules import module
import torchvision
from torch import isin, optim
import torchvision.transforms as transforms
import numpy as np
from models import SCrossEntropyLoss, SMLP3, SMLP4, SLeNet, CIFAR, FakeSCrossEntropyLoss, SAdvNet
import modules
from qmodels import QSLeNet, QCIFAR
import resnet
import qresnet
import qvgg
import qdensnet
import qresnetIN
from modules import NModel, SModule, NModule
from tqdm import tqdm
import time
import argparse
import os
from cw_attack import Attack, WCW, binary_search_c, binary_search_dist, PGD
import torch
import torch.nn as nn
import torch.optim as optim
from utils import str2bool, attack_wcw, get_dataset, get_model, prepare_model
from utils import TPMTrain, MTrain, TCEval, TMEachEval, PGD_Eval, CEval, MEachEval, MEval, MBEval
from utils import copy_model, get_logger

def GetSecond():
    model.eval()
    model.clear_noise()
    optimizer.zero_grad()
    act_grad = torch.zeros(16 * 7 * 7).to(device)
    for images, labels in tqdm(secondloader):
    # for images, labels in secondloader:
        images, labels = images.to(device), labels.to(device)
        # images = images.view(-1, 784)
        outputs, outputsS = model(images)
        loss = criteria(outputs, outputsS,labels)
        model.xx[1].retain_grad()
        loss.backward()
        act_grad += model.xx[1].grad.data.sum(axis=0)
        model.xx[1].grad.data.zero_()
    return act_grad

def GetFirst(size):
    model.eval()
    model.clear_noise()
    optimizer.zero_grad()
    act_grad = torch.zeros(size).to(device)
    act_grad_each_layer = []
    for i in range(len(act_grad)):
    # for i in tqdm(range(len(act_grad))):
        # for images, labels in tqdm(secondloader, leave=False):
        for images, labels in secondloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            # loss = (model.xx[:,i] ** 2).sum()
            loss = (model.xx[:,i]).sum()
            loss.backward()
        layers = []
        for m in model.modules():
            if isinstance(m, nn.Conv2d):# or isinstance(m, nn.Linear):
                act_grad[i] += m.weight.grad.data.abs().sum()
                layers.append(m.weight.grad.data.abs().sum())
                # act_grad[i] += m.weight.grad.data.sum()
        act_grad_each_layer.append(layers)
        optimizer.zero_grad()
    return act_grad, act_grad_each_layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epoch', action='store', type=int, default=20,
            help='# of epochs of training')
    parser.add_argument('--noise_epoch', action='store', type=int, default=100,
            help='# of epochs of noise validations')
    parser.add_argument('--train_var', action='store', type=float, default=0.0,
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
    parser.add_argument('--model', action='store', default="MLP4", #choices=["MLP3", "MLP3_2", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG", "Adv", "QVGGIN", "QResIN"],
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
    parser.add_argument('--first_path', action='store', default="./firsts",
            help='where you put the pre-calculated first derivatives')
    parser.add_argument('--save_file', action='store',type=str2bool, default=False,
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
    parser.add_argument('--attack_function', action='store', default="act", choices=["act", "cross"],
            help='function used for attack')
    parser.add_argument('--attack_distance_metric', action='store', default="l2", choices=["max", "l2", "linf", "loss"],
            help='distance metric used for attack')
    parser.add_argument('--attack_dist', action='store', type=float, default=0.03,
            help='distance used for attack')
    parser.add_argument('--load_atk', action='store',type=str2bool, default=True,
            help='if we should load the attack')
    parser.add_argument('--load_direction', action='store',type=str2bool, default=False,
            help='if we should load the noise directions')
    parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
            help='whether to use tqdm')
    parser.add_argument('--attack_name', action='store', default="C&W",
            help='# of epochs of training')
    parser.add_argument('--drop', action='store',type=float, default=0.0,
            help='random dropout ratio')
    parser.add_argument('--acc_th', action='store',type=float, default=0.0,
            help='tolerable accuracy drop')
    args = parser.parse_args()

    print(args)
    header = time.time()
    header_timer = header
    parent_path = "./"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if "LeNet" in args.model:
        BS = 10240
    elif "CIFAR" in args.model:
        BS = 5000
    elif "Res18" in args.model:
        BS = 5000
    elif "TIN" in args.model:
        BS = 128
    else:
        BS = 128
    NW = 4

    trainloader, secondloader, testloader = get_dataset(args, BS, NW)
    model = get_model(args)
    
    criteria = SCrossEntropyLoss()
    criteriaF = torch.nn.CrossEntropyLoss()

    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20])
    
    parent_path = args.model_path
    args.train_var = 0.0
    header = args.header
    model.from_first_back_second()
    state_dict = torch.load(os.path.join(parent_path, f"saved_B_{header}.pt"), map_location=device)
    if args.model == "Adv":
        model.conv1.op.weight.data, model.conv1.op.bias.data = state_dict["conv1.weight"].data, state_dict["conv1.bias"].data
        model.conv2.op.weight.data, model.conv2.op.bias.data = state_dict["conv2.weight"].data, state_dict["conv2.bias"].data
        model.fc1.op.weight.data, model.fc1.op.bias.data = state_dict["fc1.weight"].data, state_dict["fc1.bias"].data
        model.fc2.op.weight.data, model.fc2.op.bias.data = state_dict["fc2.weight"].data, state_dict["fc2.bias"].data
    else:
        model.load_state_dict(state_dict, strict=False)
    if args.model == "MLP3_2":
        model.fc1 = model.fc1.op
    large_test_loader = []
    for data, labels in testloader:
        data, labels = data, labels
        large_test_loader.append((data, labels))

    model, optimizer, w_optimizer, scheduler = prepare_model(model, device, args)
    model_group = model, criteriaF, optimizer, scheduler, device, trainloader, large_test_loader
    model.normalize()

    crr_acc = CEval(model_group)
    print(f"With mask no noise: {crr_acc:.4f}")

    model.clear_mask()
    model.to_first_only()
    kwargs = {"N":8, "m":1}
    performance_list = []
    if args.use_tqdm:
        iter_loader = tqdm(range(args.noise_epoch))
    else:
        iter_loader = range(args.noise_epoch)
    for _ in iter_loader:
        performance = MEval(model_group, args.noise_type, args.dev_var, args.rate_max, args.rate_zero, args.write_var, **kwargs)
        performance_list.append(performance)
        if args.use_tqdm:
            iter_loader.set_description(f"current: {performance:.4f}, mean: {np.mean(performance_list):.4f}, worst: {np.min(performance_list):.4f}")
    print(f"No mask noise average acc: {np.mean(performance_list):.4f}")
    print(f"No mask noise worst acc: {np.min(performance_list):.4f}")
    if args.save_file:
        torch.save(performance_list, f"Quantile_{args.model}_{header}_{args.noise_type}_{args.dev_var}_{args.rate_max}_{args.rate_zero}_{time.time()}.pt")
    q_list = [0.5, 0.1, 0.01, 0.001, 0.0001]
    for quantile in q_list:
        print(f"{quantile:6f} quantile: {np.quantile(performance_list, quantile):.4f}")
    exit()
