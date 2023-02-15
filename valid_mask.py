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
from utils import TPMTrain, MTrain, TCEval, TMEachEval, PGD_Eval, CEval, MEachEval
from utils import copy_model

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
    parser.add_argument('--mask_p', action='store', type=float, default=0.01,
            help='portion of the mask')
    parser.add_argument('--device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--verbose', action='store', type=str2bool, default=False,
            help='see training process')
    parser.add_argument('--model', action='store', default="MLP4", choices=["MLP3", "MLP3_2", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG", "Adv", "QVGGIN", "QResIN"],
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

    BS = 128
    NW = 0

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
    
    model, optimizer, w_optimizer, scheduler = prepare_model(model, device, args)
    model_group = model, criteriaF, optimizer, scheduler, device, trainloader, testloader

    crr_acc = CEval(model_group)
    print(f"With mask no noise: {crr_acc:.4f}")
    performance = 0
    print(f"With mask noise acc: {performance:.4f}")
    model.clear_noise()
    
    steps = args.attack_runs
    step_size = args.attack_dist / steps
    attacker = PGD(model, args.attack_dist, step_size=step_size, steps=steps * 10)
    attacker.set_f(args.attack_function)
    attacker(testloader, args.use_tqdm)
    # attacker.save_noise(f"lol_{header}_{args.attack_dist:.4f}.pt")
    this_accuracy = CEval(model_group)
    this_max = attacker.noise_max().item()
    this_l2 = attacker.noise_l2().item()
    print(f"PGD Results --> acc: {this_accuracy:.4f}, l2: {this_l2:.4f}, max: {this_max:.4f}")
    model.clear_noise()
    exit()

    acc, dist_max, dist_l2, c = binary_search_dist(search_runs = 100, acc_evaluator=CEval, dataloader=testloader, target_metric=args.attack_dist, attacker_class=WCW, model=model, init_c=args.attack_c, steps=args.attack_runs, lr=args.attack_lr, distance=args.attack_distance_metric, verbose=args.verbose, use_tqdm = args.use_tqdm, function = args.attack_function)
    print(f"C&W Results --> C: {c:.4e}, acc: {acc:.4f}, l2: {dist_l2:.4f},  max: {dist_max:.4f}")
    
    # target max: 0.03, header = 2
    # Model: QLeNet acc:0.5402, c = 1.9844e-09, lr = 1e-5
    # Model: QCIFAR acc:0.0245, c = 1e-5, lr = 1e-4
    # Model: QRes18 acc:0.0000, c = 10, dist = 0.0111, lr = 5e-5
    # Model: QTIN   acc:0.0000, c = 1,  dist = 0.0059, lr = 1e-4
    # Model: QVGGIN acc:0.0008, c = 1,  dist = 0.0273, lr = 1e-4
    exit()
