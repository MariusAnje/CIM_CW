# %%
from cProfile import label
from sklearn.utils import shuffle
import torch
import torchvision
from torch import optim
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
from cw_attack import Attack, WCW, binary_search_c, binary_search_dist
import torch
import torch.nn as nn
import torch.optim as optim
from utils import str2bool, attack_wcw, get_dataset, get_model, prepare_model
from utils import TPMTrain, MTrain, TCEval, TMEachEval, PGD_Eval, CEval, MEachEval, UpdateBN, MEval
from utils import copy_model

def LPLoss(model, lmd):
    loss = 0
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            w = m.weight
            shape1, shape2 = w.size(-2), w.size(-1)
            w = w.view(-1, shape1, shape2)
            loss += (w.swapaxes(-1,-2).bmm(w) - lmd * torch.eye(shape2).to(w.device)).pow(2).sum()
    return loss

def LTrain(model_group, epochs, header, lmd, beta, noise_type, dev_var, rate_max, rate_zero, write_var, verbose=False, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    for i in range(epochs):
        model.train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            model.clear_noise()
            # model.set_noise(dev_var, write_var)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            running_loss += loss.item()
            loss += beta * LPLoss(model, lmd=lmd)
            loss.backward()
            optimizer.step()
            
        test_acc = MEachEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
        # test_acc = CEval()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, loss: {running_loss / len(trainloader):.4f}")
        scheduler.step()

def str2bool(a):
    if a == "True":
        return True
    elif a == "False":
        return False
    else:
        raise NotImplementedError(f"{a}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epoch', action='store', type=int, default=20,
            help='# of epochs of training')
    parser.add_argument('--noise_epoch', action='store', type=int, default=100,
            help='# of epochs of noise validations')
    parser.add_argument('--train_var', action='store', type=float, default=0.1,
            help='device variation [std] when training')
    parser.add_argument('--dev_var', action='store', type=float, default=0.3,
            help='device variation [std] before write and verify')
    parser.add_argument('--write_var', action='store', type=float, default=0.0,
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
    parser.add_argument('--header', action='store',type=int, default=1,
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
    args = parser.parse_args()

    print(args)
    header = time.time()
    header_timer = header
    parent_path = "./"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    BS = 128
    NW = 2

    trainloader, secondloader, testloader = get_dataset(args, BS, NW)
    model = get_model(args)
    model, optimizer, w_optimizer, scheduler = prepare_model(model, device, args)

    criteria = SCrossEntropyLoss()
    criteriaF = torch.nn.CrossEntropyLoss()
    model_group = model, criteriaF, optimizer, scheduler, device, trainloader, testloader

    kwargs = {"N":8, "m":1}
    model_group = model, criteriaF, optimizer, scheduler, device, trainloader, testloader
    epochs = 30
    lmd = 1.0
    beta = 5e-4
    noise_type = "Gaussian"
    LTrain(model_group, epochs, header, lmd, beta, noise_type, dev_var=args.train_var, rate_max=1, rate_zero=1, write_var=args.write_var, verbose=args.verbose, **kwargs)
    # ATrain(args.train_epoch, header, dev_var=args.train_var, verbose=args.verbose)
    model.clear_noise()
    state_dict = torch.load(f"tmp_best_{header}.pt")
    model.load_state_dict(state_dict)
    model.from_first_back_second()
    torch.save(model.state_dict(), f"saved_B_{header}_noise_{args.train_var}.pt")
    model.clear_noise()
    model.to_first_only()
    print(f"No mask no noise: {CEval():.4f}")
    model.from_first_back_second()
    state_dict = torch.load(f"saved_B_{header}_noise_{args.train_var}.pt")
    model.load_state_dict(state_dict)
    model.clear_mask()
    model.to_first_only()
    performance = MEachEval(model_group, noise_type, dev_var=args.train_var, rate_max=1, rate_zero=1, write_var=args.write_var, **kwargs)
    print(f"No mask noise acc: {performance:.4f}")
    # mean_attack, w = attack_wcw(model, testloader, verbose=True)



