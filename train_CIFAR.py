import torch
import torchvision
from torch import optim
import torchvision.transforms as transforms
import numpy as np
from models import SCrossEntropyLoss, SMLP3, SMLP4, SLeNet, CIFAR, FakeSCrossEntropyLoss
import modules
from qmodels import QSLeNet, QCIFAR
import resnet
import qresnet
import qvgg
import qdensnet
from modules import SModule
from tqdm import tqdm
import time
import argparse
import os
from cw_attack import Attack, WCW
import torch
import torch.nn as nn
import torch.optim as optim

def CEval():
    model.eval()
    total = 0
    correct = 0
    # model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def NEval(dev_var, write_var):
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        model.set_noise(dev_var, write_var)
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def NEachEval(dev_var, write_var):
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            model.clear_noise()
            model.set_noise(dev_var, write_var)
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def NTrain(epochs, header, dev_var=0.0, write_var=0.0, verbose=False):
    best_acc = 0.0
    for i in range(epochs):
        model.train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            model.clear_noise()
            model.set_noise(dev_var, write_var)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # test_acc = NEachEval(dev_var, write_var)
        test_acc = CEval()
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
    parser.add_argument('--model', action='store', default="MLP4", choices=["MLP3", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG"],
            help='model to use')
    parser.add_argument('--method', action='store', default="SM", choices=["second", "magnitude", "saliency", "random", "SM"],
            help='method used to calculate saliency')
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
    args = parser.parse_args()

    print(args)
    header = time.time()
    header_timer = header
    parent_path = "./"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    BS = 128

    trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                            download=False, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                        download=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                shuffle=False, num_workers=2)
                                                
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            normalize])
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
            ])
    trainset = torchvision.datasets.CIFAR10(root='~/Private/data', train=True, download=False, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=4)
    secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div, shuffle=False, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='~/Private/data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=4)
    model = CIFAR()
    # model = QSLeNet()

    model.to(device)
    model.push_S_device()
    model.clear_noise()
    model.clear_mask()
    criteria = SCrossEntropyLoss()
    criteriaF = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60])
    args.train_epoch = 30
    args.dev_var = 0.3
    # args.train_var = 0.3
    args.train_var = 0.0
    args.verbose = True
    
    """
    model.to_first_only()
    NTrain(args.train_epoch, header, args.train_var, 0.0, args.verbose)
    if args.train_var > 0:
        state_dict = torch.load(f"tmp_best_{header}.pt")
        model.load_state_dict(state_dict)
    model.from_first_back_second()
    torch.save(model.state_dict(), f"saved_B_{header}_{args.train_var}.pt")
    model.clear_noise()
    print(f"No mask no noise: {CEval():.4f}")
    state_dict = torch.load(f"saved_B_{header}_{args.train_var}.pt")
    model.load_state_dict(state_dict)
    model.clear_mask()
    performance = NEachEval(args.dev_var, 0.0)
    print(f"No mask noise acc: {performance:.4f}")
    """
    args.train_var = 0.0
    header = 1
    model.from_first_back_second()
    state_dict = torch.load(f"saved_B_{header}_{args.train_var}.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.clear_mask()
    model.clear_noise()
    print(f"No mask no noise: {CEval():.4f}")
    # print(f"No mask w/ noise: {NEachEval(args.dev_var, args.write_var):.4f}")
    model.to_first_only()
    def my_target(x,y):
        return (y+1)%10
    max_list = []
    avg_list = []
    acc_list = []
    for _ in range(1):
        avg_performance = []
        model.clear_noise()
        attacker = WCW(model, c=1e-8, kappa=0, steps=10, lr=0.01, method="l2")
        # attacker.set_mode_targeted_random(n_classses=10)
        attacker.set_mode_targeted_by_function(my_target)
        attacker(testloader)
        w = attacker.get_noise()
        max_list.append(attacker.noise_max().item())
        avg_list.append(np.sqrt(attacker.noise_l2().item()))
        accuracy = CEval().item()
        acc_list.append(accuracy)
    print(f"Max diff: {np.mean(max_list):.3f}")
    print(f"L2  diff: {np.mean(avg_list):.3f}")
    print(f"Avg  acc: {np.mean(acc_list):.4f}")
