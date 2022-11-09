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

def CEval():
    model.eval()
    total = 0
    correct = 0
    # model.clear_noise()
    with torch.no_grad():
        # for images, labels in tqdm(testloader, leave=False):
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

def CEval_Dist(num_classes=10):
    model.eval()
    total = 0
    correct = 0
    res_dist = torch.LongTensor([0 for _ in range(num_classes)])
    crr_dist = torch.LongTensor([0 for _ in range(num_classes)])
    # model.clear_noise()
    with torch.no_grad():
        # for images, labels in tqdm(testloader, leave=False):
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            predict_list = predictions.tolist()
            for i in range(len(res_dist)):
                res_dist[i] += predict_list.count(i)
            for i in range(len(predict_list)):
                if correction[i] == True:
                    crr_dist[predict_list[i]] += 1
            correct += correction.sum()
            total += len(correction)
    print(f"Correction dict: {crr_dist.tolist()}")
    return (correct/total).cpu().numpy(), res_dist

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

def attack_wcw(model, val_data, verbose=False):
    def my_target(x,y):
        return (y+1)%10
    max_list = []
    avg_list = []
    acc_list = []
    for _ in range(1):
        model.clear_noise()
        model.set_noise(1e-5, 0)
        attacker = WCW(model, c=args.attack_c, kappa=0, steps=args.attack_runs, lr=args.attack_lr, distance=args.attack_distance_metric)
        # attacker.set_mode_targeted_random(n_classses=10)
        # attacker.set_mode_targeted_by_function(my_target)
        attacker.set_mode_default()
        attacker(val_data)
        max_list.append(attacker.noise_max().item())
        avg_list.append(attacker.noise_l2().item())
        attack_accuracy = CEval()
        acc_list.append(attack_accuracy)
    
    mean_attack = np.mean(acc_list)
    if verbose:
        print(f"L2 norm: {np.mean(avg_list):.4f}, max: {np.mean(max_list):.4f}, acc: {mean_attack:.4f}")
    w = attacker.get_noise()
    return mean_attack, w

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
    parser.add_argument('--header', action='store',type=int, default=1,
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

    if args.model == "CIFAR" or args.model == "Res18" or args.model == "QCIFAR" or args.model == "QRes18" or args.model == "QDENSE":
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
    elif args.model == "TIN" or args.model == "QTIN" or args.model == "QVGG":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
                [transforms.ToTensor(),
                 normalize,
                ])
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),
                transforms.ToTensor(),
                normalize,
                ])
        trainset = torchvision.datasets.ImageFolder(root='~/Private/data/tiny-imagenet-200/train', transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=8)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div, shuffle=False, num_workers=8)
        testset = torchvision.datasets.ImageFolder(root='~/Private/data/tiny-imagenet-200/val',  transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=8)
    elif args.model == "QVGGIN" or args.model == "QResIN":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pre_process = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        pre_process += [
            transforms.ToTensor(),
            normalize
        ]

        trainset = torchvision.datasets.ImageFolder('/data/data/share/imagenet/train',
                                transform=transforms.Compose(pre_process))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder('/data/data/share/imagenet/val',
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
                                ]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                    shuffle=False, num_workers=4)
    else:
        trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                                download=False, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                                shuffle=True, num_workers=2)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div,
                                                shuffle=False, num_workers=2)

        testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                            download=False, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                    shuffle=False, num_workers=2)


    if args.model == "MLP3":
        model = SMLP3()
    elif args.model == "MLP3_2":
        model = SMLP3()
    elif args.model == "MLP4":
        model = SMLP4()
    elif args.model == "LeNet":
        model = SLeNet()
    elif args.model == "CIFAR":
        model = CIFAR()
    elif args.model == "Res18":
        model = resnet.resnet18(num_classes = 10)
    elif args.model == "TIN":
        model = resnet.resnet18(num_classes = 200)
    elif args.model == "QLeNet":
        model = QSLeNet()
        FEATURESIZE = 16 * 7 * 7
    elif args.model == "QCIFAR":
        model = QCIFAR()
        FEATURESIZE = 256 * 4 * 4
    elif args.model == "QRes18":
        model = qresnet.resnet18(num_classes = 10)
    elif args.model == "QDENSE":
        model = qdensnet.densenet121(num_classes = 10)
    elif args.model == "QTIN":
        model = qresnet.resnet18(num_classes = 200)
    elif args.model == "QVGG":
        model = qvgg.vgg16(num_classes = 1000)
    elif args.model == "Adv":
        model = SAdvNet()
    elif args.model == "QVGGIN":
        model = qvgg.vgg16(num_classes = 1000)
    elif args.model == "QResIN":
        model = qresnetIN.resnet18(num_classes = 1000)
    else:
        NotImplementedError
    
    criteria = SCrossEntropyLoss()
    criteriaF = torch.nn.CrossEntropyLoss()

    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20])

    if "TIN" in args.model or "Res" in args.model or "VGG" in args.model or "DENSE" in args.model:
    # if "TIN" in args.model or "Res" in args.model:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1000])
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60])
    
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
    model.to(device)
    for m in model.modules():
        if isinstance(m, modules.FixedDropout) or isinstance(m, modules.NFixedDropout) or isinstance(m, modules.SFixedDropout):
            m.device = device
    model.normalize()
    model.clear_mask()
    model.clear_noise()
    model.push_S_device()
    model.de_select_drop()
    # act_grad = GetSecond()
    model.to_first_only()
    ori_acc = CEval()
    # print(f"No mask no noise: {CEval():.4f}")
    # model.clear_noise()
    # performance = NEachEval(args.attack_dist / 2, 0.0)
    # print(f"No mask noise acc: {performance:.4f}")
    # model.clear_noise()

    # model.to_first_only()
    # steps = 200
    # step_size = args.attack_dist / steps
    # attacker = PGD(model, step_size=step_size, steps=steps)
    # attacker.set_f(args.attack_function)
    # attacker(testloader, args.use_tqdm)
    # this_accuracy = CEval()
    # this_max = attacker.noise_max().item()
    # this_l2 = attacker.noise_l2().item()
    # print(f"PGD Results --> acc: {this_accuracy:.4f}, l2: {this_l2:.4f}， max: {this_max:.4f}")
    # model.clear_noise()

    # act_grad, act_grad_each_layer = GetFirst(FEATURESIZE)
    # torch.save([act_grad, act_grad_each_layer], f"first_gradient_{header}_layers.pt")
    # exit()
    # act_grad = torch.load(f"first_gradient_{header}_no_square.pt", map_location=device)
    # act_grad = torch.load(f"first_gradient_{header}_no_abs.pt", map_location=device)
    # act_grad, act_grad_each_layer = torch.load(os.path.join(args.first_path, args.model, f"first_gradient_{header}_layers.pt"), map_location=device)
    act_grad, act_grad_each_layer = torch.load(os.path.join(args.first_path, args.model, f"old_model/first_gradient_{header}_layers.pt"), map_location=device)
    w, h = len(act_grad_each_layer), len(act_grad_each_layer[0])
    act_layers = torch.zeros(h,w).to(device)
    for i in range(w):
        for j in range(h):
            act_layers[j,i] = act_grad_each_layer[i][j].data
    # act_grad = act_layers[-1:,:].abs().sum(0)
    act_grad = act_layers[:,:].abs().sum(0)

    act_mag = model.fc1.op.weight.data.pow(2).abs().sum(dim=0)
    # act_grad = act_grad / (act_mag + 1e-8)
    # act_grad = act_grad - (act_mag + 1e-8) * 1e7
    # act_grad = act_mag

    # print(act_grad.shape)
    # print("Layer last")
    print("Layer whole")
    
    # indices = act_grad.argsort()[FEATURESIZE - int(FEATURESIZE * args.drop):]
    th = 0
    act_grad = act_grad / act_grad.abs().max()
    act_mag = act_mag / act_mag.abs().max()
    i_sum = act_grad * (act_mag > th).to(torch.float32)
    ws_new = act_mag + ((act_mag <= th).to(torch.float32) * 20)
    # d_sum = i_sum - (ws_new - 0.5).abs()
    # d_sum = i_sum - (ws_new - 0.6).abs() * 2
    # num = int(FEATURESIZE * args.drop)
    # if num == 0:
    #     indices = []
    # else:
    #     indices = d_sum.argsort()[-num:]
    # # indices = ((act_grad > 0.3e8).to(torch.float32) * (act_mag < 13.3).to(torch.float32)).to(torch.bool)
    # # print(indices.sum())
    # # indices = act_grad.argsort()[:int(784 * args.drop)]
    # new_mask = torch.ones_like(act_grad)
    # new_mask[indices] = 0
    # # print(new_mask.sum())
    # print("AS")
    # # new_mask = ((model.fc1.weightS.grad.data * (1 + model.fc1.op.weight.data ** 2)).sum(axis=1).argsort() >= 12)
    # # print("S + W2")
    # # new_mask = ((model.fc1.weightS.grad.data).sum(axis=1).argsort() >= 12)
    # # print("S")
    # # new_mask = ((model.fc1.weightS.grad.data * (model.fc1.op.weight.data ** 2)).sum(axis=1).argsort() >= 12)
    # # print("S * W2")
    # # new_mask = ((model.fc1.op.weight.data ** 2).sum(axis=1).argsort() >= 12)
    # # print("W2")
    # # new_mask = loaded_mask
    # model.drop_feature.mask.data = new_mask
    # model.drop_feature.scale = len(new_mask) / new_mask.sum().item()

    if args.acc_th != 0:
        mid = args.drop
        high = 1.0
        low = 0.0
        for i in range(20):
            d_sum = i_sum - (ws_new - 0.5).abs() * 0
            num = int(np.round(FEATURESIZE * mid))
            if num == 0:
                indices = []
            else:
                indices = d_sum.argsort()[-num:]
            new_mask = torch.ones_like(act_grad)
            new_mask[indices] = 0
            # print("AS")
            model.drop_feature.mask.data = new_mask
            model.drop_feature.scale = len(new_mask) / new_mask.sum().item()
            crr_acc = CEval()
            diff = ori_acc - crr_acc
            if np.abs(diff - args.acc_th) > 0.003:
                # print(f"{mid:.4f}, {diff:.4f} --> With mask no noise: {crr_acc:.4f}")
                if diff > args.acc_th:
                    high = mid
                else:
                    low = mid
                mid = (low + high) / 2
            else:
                print(f"Final Dropout: {mid:.4f}")
                print(f"With mask no noise: {crr_acc:.4f}")
                break
    else:
        # d_sum = i_sum - (ws_new - 0.5).abs() * 0
        alpha = 0
        power = 6
        print(f"alpha: {alpha}, power: {power}")
        d_sum = i_sum - alpha * (ws_new) ** power
        num = int(np.round(FEATURESIZE * args.drop))
        if num == 0:
            indices = []
        else:
            indices = d_sum.argsort()[-num:]
        # left_th = 0.3558 # DR_0.06 -- H1
        # left_th = 0.22   # Dr_0.10 -- H1
        # left_th = 0.3511 # Dr_0.06 -- H2
        # left_th = 0.2778 # Dr_0.10 -- H2
        # left_th = 0.2507 # Dr_0.12 -- H2
        # left_th = 0.2156 # Dr_0.15 -- H2
        # left_th = 0.3057 # Dr_0.06 -- H3
        # left_th = 0.2764 # Dr_0.08 -- H3
        # left_th = 0.1748 # Dr_0.10 -- H3
        # left_th = 0.1417 # Dr_0.12 -- H3
        # left_th = 0.103 # Dr_0.15 -- H3
        # up_th = 0.80
        # print(f"left: {left_th}, up: {up_th}")
        # indices = ((i_sum > left_th).to(torch.float32) * (ws_new < up_th).to(torch.float32)).to(torch.bool)
        # indices = ((i_sum > 100).to(torch.float32) * (ws_new < up_th).to(torch.float32)).to(torch.bool)
        new_mask = torch.ones_like(act_grad)
        new_mask[indices] = 0
        # print("AS")
        model.drop_feature.mask.data = new_mask
        model.drop_feature.scale = len(new_mask) / new_mask.sum().item()
        crr_acc = CEval()
        print(f"With mask no noise: {crr_acc:.4f}")
    # performance = NEachEval(args.attack_dist / 2, 0.0)
    # performance = NEachEval(0.1, 0.0)
    performance = 0
    print(f"With mask noise acc: {performance:.4f}")
    model.clear_noise()
    
    model.to_first_only()
    steps = 200
    step_size = args.attack_dist / steps
    attacker = PGD(model, args.attack_dist, step_size=step_size, steps=steps * 10)
    attacker.set_f(args.attack_function)
    attacker(testloader, args.use_tqdm)
    this_accuracy = CEval()
    this_max = attacker.noise_max().item()
    this_l2 = attacker.noise_l2().item()
    print(f"PGD Results --> acc: {this_accuracy:.4f}, l2: {this_l2:.4f}， max: {this_max:.4f}")
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
