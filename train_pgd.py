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
from cw_attack import Attack, PGD, binary_search_c, binary_search_dist
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

def attack_pgd(model, val_data, verbose=False):
    def my_target(x,y):
        return (y+1)%10
    max_list = []
    avg_list = []
    acc_list = []
    for _ in range(1):
        model.clear_noise()
        model.set_noise(1e-5, 0)
        attacker = PGD(model, c=args.attack_c, kappa=0, steps=args.attack_runs, lr=args.attack_lr, method=args.attack_method)
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
    parser.add_argument('--save_file', action='store',type=str2bool, default=True,
            help='if to save the files')
    parser.add_argument('--calc_S', action='store',type=str2bool, default=True,
            help='if calculated S grad if not necessary')
    parser.add_argument('--div', action='store', type=int, default=1,
            help='division points for second')
    parser.add_argument('--layerwise', action='store',type=str2bool, default=False,
            help='if do it layer by layer')
    parser.add_argument('--attack_runs', action='store',type=int, default=10,
            help='# of runs for attack')
    parser.add_argument('--attack_lr', action='store',type=float, default=1e-3,
            help='learning rate for attack')
    parser.add_argument('--load_atk', action='store',type=str2bool, default=True,
            help='if we should load the attack')
    parser.add_argument('--load_direction', action='store',type=str2bool, default=False,
            help='if we should load the noise directions')
    parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
            help='whether to use tqdm')
    parser.add_argument('--attack_name', action='store', default="pgd",
            help='# of epochs of training')
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
    elif args.model == "QCIFAR":
        model = QCIFAR()
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

    model.to(device)
    model.push_S_device()
    model.clear_noise()
    model.clear_mask()
    criteria = SCrossEntropyLoss()
    criteriaF = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60])
    args.train_epoch = 30
    # args.dev_var = 0.3
    # args.train_var = 0.3
    args.train_var = 0.0
    args.verbose = True
    
    # model.to_first_only()
    # print(model.fc1.op.weight.shape)
    # NTrain(args.train_epoch, header, args.train_var, 0.0, args.verbose)
    # if args.train_var > 0:
    #     state_dict = torch.load(f"tmp_best_{header}.pt")
    #     model.load_state_dict(state_dict)
    # model.from_first_back_second()
    # torch.save(model.state_dict(), f"saved_B_{header}_{args.train_var}.pt")
    # model.clear_noise()
    # print(f"No mask no noise: {CEval():.4f}")
    # state_dict = torch.load(f"saved_B_{header}_{args.train_var}.pt")
    # model.load_state_dict(state_dict)
    # model.clear_mask()
    # performance = NEachEval(args.dev_var, 0.0)
    # print(f"No mask noise acc: {performance:.4f}")
    # exit()

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
        model.load_state_dict(state_dict)
    if args.model == "MLP3_2":
        model.fc1 = model.fc1.op
    model.normalize()
    model.clear_mask()
    model.clear_noise()
    model.to_first_only()
    print(f"No mask no noise: {CEval():.4f}")
    try:
        no_mask_acc_list = torch.load(os.path.join(parent_path, f"no_mask_list_{header}_{args.dev_var}.pt"))
        print(f"[{args.dev_var}] No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
    except:
        pass

    # def my_target(x,y):
    #     return (y+1)%10
    
    attacker = PGD(model, step_size=args.attack_lr, steps=args.attack_runs)
    attacker(testloader, args.use_tqdm)
    this_accuracy = CEval()
    this_max = attacker.noise_max().item()
    this_l2 = attacker.noise_l2().item()
    print(f"acc: {this_accuracy:.4f}, l2: {this_l2:.4f}， max: {this_max:.4f}")

    # binary_search_dist(search_runs = 10, acc_evaluator=CEval, dataloader=testloader, target_metric=args.attack_dist, attacker_class=PGD, model=model, init_c=args.attack_c, steps=args.attack_runs, lr=args.attack_lr, method=args.attack_method, verbose=True, use_tqdm = args.use_tqdm)
    # target max: 0.03, header = 2
    # Model: QLeNet acc:0.5402, c = 1.9844e-09, lr = 1e-5
    # Model: QCIFAR acc:0.0245, c = 1e-5, lr = 1e-4
    # Model: QRes18 acc:0.0000, c = 10, dist = 0.0111, lr = 5e-5
    # Model: QTIN   acc:0.0000, c = 1,  dist = 0.0059, lr = 1e-4
    # Model: QVGGIN acc:0.0008, c = 1,  dist = 0.0273, lr = 1e-4
    exit()

    # j = 0
    # for _ in range(10000):
    # # binary_search_c(search_runs = 10, acc_evaluator=CEval, dataloader=testloader, th_accuracy=0.001, attacker_class=PGD, model=model, init_c=1, steps=10, lr=0.01, method="l2", verbose=True)
    #     acc, w = attack_pgd(model, testloader, True)
    #     if acc < 0.01:
    #         print("Success, saving!")
    #         torch.save(w, f"noise_{args.model}_{time.time()}.pt")
    #         j += 1
    #     if j >= 3:
    #         break
    # exit()

    # parent_dir = "./pretrained/many_noise/QLeNet"
    # parent_dir = "./pretrained/many_noise/LeNet_norm"
    # parent_dir = "./pretrained/many_noise/MLP3"
    parent_dir = "./pretrained/many_noise/MLP3_2"
    file_list = os.listdir(parent_dir)
    w = torch.Tensor([]).to(device)
    if args.load_atk:
        noise = torch.load(os.path.join(parent_dir, file_list[1]), map_location=device)
        i = 0
        for name, m in model.named_modules():
            if isinstance(m, NModule) or isinstance(m, SModule):
                # m.noise.data += noise[i].data
                # m.noise = m.noise.to(device)
                m.op.weight.data += noise[i].data
                m.op.weight = m.op.weight.to(device)
                w = torch.cat([w, noise[i].data.view(-1)])
                i += 1

    # th = 0.02
    # mask = noise[0].abs()>th
    # print(mask.sum()/mask.shape.numel())
    # model.fc1.op.weight.data[mask] = state_dict["fc1.op.weight"][mask]
    # model.conv1.op.weight.data = state_dict["conv1.op.weight"]
    # model.conv2.op.weight.data = state_dict["conv2.op.weight"]
    # model.fc1.op.weight.data = state_dict["fc1.op.weight"]
    # model.fc2.op.weight.data = state_dict["fc2.op.weight"]
    # model.fc3.op.weight.data = state_dict["fc3.op.weight"]
    print(f"Attack central acc: {CEval():.4f}")
    # exit()

    noise_size = 0
    for m in model.modules():
        if isinstance(m, SModule) or isinstance(m, NModule):
            noise_size += m.op.weight.shape.numel()
    
    if not args.load_direction:
        total_noise = torch.randn(args.noise_epoch, noise_size)
        
        w = w.reshape(1,-1) * -1
        # print(((w ** 2).sum() / w.shape.numel()).sqrt().item())
        total_noise = torch.cat([total_noise, w])
        
        # total_noise = total_noise * total_noise.abs()
        scale = ((total_noise ** 2).sum(dim=-1)/len(total_noise[0])).sqrt().reshape(len(total_noise),1)
        total_noise /= scale
    else:
        total_noise = torch.load(f"pretrained/{args.model}/directions.pt")
        if len(total_noise) < args.noise_epoch:
            raise Exception("Saved direction not enough")
        else:
            total_noise = total_noise[:args.noise_epoch]
    model.to_first_only()

    max_list = []
    avg_list = []
    acc_list = []
    l2 = args.alpha

    new_loader = []
    for images, labels  in testloader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.shape[0],-1)
        o1 = model.fc1(images)
        new_loader.append((o1, labels))
    testloader = new_loader
    model.fc1 = nn.Identity()

    # for i in tqdm(range(len(total_noise))):
    for i in range(len(total_noise)):
        left = 0
        model.clear_noise()
        # model.set_noise(l2, 0.0)
        for m in model.modules():
            if isinstance(m, SModule) or isinstance(m, NModule):
                this_size = m.op.weight.shape.numel()
                m.noise.data = (total_noise[i, left:left+this_size].reshape(m.noise.shape) * l2).to(device)
                # m.noise = m.noise.to(device)
                left += this_size
        # atk = PGD(model, c=args.attack_c, kappa=0, steps=args.attack_runs, lr=args.attack_lr, method=args.attack_method)
        acc = CEval()
        acc_list.append(acc)
        # print(f"This acc: {acc:.4f}")
    print(f"L2: {l2:.1e}, Mean: {np.mean(acc_list):.4f}, Max: {np.max(acc_list):.4f}, Min: {np.min(acc_list):.4f}")
    torch.save(acc_list, f"Circle_acc_list_{l2:.1e}.pt")
