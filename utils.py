import torch
import time
from cw_attack import Attack, WCW, binary_search_c, binary_search_dist, PGD
import numpy as np
import torchvision
import torchvision.transforms as transforms
import modules
from models import SCrossEntropyLoss, SMLP3, SMLP4, SLeNet, CIFAR, FakeSCrossEntropyLoss, SAdvNet
from qmodels import QSLeNet, QCIFAR
import resnet
import qresnet
import qvgg
import qdensnet
import qresnetIN
from torch import optim

def get_dataset(args, BS, NW):
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
                                                shuffle=True, num_workers=NW)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div,
                                                shuffle=False, num_workers=NW)

        testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                            download=False, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                    shuffle=False, num_workers=NW)
    return trainloader, secondloader, testloader

def get_model(args):
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
    return model

def prepare_model(model, device, args):
    model.to(device)
    for m in model.modules():
        if isinstance(m, modules.FixedDropout) or isinstance(m, modules.NFixedDropout) or isinstance(m, modules.SFixedDropout):
            m.device = device
    model.push_S_device()
    model.clear_noise()
    model.clear_mask()
    model.to_first_only()
    model.de_select_drop()
    if "TIN" in args.model or "Res" in args.model or "VGG" in args.model or "DENSE" in args.model:
    # if "TIN" in args.model or "Res" in args.model:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1000])
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60])
    warm_optimizer = optim.SGD(model.parameters(), lr=1e-3)
    return model, optimizer, warm_optimizer, scheduler

def copy_model(old_model, args):
    new_model = get_model(args)
    state_dict = old_model.state_dict()
    for key in state_dict.keys():
        if "weight" in key:
            device = state_dict[key].device
            break
    new_model, optimizer, warm_optimizer, scheduler = prepare_model(new_model, device)
    new_model.load_state_dict(old_model.state_dict())
    return new_model, optimizer, warm_optimizer, scheduler






def CEval(model_group):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
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
    return (correct/total).cpu().item()

def PGD_Eval(model_group, steps, attack_dist, attack_function, use_tqdm = False):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    if steps == 0:
        return CEval(model_group)
    model.eval()
    model.clear_noise()
    model.normalize()
    step_size = attack_dist / steps
    attacker = PGD(model, attack_dist, step_size=step_size, steps=steps * 10)
    attacker.set_f(attack_function)
    attacker(testloader, use_tqdm)
    # attacker.save_noise(f"lol_{header}_{args.attack_dist:.4f}.pt")
    this_accuracy = CEval(model_group)
    this_max = attacker.noise_max().item()
    this_l2 = attacker.noise_l2().item()
    # print(f"PGD Results --> acc: {this_accuracy:.4f}, l2: {this_l2:.4f}, max: {this_max:.4f}")
    model.clear_noise()
    model.de_normalize()
    return this_accuracy

def CEval_Dist(model_group, num_classes=10):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
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
    return (correct/total).cpu().item(), res_dist

def NEval(model_group, dev_var, write_var):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
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
    return (correct/total).cpu().item()

def NEachEval(model_group, dev_var, write_var):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
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
    return (correct/total).cpu().item()

def MEachEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            model.clear_noise()
            model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
            # model.set_SPU(s_rate, p_rate, dev_var)
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().item()

def TMEachEval(t_model_group, noise_type, dev_var_list, rate_max, rate_zero, write_var, **kwargs):
    t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader = t_model_group
    acc_list = []
    for i in range(len(t_model)):
        model = t_model[i]
        model.eval()
        total = 0
        correct = 0
        model.clear_noise()
        with torch.no_grad():
            for images, labels in testloader:
                model.clear_noise()
                model.set_noise_multiple(noise_type, dev_var_list[i], rate_max, rate_zero, write_var, **kwargs)
                # model.set_SPU(s_rate, p_rate, dev_var)
                images, labels = images.to(device), labels.to(device)
                # images = images.view(-1, 784)
                outputs = model(images)
                if len(outputs) == 2:
                    outputs = outputs[0]
                predictions = outputs.argmax(dim=1)
                correction = predictions == labels
                correct += correction.sum()
                total += len(correction)
            acc_list.append((correct/total).cpu().item())
    return acc_list

def TPGD_Eval(t_model_group, steps, attack_dist, attack_function, use_tqdm = False):
    t_model, criteriaF, optimizer, w_optimizer, scheduler, device, trainloader, testloader = t_model_group
    acc_list = []
    for i in range(len(t_model)):
        model = t_model[i]
        model_group = [model, criteriaF, optimizer, scheduler, device, trainloader, testloader]
        if steps == 0:
            return CEval(model_group)
        model.eval()
        model.clear_noise()
        model.normalize()
        step_size = attack_dist / steps
        attacker = PGD(model, attack_dist, step_size=step_size, steps=steps * 10)
        attacker.set_f(attack_function)
        attacker(testloader, use_tqdm)
        # attacker.save_noise(f"lol_{header}_{args.attack_dist:.4f}.pt")
        this_accuracy = CEval(model_group)
        this_max = attacker.noise_max().item()
        this_l2 = attacker.noise_l2().item()
        # print(f"PGD Results --> acc: {this_accuracy:.4f}, l2: {this_l2:.4f}, max: {this_max:.4f}")
        model.clear_noise()
        model.de_normalize()
        acc_list.append(this_accuracy)
    return acc_list

def TCEval(t_model_group):
    t_model, criteriaF, optimizer, w_optimizer, scheduler, device, trainloader, testloader = t_model_group
    acc_list = []
    for model in t_model:
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
            acc_list.append((correct/total).cpu().numpy())
    return acc_list

def NTrain(model_group, epochs, header, dev_var=0.0, write_var=0.0, verbose=False):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
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
        test_acc = NEachEval(model_group, dev_var, write_var)
        # test_acc = CEval()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, loss: {running_loss / len(trainloader):.4f}")
        scheduler.step()

def MTrain(model_group, epochs, header, noise_type, dev_var, rate_max, rate_zero, write_var, verbose=False, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    for i in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            model.clear_noise()
            model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = MEachEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
        # test_acc = CEval()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            end_time = time.time()
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
        scheduler.step()

def PMTrain(model_group, epochs, header, noise_type, dev_var, rate_max, rate_zero, write_var, verbose=False, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    for i in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            model.clear_noise()
            model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = MEachEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
        pgd_acc = PGD_Eval(model_group, 5, 0.040, "act", use_tqdm = False)
        # test_acc = CEval()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            end_time = time.time()
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, pgd acc: {pgd_acc:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
        scheduler.step()

def TPMTrain(three_model_group, warm_epochs, epochs, header, noise_type, dev_var_start, dev_var_end, rate_max, rate_zero, write_var, attack_runs, attack_dist, verbose=False, **kwargs):
    t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader = three_model_group
    best_acc = 0.0
    start, end = dev_var_start, dev_var_end
    
    for ep in range(warm_epochs + epochs):
        mid = (start + end)/2
        oForth = (end - start) / 4
        left = start + oForth
        right = end - oForth
        dev_var_list = [left, mid, right]
        start_time = time.time()
        for i in range(len(t_model)):
            t_model[i].train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            for i in range(len(t_model)):
                t_model[i].clear_noise()
                t_model[i].set_noise_multiple(noise_type, dev_var_list[i], rate_max, rate_zero, write_var, **kwargs)
                t_optimizer[i].zero_grad()
                images, labels = images.to(device), labels.to(device)
                outputs = t_model[i](images)
                loss = criteriaF(outputs,labels)
                loss.backward()
                if ep >= warm_epochs:
                    t_optimizer[i].step()
                else:
                    w_optimizer[i].step()
                running_loss += loss.item()
        test_acc = TMEachEval(three_model_group, noise_type, dev_var_list, rate_max, rate_zero, write_var, **kwargs)
        best_pgd_index = 1
        if ep >= warm_epochs:
            pgd_acc = TPGD_Eval(three_model_group, attack_runs, attack_dist, "act", use_tqdm = False)
            best_pgd_index = np.argmax(pgd_acc)
            if best_pgd_index == 0:
                end = right
            elif best_pgd_index == 1:
                start = left
                end = right
            else:
                start = left
            for i in range(len(t_model)):
                t_model[i].load_state_dict(t_model[best_pgd_index].state_dict())

            # test_acc = CEval()
            if pgd_acc[best_pgd_index] > best_acc:
                best_acc = pgd_acc[best_pgd_index]
                torch.save(t_model[best_pgd_index].state_dict(), f"tmp_best_{header}.pt")
            
            for i in range(len(t_scheduler)):
                t_scheduler[i].step()
            if verbose:
                end_time = time.time()
                print(f"epoch: {ep:-3d}, test acc: {test_acc[best_pgd_index]:.4f}, pgd acc: {pgd_acc[best_pgd_index]:.4f}, left: {left:.4f}, right: {right:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
        else:
            if verbose:
                end_time = time.time()
                print(f"warm up epoch: {ep:-3d}, test acc: {test_acc[best_pgd_index]:.4f}, mid: {mid:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")


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
        attacker = WCW(model, c=args.attack_c, kappa=0, steps=args.attack_runs, lr=args.attack_lr, method=args.attack_method)
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