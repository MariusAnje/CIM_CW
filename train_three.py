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
from cw_attack import Attack, WCW, binary_search_c, binary_search_dist, PGD
import torch
import torch.nn as nn
import torch.optim as optim
from utils import str2bool, attack_wcw, get_dataset, get_model, prepare_model
from utils import TPMTrain, MTrain, TCEval, TMEachEval, PGD_Eval, CEval, MEachEval
from utils import copy_model, get_logger

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
    parser.add_argument('--attack_dist', action='store', type=float, default=0.0,
            help='method used for attack')
    parser.add_argument('--train_attack_runs', action='store',type=int, default=10,
            help='# of runs for attack during training')
    parser.add_argument('--load_atk', action='store',type=str2bool, default=True,
            help='if we should load the attack')
    parser.add_argument('--load_direction', action='store',type=str2bool, default=False,
            help='if we should load the noise directions')
    parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
            help='whether to use tqdm')
    parser.add_argument('--warm_epoch', action='store',type=int, default=0,
            help='# of epochs to warm up')
    args = parser.parse_args()

    logger = get_logger(f"logs/train_three_log_{args.model}_{args.attack_dist}_{args.rate_max}_{time.time()}")
    logger.info(f"{args}")
    header = time.time()
    header_timer = header
    parent_path = "./"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    BS = 128
    NW = 4

    trainloader, secondloader, testloader = get_dataset(args, BS, NW)
    model = get_model(args)
    model1, optimizer1, w_optimizer1, scheduler1 = prepare_model(model, device, args)
    model2, optimizer2, w_optimizer2, scheduler2 = copy_model(model, args)
    model3, optimizer3, w_optimizer3, scheduler3 = copy_model(model, args)
    t_model = [model1, model2, model3]
    t_optimizer = [optimizer1, optimizer2, optimizer3]
    w_optimizer = [w_optimizer1, w_optimizer2, w_optimizer3]
    t_scheduler = [scheduler1, scheduler2, scheduler3]
    

    criteria = SCrossEntropyLoss()
    criteriaF = torch.nn.CrossEntropyLoss()
    
    kwargs = {"N":8, "m":1}
    t_model_group = t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader
    TPMTrain(t_model_group, args.warm_epoch, args.train_epoch, header, args.noise_type, 0.0, args.train_var, args.rate_max, args.rate_zero, args.write_var, args.train_attack_runs, args.attack_dist, logger=logger, verbose=args.verbose, **kwargs)
    # ATrain(args.train_epoch, header, dev_var=args.train_var, verbose=args.verbose)
    model_group = model, criteriaF, t_optimizer[0], t_scheduler[0], device, trainloader, testloader
    model.clear_noise()
    state_dict = torch.load(f"tmp_best_{header}.pt")
    model.load_state_dict(state_dict)
    model.from_first_back_second()
    torch.save(model.state_dict(), f"saved_B_{header}_dist_{args.attack_dist}_noise_{args.rate_max}_{args.train_var}.pt")
    model.clear_noise()
    model.to_first_only()
    # logger.info(f"No mask no noise: {', '.join([ f'{acc:.4f}' for acc in TCEval(t_model_group)])}")
    logger.info(f"No mask no noise: {CEval(model_group):.4f}")
    model.from_first_back_second()
    state_dict = torch.load(f"saved_B_{header}_dist_{args.attack_dist}_noise_{args.rate_max}_{args.train_var}.pt")
    model.load_state_dict(state_dict)
    model.clear_mask()
    model.to_first_only()
    performance, pgd_max, pgd_l2 = PGD_Eval(model_group, args.attack_runs, args.attack_dist, "act")
    logger.info(f"No mask PGD acc: {performance:.4f}, max distance: {pgd_max}")
    # mean_attack, w = attack_wcw(model, testloader, verbose=True)
    exit()

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
    
    # binary_search_c(search_runs = 10, acc_evaluator=CEval, dataloader=testloader, th_accuracy=0.15, attacker_class=WCW, model=model, init_c=args.attack_c, steps=args.attack_runs, lr=args.attack_lr, method=args.attack_method, verbose=True)
    binary_search_dist(search_runs = 10, acc_evaluator=CEval, dataloader=testloader, target_metric=0.03, attacker_class=WCW, model=model, init_c=args.attack_c, steps=args.attack_runs, lr=args.attack_lr, method=args.attack_method, verbose=True, use_tqdm = args.use_tqdm)
    # target max: 0.03, header = 2
    # Model: QLeNet acc:0.5402, c = 1.9844e-09, lr = 1e-5
    # Model: QCIFAR acc:0.0245, c = 1e-5, lr = 1e-4
    # Model: QRes18 acc:0.0000, c = 10, dist = 0.0111, lr = 5e-5
    # Model: QTIN   acc:0.0000, c = 1,  dist = 0.0059, lr = 1e-4
    # Model: QVGGIN acc:0.0008, c = 1,  dist = 0.0273, lr = 1e-4
    exit()

    # j = 0
    # for _ in range(10000):
    # # binary_search_c(search_runs = 10, acc_evaluator=CEval, dataloader=testloader, th_accuracy=0.001, attacker_class=WCW, model=model, init_c=1, steps=10, lr=0.01, method="l2", verbose=True)
    #     acc, w = attack_wcw(model, testloader, True)
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
        # atk = WCW(model, c=args.attack_c, kappa=0, steps=args.attack_runs, lr=args.attack_lr, method=args.attack_method)
        acc = CEval()
        acc_list.append(acc)
        # print(f"This acc: {acc:.4f}")
    print(f"L2: {l2:.1e}, Mean: {np.mean(acc_list):.4f}, Max: {np.max(acc_list):.4f}, Min: {np.min(acc_list):.4f}")
    torch.save(acc_list, f"Circle_acc_list_{l2:.1e}.pt")
