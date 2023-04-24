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
from utils import TQMTrain, MTrain, TCEval, TMEachEval, PGD_Eval, CEval, MEachEval, MEval, MEachEval
from utils import copy_model, get_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epoch', action='store', type=int, default=20,
            help='# of epochs of training')
    parser.add_argument('--noise_epoch', action='store', type=int, default=100,
            help='# of epochs of noise validations')
    parser.add_argument('--quantile', action='store', type=float, default=0.5,
            help='quantile of noise validations')
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
    parser.add_argument('--test_zero', action='store', type=float, default=0,
            help='for test, pepper rate, rate of noise being zero')
    parser.add_argument('--test_max', action='store', type=float, default=1,
            help='for test, salt rate, rate of noise being one')
    parser.add_argument('--test_noise_type', action='store', default="Gaussian",
            help='for test, type of noise used')
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

    logger = get_logger(f"logs/Quantile/train_noise_three_quantile_log_{args.model}_{args.attack_dist}_{args.rate_max}_{time.time()}")
    logger.info(f"{args}")
    header = time.time()
    header_timer = header
    parent_path = "./"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    BS = 128
    if "LeNet" in args.model:
        LARGE_BS = 10240
    elif "CIFAR" in args.model:
        LARGE_BS = 5000
    elif "Res18" in args.model:
        LARGE_BS = 1024
    elif "TIN" in args.model:
        LARGE_BS = 128
    else:
        LARGE_BS = 128
    NW = 4

    trainloader, secondloader, testloader = get_dataset(args, BS, NW)
    _, _, testloader_large = get_dataset(args, LARGE_BS, NW)
    memory_testloader = []
    for data, labels in testloader:
        data, labels = data, labels
        memory_testloader.append((data, labels))
    memory_testloader_large = []
    for data, labels in testloader_large:
        data, labels = data, labels
        memory_testloader_large.append((data, labels))
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
    t_model_group = t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, memory_testloader, memory_testloader_large
    TQMTrain(t_model_group, args.warm_epoch, args.train_epoch, args.train_attack_runs, args.quantile, header, 
             args.noise_type, 0.0, args.train_var, args.rate_max, args.rate_zero, args.write_var, 
             args.train_attack_runs, args.test_noise_type, args.test_max, args.test_zero, args.attack_dist, logger=logger, verbose=args.verbose, **kwargs)
    # ATrain(args.train_epoch, header, dev_var=args.train_var, verbose=args.verbose)
    model_group = model, criteriaF, t_optimizer[0], t_scheduler[0], device, trainloader, memory_testloader_large
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
    average_performance = MEachEval(model_group, args.test_noise_type, args.attack_dist, args.test_max, args.test_zero, 0, **kwargs)
    logger.info(f"No mask noise average acc: {average_performance:.4f}")
    performance_list = []
    for _ in range(args.attack_runs):
        performance = MEval(model_group, args.test_noise_type, args.attack_dist, args.test_max, args.test_zero, 0, **kwargs)
        performance_list.append(performance)
    performance = np.quantile(performance_list, args.quantile)
    logger.info(f"No mask noise acc: {performance:.4f}, distance: {args.attack_dist}, quantile: {args.quantile}")
    q_list = [0.5, 0.1, 0.01, 0.001, 0.0001]
    for quantile in q_list:
        logger.info(f"{quantile:6f} quantile: {np.quantile(performance_list, quantile):.4f}")
    exit()
