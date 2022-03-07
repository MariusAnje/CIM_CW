import argparse
import csv
import logging
import os
import time

import torch

import child
import data
import backend
from controller import Agent
from config import ARCH_SPACE, QUAN_SPACE, CLOCK_FREQUENCY
from utility import BestSamples
import utility
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from cw_attack import WCW


# def get_args():
parser = argparse.ArgumentParser('Parser User Input Arguments')
parser.add_argument(
    '-d', '--dataset',
    default='CIFAR10',
    help="supported dataset including : 1. MNIST, 2. CIFAR10 (default)"
    )
parser.add_argument(
    '-l', '--layers',
    type=int,
    default=6,
    help="the number of child network layers, default is 6"
    )
parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=30,
    help="the total epochs for model fitting, default is 30"
    )
parser.add_argument(
    '-ep', '--episodes',
    type=int,
    default=2000,
    help='''the number of episodes for training the policy network, default
        is 2000'''
    )
parser.add_argument(
    '-ep1', '--train_episode',
    type=int,
    default=50,
    help='''the number of episodes for training the policy network, default
        is 2000'''
    )
parser.add_argument(
    '-lr', '--learning_rate',
    type=float,
    default=0.2,
    help="learning rate for updating the controller, default is 0.2")
parser.add_argument(
    '--dev_var',
    type=float,
    default=0.3,
    help="device variation")
parser.add_argument(
    '-b', '--batch_size',
    type=int,
    default=128,
    help="the batch size used to train the child CNN, default is 128"
    )
parser.add_argument(
    '-s', '--seed',
    type=int,
    default=1,
    help="seed for randomness, default is 0"
    )
parser.add_argument(
    '-g', '--gpu',
    type=int,
    default=0,
    help="in single gpu mode the id of the gpu used, default is 0"
    )
parser.add_argument(
    '-v', '--verbosity',
    type=int,
    choices=range(3),
    default=0,
    help="verbosity level: 0 (default), 1 and 2 with 2 being the most verbose"
    )
parser.add_argument(
    '-est', '--estimate',
    type=bool,
    default=False,
    help="something"
    )
parser.add_argument(
    '-gt', '--test_gt',
    type=bool,
    default=False,
    help="something"
    )
parser.add_argument(
    '--mode',
    default='nas',
    choices=['nas', 'joint', 'nested', 'quantization'],
    help="supported dataset including : 1. nas (default), 2. joint"
    )
parser.add_argument(
    '-opt', '--optimizer',
    default='Adam',
    choices=['Adam', 'SGD'],
    help="Optimizer used, just an indicator, no effecct, please edit the code to change"
    )
parser.add_argument(
    '-atk', '--attack',
    default='l2',
    choices=['l2', 'max'],
    help="Distance for CW attack"
    )
parser.add_argument(
    '-c', '--cw_c',
    type=float,
    default=1e-6,
    help="hyper parameter for CW attack"
    )
parser.add_argument(
    '-a', '--alpha',
    type=float,
    default=0.5,
    help="hyper parameter for CW attack"
    )

args = parser.parse_args()


def get_logger(filepath=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    if filepath is not None:
        file_handler = logging.FileHandler(filepath+'.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
    return logger


def main():
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    print(f"using device {device}")
    dir = os.path.join(
        f'experiment',
        args.dataset + f"({args.layers} layers)"
        )
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    SCRIPT[args.mode](device, dir)

def generate_legnth(space: dict):
    length = []
    for key in space:
        length.append(len(space[key]))
    length = [length] * args.layers
    length = np.array(length)
    length = length.reshape(-1)
    length = length.tolist()
    return length


def generate_hot(r, length):
    hot = []
    for i in range(len(length)):
        this = [0 for _ in range(length[i])]
        this[r[i]] = 1
        hot += this
    return hot

def attack_wcw(logger, model, val_data, quan_paras):
    def my_target(x,y):
        return (y+1)%10
    max_list = []
    avg_list = []
    acc_list = []
    for _ in range(9):
        model.clear_noise()
        attacker = WCW(model, c=args.cw_c, kappa=0, steps=10, lr=0.01, method=args.attack)
        # attacker.set_mode_targeted_random(n_classses=10)
        attacker.set_mode_targeted_by_function(my_target)
        attacker(val_data)
        max_list.append(attacker.noise_max().item())
        avg_list.append(np.sqrt(attacker.noise_l2().item()))
        loss, attack_accuracy = backend.epoch_fit(model, val_data, optimizer=None, quan_paras=quan_paras, verbosity=0, dev_var=0.0)
        acc_list.append(attack_accuracy)
    logger.info(f"L2 norm: {np.mean(avg_list):.4f}")
    mean_attack = np.mean(acc_list)
    return mean_attack


def reware_function(mean_noise, mean_attack, alpha):
    reward = (1-alpha) * mean_noise + alpha * mean_attack
    return reward

def nas(device, dir='experiment'):
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    filepath = os.path.join(dir, f"nas ({args.episodes} episodes)")
    logger = get_logger(filepath)
    csvfile = open(filepath+'.csv', mode='w+', newline='')
    writer = csv.writer(csvfile)
    logger.info(f"INFORMATION")
    logger.info(f"mode: \t\t\t\t\t {'nas'}")
    logger.info(f"dataset: \t\t\t\t {args.dataset}")
    logger.info(f"number of child network layers: \t {args.layers}")
    logger.info(f"training epochs: \t\t\t {args.epochs}")
    logger.info(f"batch size: \t\t\t\t {args.batch_size}")
    logger.info(f"controller learning rate: \t\t {args.learning_rate}")
    logger.info(f"architecture episodes: \t\t\t {args.episodes}")
    logger.info(f"use estimation method: \t\t\t {args.estimate}")
    logger.info(f"architecture space: ")
    for name, value in ARCH_SPACE.items():
        logger.info(name + f": \t\t\t\t {value}")
    length = generate_legnth(ARCH_SPACE)
    logger.info(args)
    agent = Agent(ARCH_SPACE, args.layers,
                  lr=args.learning_rate,
                  device=torch.device('cpu'), skip=False)
    
    train_data, val_data = data.get_data(
        args.dataset, device, shuffle=True,
        batch_size=args.batch_size, augment=True)
    input_shape, num_classes = data.get_info(args.dataset)
    writer.writerow(["ID"] +
                    ["Layer {}".format(i) for i in range(args.layers)] +
                    ["Accuracy"] +
                    ["Partition (Tn, Tm)", "Partition (#LUTs)",
                    "Partition (#cycles)", "Total LUT", "Total Throughput"] +
                    ["Time"])
    child_id, total_time = 0, 0
    logger.info('=' * 50 +
                "Start exploring architecture & quantization space" + '=' * 50)
    best_samples = BestSamples(5)
    X = []
    y = []
    clf = RandomForestRegressor()
    for e in range(args.episodes):
        logger.info('-' * 130)
        child_id += 1
        start = time.time()
        rollout, paras = agent.rollout()
        # rollout = [1, 1, 0, 0, 1, 0,  1, 1, 0, 0, 1, 1,  1, 1, 0, 0, 2, 0,  1, 1, 0, 0, 2, 1,  1, 1, 0, 0, 3, 0,  1, 1, 0, 0, 3, 1]
        # paras = agent.agent._format_rollout(rollout)
        logger.info("Sample Architecture ID: {}, Sampled actions: {}".format(
                    child_id, rollout))
        arch_paras, quan_paras = utility.split_paras(paras)
        XX = generate_hot(rollout, length)

        if args.estimate and e == args.train_episode:
            clf.fit(X, y)

        if (not args.estimate) or e < args.train_episode:
            model, optimizer = child.get_model(
                input_shape, arch_paras, num_classes, device,
                multi_gpu=False, do_bn=False)
            model.push_S_device()
            model.to_first_only()
            _, mean_noise = backend.fit(
                model, optimizer, train_data, val_data, quan_paras=quan_paras,
                epochs=args.epochs, verbosity=args.verbosity, dev_var = args.dev_var)
            mean_attack = attack_wcw(logger, model, val_data, quan_paras)
            reward = reware_function(mean_noise, mean_attack, args.alpha)
            X.append(XX)
            y.append(reward)

        if args.estimate and e >= args.train_episode:
            reward = clf.predict([XX])[0].item()
            if args.test_gt:
                model, optimizer = child.get_model(
                    input_shape, arch_paras, num_classes, device,
                    multi_gpu=False, do_bn=False)
                model.push_S_device()
                model.to_first_only()
                _, GT_reward = backend.fit(
                    model, optimizer, train_data, val_data, quan_paras=quan_paras,
                    epochs=args.epochs, verbosity=args.verbosity, dev_var = args.dev_var)
                logger.info(f"GT Reward: {GT_reward}, ")

        agent.store_rollout(rollout, reward)
        end = time.time()
        ep_time = end - start
        total_time += ep_time
        best_samples.register(child_id, rollout, reward)
        writer.writerow(
            [child_id] +
            [str(paras[i]) for i in range(args.layers)] +
            [reward] + [ep_time]
            )
        logger.info(f"Reward: {reward:.4f}, " +
                    f"Noise: {mean_noise:.4f}, " +
                    f"Attack: {mean_attack:.4f}, " + 
                    f"Elasped time: {ep_time}, " +
                    f"Average time: {total_time/(e+1)}")
        logger.info(f"Best Reward: {best_samples.reward_list[0]}, " +
                    f"ID: {best_samples.id_list[0]}, " +
                    f"Rollout: {best_samples.rollout_list[0]}")
    logger.info(
        '=' * 50 +
        "Architecture & quantization sapce exploration finished" +
        '=' * 50)
    logger.info(f"Total elasped time: {total_time}")
    logger.info(f"Best samples: {best_samples}")
    csvfile.close()

def sync_search(device, dir='experiment'):
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    filepath = os.path.join(dir, f"joint ({args.episodes} episodes)")
    logger = get_logger(filepath)
    csvfile = open(filepath+'.csv', mode='w+', newline='')
    writer = csv.writer(csvfile)
    logger.info(f"INFORMATION")
    logger.info(f"mode: \t\t\t\t\t {'joint'}")
    logger.info(f"dataset: \t\t\t\t {args.dataset}")
    logger.info(f"number of child network layers: \t {args.layers}")
    logger.info(f"training epochs: \t\t\t {args.epochs}")
    logger.info(f"batch size: \t\t\t\t {args.batch_size}")
    logger.info(f"controller learning rate: \t\t {args.learning_rate}")
    logger.info(f"architecture episodes: \t\t\t {args.episodes}")
    logger.info(f"use estimation method: \t\t\t {args.estimate}")
    logger.info(f"architecture space: ")
    for name, value in ARCH_SPACE.items():
        logger.info(name + f": \t\t\t\t {value}")
    logger.info(f"quantization space: ")
    for name, value in QUAN_SPACE.items():
        logger.info(name + f": \t\t\t {value}")
    logger.info(args)
    agent = Agent({**ARCH_SPACE, **QUAN_SPACE}, args.layers,
                  lr=args.learning_rate,
                  device=torch.device('cpu'), skip=False)
    length = generate_legnth({**ARCH_SPACE, **QUAN_SPACE})
    train_data, val_data = data.get_data(
        args.dataset, device, shuffle=True,
        batch_size=args.batch_size, augment=True)
    input_shape, num_classes = data.get_info(args.dataset)
    writer.writerow(["ID"] +
                    ["Layer {}".format(i) for i in range(args.layers)] +
                    ["Accuracy"] +
                    ["Partition (Tn, Tm)", "Partition (#LUTs)",
                    "Partition (#cycles)", "Total LUT", "Total Throughput"] +
                    ["Time"])
    child_id, total_time = 0, 0
    logger.info('=' * 50 +
                "Start exploring architecture & quantization space" + '=' * 50)
    best_samples = BestSamples(5)
    X = []
    y = []
    clf = RandomForestRegressor()
    for e in range(args.episodes):
        logger.info('-' * 130)
        child_id += 1
        start = time.time()
        rollout, paras = agent.rollout()
        logger.info("Sample Architecture ID: {}, Sampled actions: {}".format(
                    child_id, rollout))
        arch_paras, quan_paras = utility.split_paras(paras)
        XX = generate_hot(rollout, length)

        if args.estimate and e == args.train_episode:
            clf.fit(X, y)

        if (not args.estimate) or e < args.train_episode:
            model, optimizer = child.get_model(
                input_shape, arch_paras, num_classes, device,
                multi_gpu=False, do_bn=False)
            model.push_S_device()
            model.to_first_only()
            _, mean_noise = backend.fit(
                model, optimizer, train_data, val_data, quan_paras=quan_paras,
                epochs=args.epochs, verbosity=args.verbosity, dev_var = args.dev_var)
            mean_attack = attack_wcw(logger, model, val_data, quan_paras)
            reward = reware_function(mean_noise, mean_attack, args.alpha)
            X.append(XX)
            y.append(reward)
            
        if args.estimate and e >= args.train_episode:
            reward = clf.predict([XX])[0].item()
            if args.test_gt:
                model, optimizer = child.get_model(
                    input_shape, arch_paras, num_classes, device,
                    multi_gpu=False, do_bn=False)
                model.push_S_device()
                model.to_first_only()
                _, GT_reward = backend.fit(
                    model, optimizer, train_data, val_data, quan_paras=quan_paras,
                    epochs=args.epochs, verbosity=args.verbosity, dev_var = args.dev_var)
                logger.info(f"GT Reward: {GT_reward}")

        agent.store_rollout(rollout, reward)
        end = time.time()
        ep_time = end - start
        total_time += ep_time
        best_samples.register(child_id, rollout, reward)
        writer.writerow(
            [child_id] +
            [str(paras[i]) for i in range(args.layers)] +
            [reward] + [ep_time]
            )
        logger.info(f"Reward: {reward:.4f}, " +
                    f"Noise: {mean_noise:.4f}, " +
                    f"Attack: {mean_attack:.4f}, " + 
                    f"Elasped time: {ep_time}, " +
                    f"Average time: {total_time/(e+1)}")
        logger.info(f"Best Reward: {best_samples.reward_list[0]}, " +
                    f"ID: {best_samples.id_list[0]}, " +
                    f"Rollout: {best_samples.rollout_list[0]}")
    logger.info(
        '=' * 50 +
        "Architecture & quantization sapce exploration finished" +
        '=' * 50)
    logger.info(f"Total elasped time: {total_time}")
    logger.info(f"Best samples: {best_samples}")
    csvfile.close()

SCRIPT = {
    'nas': nas,
    'joint': sync_search,
}

if __name__ == '__main__':
    import random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    main()

