import os
import time
import argparse
import numpy as np
import logging

class BoundedSearch():
    def __init__(self, max_iter, logger) -> None:
        self.replay_memory = {}
        self.crr_iter = 0
        self.max_iter = max_iter
        self.logger = logger

    def bounded_search_float(self, func, start, end): 
        mid = (start + end) / 2
        self.crr_iter += 1
        if self.crr_iter > self.max_iter:
            if f"{mid:.4f}" in self.replay_memory:
                probMid = self.replay_memory[f"{mid:.4f}"]
            else:
                probMid = func(mid, self.logger)
                self.replay_memory[f"{mid:.4f}"] = probMid
            return probMid

        oForth = (end - start) / 4
        left = start + oForth
        right = end - oForth
        if f"{mid:.4f}" in self.replay_memory:
            probMid = self.replay_memory[f"{mid:.4f}"]
        else:
            probMid = func(mid, self.logger)
            self.replay_memory[f"{mid:.4f}"] = probMid

        if f"{left:.4f}" in self.replay_memory:
            probLeft = self.replay_memory[f"{left:.4f}"]
        else:
            probLeft = func(left, self.logger)
            self.replay_memory[f"{left:.4f}"] = probLeft

        if f"{right:.4f}" in self.replay_memory:
            probRight = self.replay_memory[f"{right:.4f}"]
        else:
            probRight = func(right, self.logger)
            self.replay_memory[f"{right:.4f}"] = probRight
        
        self.logger.info(self.replay_memory)

        theMax = max([probMid, probLeft, probRight])
        if theMax == probMid:
            return self.bounded_search_float(func, left, right)
        elif theMax == probLeft:
            return self.bounded_search_float(func, start, right)
        else:
            return self.bounded_search_float(func, left, end)

def read_using_keyword(line:str, keyword:str, offset:int):
    index = line.find(keyword) + len(keyword)
    return line[index:index+offset]

def collect_result_from_file(filename):
    with open(filename, "r") as f:
        file_content = f.read().splitlines()
    for line in file_content:
        if "PGD" in line and "acc" in line:
            acc = float(read_using_keyword(line, "acc: ", 6))
    return acc

class SGEval():
    def __init__(self, th, distance, epochs, attack_runs, eval_runs, model, device) -> None:
        self.th = th
        self.distance= distance
        self.epochs = epochs
        self.attack_runs = attack_runs
        self.eval_runs = eval_runs
        self.model = model
        self.device = device

    def evaluate_dev_var(self, dev_var, logger):
        acc_list = []
        for _ in range(self.eval_runs):
            header = time.time()
            tmp_filename = f"tmp_{header}.txt"
            os.system(f"python train_multi.py --model {self.model} --noise_type SG --train_var {dev_var} --dev_var {dev_var} --rate_max {self.th} --train_epoch {self.epochs} --verbose True --header {header} --device {self.device} > {tmp_filename}")
            right = f"{header}_noise_{self.th}_{dev_var}"
            output_filename = f"PGD_result_{header}.txt"
            os.system(f"python valid_mask.py --model {self.model} --model_path ./ --attack_runs {self.attack_runs} --attack_distance_metric max --attack_dist {self.distance} --drop 0.00 --header {right} --attack_function act --device {self.device} > {output_filename}")
            this_acc = collect_result_from_file(output_filename)
            acc_list.append(this_acc)
            os.remove(f"tmp_best_{header}.pt")
            os.remove(f"saved_B_{right}.pt")
            os.remove(output_filename)
            os.remove(tmp_filename)
        logger.info(f"dev var: {dev_var:.4f}, PGD acc: {np.mean(acc_list):.4f}")
        return np.mean(acc_list)

def get_logger(filepath=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)
    if filepath is not None:
        file_handler = logging.FileHandler(filepath+'.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_search_iter', action='store', type=int, default=5,
            help='# of search iteration')
    parser.add_argument('--eval_runs', action='store', type=int, default=3,
            help='# of evaluations for one hyper-parameter')
    parser.add_argument('--train_epoch', action='store', type=int, default=20,
            help='# of epochs of training')
    parser.add_argument('--train_var_start', action='store', type=float, default=None,
            help='device variation [std] before write and verify')
    parser.add_argument('--train_var_end', action='store', type=float, default=None,
            help='device variation [std] before write and verify')
    parser.add_argument('--train_th', action='store', type=float, default=None,
            help='threshold for PTGaussian')
    parser.add_argument('--device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--model', action='store', default="MLP4", choices=["MLP3", "MLP3_2", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG", "Adv", "QVGGIN", "QResIN"],
            help='model to use')
    parser.add_argument('--attack_function', action='store', default="act", choices=["act", "cross"],
            help='function used for attack')
    parser.add_argument('--attack_runs', action='store', type=int, default=20,
            help='# of epochs of training')
    parser.add_argument('--attack_distance_metric', action='store', default="l2", choices=["max", "l2", "linf", "loss"],
            help='distance metric used for attack')
    parser.add_argument('--attack_dist', action='store', type=float, default=0.03,
            help='distance used for attack')
    parser.add_argument('--attack_name', action='store', default="C&W",
            help='# of epochs of training')
    args = parser.parse_args()

    # bounded_search_float(func,1,39,0,10)
    logger = get_logger(f"logs/Search_SG_log_{args.model}_{args.attack_dist}_{args.train_th}_{time.time()}")
    logger.info(f"{args}")
    evaluator = SGEval(args.train_th, args.attack_dist, args.train_epoch, args.attack_runs, args.eval_runs, args.model, args.device)
    # search.evaluate_dev_var(0.2, eval_runs=args.eval_runs)
    searcher = BoundedSearch(args.max_search_iter, logger)
    res = searcher.bounded_search_float(evaluator.evaluate_dev_var, args.train_var_start, args.train_var_end)
    max_key = max(searcher.replay_memory, key=searcher.replay_memory.get)
    max_value = max(searcher.replay_memory.values())
    logger.info(f"best variation: {max_key}, best accuracy: {max_value:.4f}")