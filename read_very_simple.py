import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder_name', default="./")
parser.add_argument('file_name', default="CW.o")
args = parser.parse_args()
print(args)

file_list = os.listdir(args.folder_name)
num_list = []
for fn in file_list:
    if args.file_name in fn:
        with open(os.path.join(args.folder_name, fn), "r") as f:
            data = f.read()
            data = data.splitlines()
        for line in data:
            if (("PGD" in line or "C&W" in line or "noise worst acc: " in line or "quantile" in line) and "acc: " in line):
                index = line.find("acc: ") + len("acc: ")
                num_list.append(float(line[index:index+6]))
print(f"Everything: {num_list}")
print(f"Average: {np.mean(num_list):.4f}, Worst: {np.min(num_list):.4f}")