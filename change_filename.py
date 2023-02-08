import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dev_var', action='store')
args = parser.parse_args()

fn_list = os.listdir("./")

i = 1
for fn in fn_list:
    if "saved_B_" in fn and ".pt" in fn and "_" + args.dev_var in fn:
        new_fn = f"saved_B_{i}.pt"
        os.system(f"mv {fn} {new_fn}")
        i += 1
