import os
import time

def bounded_search_float(func, start, end, crr_iter, max_iter): 
    print(start, end)
    mid = (start + end) / 2
    crr_iter += 1
    if crr_iter > max_iter:
        return func(mid)
    oForth = (end - start) / 4
    left = start + oForth
    right = end - oForth
    probMid = func(mid)
    probLeft = func(left)
    probRight = func(right)
    theMax = max([probMid, probLeft, probRight])
    if theMax == probMid:
        return bounded_search_float(func, left, right, crr_iter, max_iter)
    elif theMax == probLeft:
        return bounded_search_float(func, left, right, crr_iter, max_iter)
    else:
        return bounded_search_float(func, left, right, crr_iter, max_iter)

class SGSearch():
    def __init__(self, th, distance, epochs, attack_runs) -> None:
        self.th = th
        self.distance= distance
        self.epochs = epochs
        self.attack_runs = attack_runs

    def evaluate_dev_var(self, dev_var):
        header = time.time()
        os.system(f"python train_multi.py --model QLeNet --noise_type SG --train_var {dev_var} --dev_var {dev_var} --rate_max {self.th} --train_epoch {self.epochs} --verbose True --header {header}")    
        right = f"{header}_noise_{dev_var}"
        os.system(f"python valid_mask.py --model QLeNet --model_path ./ --attack_runs {self.attack_runs} --attack_distance_metric max --attack_dist {self.distance} --drop 0.00 --header {right} --attack_function act --use_tqdm True")
        os.system(f"rm tmp_best_{header}.pt")
        os.system(f"rm saved_B_{right}.pt")

# bounded_search_float(func,1,39,0,10)
search = SGSearch(1, 0.01, 5, 100)
search.evaluate_dev_var(0.2)