import logging
from config import ARCH_SPACE, QUAN_SPACE


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


def split_paras(paras):
    num_layers = len(paras)
    arch_paras = []
    quan_paras = []
    for i in range(num_layers):
        para = paras[i]
        arch_para = {}
        quan_para = {}
        for name, _ in ARCH_SPACE.items():
            if name in para:
                arch_para[name] = para[name]
            if 'anchor_point' in para:
                arch_para['anchor_point'] = para['anchor_point']
        for name, _ in QUAN_SPACE.items():
            if name in para:
                quan_para[name] = para[name]
        if arch_para != {}:
            arch_paras.append(arch_para)
        if quan_para != {}:
            quan_paras.append(quan_para)
        if arch_paras == []:
            arch_paras = None
        if quan_paras == []:
            quan_paras = None
    return arch_paras, quan_paras


def combine_rollout(arch_rollout, quan_rollout, num_layers):
    arch_num_paras_per_layer = int(len(arch_rollout) / num_layers)
    quan_num_paras_per_layer = int(len(quan_rollout) / num_layers)
    result = []
    for i in range(num_layers):
        result += arch_rollout[i * arch_num_paras_per_layer:
                               arch_num_paras_per_layer * (i + 1)]
        result += quan_rollout[i * quan_num_paras_per_layer:
                               quan_num_paras_per_layer * (i + 1)]
    return result


class BestSamples(object):
    def __init__(self, length=5):
        self.length = length
        self.id_list = [i for i in range(length)]
        self.rollout_list = [[] for _ in range(length)]
        self.reward_list = [-1 for _ in range(length)]

    def register(self, id, rollout, reward):
        for i in range(self.length):
            if reward > self.reward_list[i]:
                self.id_list = self.insert(id, self.id_list, i)
                self.rollout_list = self.insert(rollout, self.rollout_list, i)
                self.reward_list = self.insert(reward, self.reward_list, i)
                break
    
    def insert(self, data, t_list, index):
        if index >= len(t_list):
            return t_list
        else:
            t_list = t_list[:-1]
            t_list = t_list[:index] + [data] + t_list[index:]
            return t_list

    def __repr__(self):
        return str(dict(zip(self.id_list, self.reward_list)))



if __name__ == '__main__':
    arch_rollout = [3, 3, 0, 0, 0, 1, 2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 2, 1, 3, 3, 1, 1, 3, 0, 1, 2, 0, 2, 0, 1, 3, 2, 2, 2, 1, 1]
    quan_rollout = [2, 1, 3, 4, 0, 0, 0, 6, 3, 2, 0, 4, 2, 4, 1, 2, 0, 2, 1, 1, 3, 6, 1, 2]
    combine_rollout(arch_rollout, quan_rollout, 6)
    best_samples = BestSamples(5)
    i  = [1,2,3,4,5]
    r  = [2,3,4,12,23]
    ro = [[1],[2],[3],[4],[5]]
    print(best_samples)
    for id in range(len(i)):
        best_samples.register(i[id], ro[id], r[id])
        print(best_samples)



