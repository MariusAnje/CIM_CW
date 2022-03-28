# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import math

import torch
from torch import nn
from torch import functional
from torch._C import device
from torch.nn.modules.pooling import MaxPool2d
import numpy as np


class MyModule(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

class NModule(nn.Module):
    def set_noise(self, dev_var, write_var, N, m):
        # N: number of bits per weight, m: number of bits per device
        # Dev_var: device variation before write and verify
        # write_var: device variation after write and verity
        scale = self.op.weight.abs().max()
        noise_dev = torch.zeros_like(self.noise).to(self.op.weight.device)
        # noise_write = torch.zeros_like(self.noise).to(self.op.weight.device)
        # for i in range(1, N//m + 1):
        #     if dev_var != 0:
        #         noise_dev   += (torch.normal(mean=0., std=dev_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
        #     if write_var != 0:
        #         noise_write += (torch.normal(mean=0., std=write_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
        
        # noise_dev = noise_dev.to(self.op.weight.device) * scale
        # noise_write = noise_write.to(self.op.weight.device) * scale
        # self.noise = noise_dev * self.mask + noise_write * (1 - self.mask)

        self.noise = torch.normal(mean=0., std=dev_var, size=self.noise.size()).to(self.op.weight.device) * scale
        
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def push_S_device(self):
        # self.mask = self.mask.to(self.op.weight.device)
        self.noise = self.noise.to(self.op.weight.device)

class NLinear(NModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.noise = torch.zeros_like(self.op.weight)
        # self.mask = torch.ones_like(self.op.weight)
        self.function = nn.functional.linear

    def forward(self, x):
        x = self.function(x, self.op.weight + self.noise, self.op.bias)
        return x

class NConv2d(NModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.noise = torch.zeros_like(self.op.weight)
        # self.mask = torch.ones_like(self.op.weight)
        self.function = nn.functional.conv2d
        self.in_channels = self.op.in_channels
        self.out_channels = self.op.out_channels
        self.kernel_size = self.op.kernel_size
        self.stride = self.op.stride
        self.padding = self.op.padding
        self.dilation = self.op.dilation
        self.groups = self.op.groups
        self.padding_mode = self.op.padding_mode

    def forward(self, x):
        x = self.function(x, self.op.weight + self.noise, self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        return x

class NAct(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.noise = torch.zeros(self.size)
        # self.mask = torch.ones(self.size)
    
    def clear_noise(self):
        self.noise = torch.zeros(self.size)
    
    def set_noise(self, var):
        self.noise = torch.randn(self.size) * var
        
    def push_S_device(self, device):
        # self.mask = self.mask.to(device)
        self.noise = self.noise.to(device)

    def forward(self, x):
        return x + self.noise #* self.mask

class NModel(nn.Module):
    def __init__(self):
        super().__init__()

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def unpack_flattern(self, x):
        return x.view(-1, self.num_flat_features(x))

    def set_noise(self, dev_var, write_var, N=8, m=1):
        for mo in self.modules():
            if isinstance(mo, NModule):
                mo.set_noise(dev_var, write_var, N, m)

    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, NModule) or isinstance(m, NAct):
                m.clear_noise()
    
    def push_S_device(self):
        for m in self.modules():
            if isinstance(m, NModule):
                m.push_S_device()
                device = m.op.weight.device
            if isinstance(m, NAct):
                m.push_S_device(device)




class MyNetwork(MyModule, NModel):

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    def init_model(self, model_init, init_div_groups=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_parameters(self, keys=None, mode='include'):
        if keys is None:
            for name, param in self.named_parameters():
                yield param
        elif mode == 'include':
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag:
                    yield param
        elif mode == 'exclude':
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag:
                    yield param
        else:
            raise ValueError('do not support: %s' % mode)

    def weight_parameters(self):
        return self.get_parameters()
