import torch
from torch import nn
from torch import functional
from torch._C import device
from torch.nn.modules.pooling import MaxPool2d
from Functions import SLinearFunction, SConv2dFunction, SMSEFunction, SCrossEntropyLossFunction, SBatchNorm2dFunction, TimesFunction, SDropout
import numpy as np
from scipy import stats

def create_sign_map(self):
    return (torch.bernoulli(torch.ones_like(self.noise) * 0.5) - 0.5) * 2

def set_noise_old(self, dev_var, write_var, N, m):
    # N: number of bits per weight, m: number of bits per device
    # Dev_var: device variation before write and verify
    # write_var: device variation after write and verity
    scale = self.op.weight.abs().max().item()
    noise_dev = torch.zeros_like(self.noise).to(self.op.weight.device)
    noise_write = torch.zeros_like(self.noise).to(self.op.weight.device)
    for i in range(1, N//m + 1):
        if dev_var != 0:
            noise_dev   += (torch.normal(mean=0., std=dev_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
        if write_var != 0:
            noise_write += (torch.normal(mean=0., std=write_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
    noise_dev = noise_dev.to(self.op.weight.device) * scale
    noise_write = noise_write.to(self.op.weight.device) * scale

    self.noise = noise_dev * self.mask + noise_write * (1 - self.mask)

def set_noise(self, dev_var, write_var, N, m):
    # N: number of bits per weight, m: number of bits per device
    # Dev_var: device variation before write and verify
    # write_var: device variation after write and verity
    scale = self.op.weight.abs().max().item()
    noise_dev = torch.zeros_like(self.noise).to(self.op.weight.device)
    noise_write = torch.zeros_like(self.noise).to(self.op.weight.device)
    new_sigma = 0
    for i in range(1, N//m + 1):
        new_sigma += pow(2, - i*m) ** 2
    new_sigma = np.sqrt(new_sigma)
    noise_dev = torch.randn_like(self.noise) * new_sigma * dev_var
    noise_write = torch.randn_like(self.noise) * new_sigma * write_var
    noise_dev = noise_dev.to(self.op.weight.device) * scale
    noise_write = noise_write.to(self.op.weight.device) * scale

    self.noise = noise_dev * self.mask + noise_write * (1 - self.mask)

def set_noise_multiple(self, noise_type, dev_var, rate_max=0, rate_zero=0, write_var=0, **kwargs):
    if   noise_type == "Gaussian":
        set_noise(self, dev_var, write_var, kwargs["N"], kwargs["m"])
    elif noise_type == "pepper":
        set_pepper(self, dev_var, rate_max)
    elif noise_type == "uni":
        set_uni(self, dev_var)
    elif noise_type == "SPU":
        set_SPU(self, rate_max, rate_zero, dev_var)
    elif noise_type == "SU":
        set_SU(self, rate_max, dev_var)
    elif noise_type == "SG":
        set_SG(self, rate_max, dev_var)
    elif noise_type == "LSG":
        set_LSG(self, rate_max, dev_var)
    elif noise_type == "FSG":
        set_FSG(self, rate_max, rate_zero, dev_var)
    elif noise_type == "CSG":
        set_CSG(self, rate_max, dev_var)
    elif noise_type == "BSG":
        set_BSG(self, rate_max, dev_var)
    elif noise_type == "BG":
        set_BG(self, rate_max, dev_var)
    elif noise_type == "FG":
        set_BG(self, rate_max, dev_var)
    elif noise_type == "TG":
        set_TG(self, rate_max, dev_var)
    elif noise_type == "LTG":
        set_TG(self, rate_max, dev_var)
    elif noise_type == "ATG":
        set_ATG(self, rate_max, dev_var)
    elif noise_type == "powerlaw":
        set_powerlaw(self, dev_var, rate_max)
    elif noise_type == "SL":
        set_SL(self, dev_var, rate_max, rate_zero)
    elif noise_type == "Four":
        set_four(self, dev_var, rate_max, kwargs["N"], kwargs["m"])
    else:
        raise NotImplementedError(f"Noise type: {noise_type} is not supported")

def set_four(self, dev_var, s_rate, N, m):
    dev_var = dev_var / np.sqrt((s_rate**2 * 0.4 + 0.6))
    dev_var_list = [1., s_rate, s_rate, 1.]
    scale = self.op.weight.abs().max().item()
    mask = ((0.25 < (self.op.weight.abs() / scale)) * ((self.op.weight.abs() / scale) < 0.75)).float()
    new_sigma = 0
    for i in range(1, N//m + 1):
        new_sigma += pow(2, - i*m) ** 2
    new_sigma = np.sqrt(new_sigma)
    noise_dev = torch.randn_like(self.noise) * new_sigma * dev_var
    noise_dev = noise_dev.to(self.op.weight.device) * scale
    self.noise = noise_dev * mask * dev_var_list[1] + noise_dev * (1-mask) * dev_var_list[0]

def set_pepper(self, dev_var, rate):

    scale = self.op.weight.abs().max().item()
    rate_mat = torch.ones_like(self.noise).to(self.op.weight.device) * rate
    sign_bit = torch.randn_like(self.noise).sign().to(self.op.weight.device)
    noise_dev = torch.bernoulli(rate_mat).to(self.op.weight.device) * sign_bit * dev_var * scale

    self.noise = noise_dev

def set_uni(self, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = (torch.rand_like(self.noise) - 0.5) * 2 * dev_var * scale

def set_SPU(self, s_rate, p_rate, dev_var):
    assert s_rate + p_rate < 1
    scale = self.op.weight.abs().max().item()
    self.noise = (torch.rand_like(self.noise) - 0.5) * 2
    rate_mat = torch.rand_like(self.noise)
    zero_mat = rate_mat < p_rate
    th_mat = rate_mat > (1 - s_rate)
    self.noise[zero_mat] = 0
    self.noise[th_mat] = self.noise[th_mat].data.sign()
    self.noise = self.noise * scale * dev_var

def set_SU(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = (torch.rand_like(self.noise) - 0.5) * 2
    rate_mat = torch.rand_like(self.noise)
    th_mat = rate_mat > (1 - s_rate)
    self.noise[th_mat] = self.noise[th_mat].data.sign()
    self.noise = self.noise * scale * dev_var

def set_powerlaw(self, dev_var, s_rate, p_rate=0.1 ):
    # here s_rate means alpha of lognormal distribution
    scale = self.op.weight.abs().max().item()
    lognorm_scale = p_rate
    np_noise = np.random.power(s_rate, self.noise.shape)
    self.noise = torch.Tensor(np_noise).to(torch.float32).to(self.noise.device) / lognorm_scale * create_sign_map(self)
    self.noise = self.noise * scale * dev_var

def set_SG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    self.noise[self.noise > s_rate] = s_rate
    # self.noise[self.noise < -s_rate] = -s_rate
    self.noise = self.noise * scale * dev_var

def set_LSG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    # self.noise[self.noise > s_rate] = s_rate
    self.noise[self.noise < -s_rate] = -s_rate
    self.noise = self.noise * scale * dev_var

def set_FSG(self, s_rate, f_rate, dev_var):
    dev_var = dev_var / np.sqrt((f_rate**2 * 0.4 + 0.6))
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    self.noise[self.noise > s_rate] = s_rate
    self.noise = self.noise * scale * dev_var

    dev_var_list = [1., f_rate, f_rate, 1.]
    mask = ((0.25 < (self.op.weight.abs() / scale)) * ((self.op.weight.abs() / scale) < 0.75)).float()
    self.noise = self.noise * mask * dev_var_list[1] + self.noise * (1-mask) * dev_var_list[0]

def set_CSG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    self.noise[self.noise > s_rate] = s_rate
    self.noise[self.noise < -s_rate] = -s_rate
    self.noise = self.noise * scale * dev_var

def set_BSG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    self.noise[self.noise > s_rate] = s_rate
    cdf = stats.norm.cdf(s_rate)
    bias = (1 - cdf) * s_rate - 1 / np.sqrt(2 * np.pi) * np.exp(0-(s_rate**2 / 2))
    self.noise = self.noise - bias
    self.noise = self.noise * scale * dev_var

def set_BG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    cdf = stats.norm.cdf(s_rate)
    bias = (1 - cdf) * s_rate - 1 / np.sqrt(2 * np.pi) * np.exp(0-(s_rate**2 / 2))
    # bias = 0
    # bias = -0.19779655740130608 # 0.5
    # bias = -0.142879 # 0.7
    # bias = -0.083 # 1.0
    # bias = -0.0085 # 2.0
    self.noise = self.noise + bias
    self.noise = self.noise * scale * dev_var

def set_FG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    noise = (stats.foldnorm.rvs(s_rate, size=self.op.weight.shape) - s_rate) * -1
    self.noise.data = torch.Tensor(noise).to(self.noise.device).data
    self.noise = self.noise * scale * dev_var


def set_SL(self, dev_var, s_rate, p_rate=0.1):
    # here s_rate means alpha of lognormal distribution
    scale = self.op.weight.abs().max().item()
    lognorm_scale = p_rate
    np_noise = np.random.power(s_rate, self.noise.shape)
    self.noise = torch.Tensor(np_noise).to(torch.float32).to(self.noise.device) / lognorm_scale * create_sign_map(self)
    self.noise[self.noise > 1] = 1
    self.noise = self.noise * scale * dev_var

def set_TG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    noise = stats.truncnorm.rvs(-np.inf, s_rate, size = self.op.weight.shape)
    self.noise.data = torch.Tensor(noise).to(self.noise.device).data
    self.noise = self.noise * scale * dev_var

def set_LTG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    noise = stats.truncnorm.rvs(-s_rate, np.inf, size = self.op.weight.shape)
    self.noise.data = torch.Tensor(noise).to(self.noise.device).data
    self.noise = self.noise * scale * dev_var

# def set_TG(self, s_rate, dev_var):
#     scale = self.op.weight.abs().max().item()
#     def oversample_Gaussian(target_size, th):
#         tmp = np.random.normal(size=int(target_size*1/th*2))
#         index = np.abs(tmp) < 1*th
#         tmp = tmp[index][:target_size]
#         return tmp
#     target_size = self.noise.shape.numel()
#     for _ in range(10):
#         sampled_Gaussian = oversample_Gaussian(target_size, s_rate)
#         if len(sampled_Gaussian) == target_size:
#             break
#         else:
#             sampled_Gaussian = oversample_Gaussian(target_size, s_rate)
#     assert len(sampled_Gaussian) == target_size
#     self.noise = torch.Tensor(sampled_Gaussian).view(self.noise.size()).to(device=self.op.weight.device)
#     self.noise = self.noise * scale * dev_var

def set_ATG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    def oversample_Gaussian(target_size, th):
        tmp = np.random.normal(size=int(target_size*1/th*2))
        index = tmp < (1*th)
        tmp = tmp[index][:target_size]
        return tmp
    target_size = self.noise.shape.numel()
    for _ in range(10):
        sampled_Gaussian = oversample_Gaussian(target_size, s_rate)
        if len(sampled_Gaussian) == target_size:
            break
        else:
            sampled_Gaussian = oversample_Gaussian(target_size, s_rate)
    assert len(sampled_Gaussian) == target_size
    self.noise = torch.Tensor(sampled_Gaussian).view(self.noise.size()).to(device=self.op.weight.device)
    self.noise = self.noise * scale * dev_var

class SModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def create_helper(self):
        self.weightS = nn.Parameter(torch.ones(self.op.weight.size()).requires_grad_())
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.original_w = None
        self.original_b = None
        self.scale = 1.0
    
    def set_noise(self, dev_var, write_var, N, m):
        # N: number of bits per weight, m: number of bits per device
        # Dev_var: device variation before write and verify
        # write_var: device variation after write and verity
        scale = self.op.weight.abs().max().item()
        noise_dev = torch.zeros_like(self.noise).to(self.op.weight.device)
        noise_write = torch.zeros_like(self.noise).to(self.op.weight.device)
        for i in range(1, N//m + 1):
            if dev_var != 0:
                noise_dev   += (torch.normal(mean=0., std=dev_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
            if write_var != 0:
                noise_write += (torch.normal(mean=0., std=write_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
        noise_dev = noise_dev.to(self.op.weight.device) * scale
        noise_write = noise_write.to(self.op.weight.device) * scale

        self.noise = noise_dev * self.mask + noise_write * (1 - self.mask)
    
    def set_noise_multiple(self, noise_type, dev_var, rate_max=0, rate_zero=0, write_var=0, **kwargs):
        set_noise_multiple(self, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
    
    def set_pepper(self, dev_var, rate):

        scale = self.op.weight.abs().max().item()
        rate_mat = torch.ones_like(self.noise).to(self.op.weight.device) * rate
        sign_bit = torch.randn_like(self.noise).sign().to(self.op.weight.device)
        noise_dev = torch.bernoulli(rate_mat).to(self.op.weight.device) * sign_bit * dev_var * scale

        self.noise = noise_dev
    
    def set_uni(self, dev_var):
        scale = self.op.weight.abs().max().item()
        self.noise = (torch.rand_like(self.noise) - 0.5) * 2 * dev_var * scale
    
    def set_SPU(self, s_rate, p_rate, dev_var):
        assert s_rate + p_rate < 1
        scale = self.op.weight.abs().max().item()
        self.noise = (torch.rand_like(self.noise) - 0.5) * 2
        rate_mat = torch.rand_like(self.noise)
        zero_mat = rate_mat < p_rate
        th_mat = rate_mat > (1 - s_rate)
        self.noise[zero_mat] = 0
        self.noise[th_mat].data = self.noise[th_mat].data.sign()
        self.noise = self.noise * scale * dev_var
    
    def set_SG(self, s_rate, dev_var):
        scale = self.op.weight.abs().max().item()
        self.noise = torch.randn_like(self.noise)
        rate_mat = torch.rand_like(self.noise)
        th_mat = rate_mat < s_rate
        self.noise[th_mat] = self.noise[th_mat].data.sign() * 3
        self.noise = self.noise * scale * dev_var
    
    def set_add(self, dev_var, write_var, N, m):
        # N: number of bits per weight, m: number of bits per device
        # Dev_var: device variation before write and verify
        # write_var: device variation after write and verity
        scale = self.op.weight.abs().max()
        noise_dev = torch.ones_like(self.noise).to(self.op.weight.device) * dev_var
        noise_write = torch.ones_like(self.noise).to(self.op.weight.device) * write_var
        noise_dev = noise_dev.to(self.op.weight.device) * scale
        noise_write = noise_write.to(self.op.weight.device) * scale

        self.noise = noise_dev * self.mask + noise_write * (1 - self.mask)
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def mask_indicator(self, method, alpha=None):
        if method == "second":
            return self.weightS.grad.data.abs()
        if method == "magnitude":
            # return 1 / (self.op.weight.data.abs() + 1e-8)
            return self.op.weight.data.abs()
        if method == "saliency":
            if alpha is None:
                alpha = 2
            return self.weightS.grad.data.abs() * (self.op.weight.data ** alpha).abs()
        if method == "r_saliency":
            if alpha is None:
                alpha = 2
            return self.weightS.grad.abs() * self.op.weight.abs().max() / (self.op.weight.data ** alpha + 1e-8).abs()
        if method == "subtract":
            return self.weightS.grad.data.abs() - alpha * self.weightS.grad.data.abs() * (self.op.weight.data ** 2)
        if method == "SM":
            # return self.weightS.grad.data.abs() * alpha - self.op.weight.data.abs()
            return self.weightS.grad.data.abs() * self.op.weight.abs().max() * alpha + self.op.weight.data.abs()
        else:
            raise NotImplementedError(f"method {method} not supported")
    
    def get_mask_info(self):
        total = (self.mask != 10).sum()
        RM = (self.mask == 0).sum()
        return total, RM

    def set_mask(self, portion, mode):
        if mode == "portion":
            th = self.weightS.grad.abs().view(-1).quantile(1-portion)
            mask = (self.weightS.grad.data.abs() <= th).to(torch.float)
        elif mode == "th":
            mask = (self.weightS.grad.data.abs() <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
        self.mask = self.mask * mask
    
    def set_mask_mag(self, portion, mode):
        if mode == "portion":
            th = self.op.weight.abs().view(-1).quantile(1-portion)
            self.mask = (self.op.weight.data.abs() <= th).to(torch.float)
        elif mode == "th":
            self.mask = (self.op.weight.data.abs() <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
    
    def set_mask_sail(self, portion, mode, method, alpha=None):
        if mode == "random":
            size = len(self.mask.view(-1))
            self.mask = torch.Tensor(np.random.binomial(1,1-portion,size)).to(self.mask.dtype).to(self.mask.device).view(self.mask.shape)
        else:
            saliency = self.mask_indicator(method, alpha)
            if mode == "portion":
                th = saliency.view(-1).quantile(1-portion)
                mask = (saliency <= th).to(torch.float)
            elif mode == "th":
                mask = (saliency <= portion).to(torch.float)                
            else:
                raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
            self.mask = self.mask * mask
    
    def clear_mask(self):
        self.mask = torch.ones_like(self.op.weight)

    def push_S_device(self):
        # self.weightS = self.weightS.to(self.op.weight.device)
        self.mask = self.mask.to(self.op.weight.device)
        self.noise = self.noise.to(self.op.weight.device)
        try:
            self.input_range = self.input_range.to(self.op.weight.device)
        except:
            pass

    def clear_S_grad(self):
        with torch.no_grad():
            if self.weightS.grad is not None:
                self.weightS.grad.data *= 0
    
    def fetch_S_grad(self):
        return (self.weightS.grad.abs() * self.mask).sum()
    
    def fetch_S_grad_list(self):
        return (self.weightS.grad.data * self.mask)

    def do_second(self):
        self.op.weight.grad.data = self.op.weight.grad.data / (self.weightS.grad.data + 1e-10)
    
    def normalize(self):
        if self.original_w is None:
            self.original_w = self.op.weight.data
        if (self.original_b is None) and (self.op.bias is not None):
            self.original_b = self.op.bias.data
        scale = self.op.weight.data.abs().max().item()
        self.scale = scale
        self.op.weight.data = self.op.weight.data / scale
        # if self.op.bias is not None:
        #     self.op.bias.data = self.op.bias.data / scale

class SLinear(SModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.create_helper()
        self.function = SLinearFunction.apply
        self.times_function = TimesFunction.apply
    
    def copy_N(self):
        new = NLinear(self.op.in_features, self.op.out_features, False if self.op.bias is None else True)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        new.scale = self.scale
        new.original_w = self.original_w
        new.original_b = self.original_b
        return new

    def forward(self, xC):
        x, xS = xC
        # x, xS = self.function(x * self.scale, xS * self.scale, self.op.weight + self.noise, self.weightS)
        x, xS = self.function(x, xS, self.op.weight + self.noise, self.weightS)
        x, xS = self.times_function(x, xS, self.scale)
        if self.op.bias is not None:
            x += self.op.bias
        if self.op.bias is not None:
            xS += self.op.bias
        return x, xS

class SConv2d(SModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = SConv2dFunction.apply
        self.times_function = TimesFunction.apply

    def copy_N(self):
        new = NConv2d(self.op.in_channels, self.op.out_channels, self.op.kernel_size, self.op.stride, self.op.padding, self.op.dilation, self.op.groups, False if self.op.bias is None else True, self.op.padding_mode)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        new.scale = self.scale
        new.original_w = self.original_w
        new.original_b = self.original_b
        return new

    def forward(self, xC):
        x, xS = xC
        # x, xS = self.function(x * self.scale, xS * self.scale, self.op.weight + self.noise, self.weightS, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        x, xS = self.function(x, xS, self.op.weight + self.noise, self.weightS, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        x, xS = self.times_function(x, xS, self.scale)
        if self.op.bias is not None:
            x += self.op.bias.reshape(1,-1,1,1).expand_as(x)
        if self.op.bias is not None:
            xS += self.op.bias.reshape(1,-1,1,1).expand_as(xS)
        return x, xS

class NModule(nn.Module):
    def set_noise(self, dev_var, write_var, N, m):
        # N: number of bits per weight, m: number of bits per device
        # Dev_var: device variation before write and verify
        # write_var: device variation after write and verity
        scale = self.op.weight.abs().max().item()
        noise_dev = torch.zeros_like(self.noise).to(self.op.weight.device)
        noise_write = torch.zeros_like(self.noise).to(self.op.weight.device)
        for i in range(1, N//m + 1):
            if dev_var != 0:
                noise_dev   += (torch.normal(mean=0., std=dev_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
            if write_var != 0:
                noise_write += (torch.normal(mean=0., std=write_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
        noise_dev = noise_dev.to(self.op.weight.device) * scale
        noise_write = noise_write.to(self.op.weight.device) * scale

        self.noise = noise_dev * self.mask + noise_write * (1 - self.mask)
    
    def set_noise_multiple(self, noise_type, dev_var, rate_max=0, rate_zero=0, write_var=0, **kwargs):
        set_noise_multiple(self, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
    
    def set_pepper(self, dev_var, rate):

        scale = self.op.weight.abs().max().item()
        rate_mat = torch.ones_like(self.noise).to(self.op.weight.device) * rate
        sign_bit = torch.randn_like(self.noise).sign().to(self.op.weight.device)
        noise_dev = torch.bernoulli(rate_mat).to(self.op.weight.device) * sign_bit * dev_var * scale

        self.noise = noise_dev
    
    def set_uni(self, dev_var):
        scale = self.op.weight.abs().max().item()
        self.noise = (torch.rand_like(self.noise) - 0.5) * 2 * dev_var * scale
    
    def set_SPU(self, s_rate, p_rate, dev_var):
        assert s_rate + p_rate < 1
        scale = self.op.weight.abs().max().item()
        self.noise = (torch.rand_like(self.noise) - 0.5) * 2
        rate_mat = torch.rand_like(self.noise)
        zero_mat = rate_mat < p_rate
        th_mat = rate_mat > (1 - s_rate)
        self.noise[zero_mat] = 0
        self.noise[th_mat] = self.noise[th_mat].data.sign()
        self.noise = self.noise * scale * dev_var
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def clear_mask(self):
        self.mask = torch.ones_like(self.op.weight)
    
    def push_S_device(self):
        self.mask = self.mask.to(self.op.weight.device)
        self.noise = self.noise.to(self.op.weight.device)
        try:
            self.input_range = self.input_range.to(self.op.weight.device)
        except:
            pass
    
    def normalize(self):
        if self.original_w is None:
            self.original_w = self.op.weight.data
        if (self.original_b is None) and (self.op.bias is not None):
            self.original_b = self.op.bias.data
        scale = self.op.weight.data.abs().max().item()
        self.scale = scale
        self.op.weight.data = self.op.weight.data / scale

class NLinear(NModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.function = nn.functional.linear
        self.scale = 1.0

    def copy_S(self):
        new = SLinear(self.op.in_features, self.op.out_features, False if self.op.bias is None else True)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        new.scale = self.scale
        return new

    def forward(self, x):
        
        x = self.function(x, self.op.weight + self.noise, None)
        x = x * self.scale
        if self.op.bias is not None:
            x += self.op.bias
        return x

class NConv2d(NModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.function = nn.functional.conv2d
        self.scale = 1.0
    
    def copy_S(self):
        new = SConv2d(self.op.in_channels, self.op.out_channels, self.op.kernel_size, self.op.stride, self.op.padding, self.op.dilation, self.op.groups, False if self.op.bias is None else True, self.op.padding_mode)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        new.scale = self.scale
        return new

    def forward(self, x):
        x = self.function(x, self.op.weight + self.noise, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        x = x * self.scale
        if self.op.bias is not None:
            x += self.op.bias.reshape(1,-1,1,1).expand_as(x)
        return x

class SReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.op = nn.ReLU(inplace)
    
    def forward(self, xC):
        x, xS = xC
        with torch.no_grad():
            mask = (x > 0).to(torch.float)
        return self.op(x), xS * mask

class SMaxpool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        return_indices = True
        self.op = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def parse_indice(self, indice):
        bs, ch, w, h = indice.shape
        length = w * h
        BD = torch.LongTensor(list(range(bs))).expand([length,ch,bs]).swapaxes(0,2).reshape(-1)
        CD = torch.LongTensor(list(range(ch))).expand([bs,length,ch]).swapaxes(1,2).reshape(-1)
        shape = [bs, ch, -1]
        return shape, [BD, CD, indice.view(-1)]

    
    def forward(self, xC):
        x, xS = xC
        x, indices = self.op(x)
        shape, indices = self.parse_indice(indices)
        xS = xS.view(shape)[indices].view(x.shape)
        return x, xS

class SAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.op = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, xC):
        x, xS = xC
        return self.op(x), self.op(xS)

class SAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super().__init__()
        self.op = nn.AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, xC):
        x, xS = xC
        return self.op(x), self.op(xS)

class SBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.op = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.function = SBatchNorm2dFunction.apply
    
    def forward(self, xC):
        x, xS = xC
        x, xS = self.function(x, xS, self.op.running_mean, self.op.running_var, self.op.weight, self.op.bias, self.op.training, self.op.momentum, self.op.eps)
        return x, xS

class SAct(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.noise = torch.zeros(self.size)
        self.mask = torch.ones(self.size)
    
    def copy_N(self):
        new = NAct(self.size)
        new.noise = self.noise
        new.mask = self.mask
        return new
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.noise)
    
    def clear_mask(self):
        self.mask = torch.ones_like(self.mask)
    
    def set_noise(self, var):
        self.noise = torch.randn_like(self.noise) * var
        
    def push_S_device(self, device):
        self.mask = self.mask.to(device)
        self.noise = self.noise.to(device)
        # self.input_range = self.input_range.to(self.op.weight.device)

    def forward(self, xC):
        x, xS = xC
        x = (x + self.noise) * self.mask
        xS = xS * self.mask
        return x, xS

class NAct(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.noise = torch.zeros(self.size)
        self.mask = torch.ones(self.size)
    
    def copy_S(self):
        new = SAct(self.size)
        new.noise = self.noise
        new.mask = self.mask
        return new
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.noise)
    
    def clear_mask(self):
        self.mask = torch.ones_like(self.mask)
    
    def set_noise(self, var):
        self.noise = torch.randn_like(self.noise) * var
        
    def push_S_device(self, device):
        self.mask = self.mask.to(device)
        self.noise = self.noise.to(device)
        # self.input_range = self.input_range.to(self.op.weight.device)

    def forward(self, x):
        return x + self.noise * self.mask

class FakeSModule(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op
        if isinstance(self.op, nn.MaxPool2d):
            self.op.return_indices = False
    
    def forward(self, xC):
        x, xS = xC
        x = self.op(x)
        return x, None



class NModel(nn.Module):
    def __init__(self):
        super().__init__()

    def select_drop(self, p):
        for mo in self.modules():
            if isinstance(mo, NFixedDropout) or isinstance(mo, SFixedDropout):
                mo.select(p)
    
    def de_select_drop(self):
        for mo in self.modules():
            if isinstance(mo, NFixedDropout) or isinstance(mo, SFixedDropout):
                mo.de_select()
    
    def set_noise(self, dev_var, write_var, N=8, m=1):
        for mo in self.modules():
            if isinstance(mo, NModule) or isinstance(mo, SModule):
                mo.set_noise(dev_var, write_var, N, m)
    
    def set_noise_multiple(self, noise_type, dev_var, rate_max=0, rate_zero=0, write_var=0, **kwargs):
        for mo in self.modules():
            if isinstance(mo, NModule) or isinstance(mo, SModule):
                mo.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)

    def set_pepper(self, dev_var, rate):
        for mo in self.modules():
            if isinstance(mo, NModule) or isinstance(mo, SModule):
                mo.set_pepper(dev_var, rate)
    
    def set_uni(self, dev_var):
        for mo in self.modules():
            if isinstance(mo, NModule) or isinstance(mo, SModule):
                mo.set_uni(dev_var)
    
    def set_SPU(self, s_rate, p_rate, dev_var):
        for mo in self.modules():
            if isinstance(mo, NModule) or isinstance(mo, SModule):
                mo.set_SPU(s_rate, p_rate, dev_var)

    def set_noise_act(self, dev_var):
        for mo in self.modules():
            if isinstance(mo, NAct) or isinstance(mo, SAct):
                mo.set_noise(dev_var)
   
    def set_add(self, dev_var, write_var, N=8, m=1):
        for mo in self.modules():
            if isinstance(mo, NModule) or isinstance(mo, SModule):
                mo.set_add(dev_var, write_var, N, m) 

    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, NModule) or isinstance(m, SModule) or isinstance(m, SAct) or isinstance(m, NAct):
                m.clear_noise()
    
    def push_S_device(self):
        for m in self.modules():
            if isinstance(m, NModule) or isinstance(m, SModule):
            # if isinstance(m, NLinear) or isinstance(m, NConv2d):
                m.push_S_device()
                device = m.op.weight.device
            if isinstance(m, SAct) or isinstance(m, NAct):
                m.push_S_device(device)
    
    def de_normalize(self):
        for mo in self.modules():
            # if isinstance(mo, SLinear) or isinstance(mo, SConv2d):
            if isinstance(mo, SModule) or isinstance(mo, NModule):
                if mo.original_w is None:
                    raise Exception("no original weight")
                else:
                    mo.scale = 1.0
                    mo.op.weight.data = mo.original_w
                    mo.original_w = None
                    if mo.original_b is not None:
                        mo.op.bias.data = mo.original_b
                        mo.original_b = None
    
    def normalize(self):
        for mo in self.modules():
            # if isinstance(mo, SLinear) or isinstance(mo, SConv2d):
            if isinstance(mo, SModule) or isinstance(mo, NModule):
                mo.normalize()

class SModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_only = False
    
    def select_drop(self, p):
        for mo in self.modules():
            if isinstance(mo, SFixedDropout) or isinstance(mo, NFixedDropout):
                mo.select(p)
    
    def de_select_drop(self):
        for mo in self.modules():
            if isinstance(mo, SFixedDropout) or isinstance(mo, NFixedDropout):
                mo.de_select()
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def unpack_flattern(self, x):
        if self.first_only:
            return x.view(-1, self.num_flat_features(x))
        else:
            x, xS = x
            x = x.view(-1, self.num_flat_features(x))
            if xS is not None:
                xS = xS.view(-1, self.num_flat_features(xS))
            return x, xS

    def push_S_device(self):
        for m in self.modules():
            if isinstance(m, SModule) or isinstance(m, NModule):
                m.push_S_device()
                device = m.op.weight.device
            if isinstance(m, SAct) or isinstance(m, NAct):
                m.push_S_device(device)

    def clear_S_grad(self):
        for m in self.modules():
            if isinstance(m, SModule):
                m.clear_S_grad()

    def do_second(self):
        for m in self.modules():
            if isinstance(m, SModule):
                m.do_second()

    def fetch_S_grad(self):
        S_grad_sum = 0
        for m in self.modules():
            if isinstance(m, SModule):
                S_grad_sum += m.fetch_S_grad()
        return S_grad_sum
    
    def calc_S_grad_th(self, quantile):
        S_grad_list = None
        for m in self.modules():
            if isinstance(m, SModule):
                if S_grad_list is None:
                    S_grad_list = m.fetch_S_grad_list().view(-1)
                else:
                    S_grad_list = torch.cat([S_grad_list, m.fetch_S_grad_list().view(-1)])
        th = torch.quantile(S_grad_list, 1-quantile)
        # print(th)
        return th
    
    def calc_sail_th(self, quantile, method, alpha=None):
        sail_list = None
        for m in self.modules():
            if isinstance(m, SModule):
                sail = m.mask_indicator(method, alpha).view(-1)
                if sail_list is None:
                    sail_list = sail
                else:
                    sail_list = torch.cat([sail_list, sail])
        if method == "second":
            import time
            torch.save(sail_list, f"S_grad_{time.time()}.pt")
        th = torch.quantile(sail_list, 1-quantile)
        # print(th)
        return th

    def set_noise(self, dev_var, write_var, N=8, m=1):
        for mo in self.modules():
            if isinstance(mo, SModule) or isinstance(mo, NModule):
                mo.set_noise(dev_var, write_var, N, m)
    
    def set_noise_multiple(self, noise_type, dev_var, rate_max=0, rate_zero=0, write_var=0, **kwargs):
        for mo in self.modules():
            if isinstance(mo, NModule) or isinstance(mo, SModule):
                mo.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)

    def set_pepper(self, dev_var, rate):
        for mo in self.modules():
            if isinstance(mo, NModule) or isinstance(mo, SModule):
                mo.set_pepper(dev_var, rate)

    def set_uni(self, dev_var):
        for mo in self.modules():
            if isinstance(mo, NModule) or isinstance(mo, SModule):
                mo.set_uni(dev_var)
    
    def set_SPU(self, s_rate, p_rate, dev_var):
        for mo in self.modules():
            if isinstance(mo, NModule) or isinstance(mo, SModule):
                mo.set_SPU(s_rate, p_rate, dev_var)
    
    def set_noise_act(self, dev_var):
        for mo in self.modules():
            if isinstance(mo, NAct) or isinstance(mo, SAct):
                mo.set_noise(dev_var)
    
    def set_add(self, dev_var, write_var, N=8, m=1):
        for mo in self.modules():
            if isinstance(mo, SModule) or isinstance(mo, NModule):
                mo.set_add(dev_var, write_var, N, m)    

    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, SModule) or isinstance(m, NModule) or isinstance(m, SAct) or isinstance(m, NAct):
                m.clear_noise()
    
    def get_mask_info(self):
        total = 0
        RM = 0
        for m in self.modules():
            if isinstance(m, SModule) or isinstance(m, NModule):
                t, r = m.get_mask_info()
                total += t
                RM += r
        return total, RM

    def set_mask(self, th, mode):
        for m in self.modules():
            if isinstance(m, SModule):
                m.set_mask(th, mode)
    
    def set_mask_mag(self, th, mode):
        for m in self.modules():
            if isinstance(m, SModule):
                m.set_mask_mag(th, mode)
    
    def set_mask_sail(self, th, mode, method, alpha=None):
        for m in self.modules():
            if isinstance(m, SModule):
                m.set_mask_sail(th, mode, method, alpha)
    
    def clear_mask(self):
        for m in self.modules():
            if isinstance(m, SModule) or isinstance(m, NModule) or isinstance(m, SAct) or isinstance(m, NAct):
                m.clear_mask()
    
    def to_fake(self, device):
        for name, m in self.named_modules():
            if isinstance(m, SModule) or isinstance(m, SMaxpool2D) or isinstance(m, SReLU):
                new = FakeSModule(m.op)
                self._modules[name] = new
        self.to(device)
    
    def to_first_only(self):
        self.first_only = True
        for m in self.modules():
            if isinstance(m, SModel):
                m.first_only = True
        for n, m in self.named_modules():
            if isinstance(m, SModule) or isinstance(m, SAct) or isinstance(m, SFixedDropout):
                n = n.split(".")
                father = self
                for i in range(len(n) - 1):
                    father = father._modules[n[i]]
                mo = father._modules[n[-1]]
                new = mo.copy_N()
                father._modules[n[-1]] = new
            if isinstance(m, SReLU) or isinstance(m, SMaxpool2D) or isinstance(m, SBatchNorm2d) or isinstance(m, SAdaptiveAvgPool2d) or isinstance(m, SAvgPool2d):
                n = n.split(".")
                father = self
                for i in range(len(n) - 1):
                    father = father._modules[n[i]]
                father._modules[n[-1]] = m.op
                if isinstance(m, SMaxpool2D):
                    father._modules[n[-1]].return_indices = False

    def from_first_back_second(self):
        if self.first_only:
            self.first_only = False
            for m in self.modules():
                if isinstance(m, SModel):
                    m.first_only = False
            for n, m in self.named_modules():
                if isinstance(m, NModule) or isinstance(m, NFixedDropout) or isinstance(m, NAct):
                    n = n.split(".")
                    father = self
                    for i in range(len(n) - 1):
                        father = father._modules[n[i]]
                    mo = father._modules[n[-1]]
                    new = mo.copy_S()
                    father._modules[n[-1]] = new
                if isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.AvgPool2d):
                    n = n.split(".")
                    father = self
                    for i in range(len(n) - 1):
                        father = father._modules[n[i]]
                    if isinstance(m, nn.ReLU):
                        new = SReLU(m.inplace)
                    elif isinstance(m, nn.MaxPool2d):
                        new = SMaxpool2D(m.kernel_size, m.stride, m.padding, m.dilation, True, m.ceil_mode)
                    elif isinstance(m, nn.AdaptiveAvgPool2d):
                        new = SAdaptiveAvgPool2d(m.output_size)
                        new.op = m
                    elif isinstance(m, nn.AvgPool2d):
                        new = SAvgPool2d(m.kernel_size, m.stride, m.padding, m.ceil_mode, m.count_include_pad, m.divisor_override)
                        new.op = m
                    elif isinstance(m, nn.BatchNorm2d):
                        new = SBatchNorm2d(m.num_features)
                        new.op = m
                    # TODO: Other modules specified above
                    father._modules[n[-1]] = new

    def normalize(self):
        for mo in self.modules():
            # if isinstance(mo, SLinear) or isinstance(mo, SConv2d):
            if isinstance(mo, SModule) or isinstance(mo, NModule):
                mo.normalize()

    def get_scale(self):
        scale = 1.0
        for m in self.modules():
            if isinstance(m, SModule):
                scale *= m.scale
        return scale

    def de_normalize(self):
        for mo in self.modules():
            # if isinstance(mo, SLinear) or isinstance(mo, SConv2d):
            if isinstance(mo, SModule) or isinstance(mo, NModule):
                if mo.original_w is None:
                    raise Exception("no original weight")
                else:
                    mo.scale = 1.0
                    mo.op.weight.data = mo.original_w
                    mo.original_w = None
                    if mo.original_b is not None:
                        mo.op.bias.data = mo.original_b
                        mo.original_b = None
    
    def fine_S_grad(self):
        for m in self.modules():
            if isinstance(m, SModule):
                m.weightS.grad.data += m.op.weight.grad.data * 2


    def back_real(self, device):
        for name, m in self.named_modules():
            if isinstance(m, FakeSModule):
                if isinstance(m.op, nn.Linear):
                    if m.op.bias is not None:
                        bias = True
                    new = SLinear(m.op.in_features, m.op.out_features, bias)
                    new.op = m.op
                    self._modules[name] = new

                elif isinstance(m.op, nn.Conv2d):
                    if m.op.bias is not None:
                        bias = True
                    new = SConv2d(m.op.in_channels, m.op.out_channels, m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.groups, bias, m.op.padding_mode)
                    new.op = m.op
                    self._modules[name] = new

                elif isinstance(m.op, nn.MaxPool2d):
                    new = SMaxpool2D(m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.return_indices, m.op.ceil_mode)
                    new.op = m.op
                    new.op.return_indices = True
                    self._modules[name] = new

                elif isinstance(m.op, nn.ReLU):
                    new = SReLU()
                    new.op = m.op
                    self._modules[name] = new

                else:
                    raise NotImplementedError
        self.to(device)

class FixedDropout(nn.Module):
    def __init__(self, shape:torch.Size):
        super().__init__()
        self.shape = shape
        self.scale = 1
        self.device = torch.device("cpu")
        self.mask = torch.ones(shape).to(self.device)
    
    def select(self, p):
        p_ext = torch.ones(self.shape) * (1 - p)
        self.mask = torch.bernoulli(p_ext).to(self.device)
        self.scale = (self.shape.numel()/self.mask.sum()).item()
    
    def de_select(self):
        self.mask = torch.ones(self.shape).to(self.device)

class SFixedDropout(FixedDropout):
    def __init__(self, shape:torch.Size):
        super().__init__(shape)
        self.function = SDropout.apply
    
    def copy_N(self):
        new = NFixedDropout(self.shape)
        new.scale = self.scale
        new.mask = self.mask
        new.device = self.device
        return new
    
    def forward(self, xC):
        x, xS = xC
        x, xS = self.function(x, xS, self.mask, self.scale)
        return x, xS

class NFixedDropout(FixedDropout):
    def __init__(self, shape:torch.Size):
        super().__init__(shape)
    
    def copy_S(self):
        new = SFixedDropout(self.shape)
        new.scale = self.scale
        new.mask = self.mask
        new.device = self.device
        return new
    
    def forward(self, x):
        # ext_mask = self.mask.expand_as(x)
        return x * self.mask * self.scale
