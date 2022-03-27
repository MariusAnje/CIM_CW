import numpy as np
from sklearn import datasets
import torch
from matplotlib import pyplot as plt
from models import *

class SMLP1(SModel):
    def __init__(self):
        super().__init__()
        self.fc1 = SLinear(2,1)
        self.relu = SReLU()
        self.init_weight()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        xS = torch.zeros_like(x)
        if not self.first_only:
            x = (x, xS)
        x = self.fc1(x)
        return torch.cat([x.reshape(-1,1),torch.zeros_like(x).reshape(-1,1)], dim=1)

class SMLP2(SModel):
    def __init__(self):
        super().__init__()
        self.fc1 = SLinear(2,4)
        self.fc2 = SLinear(4,2)
        self.relu = SReLU()
        self.init_weight()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        xS = torch.zeros_like(x)
        if not self.first_only:
            x = (x, xS)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SMLP3(SModel):
    def __init__(self):
        super().__init__()
        self.fc1 = SLinear(2,8)
        self.fc2 = SLinear(8,8)
        self.fc3 = SLinear(8,2)
        self.relu = nn.ReLU()
        self.init_weight()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        xS = torch.zeros_like(x)
        if not self.first_only:
            x = (x, xS)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

xs = torch.linspace(6, 12, steps=100)
ys = torch.linspace(0, 10, steps=100)
x, y = torch.meshgrid(xs, ys, indexing='xy')
XX = torch.cat([x.reshape(-1,1),y.reshape(-1,1)],dim=1)
yy = (y > (x - 9) ** 2).reshape(-1).to(torch.long)

model = SMLP1()
model = SMLP2()
# model = SMLP3()
model.to_first_only()
criteria = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(),lr=1e-2, momentum=0.9, weight_decay=1e-5)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-1, weight_decay=1e-5)

state_dict = torch.load("NMLP2.pt")
model.load_state_dict(state_dict)

outputs = model(XX)
loss = criteria(outputs, yy)
acc = (outputs.argmax(dim=1) == yy).sum() / len(yy)
print(acc.item(), loss.item())

a = torch.randn(100,2,8)
scale = (a ** 2).sum(dim=-1).sum(dim=-1).sqrt()
for i in range(len(scale)):
    a[i,:,:] = a[i,:,:] / scale[i]

model.clear_noise()
pred_GT = model(XX).argmax(dim=1)

for l2 in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]:
    acc_list = []
    for i in range(len(a)):
        model.clear_noise()
        model.fc1.noise += (a[i,0,:] * l2).to(torch.float32).view(4,2)
        model.fc2.noise += (a[i,0,:] * l2).to(torch.float32).view(2,4)
        pred = model(XX).argmax(dim=1)
        acc = (pred == pred_GT).sum() / len(pred_GT)
        acc_list.append(acc.item())
    print(f"L2: {l2:.1e}, Mean: {np.mean(acc_list):.4f}, Max: {np.max(acc_list):.4f}, Min: {np.min(acc_list):.4f}")