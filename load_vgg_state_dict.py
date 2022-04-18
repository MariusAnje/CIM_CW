import torch
import torchvision
import qvgg
model_ori = torchvision.models.vgg16(pretrained=True)
model_new = qvgg.vgg16(num_classes=1000)
model_ori.eval()
model_new.eval()
new_state_dict = model_new.state_dict()
old_state_dict = model_ori.state_dict()
for key in old_state_dict:
    sp_key = key.split(".")
    new_key = sp_key[0] + "." + sp_key[1] + "." + "op." + sp_key[2]
    new_state_dict[new_key] = old_state_dict[key]
model_new.load_state_dict(new_state_dict)
x = torch.randn(1,3,224,224)
print(model_new(x)[0][0,:10])
print(model_ori(x)[0,:10])