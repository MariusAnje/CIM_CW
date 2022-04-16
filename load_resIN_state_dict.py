import torch
import torchvision
import qvgg
import qresnetIN
model_ori = torchvision.models.resnet18(pretrained=True)
model_new = qresnetIN.resnet18(num_classes=1000)
# model_ori.eval()
# model_new.eval()
new_state_dict = model_new.state_dict()
old_state_dict = model_ori.state_dict()
for key in old_state_dict:
    sp_key = key.split(".")
    if len(sp_key) == 2:
        new_key = sp_key[0] + ".op." + sp_key[1]
    elif len(sp_key) == 4:
        new_key = sp_key[0] + "." + sp_key[1] + "." + sp_key[2] + ".op." + sp_key[3]
    elif len(sp_key) == 5:
        new_key = sp_key[0] + "." + sp_key[1] + "." + sp_key[2] + "." + sp_key[3] + ".op." + sp_key[4]
    else:
        print(key)
    new_state_dict[new_key] = old_state_dict[key]
model_new.load_state_dict(new_state_dict)
model_new.to_first_only()
model_ori.eval()
model_new.eval()
x = torch.randn(1,3,224,224)
print(model_new(x)[0,:10])
print(model_ori(x)[0,:10])