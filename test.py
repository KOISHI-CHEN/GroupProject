from torchvision import datasets,transforms,utils
from torch.utils import data
import torch
import torchvision
from Network import SegmentationNetwork
train_data_test = datasets.ImageFolder('./Data',transform=torchvision.transforms.ToTensor())

train_loader = data.DataLoader(train_data_test,batch_size=1)
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model = SegmentationNetwork()
model.to(device)
for img,label in (train_loader):
    img = img.to(device)
    out = model(img)
    break
    # if i_batch==0:
    #     print(i_batch)
    #     print(img)
    #     print(label)
    # break

# print('img')
# print(img)
# print('label')
# print(label)
