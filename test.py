from torchvision import datasets,transforms,utils
from torch.utils import data
import torch
import torchvision
from Network import SegmentationNetwork
import torch.nn.functional as F
train_data_test = datasets.ImageFolder('./Data',transform=torchvision.transforms.ToTensor())

train_loader = data.DataLoader(train_data_test,batch_size=2)
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model = SegmentationNetwork()
model.to(device)

for img,label in (train_loader):
    img = img.to(device)
    local_feature,global_feature_one,global_feature_two = model(img)
    print(local_feature.size())
    print(global_feature_one.size())
    print(global_feature_two.size())

    feature_one = global_feature_one
    feature_one = feature_one.unsqueeze(2)
    feature_one = feature_one.unsqueeze(3)

    feature_two = global_feature_two
    feature_two = feature_two.unsqueeze(2)
    feature_two = feature_two.unsqueeze(3)

    dot_sum_one = torch.mul(local_feature, feature_one)
    dot_sum_one = torch.sum(dot_sum_one,dim=1)
    dot_sum_two = torch.mul(local_feature, feature_two)
    dot_sum_two = torch.sum(dot_sum_two,dim=1)

    stack_dot_sum = torch.stack([dot_sum_one, dot_sum_two], axis=1)
    result = F.softmax(stack_dot_sum,1)

    class_one_possibility = result[:,0]
    class_two_possibility = result[:,1]

    class_one_possibility = class_one_possibility.unsqueeze(1)
    class_two_possibility = class_two_possibility.unsqueeze(1)
    feature_recover = torch.add(torch.mul(class_one_possibility,feature_one),torch.mul(class_two_possibility,feature_two))
    print(feature_recover.size())


    # print(class_one_possibility.size())
    # recover_feature = torch.mul(class_one_possibility,global_feature_one)

    # print(class_one_possibility.size())
    # NOTE:check correct ---- True if all 1 else False
    # print(class_one_possibility+class_two_possibility)

    break
     # = global_feature_one.unsqueeze(2)
    # b = global_feature_two.unsqueeze(3)
    # if i_batch==0:
    #     print(i_batch)
    #     print(img)
    #     print(label)
    break

# print('img')
# print(img)
# print('label')
# print(label)
