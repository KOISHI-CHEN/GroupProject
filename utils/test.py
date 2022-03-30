from torchvision import datasets,transforms,utils
from torch.utils import data
import torch
import torchvision
from Network import SegmentationNetwork
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from DataProcess import DCE_MRI_2d
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "2"




lr =0.001
epoches = 100
batch_size =24
test_batch_size = 2

def MI_loss(local_feature, recover_feature):
    product = torch.mul(local_feature,recover_feature)
    product = torch.sum(product,dim=1)
    temp_loss = -F.softplus(-product)-F.softplus(product)
    temp_loss = torch.sum(temp_loss,dim=0)/batch_size
    length = len(temp_loss[0])
    loss = torch.sum(temp_loss)
    loss = -loss/(length*length)
    return loss
    # print(product.size())



data_set = DCE_MRI_2d()
length=len(data_set)
print(length)
train_size=int(0.9*length)
validate_size = length - train_size
train_set,validate_set=torch.utils.data.random_split(data_set,[train_size,validate_size])
train_loader = data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader = data.DataLoader(validate_set,batch_size=test_batch_size,shuffle=True)

device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model = SegmentationNetwork()
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

num = 0
losses = []
test_case_id =0
for epoch in range(epoches):
    for batch in (train_loader):
        img = batch['image']
        img = img.unsqueeze(1)
        true_masks = batch['mask']
        num+=1
        model.train()
        img = img.to(device)
        local_feature,global_feature_one,global_feature_two = model(img)
        # print(local_feature.size())
        # print(global_feature_one.size())
        # print(global_feature_two.size())

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

        # NOTE:check correct ---- True if all 1 else False
        # print(class_one_possibility+class_two_possibility)

        feature_recover = torch.add(torch.mul(class_one_possibility,feature_one),torch.mul(class_two_possibility,feature_two))
        # print(feature_recover.size())

        loss = MI_loss(local_feature,feature_recover)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(num)

    model.eval()
    for test_batch in (test_loader):
        test_img_origin = test_batch['image']
        test_img = test_img_origin.unsqueeze(1)
        test_true_masks = test_batch['mask']
        test_img = test_img.to(device)
        test_local_feature,test_global_feature_one,test_global_feature_two = model(test_img)

        test_feature_one = test_global_feature_one
        test_feature_one = test_feature_one.unsqueeze(2)
        test_feature_one = test_feature_one.unsqueeze(3)

        test_feature_two = test_global_feature_two
        test_feature_two = test_feature_two.unsqueeze(2)
        test_feature_two = test_feature_two.unsqueeze(3)

        test_dot_sum_one = torch.mul(test_local_feature, test_feature_one)
        test_dot_sum_one = torch.sum(test_dot_sum_one,dim=1)
        test_dot_sum_two = torch.mul(test_local_feature, test_feature_two)
        test_dot_sum_two = torch.sum(test_dot_sum_two,dim=1)

        test_stack_dot_sum = torch.stack([test_dot_sum_one, test_dot_sum_two], axis=1)
        test_result = F.softmax(test_stack_dot_sum,1)

        test_class_one_possibility = test_result[:,0]
        test_class_two_possibility = test_result[:,1]

        # print("====================================================")
        # print(test_class_one_possibility)
        
        mask = test_class_one_possibility > test_class_two_possibility
        mask = mask.unsqueeze(1)
        mask = mask*255.0
        mask = nn.Upsample(scale_factor=4,mode='bilinear')(mask)
        # mask = mask >127.0
        mask = mask[0,:]
        test_true_masks = test_true_masks[0].unsqueeze(0)
        # print(test_true_masks.size())
        # print(mask.size())
        # print(mask)
        toPIL = transforms.ToPILImage()
        predict_mask = toPIL(mask)
        true_mask = toPIL(test_true_masks)
        # predict_image = Image.fromarray((mask * 255).astype(np.uint8))
        print(test_img_origin.size())

        origin_img = test_img_origin[0].unsqueeze(0)
        print(origin_img.size())
        origin_img = toPIL(origin_img)
        origin_img.save('checkpoint/original/original'+str(test_case_id)+'.jpg')
        predict_mask.save('checkpoint/predict/predict'+str(test_case_id)+'.jpg')
        true_mask.save('checkpoint/true/true_mask'+str(test_case_id)+'.jpg')
        # print(mask)

        test_case_id += 1
        break
        # print(test_class_one_possibility.size())

        # class_one_possibility = class_one_possibility.unsqueeze(1)
        # class_two_possibility = class_two_possibility.unsqueeze(1)





        # print(class_one_possibility.size())
        # recover_feature = torch.mul(class_one_possibility,global_feature_one)

        # print(class_one_possibility.size())

        # break
        # = global_feature_one.unsqueeze(2)
        # b = global_feature_two.unsqueeze(3)
        # if i_batch==0:
        #     print(i_batch)
        #     print(img)
        #     print(label)
        # break

