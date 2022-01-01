import os
import torch
import torch.nn as nn
import torch.nn.functional as F

input_channel = 3
input_size = 320
out_put_size_A = 78

hidden_layer_size_B = 38
P = 256  #1024 in that paper
size = 1


class SegmentationNetwork(nn.Module):
    def __init__(self):
        super(SegmentationNetwork,self).__init__()

        self.convA1 = nn.Conv2d(in_channels=input_channel,out_channels=64,kernel_size=4,stride=2)
        self.bnA1 = nn.BatchNorm2d(64)
        self.convA2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2)
        self.bnA2 = nn.BatchNorm2d(128)

        self.convB1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2)
        self.bnB1 = nn.BatchNorm2d(256)
        self.fcB1 = nn.Linear(256*hidden_layer_size_B*hidden_layer_size_B,1024)
        self.bnB2 = nn.BatchNorm1d(1024)
        self.fcB3 = nn.Linear(1024,64)

        self.convLC1 = nn.Conv2d(in_channels=128,out_channels=P,kernel_size=1)
        self.bnC1 = nn.BatchNorm2d(P)
        self.convLC2 = nn.Conv2d(in_channels=P,out_channels=P,kernel_size=1)
        self.convLC3 = nn.Conv2d(in_channels=128,out_channels=P,kernel_size=1)
        self.convLC4 = nn.Conv2d(in_channels=P,out_channels=P,kernel_size=1)

        self.convHC1 = nn.Conv2d(in_channels=1,out_channels=P,kernel_size=1)
        self.bnHC1 = nn.BatchNorm2d(P)
        self.convHC2 = nn.Conv2d(in_channels=P,out_channels=P,kernel_size=1)
        self.convHC3 = nn.Conv2d(in_channels=1,out_channels=P,kernel_size=1)
        self.convHC4 = nn.Conv2d(in_channels=P,out_channels=P,kernel_size=1)

        self.HD_one_fc = nn.Linear(64*P,P)
        self.bnHD_one = nn.BatchNorm1d(P)

        self.HD_two_fc = nn.Linear(64*P,P)
        self.bnHD_two = nn.BatchNorm1d(P)

    def Block_A(self,x):
        x = self.convA1(x)
        h1 = F.relu(self.bnA1(x))
        h2 = F.relu(self.bnA2(self.convA2(h1)))
        return h2

    def Block_B(self,x):
        x = self.convB1(x)
        B_h1 = F.relu(self.bnB1(x)).view(-1, int(x.nelement() / x.shape[0]))
        B_h1 = self.fcB1(B_h1)
        B_h2 = F.relu(self.bnB2(B_h1))
        return self.fcB3(B_h2)

    def Block_LC(self,x):
        LC_h1 = F.relu(self.bnC1(self.convLC1(x)))
        LC_h2 = torch.add(self.convLC2(LC_h1),self.convLC3(x))
        LC_output = self.convLC4(LC_h2)
        return LC_output

    def Block_HC(self,x):
        HC_h1 = F.relu(self.bnHC1(self.convHC1(x)))
        HC_h2 = torch.add(self.convHC2(HC_h1),self.convHC3(x))
        return self.convHC4(HC_h2)

    def Block_HD_One(self,x):
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        return F.relu(self.bnHD_one(self.HD_one_fc(x)))

    def Block_HD_Two(self,x):
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        return F.relu(self.bnHD_two(self.HD_two_fc(x)))

    def forward(self,x):
        z1 = self.Block_A(x)
        local_feature = self.Block_LC(z1)
        z2 = self.Block_B(z1)
        z2 = z2.unsqueeze(1)
        z2 = z2.unsqueeze(1)
        z3 = self.Block_HC(z2)
        global_feature_one = self.Block_HD_One(z3)
        global_feature_two = self.Block_HD_Two(z3)

        return local_feature,global_feature_one,global_feature_two

