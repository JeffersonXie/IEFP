import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.nn import Parameter
import os
import numpy as np
import torchvision.models as models
import math



class ArcMarginProduct(nn.Module):
    ### arcface ###
    def __init__(self, out_feature=10575, in_feature=512, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # cos(theta)
        cosine_theta = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine_theta = cosine_theta.clamp(-1,1) # for numerical stability
        # cos(theta + m)
        sine_theta = torch.sqrt(1.0 - torch.pow(cosine_theta, 2))
        cosine_theta_m = cosine_theta * self.cos_m - sine_theta * self.sin_m

        if self.easy_margin:
            cosine_theta_m = torch.where(cosine_theta > 0, cosine_theta_m, cosine_theta)
        else:
            cosine_theta_m = torch.where((cosine_theta - self.th) > 0, cosine_theta_m, cosine_theta - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * cosine_theta_m) + ((1.0 - one_hot) * cosine_theta)
        output = output * self.s

        return output



class AddMarginProduct(nn.Module):
    ### cosface ###
    def __init__(self, out_features=10575, in_features=512, s=64.0, m=0.35):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
       # print(one_hot.size())
       # os._exit(0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  
        output *= self.s

        return output


