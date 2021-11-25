import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.nn import Parameter
import os
import numpy as np
from utils import accuracy
import torchvision.models as models
import math



class PU_1(nn.Module):
    '''
    purification unit 1.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.seq = nn.Sequential(
                                 nn.BatchNorm1d(n_in),
                                 nn.Linear(n_in, n_in),
                                 nn.ReLU(inplace=True)
                                )
    def forward(self, xs):
        ys = xs - self.seq(xs)
        return ys



class PU_2(nn.Module):
    '''
    purification unit 2.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.seq = nn.Sequential(
                                 nn.BatchNorm1d(n_in),
                                 nn.Linear(n_in, n_in),
                                 nn.ReLU(inplace=True)
                                )
    def forward(self, xs):
        ys = xs + self.seq(xs)
        return ys



class PU_3(nn.Module):
    '''
    purification unit 3.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.seq = nn.Sequential(
                                 nn.BatchNorm1d(n_in),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(n_in, n_in),
                                )
    def forward(self, xs):
        ys = xs - self.seq(xs)
        return ys




class PU_3_time2(nn.Module):
    '''
    purification unit 3.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.seq = nn.Sequential(
                                 nn.BatchNorm1d(n_in),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(n_in, n_in),
                                 nn.BatchNorm1d(n_in),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(n_in, n_in),
                                )
    def forward(self, xs):
        ys = xs - self.seq(xs)
        return ys



    
class PU_4(nn.Module):
    '''
    purification unit 4.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.seq = nn.Sequential(
                                 nn.BatchNorm1d(n_in),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(n_in, n_in),
                                )
    def forward(self, xs):
        ys = xs + self.seq(xs)
        return ys


class PU_5(nn.Module):
    '''
    purification unit 5.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.seq = nn.Sequential(
                                 nn.BatchNorm1d(n_in),
                                 nn.Linear(n_in, n_in),
                                 nn.ReLU(inplace=True)
                                )
    def forward(self, xs):
        ys = self.seq(xs)
        return ys



class PU_6(nn.Module):
    '''
    purification unit 6.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.seq = nn.Sequential(
                                 nn.BatchNorm1d(n_in),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(n_in, n_in)
                                )
    def forward(self, xs):
        ys = self.seq(xs)
        return ys



class PU_7(nn.Module):
    '''
    purification unit 7.
    '''
    def __init__(self, n_in):
        super().__init__()
    def forward(self, xs):
        ys = xs * 1.0
        return ys
