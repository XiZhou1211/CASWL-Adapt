import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from meta_net import *

class FeatureExtracter(nn.Module):
    def __init__(self, N_channels):
        super(FeatureExtracter, self).__init__()
        self.conv1 = nn.Conv1d(N_channels, 128, kernel_size=8, stride=2, bias=False)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=2, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward( self, x):
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x)

        x = x.reshape(x.shape[0], x.shape[1])
        return x   
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.L1 = nn.Linear(128, 500, bias=False)
        self.bn1 = nn.BatchNorm1d(500)

        self.L2 = nn.Linear(500, 500, bias=False)
        self.bn2 = nn.BatchNorm1d(500)

        self.out = nn.Linear(500, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.L1(x)))
        x = F.relu(self.bn2(self.L2(x)))

        x = self.out(x)
        return x.reshape(-1)

class ActivityClassifier(nn.Module):
    def __init__(self, N_classes):
        super(ActivityClassifier, self).__init__()
        self.out = nn.Linear(128, N_classes)
        
    def forward(self, x):
        x = self.out(x)
        return x

class ClassAwareWeightNetwork(nn.Module):
    def __init__(self, input, hidden1, hidden2, output, num_classes):
        super(ClassAwareWeightNetwork, self).__init__()
        self.feature = share(input, hidden1, hidden2)
        self.classfier = task(hidden2, output, num_classes)

    def forward(self, x, num, c):
        # num = torch.argmax(num, -1)
        output = self.classfier(self.feature(x), num, c)

        return output

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
