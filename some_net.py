import torch
import cv2
import numpy as np
import torchvision
import torch.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from tensorboardX import SummaryWriter
import time

class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, np.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(200, 400, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(400)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p = 0.3)
        self.conv2 = nn.Conv2d(400, 600, kernel_size = 1)
        self.bn2 = nn.BatchNorm2d(600)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p = 0.3)
        self.conv3 = nn.Conv2d(600, 800, kernel_size = 1)
        self.bn3 = nn.BatchNorm2d(800)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(p = 0.3)

        self.conv4 = nn.Conv2d(800, 1000, kernel_size = 1)
        self.bn4 = nn.BatchNorm2d(1000)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(p = 0.3)
        #self.conv_fin = nn.Conv2d(800, 8, kernel_size = 1)
        self.conv_fin = nn.Linear(1000, 100)
        self.bn5 = nn.BatchNorm1d(100)

        self.conv_fin2 = nn.Linear(100, 8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        #print(x.shape)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.maxpool1(x)
        #print(x.shape)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = self.maxpool2(x)
        #print(x.shape)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        x = self.maxpool3(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.drop4(x)
        x = torch.squeeze(x)

        x = self.conv_fin(x)
        x = self.relu4(self.bn5(x))
        x = self.conv_fin2(x)
        return x

def fire_layer(inp, s, e):
    f = fire(inp, s, e)
    return f

def squeezenet(pretrained=False):
    net = Net()
    # inp = Variable(torch.randn(64,3,32,32))
    # out = net.forward(inp)
    # print(out.size())
    return net

log_path = r'.\logs\\'
writer = SummaryWriter(log_path)
