#copied from pytorch model zoo and modified for 200 channels

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


class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.n0 = 40
        self.bn0 = nn.BatchNorm2d(200)
        self.conv1 = nn.Conv2d(200, 4*self.n0, kernel_size=1, stride=1)#, padding=1) # 32
        self.bn1 = nn.BatchNorm2d(4*self.n0)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire2 = fire(4*self.n0, 4*self.n0, 16*self.n0)
        self.fire3 = fire(32*self.n0, 4*self.n0, 16*self.n0)
        self.fire4 = fire(32*self.n0, 8*self.n0, 32*self.n0)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire5 = fire(64*self.n0, 8*self.n0, 32*self.n0)
        self.fire6 = fire(64*self.n0, 12*self.n0, 48*self.n0)
        self.fire7 = fire(96*self.n0, 12*self.n0, 48*self.n0)
        self.fire8 = fire(96*self.n0, 16*self.n0, 64*self.n0)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        # self.fire9 = fire(128*self.n0, 16*self.n0, 64*self.n0)
        # self.fire10 = fire(128*self.n0, 24*self.n0, 96*self.n0)
        # self.fire11 = fire(192*self.n0, 24*self.n0, 96*self.n0)
        # self.fire12 = fire(192*self.n0, 32*self.n0, 128*self.n0)
        # self.maxpool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(96*self.n0, 8, kernel_size=1, stride=1)
        self.avg_pool = nn.MaxPool2d(kernel_size=5)


    #    self.FC1 = nn.Linear(289, 100)
    #    self.FC2 = nn.Linear(100,1)
        #self.FC2 = nn.Linear(100, 1)
        #self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p = 0.3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        #print(x.shape)
        x = self.bn0(x)
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool1(x)
        #print(x.shape)
        x = self.fire2(x)
        #print(x.shape)
        #x = self.drop(x)
        x = self.fire3(x)
        #print(x.shape)
        x = self.fire4(x)
        #print(x.shape)

        # x = self.maxpool2(x)
        # #print(x.shape)
        #x = self.drop(x)
        x = self.fire5(x)
        # #print(x.shape)
        #
        x = self.fire6(x)
        # #print(x.shape)
        #x = self.drop(x)
        #x = self.fire7(x)
        # #print(x.shape)
        #
        #x = self.fire8(x)
        # #print(x.shape)
        #
        # x = self.maxpool3(x)
        # #print(x.shape)
        # x = self.drop(x)
        #
        # x = self.fire9(x)
        # #print(x.shape)
        # x = self.fire10(x)
        # x= self.fire11(x)
        # x = self.fire12(x)
        # x = self.maxpool4(x)

        x = self.conv2(x)
        #print(x.shape)

        #x = self.avg_pool(x)
        #print(x.shape)

        #x = x.view((batch_size, num_classes, -1))
        #print(x.shape)
        #x = self.drop(x)
        x = self.avg_pool(x)
        # x = self.relu(x)
        #
        # x = self.FC2(x)
        #x = self.softmax(x)

        return x

def fire_layer(inp, s, e):
    f = fire(inp, s, e)
    return f

def squeezenet(pretrained=False):
    net = SqueezeNet()
    # inp = Variable(torch.randn(64,3,32,32))
    # out = net.forward(inp)
    # print(out.size())
    return net

log_path = r'.\logs\\'
writer = SummaryWriter(log_path)
