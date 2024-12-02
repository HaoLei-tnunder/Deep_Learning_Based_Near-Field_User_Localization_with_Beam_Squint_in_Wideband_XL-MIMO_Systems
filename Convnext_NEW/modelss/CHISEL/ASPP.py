import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ASPP(nn.Module):
    def __init__(self, in_channel):
        super(ASPP,self).__init__()        
        depth = in_channel
        self.mean = nn.AdaptiveAvgPool2d(1)       # 自适应均值池化
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        # self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 4, depth, 1, 1)


    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        # atrous_block18 = self.atrous_block18(x)
 
        # cat = torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1)
        cat = torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12], dim=1)
        out = self.conv_1x1_output(cat)
        return out




class ASPP1(nn.Module):
    def __init__(self, in_channel):
        super(ASPP1, self).__init__()
        depth = in_channel
        self.mean = nn.AdaptiveAvgPool2d(1)  # 自适应均值池化
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        # self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth *4, 1, 1, 1,padding=depth)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        # atrous_block18 = self.atrous_block18(x)

        # cat = torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1)
        cat = torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12], dim=1)
        out = self.conv_1x1_output(cat)
        return out


class Net(nn.Module):
    def __init__(self,in_channel):
        super(Net, self).__init__()
        self.numofkernels = in_channel
        self.conv1 = nn.Conv2d(self.numofkernels, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        # print(np.shape(x))
        x = x.view(x.size(0), -1)
        # print(np.shape(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

