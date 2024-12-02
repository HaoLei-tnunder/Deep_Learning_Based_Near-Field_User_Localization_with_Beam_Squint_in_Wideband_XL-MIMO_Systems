from modelss.mrdn.subNets import *
from modelss.mrdn.cbam import *
import numpy as np
import torch.nn.functional as F
from modelss.mrdn.ASPP import *
from modelss.CHISEL.Encoder import *

class CHISEL(nn.Module):
    def __init__(self, input_channel, layersize=6, numoffilters=60, t = 1):
        super(CHISEL, self).__init__()


        self.layersize = layersize
        self.numofkernels = numoffilters*t

        self.encoder = encoder(input_channel, self.layersize, self.numofkernels)
        self.decoder = decoder(input_channel, self.layersize, self.numofkernels)
        self.Classifier = Classifier(input_channel, self.layersize, self.numofkernels)

        self.flatten = nn.Flatten()  # 展平层
        self.fc1 = nn.Linear(32 * 16 * 16 * t, 512  )  # 第一个全连接层
        self.ln1 = nn.LayerNorm(512)  # 层归一化
        self.fc2 = nn.Linear(512, 2)  # 第二个全连接层



    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.Classifier(out)

        out = self.flatten(out)  # 展平输入

        out = self.fc1(out)  # 第一层全连接层
        out = self.ln1(out)
        out = self.fc2(out)  # 第二层全连接层
        return out

