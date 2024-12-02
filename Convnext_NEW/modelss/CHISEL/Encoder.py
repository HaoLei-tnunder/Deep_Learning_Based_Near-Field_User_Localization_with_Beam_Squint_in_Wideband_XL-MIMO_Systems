import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, input_channel, layersize=6, numoffilters=8):
        super(encoder, self).__init__()

        # Define the first set of 24 CNN layers with 16 channels
        self.conv1 = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d( numoffilters*2 if i > 0 else input_channel, numoffilters*2, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ) for i in range(layersize*4)]
        )

        # Define the max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the second set of 12 CNN layers with 8 channels
        self.conv2 = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(numoffilters if i > 0 else numoffilters*2, numoffilters, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ) for i in range(layersize*2)]
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_channel, layersize=6, numoffilters=8):
        super(decoder, self).__init__()

        # Define the upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Define the 24 CNN layers with 1 channel

        self.conv = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(numoffilters*2 if i > 0 else numoffilters, numoffilters*2, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ) for i in range(layersize*4)]
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_channel, layersize=6, numoffilters=8):
        super(Classifier, self).__init__()

        # Define the first set of 24 CNN layers with 16 channels

        # self.conv0 = nn.Sequential(
        #     *[nn.Sequential(
        #         nn.Conv2d(numoffilters if i > 0 else 1, numoffilters, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU()
        #     ) for i in range(layersize*2)]
        # )

        self.conv1 = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(numoffilters*2 , numoffilters*2, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ) for i in range(layersize*4)]
        )

        # Define the first max-pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the second set of 12 CNN layers with 32 channels
        self.conv2 = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(numoffilters*4 if i > 0 else numoffilters*2, numoffilters*4, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ) for i in range(layersize*2)]
        )

        # Define the second max-pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        # x = self.conv0(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x



