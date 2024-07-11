import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np

# weight initialization (random from normal distribution with mean at 0 and std dev at 0.02)

def init_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, d_input, d_features):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            # input z is shape d_input
            nn.ConvTranspose2d(d_input, d_features*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d_features*16),
            nn.ReLU(True),
            # (d_features * 16, 4, 4)
            nn.ConvTranspose2d(d_features*16, d_features*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features*8),
            nn.ReLU(True),
            # (d_features * 8, 8, 8)
            nn.ConvTranspose2d(d_features*8, d_features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features*4),
            nn.ReLU(True),
            # (d_features * 4, 16, 16)
            nn.ConvTranspose2d(d_features*4, d_features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features*2),
            nn.ReLU(True),
            # (d_features * 2, 32, 32)
            nn.ConvTranspose2d(d_features*2, d_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features),
            nn.ReLU(True),
            # (d_features, 64, 64)
            nn.ConvTranspose2d(d_features, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # (num_channels (3), 128, 128)
        )

    def forward(self, x):
        return self.layers(x)
        

class Discriminator(nn.Module):
    def __init__(self, num_channels, d_features):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            # Input is (N, num_channels, 128, 128)
            nn.Conv2d(num_channels, d_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output is (N, d_features, 64, 64)
            
            nn.Conv2d(d_features, d_features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features*2),
            nn.LeakyReLU(0.2, inplace=True),
            # Output is (N, d_features*2, 32, 32)
            
            nn.Conv2d(d_features*2, d_features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features*4),
            nn.LeakyReLU(0.2, inplace=True),
            # Output is (N, d_features*4, 16, 16)
            
            nn.Conv2d(d_features*4, d_features*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features*8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output is (N, d_features*8, 4, 4)
            
            nn.Conv2d(d_features*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output is (N, 1, 1, 1)
        )

    def forward(self, input):
        return self.layers(input)
