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
import matplotlib.pyplot as plt

from model import init_weights
from model import Generator, Discriminator

data = "dataset/"

# hyperparameters
batch_size = 128
num_epochs = 5
image_size = 128
nc = 3 # num channels
nz = 100 # size of latent vector
ngf = 128 # num feature maps in generator
ndf = 128 # num feature maps in the discriminator
lr = 0.0002

dataset = torchvision.datasets.ImageFolder(
    root = data,
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True)
device = torch.device("mps")
# device = torch.device("cpu")

# setup model
generator = Generator(d_input=nz, d_features=ngf)
generator.apply(init_weights)
discriminator = Discriminator(num_channels=nc, d_features=ndf)
discriminator.apply(init_weights)

# loss function and optimizer
criterion = nn.BCELoss()
rand_noise = torch.randn(64, nz, 1, 1)

real_label = 1.
fake_label = 0.

optimizerG = optim.Adam(generator.parameters(), lr = lr)
optimizerD = optim.Adam(discriminator.parameters(), lr = lr)

# training loop
imgs_generated = []
gen_losses = []
disc_losses = []
iters = 0

print("Starting Training...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # train the discriminator: try to maximize log(D(x)) + log(1-D(G(z)))
        real_img = data[0]
        label = torch.full((real_img.size(0),), real_label, dtype=torch.float)
        output = discriminator(real_img)
        dx = criterion(output, label)
        print(dx)
        break
        
    break
        
        
        
        
        
        


