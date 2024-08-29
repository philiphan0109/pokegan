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


def main():
    data = "data/"
    os.makedirs("artifacts/checkpoints", exist_ok=True)
    os.makedirs("artifacts/losses", exist_ok=True)

    # random seed for reproducability
    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)

    # hyperparameters
    batch_size = 128
    num_epochs = 100
    image_width = 184
    image_height = 256
    nc = 3 # num channels
    nz = 100 # size of latent vector
    ngf = 64 # num feature maps in generator
    ndf = 64 # num feature maps in the discriminator
    lr = 0.0001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torchvision.datasets.ImageFolder(
        root=data,
        transform=transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # setup model
    generator = Generator(d_input=nz, d_features=ngf, img_size=(image_height, image_width)).to(device)
    discriminator = Discriminator(num_channels=nc, d_features=ndf, img_size=(image_height, image_width)).to(device)

    # load in model
    # generator_checkpoint = torch.load('artifacts/checkpoints/generator_6159.pth')
    # discriminator_checkpoint = torch.load('artifacts/checkpoints/discriminator_6159.pth')

    # generator.load_state_dict(generator_checkpoint['model_state_dict'])
    # discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])


    # loss function and optimizer
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # optimizerG.load_state_dict(generator_checkpoint['optimizer_state_dict'])
    # optimizerD.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])

    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    # training loop
    imgs_generated = []
    gen_losses = []
    disc_losses = []
    iters = 0

    print("Starting Training...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # train the discriminator
            discriminator.zero_grad()
            real_imgs = data[0].to(device)
            label = torch.full((real_imgs.size(0),), real_label, dtype=torch.float, device=device)
            output = discriminator(real_imgs).view(-1)
            lossd_real = criterion(output, label)
            lossd_real.backward()
            
            D_x = output.mean().item()
            
            noise = torch.randn(real_imgs.size(0), nz, 1, 1, device=device)
            fake_imgs = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_imgs.detach()).view(-1)
            lossd_fake = criterion(output, label)
            lossd_fake.backward()
            
            D_G_z1 = output.mean().item()
            
            lossd = lossd_fake + lossd_real
            optimizerD.step()
            
            # train the generator
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake_imgs).view(-1)
            lossg = criterion(output, label)
            lossg.backward()
            D_G_z2 = lossg.mean().item()
            
            optimizerG.step()
            
            gen_losses.append(lossg.item())
            disc_losses.append(lossd.item())
            
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(dataloader), lossd.item(), lossg.item(), D_x, D_G_z1, D_G_z2))
            
            iters += 1
        
        # Save checkpoints at the end of each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict(),
        }, f'artifacts/checkpoints/generator_epoch_{epoch}.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
        }, f'artifacts/checkpoints/discriminator_epoch_{epoch}.pth')
        
        # Generate and save sample images at the end of each epoch
        with torch.no_grad():
            fake = generator(torch.randn(64, nz, 1, 1, device=device)).detach().cpu()
        img_grid = torchvision.utils.make_grid(fake, padding=2, normalize=True)
        imgs_generated.append(img_grid)
        
        plt.figure(figsize=(15, 15))
        plt.imshow(np.transpose(img_grid, (1, 2, 0)))
        plt.axis('off')
        plt.title(f"Generated Images - Epoch {epoch}")
        plt.savefig(f'artifacts/generated_images/generated_image_epoch_{epoch}.png')
        plt.close()

        # Save loss plot at the end of each epoch
        plt.figure(figsize=(10, 5))
        plt.plot(gen_losses, label="generator losses")
        plt.plot(disc_losses, label="discriminator losses")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'artifacts/losses/loss_plot_epoch_{epoch}.png')
        plt.close()
        
if __name__ == "__main__":
    main()