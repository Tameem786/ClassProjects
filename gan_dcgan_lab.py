import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device:{device}\n')

batch_size = 128
image_size = 28
latent_size = 100
learning_rate = 0.0002

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', download=True, transform=transforms, train=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_size * image_size),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.model(x)
        return output.view(-1, 1, image_size, image_size)

class GeneratorDCGAN(nn.Module):
    def __init__(self):
        super(GeneratorDCGAN, self).__init__()
        self.model = nn.Sequential(
            # Step 1: (100, 1, 1) -> (128, 7, 7)
            nn.ConvTranspose2d(latent_size, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Step 2: (128, 7, 7) -> (64, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Step 3: (64, 14, 14) -> (1, 28, 28)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Outputs pixel values in [-1, 1]
        )

    def forward(self, x):
        x = x.view(-1, latent_size, 1, 1)  # Reshape noise for ConvTranspose2d
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

class DiscriminatorDCGAN(nn.Module):
    def __init__(self):
        super(DiscriminatorDCGAN, self).__init__()
        self.model = nn.Sequential(
            # Step 1: (1, 28, 28) -> (64, 14, 14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Step 2: (64, 14, 14) -> (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Step 3: (128, 7, 7) -> (1, 1, 1) (Final decision)
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # Probability of real (1) or fake (0)
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)  # Flatten output

def show_generated_images(generator, save_only=True, epoch=None):
    z = torch.randn(16, latent_size).to(device)
    fake_images = generator(z).detach()
    fake_images = (fake_images + 1) / 2

    grid = make_grid(fake_images, nrow=4)
    if save_only:
        save_image(grid, f'fake_images_at_epoch_{epoch}.png')
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title('Final Generated Images')
        plt.show()

def plot_losses(d_losses, g_losses, title):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title(f'{title} Losses')
    plt.legend()
    plt.grid(True)
    plt.show()

def train(epochs, model='DCGAN'):
    if model == 'DCGAN':
        generator = GeneratorDCGAN().to(device)
        discriminator = DiscriminatorDCGAN().to(device)
    else:
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

    d_losses = []
    g_losses = []

    for epoch in tqdm(range(epochs)):
        total_d_loss = 0
        total_g_loss = 0
        for i, (images, _) in enumerate(train_dataloader):
            images = images.to(device)

            # Generate noise and fake images
            z = torch.randn(images.size(0), latent_size).to(device)
            fake_images = generator(z)

            # Train the discriminator
            d_optimizer.zero_grad()
            real_loss = criterion(discriminator(images), torch.ones(images.size(0), 1).to(device))
            fake_loss = criterion(discriminator(fake_images.detach()), torch.zeros(images.size(0), 1).to(device))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train the generator
            g_optimizer.zero_grad()
            g_loss = criterion(discriminator(fake_images), torch.ones(images.size(0), 1).to(device))
            g_loss.backward()
            g_optimizer.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

            if i%100 == 0:
                print(f'\nEpoch [{epoch}/{epochs}], Step [{i}/{len(train_dataloader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
        
        d_losses.append(total_d_loss/len(train_dataloader))
        g_losses.append(total_g_loss/len(train_dataloader))

        if epoch%5 == 0:
            show_generated_images(generator, epoch=epoch)
    
    print(f'\nTraining completed!')
    print(f'\nPlotting Graphs...')
    plot_losses(d_losses, g_losses, model)
    print(f'\nGenerating Final Output...')
    show_generated_images(generator, save_only=False)

if __name__ == '__main__':
    choice = input('Enter model choice: \n1. GAN\n2. DCGAN\n')
    if choice == '1':
        train(epochs=200, model='GAN')
    elif choice == '2':
        train(epochs=30, model='DCGAN')
    else:
        print('Invalid choice!')