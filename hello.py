import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader

# Data Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split trainset into training and validation sets
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size])

# Data Loaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Training function
def train_model(model, criterion, optimizer, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}: Loss = {running_loss / len(trainloader):.4f}, Train Accuracy = {train_acc:.2f}%")

    return model


# Testing function
def test_model(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct, total = 0, 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100 * correct / total
    test_loss /= len(testloader)
    return test_acc, test_loss


# Conduct 10 Experiments
num_experiments = 10
test_accuracies, test_losses = [], []

for i in range(num_experiments):
    print(f"Experiment {i + 1}")
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train_model(model, criterion, optimizer, epochs=10)
    acc, loss = test_model(model)
    test_accuracies.append(acc)
    test_losses.append(loss)
    print(f"Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}\n")

# Compute Mean and Standard Deviation
mean_acc, std_acc = np.mean(test_accuracies), np.std(test_accuracies)
mean_loss, std_loss = np.mean(test_losses), np.std(test_losses)

# Plot Test Accuracy
plt.figure(figsize=(8, 5))
plt.errorbar(range(1, num_experiments + 1), test_accuracies, yerr=std_acc, fmt='-o', capsize=5, label='Test Accuracy')
plt.xlabel("Experiment Number")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy Across 10 Runs")
plt.legend()
plt.grid()
plt.savefig("test_accuracy.png")
plt.show()

# Plot Test Loss
plt.figure(figsize=(8, 5))
plt.errorbar(range(1, num_experiments + 1), test_losses, yerr=std_loss, fmt='-o', capsize=5, label='Test Loss',
             color='r')
plt.xlabel("Experiment Number")
plt.ylabel("Loss")
plt.title("Test Loss Across 10 Runs")
plt.legend()
plt.grid()
plt.savefig("test_loss.png")
plt.show()

print(f"Mean Accuracy: {mean_acc:.2f}%, Std Dev: {std_acc:.2f}")
print(f"Mean Loss: {mean_loss:.4f}, Std Dev: {std_loss:.4f}")
