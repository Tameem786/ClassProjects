import torch
import itertools
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train():
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += torch.sum(predicted == labels.data).item()
            total += labels.size(0)
        train_losses.append(total_loss/len(train_loader))
        train_accuracy = 100 * correct / total
        train_accuracies.append(correct / total)
        val_accuracy, val_loss = validate()
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:4f}, Val Acc: {100*val_accuracy:.2f}%')

    return train_losses, train_accuracies, val_losses, val_accuracies

def validate():
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += torch.sum(predicted == labels.data).item()
            total += labels.size(0)
    val_accuracy = (correct / total)
    val_loss = total_loss / len(val_loader)
    return val_accuracy, val_loss

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += torch.sum(predicted == labels.data).item()
            total += labels.size(0)
    accuracy = correct / total
    print(f'Test Accuracy: {100 * accuracy:.2f}%, Test Loss: {100 * (1-accuracy):.2f}%')

def plot(val_acc_tensor):
    mean_acc = torch.mean(val_acc_tensor, dim=0)
    std_acc = torch.std(val_acc_tensor, dim=0)

    # Plot the average curve with standard deviation
    epochs = range(len(mean_acc))
    plt.plot(epochs, mean_acc.numpy(), label="Average Test Accuracy", color='blue')
    plt.fill_between(epochs, (mean_acc - std_acc).numpy(), (mean_acc + std_acc).numpy(), alpha=0.3, color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy with Standard Deviation")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_rates = [0.1, 0.5, 0.01, 0.001, 0.0001]
    batch_sizes = [64, 128]

    num_epochs = 20
    num_classes = 10

    transformation = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                                 transform=transformation['train'])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                                transform=transformation['test'])

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    val_accuracy_list = []

    for lr_rate,batch_size in itertools.product(lr_rates, batch_sizes):
        print(f'\nRunning experiment with learning rate {lr_rate}, batch size {batch_size}')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = CifarCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

        train_losses, train_accuracies, val_losses, val_accuracies = train()
        test()

        val_accuracy_list.append(val_accuracies)

    val_accuracy_list = [torch.tensor(val) for val in val_accuracy_list]
    val_accuracy_tensor = torch.stack(val_accuracy_list)

    plot(val_accuracy_tensor)
