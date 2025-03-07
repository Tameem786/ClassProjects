import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 50
learning_rate = 0.01
input_size = 3*32*32
num_classes = 10

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
print(f'Training Size: {len(train_dataset)}, Validation Size: {len(val_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class CifarNN(nn.Module):
    def __init__(self):
        super(CifarNN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.fc_layers(x)
        return x

model = CifarNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    plot(train_losses, train_accuracies, val_losses, val_accuracies)

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
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

def plot(train_losses, train_accuracies, val_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(train_losses)
    ax1.plot(val_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(['Train Loss', 'Val Loss'])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax2.plot(train_accuracies)
    ax2.plot(val_accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(['Train Accuracy', 'Val Accuracy'])
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == '__main__':
    train()
    test()