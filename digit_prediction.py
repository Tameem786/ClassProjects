import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f'Using {device}')

class DigitPredictionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def data_loading():
    training_data = datasets.MNIST(
        root='./data',
        train=True,
        # download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root='./data',
        train=False,
        # download=True,
        transform=ToTensor()
    )
    return training_data, test_data

def data_loader():
    training_data, test_data = data_loading()
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader

def testImagePrediction(m, cnt):
    figure = plt.figure(figsize=(10, 10))
    cols, rows = cnt, cnt
    for i in range(1, cols * rows + 1):
        index = torch.randint(len(data_loading()[0]), (1,)).item()
        image, label = data_loading()[0][index]
        image = image.unsqueeze_(0).to(device)
        with torch.no_grad():
            outputs = m(image)
            predicted_label = torch.argmax(outputs, dim=1).item()
        figure.add_subplot(rows, cols, i)
        plt.axis('off')
        plt.imshow(image.cpu().squeeze(), cmap='gray')
        plt.title(f'Model Prediction: {predicted_label} (Actual: {label})')
    plt.show()

def testTrainedModel():
    model = DigitPredictionNN().to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    testImagePrediction(model, 4)

def train():
    model = DigitPredictionNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    for epoch in range(epochs):
        for batch, (X, y) in enumerate(data_loader()[0]):
            X, y = X.to(device), y.to(device)

            output = model(X)
            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    print('Training Completed!')
    torch.save(model.state_dict(), 'model.pt')

def test():
    model = DigitPredictionNN().to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader()[0]:
            X, y = X.to(device), y.to(device)
            output = model(X)
            _, predicted = torch.max(output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy*100:.2f}%')

if __name__ == '__main__':
    testTrainedModel()