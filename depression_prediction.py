import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

torch.manual_seed(42)
label_encoder = LabelEncoder()
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

df = pd.read_csv('depression.csv')
df = df.dropna()
df['Depression State'] = df['Depression State'].str.strip()
df['Depression State'] = df['Depression State'].str.replace(r'^\d+\tNo depression$', 'No depression', regex=True)
df['Depression State'] = label_encoder.fit_transform(df['Depression State'])
label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

X = df.drop(columns=['Number ','Depression State']).values
y = df['Depression State'].values

X = X.to_numpy()
y = y.to_numpy()
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

# print(X_tensor.shape, y_tensor.shape)
dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class DepressionNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

input_size = 14
num_classes = len(torch.unique(y_train_tensor))

model = DepressionNN(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10000

# for epoch in range(epochs):
#     epoch_loss = 0
#     for batch_X, batch_y in dataloader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#         output = model(batch_X)
#         loss = criterion(output, batch_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     # if (epoch + 1) % 10 == 0:
#     print(f'Epoch {epoch + 1} / {epochs}, Loss: {epoch_loss/len(dataloader):.4f}')
#
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
print('Training Completed!')

with torch.no_grad():
    output = model(X_test_tensor)
    _, predicted = torch.max(output, 1)
accuracy = (predicted == y_test_tensor).sum().item()/y_test_tensor.size(0)
print(f'Accuracy: {accuracy*100:.2f}%')

plt.plot([x for x in range(epochs)], losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# print(torch.unique(y_tensor))