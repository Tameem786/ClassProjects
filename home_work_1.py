from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

X = load_iris().data
y = load_iris().target

y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1,1))

n_samples, n_features = X.shape
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

w = np.zeros((n_features, n_classes))
b = np.zeros((1, n_classes))
lr = 0.01
epochs = 1000

training_accuracy = []
testing_accuracy = []

for i in range(epochs):
    z = np.dot(X_train, w) + b
    y_pred = softmax(z)

    error = y_pred - y_train
    dw = np.dot(X_train.T, error) / n_samples
    db = np.sum(error, axis=0) / n_samples
    # Updating weight and bias.
    w -= lr * dw
    b -= lr * db
    # Calculating loss and accuracy error in Training.
    loss = mean_squared_error(y_train, y_pred)
    accuracy_error = 1 - accuracy_score(np.argmax(y_train, axis=1), np.argmax(y_pred, axis=1))
    training_accuracy.append(accuracy_error)
    # Calculating accuracy error in Testing.
    z_test = np.dot(X_test, w) + b
    y_pred_test = softmax(z_test)
    accuracy_error = 1 - accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_test, axis=1))
    testing_accuracy.append(accuracy_error)

# Ploting Data.
plt.figure(figsize=(10, 6))
plt.title(f'Training Error vs Testing Error Over {epochs} Epochs', fontsize=20)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.plot(training_accuracy, label='Training Error', color='blue', linewidth=2)
plt.plot(testing_accuracy, label='Testing Error', color='red', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=14)
plt.show()