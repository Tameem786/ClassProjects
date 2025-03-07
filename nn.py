import numpy as np

L = 3
n = [2, 3, 3, 1]

W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])

b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)

def prepare_data():
    X = np.array([
        [150, 70], # it's our boy Jimmy again! 150 pounds, 70 inches tall.
        [254, 73],
        [312, 68],
        [120, 60],
        [154, 61],
        [212, 65],
        [216, 67],
        [145, 67],
        [184, 64],
        [130, 69]
    ])
    y = np.array([
        0,  # whew, thank God Jimmy isn't at risk for cardiovascular disease.
        1,   # damn, this guy wasn't as lucky
        1, # ok, this guy should have seen it coming. 5"8, 312 lbs isn't great.
        0,
        0,
        1,
        1,
        0,
        1,
        0
    ])
    m = 10
    y = y.reshape(n[3], m)
    A0 = X.T
    return A0, y, m

def sigmoid(x):
    return 1/(1+np.exp(-x))

def feed_forward(A0):
    Z1 = W1 @ A0 + b1
    A1 = sigmoid(Z1)

    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    cache = {
        "A0": A0,
        "A1": A1,
        "A2": A2
    }
    return A3, cache

def cost(A3, y):
    losses = -((y*np.log(A3)) + (1-y)*np.log(1-A3))
    m = y.reshape(-1).shape[0]
    summed_loss = (1/m)*np.sum(losses, axis=1)
    return np.sum(summed_loss)

A0, y, m = prepare_data()

def backprop_layer_3(y_hat, Y, m, A2, A3):
    A3 = y_hat
    dC_dZ3 = (1/m)*(A3-Y)
    