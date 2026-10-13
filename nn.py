import numpy as np
from sklearn.datasets import fetch_openml


mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float32)/255   # Form (70000, 784)
y = mnist.target.astype(np.int64)   # Form (70000,)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


def parameters():
    np.random.seed(1)
    W1 = np.random.randn(784,128) * 0.01
    W2 = np.random.randn(128,10) * 0.01
    b1 = np.zeros((1,128))
    b2 = np.zeros((1,10))
    return W1, W2, b1, b2

def ReLu(x):
    max = np.maximum(0,x)
    return max

def Softmax(Z2):
    Z2 = Z2 - np.max(Z2, axis=1, keepdims=True)
    exp = np.exp(Z2)
    return exp / np.sum(exp,axis=1,keepdims=True)

def forwardpass(X,W1, W2, b1, b2):
    Z1 = np.dot(X,W1)+b1
    A1 = ReLu(Z1)
    Z2 = np.dot(A1, W2)+b2
    A2 = Softmax(Z2)
    return A1,A2,Z1,Z2

def execute_forwardpropagation(W1,W2,b1,b2):
    W1,W2,b1,b2 = parameters()
    A = forwardpass(X_train[:5],W1,W2,b1,b2)
    return W1,W2,b1,b2,A


W1, W2, b1, b2 = parameters()
A1, A2, Z1, Z2 = forwardpass(X_train[:5], W1, W2, b1, b2)

print(A2.shape)
print(A2.sum(axis=1))
print(A1.min())

Y_train = np.eye(10)[y_train]

def cross_entropy(A2,Y):
    p = np.sum(A2*Y, axis=1)
    loss = np.mean(-np.log(p + 1e-9))  
    return loss

loss = cross_entropy(A2, Y_train[:5])
print(loss)

def backward(X, y, A1, A2, Z1, W2):
    dZ2 = (A2 - y) / X.shape[0]
    dW2 = np.dot(A1.T,dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1*(Z1>0).astype(float)
    dW1 = np.dot(X.T,dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

dW1, db1, dW2, db2 = backward(X_train[:5], Y_train[:5], A1, A2, Z1, W2)
print(dW1.shape, db1.shape, dW2.shape, db2.shape)


def update(W1,W2,b1,b2,dW1, db1, dW2, db2,lr):
    W1= W1 - lr * dW1
    W2= W2 - lr * dW2
    b1 = b1 - lr * db1
    b2 = b2 - lr * db2
    return W1,W2,b1,b2


def train():
    W1,W2,b1,b2 = parameters()
    lr = 0.1
    for epoch in range(10):
        idx = np.random.permutation(60000)
        X_shuffled = X_train[idx]
        Y_shuffled = Y_train[idx]
        for start in range(0, 60000, 128):
            X_batch = X_shuffled[start:start+128]
            Y_batch = Y_shuffled[start:start+128]
            A1, A2, Z1, Z2 = forwardpass(X_batch, W1, W2, b1, b2)
            loss = cross_entropy(A2, Y_batch)
            dW1, db1, dW2, db2 = backward(X_batch, Y_batch, A1, A2, Z1, W2)
            W1,W2,b1,b2 = update(W1,W2,b1,b2,dW1, db1, dW2, db2,lr)
        print(f"Epoche {epoch+1}/10 abgeschlossen, Loss: {loss:.4f}")
    return W1, W2, b1, b2

W1, W2, b1, b2 = train()
