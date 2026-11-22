"""
A 784 -> 128 (ReLU) -> 10 (softmax) neural network in pure NumPy.

Usage from the API:
    from network import predict
    result = predict(pixels)   # pixels: 784 numbers between 0 and 1

Weights are loaded once, when this file is first imported.
"""
import os

import numpy as np

# Build the path from this file's location so it works from any folder
# (the API will run from web/, not model/).
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")

# Global parameters, same names as in the notebook
w1 = b1 = w2 = b2 = None


# ---------------------------------------------------------------- parameters
def init_params(seed=42):
    """He initialization: random weights scaled by sqrt(2 / inputs), zero biases."""
    global w1, b1, w2, b2
    np.random.seed(seed)
    w1 = np.random.randn(784, 128) * np.sqrt(2 / 784)
    b1 = np.zeros((1, 128))
    w2 = np.random.randn(128, 10) * np.sqrt(2 / 128)
    b2 = np.zeros((1, 10))


def save_weights(path=WEIGHTS_PATH):
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2)


def load_weights(path=WEIGHTS_PATH):
    global w1, b1, w2, b2
    data = np.load(path)
    w1, b1, w2, b2 = data["w1"], data["b1"], data["w2"], data["b2"]


# ------------------------------------------------------------------ the math
def forward(x):
    z1 = np.matmul(x, w1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.matmul(a1, w2) + b2

    shifted = z2 - z2.max(axis=1, keepdims=True)  # stops exp() overflowing
    e = np.exp(shifted)
    a2 = e / e.sum(axis=1, keepdims=True)

    return z1, a1, z2, a2


def loss(a2, y):
    """Cross-entropy. y is one-hot, shape (m, 10)."""
    return -np.mean(np.sum(y * np.log(a2 + 1e-9), axis=1))


def accuracy(a2, y):
    """y is the plain integer labels, shape (m,)."""
    return (np.argmax(a2, axis=1) == y).mean()


def backward(x, y, z1, a1, a2):
    """y is one-hot. Returns dw1, db1, dw2, db2."""
    m = x.shape[0]

    dz2 = (a2 - y) / m
    dw2 = np.matmul(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = np.matmul(dz2, w2.T)
    dz1 = da1 * (z1 > 0)
    dw1 = np.matmul(x.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return dw1, db1, dw2, db2


# ------------------------------------------------------------------ training
def train(x_train, y_train, y_train_oh, epochs=10, lr=0.1, batch_size=128):
    """
    x_train:     (m, 784) pixels scaled to 0-1
    y_train:     (m,) integer labels
    y_train_oh:  (m, 10) one-hot labels
    Returns (loss_history, acc_history). Call init_params() first.
    """
    global w1, b1, w2, b2
    loss_history, acc_history = [], []

    for epoch in range(epochs):
        indices = np.random.permutation(len(x_train))

        for i in range(0, len(x_train), batch_size):
            batch_indices = indices[i:i + batch_size]
            x_batch = x_train[batch_indices]
            y_batch = y_train_oh[batch_indices]

            z1, a1, z2, a2 = forward(x_batch)
            dw1, db1, dw2, db2 = backward(x_batch, y_batch, z1, a1, a2)

            w1 -= lr * dw1
            b1 -= lr * db1
            w2 -= lr * dw2
            b2 -= lr * db2

        _, _, _, a2 = forward(x_train)
        train_loss = loss(a2, y_train_oh)
        train_acc = accuracy(a2, y_train)
        loss_history.append(train_loss)
        acc_history.append(train_acc)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    return loss_history, acc_history


# ---------------------------------------------------------------- prediction
def predict(pixels):
    """
    pixels: 784 numbers between 0 and 1 (list or array).
    Returns {"digit": int, "probabilities": [10 floats]} (JSON-friendly).
    """
    if w1 is None:
        raise RuntimeError("Weights not loaded. Train and save weights.npz first.")

    x = np.asarray(pixels, dtype=float)
    if x.size != 784:
        raise ValueError(f"Expected 784 pixel values, got {x.size}")
    x = x.reshape(1, 784)

    _, _, _, a2 = forward(x)
    probs = a2[0]

    return {
        "digit": int(np.argmax(probs)),
        "probabilities": [float(p) for p in probs],
    }


# Load saved weights automatically, once, on import (if the file exists)
if os.path.exists(WEIGHTS_PATH):
    load_weights()