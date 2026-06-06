// Runnable Python exports for the Neural Network labs.

export const mlpPython = (sizes: number[], act: string, lr: number, dataset: string) => `import numpy as np

# MLP with backprop (NumPy) — mirrors the lab
SIZES = ${JSON.stringify(sizes)}   # e.g. [2, h, h, 1];  dataset = ${dataset}
ACT, LR = "${act}", ${lr}

def act(z):
    if ACT == "relu": return np.maximum(0, z)
    if ACT == "tanh": return np.tanh(z)
    return 1 / (1 + np.exp(-z))
def dact(a):
    if ACT == "relu": return (a > 0).astype(float)
    if ACT == "tanh": return 1 - a ** 2
    return a * (1 - a)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

rng = np.random.default_rng(0)
W = [rng.standard_normal((SIZES[i], SIZES[i + 1])) * np.sqrt(2 / SIZES[i]) for i in range(len(SIZES) - 1)]
b = [np.zeros(SIZES[i + 1]) for i in range(len(SIZES) - 1)]

def forward(X):
    a = [X]
    for l in range(len(W)):
        z = a[-1] @ W[l] + b[l]
        a.append(sigmoid(z) if l == len(W) - 1 else act(z))
    return a

def train(X, y, epochs=2000):
    for _ in range(epochs):
        a = forward(X)
        delta = (a[-1] - y[:, None]) / len(X)         # sigmoid + BCE
        for l in reversed(range(len(W))):
            gW, gb = a[l].T @ delta, delta.sum(0)
            if l > 0: delta = (delta @ W[l].T) * dact(a[l])
            W[l] -= LR * gW; b[l] -= LR * gb

if __name__ == "__main__":
    X = rng.uniform(-1, 1, (300, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)  # XOR
    train(X, y)
    acc = ((forward(X)[-1][:, 0] > 0.5) == y).mean()
    print("train acc:", acc)
`;

export const activationsPython = (fn: string) => `import numpy as np

# Activation "${fn}" and its derivative
def f(x):
    if "${fn}" == "relu":    return np.maximum(0, x)
    if "${fn}" == "leaky":   return np.where(x > 0, x, 0.1 * x)
    if "${fn}" == "tanh":    return np.tanh(x)
    if "${fn}" == "gelu":    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    return 1 / (1 + np.exp(-x))   # sigmoid

x = np.linspace(-5, 5, 200)
y = f(x)
dy = np.gradient(y, x)            # numerical derivative
print("f(0) =", f(0.0), " f'(0) ≈", dy[len(dy)//2])
`;

export const perceptronPython = () => `import numpy as np

# Perceptron learning rule — mirrors the lab
def train(X, y, eta=0.5, epochs=50):   # y in {-1, +1}
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi + b) <= 0:
                w += eta * yi * xi; b += eta * yi; errors += 1
        if errors == 0:
            break                       # converged (data was separable)
    return w, b

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.standard_normal((25, 2)) * 0.13 + [-0.6, -0.6]
    B = rng.standard_normal((25, 2)) * 0.13 + [0.6, 0.6]
    X = np.vstack([A, B]); y = np.array([-1] * 25 + [1] * 25)
    print(train(X, y))
`;
