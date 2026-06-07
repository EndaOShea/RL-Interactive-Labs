// Runnable Python exports for the Neural Network labs.

export const mlpPython = (sizes: number[], act: string, lr: number, dataset: string, optimizer = 'sgd', l2 = 0) => `import numpy as np

# MLP with backprop (NumPy) — mirrors the lab
SIZES = ${JSON.stringify(sizes)}   # e.g. [2, h, h, 1];  dataset = ${dataset}
ACT, LR = "${act}", ${lr}
OPTIMIZER, L2 = "${optimizer}", ${l2}

def gelu(z):  return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
def act(z):
    if ACT == "relu":  return np.maximum(0, z)
    if ACT == "leaky": return np.where(z > 0, z, 0.1 * z)
    if ACT == "gelu":  return gelu(z)
    if ACT == "tanh":  return np.tanh(z)
    return 1 / (1 + np.exp(-z))
def dact(z):                                   # derivative wrt pre-activation z
    if ACT == "relu":  return (z > 0).astype(float)
    if ACT == "leaky": return np.where(z > 0, 1.0, 0.1)
    if ACT == "gelu":  return (gelu(z + 1e-3) - gelu(z - 1e-3)) / 2e-3
    if ACT == "tanh":  return 1 - np.tanh(z) ** 2
    s = 1 / (1 + np.exp(-z)); return s * (1 - s)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

rng = np.random.default_rng(0)
W = [rng.standard_normal((SIZES[i], SIZES[i + 1])) * np.sqrt(2 / SIZES[i]) for i in range(len(SIZES) - 1)]
b = [np.zeros(SIZES[i + 1]) for i in range(len(SIZES) - 1)]
# optimizer state
mW = [np.zeros_like(w) for w in W]; vW = [np.zeros_like(w) for w in W]
mb = [np.zeros_like(x) for x in b]; vb = [np.zeros_like(x) for x in b]
t = 0

def forward(X):
    a, zs = [X], [None]
    for l in range(len(W)):
        z = a[-1] @ W[l] + b[l]; zs.append(z)
        a.append(sigmoid(z) if l == len(W) - 1 else act(z))
    return a, zs

def update(l, gW, gb):
    global t
    b1, b2, eps = 0.9, 0.999, 1e-8
    gW = gW + L2 * W[l]                          # L2 weight decay
    if OPTIMIZER == "momentum":
        mW[l][:] = b1 * mW[l] + gW; mb[l][:] = b1 * mb[l] + gb
        W[l] -= LR * mW[l]; b[l] -= LR * mb[l]
    elif OPTIMIZER == "adam":
        mW[l][:] = b1 * mW[l] + (1 - b1) * gW; vW[l][:] = b2 * vW[l] + (1 - b2) * gW**2
        mb[l][:] = b1 * mb[l] + (1 - b1) * gb; vb[l][:] = b2 * vb[l] + (1 - b2) * gb**2
        mWh, vWh = mW[l] / (1 - b1**t), vW[l] / (1 - b2**t)
        mbh, vbh = mb[l] / (1 - b1**t), vb[l] / (1 - b2**t)
        W[l] -= LR * mWh / (np.sqrt(vWh) + eps); b[l] -= LR * mbh / (np.sqrt(vbh) + eps)
    else:                                        # sgd
        W[l] -= LR * gW; b[l] -= LR * gb

def train(X, y, epochs=2000):
    global t
    for _ in range(epochs):
        t += 1
        a, zs = forward(X)
        delta = (a[-1] - y[:, None]) / len(X)    # sigmoid + BCE
        for l in reversed(range(len(W))):
            gW, gb = a[l].T @ delta, delta.sum(0)
            if l > 0: delta = (delta @ W[l].T) * dact(zs[l])
            update(l, gW, gb)

if __name__ == "__main__":
    X = rng.uniform(-1, 1, (300, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)  # XOR
    train(X, y)
    acc = ((forward(X)[0][-1][:, 0] > 0.5) == y).mean()
    print("train acc:", acc)
`;

export const activationsPython = (fn: string) => `import numpy as np

# Activation "${fn}" and its derivative
def f(x):
    sig = 1 / (1 + np.exp(-x))
    if "${fn}" == "relu":    return np.maximum(0, x)
    if "${fn}" == "leaky":   return np.where(x > 0, x, 0.1 * x)
    if "${fn}" == "elu":     return np.where(x > 0, x, np.exp(x) - 1)
    if "${fn}" == "tanh":    return np.tanh(x)
    if "${fn}" == "silu":    return x * sig                         # SiLU / Swish
    if "${fn}" == "gelu":    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    return sig   # sigmoid

x = np.linspace(-5, 5, 200)
y = f(x)
dy = np.gradient(y, x)            # numerical derivative
print("f(0) =", f(0.0), " f'(0) ≈", dy[len(dy)//2])
`;

export const perceptronPython = (rule = 'perceptron', eta = 0.5, margin = 0.2) => `import numpy as np

# Perceptron learning rule — mirrors the lab
RULE, ETA, GAMMA = "${rule}", ${eta}, ${margin}

def accuracy(X, y, w, b):
    return (np.sign(X @ w + b) == y).mean()

def train(X, y, epochs=50):            # y in {-1, +1}
    w = np.zeros(X.shape[1]); b = 0.0
    best_w, best_b, best_acc = w.copy(), b, accuracy(X, y, w, b)
    thresh = GAMMA if RULE == "margin" else 0.0
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi + b) <= thresh:
                w = w + ETA * yi * xi; b = b + ETA * yi; errors += 1
                if RULE == "pocket":                 # keep best-accuracy weights
                    acc = accuracy(X, y, w, b)
                    if acc > best_acc:
                        best_acc, best_w, best_b = acc, w.copy(), b
        if errors == 0 and RULE != "pocket":
            break                       # converged (data was separable)
    return (best_w, best_b) if RULE == "pocket" else (w, b)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.standard_normal((25, 2)) * 0.13 + [-0.6, -0.6]
    B = rng.standard_normal((25, 2)) * 0.13 + [0.6, 0.6]
    X = np.vstack([A, B]); y = np.array([-1] * 25 + [1] * 25)
    print(train(X, y))
`;
