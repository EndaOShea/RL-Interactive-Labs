// Runnable Python exports for the Classic ML labs (template strings — not LLM
// generated), mirroring the on-screen algorithm/parameters.

export const knnPython = (k: number, metric: 'l1' | 'l2', perClass: number) => `import numpy as np
from collections import Counter

# k-Nearest Neighbours (from scratch) — mirrors the lab config
K = ${k}
P = ${metric === 'l1' ? 1 : 2}  # Minkowski power: 1 = Manhattan (L1), 2 = Euclidean (L2)

def make_blobs(centers, spread=0.1, per_class=${perClass}):
    X, y = [], []
    for cls, (cx, cy) in enumerate(centers):
        pts = np.random.randn(per_class, 2) * spread + [cx, cy]
        X.append(pts); y += [cls] * per_class
    return np.clip(np.vstack(X), 0, 1), np.array(y)

def predict(X_train, y_train, q, k=K, p=P):
    d = (np.abs(X_train - q) ** p).sum(axis=1) ** (1 / p)
    idx = np.argsort(d)[:k]
    return Counter(y_train[idx]).most_common(1)[0][0]

if __name__ == "__main__":
    X, y = make_blobs([(0.25, 0.30), (0.72, 0.35), (0.50, 0.75)])
    q = np.array([0.5, 0.5])
    print("prediction at", q, "->", predict(X, y, q))
`;

export const linregPython = (alpha: number) => `import numpy as np

# Linear Regression via batch gradient descent — mirrors the lab
ALPHA = ${alpha}
EPOCHS = 400

def fit(x, y, alpha=ALPHA, epochs=EPOCHS):
    w, b = 0.0, 0.0
    n = len(x)
    for _ in range(epochs):
        y_hat = w * x + b
        err = y_hat - y
        dw = (err * x).mean()      # dJ/dw
        db = err.mean()            # dJ/db
        w -= alpha * dw
        b -= alpha * db
    return w, b

if __name__ == "__main__":
    x = np.random.rand(60)
    y = 0.8 * x + 0.1 + np.random.randn(60) * 0.05
    w, b = fit(x, y)
    mse = 0.5 * np.mean((w * x + b - y) ** 2)
    print(f"w={w:.3f} b={b:.3f} J={mse:.4f}")
`;

export const logregPython = (alpha: number) => `import numpy as np

# Logistic Regression (binary) via gradient descent — mirrors the lab
ALPHA = ${alpha}
EPOCHS = 400

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def fit(X, y, alpha=ALPHA, epochs=EPOCHS):
    w = np.zeros(X.shape[1]); b = 0.0
    n = len(y)
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        grad_w = X.T @ (p - y) / n     # gradient of cross-entropy
        grad_b = (p - y).mean()
        w -= alpha * grad_w
        b -= alpha * grad_b
    return w, b

if __name__ == "__main__":
    X = np.random.randn(120, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    w, b = fit(X, y)
    acc = ((sigmoid(X @ w + b) > 0.5) == y).mean()
    print(f"w={w} b={b:.3f} acc={acc:.2f}")
`;

export const kmeansPython = (k: number, init: 'random' | 'kpp') => `import numpy as np

# k-Means clustering — mirrors the lab (init = ${init})
K = ${k}

def init_centroids(X, k, method="${init === 'kpp' ? 'k-means++' : 'random'}"):
    if method == "random":
        return X[np.random.choice(len(X), k, replace=False)].copy()
    # k-means++
    centers = [X[np.random.randint(len(X))]]
    for _ in range(1, k):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        probs = d2 / d2.sum()
        centers.append(X[np.random.choice(len(X), p=probs)])
    return np.array(centers)

def kmeans(X, k=K, iters=50):
    C = init_centroids(X, k)
    for _ in range(iters):
        labels = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), axis=1)
        newC = np.array([X[labels == j].mean(0) if (labels == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    inertia = ((X - C[labels]) ** 2).sum()
    return labels, C, inertia

if __name__ == "__main__":
    X = np.random.rand(150, 2)
    labels, C, inertia = kmeans(X)
    print("inertia:", round(float(inertia), 3))
`;

export const pcaPython = () => `import numpy as np

# Principal Component Analysis (2-D) — mirrors the lab
def pca(X):
    Xc = X - X.mean(axis=0)             # centre
    cov = np.cov(Xc, rowvar=False)      # covariance matrix
    vals, vecs = np.linalg.eigh(cov)    # eigen-decomposition
    order = np.argsort(vals)[::-1]      # largest variance first
    vals, vecs = vals[order], vecs[:, order]
    explained = vals / vals.sum()
    return vecs, vals, explained

if __name__ == "__main__":
    X = np.random.randn(200, 2) @ np.array([[2.0, 0.6], [0.6, 0.5]])
    components, variances, explained = pca(X)
    print("PC1", components[:, 0], "explains", round(float(explained[0]), 3))
`;
