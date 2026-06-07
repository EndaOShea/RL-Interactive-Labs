// Runnable Python exports for the Classic ML labs (template strings — not LLM
// generated), mirroring the on-screen algorithm/parameters.

export const knnPython = (
  k: number,
  metric: 'l1' | 'l2' | 'cheb',
  perClass: number,
  weighted: boolean,
) => `import numpy as np
from collections import defaultdict

# k-Nearest Neighbours (from scratch) — mirrors the lab config
K = ${k}
METRIC = "${metric}"        # l1 = Manhattan, l2 = Euclidean, cheb = Chebyshev (L∞)
WEIGHTED = ${weighted ? 'True' : 'False'}  # distance-weighted vote (1/d) vs plain majority

def make_blobs(centers, spread=0.1, per_class=${perClass}):
    X, y = [], []
    for cls, (cx, cy) in enumerate(centers):
        pts = np.random.randn(per_class, 2) * spread + [cx, cy]
        X.append(pts); y += [cls] * per_class
    return np.clip(np.vstack(X), 0, 1), np.array(y)

def distances(X_train, q, metric=METRIC):
    diff = np.abs(X_train - q)
    if metric == "l1":
        return diff.sum(axis=1)                # Manhattan (L1)
    if metric == "cheb":
        return diff.max(axis=1)                # Chebyshev (L∞)
    return np.sqrt((diff ** 2).sum(axis=1))    # Euclidean (L2)

def predict(X_train, y_train, q, k=K, weighted=WEIGHTED):
    d = distances(X_train, q)
    idx = np.argsort(d)[:k]
    votes = defaultdict(float)
    for i in idx:
        w = 1.0 / (d[i] + 1e-9) if weighted else 1.0   # closer neighbours count more
        votes[y_train[i]] += w
    return max(votes, key=votes.get)

if __name__ == "__main__":
    X, y = make_blobs([(0.25, 0.30), (0.72, 0.35), (0.50, 0.75)])
    q = np.array([0.5, 0.5])
    print("prediction at", q, "->", predict(X, y, q))
`;

export const linregPython = (alpha: number, degree: number, ridge: number) => `import numpy as np

# Polynomial / Ridge Regression via batch gradient descent — mirrors the lab
ALPHA  = ${alpha}
DEGREE = ${degree}      # 1 = straight line, >1 = polynomial features
LAMBDA = ${ridge}      # L2 (ridge) penalty; 0 = ordinary least squares
EPOCHS = 400

def design(x, degree=DEGREE):
    # Vandermonde-style feature matrix [1, x, x^2, ...] (bias handled separately below)
    return np.stack([x ** d for d in range(1, degree + 1)], axis=1)

def fit(x, y, alpha=ALPHA, lam=LAMBDA, epochs=EPOCHS):
    X = design(x)
    w = np.zeros(X.shape[1]); b = 0.0
    n = len(x)
    for _ in range(epochs):
        y_hat = X @ w + b
        err = y_hat - y
        dw = X.T @ err / n + lam * w   # ridge: shrink the weights toward 0
        db = err.mean()
        w -= alpha * dw
        b -= alpha * db
    return w, b

if __name__ == "__main__":
    x = np.random.rand(60)
    y = 0.8 * x + 0.1 + np.random.randn(60) * 0.05
    w, b = fit(x, y)
    X = design(x)
    mse = 0.5 * np.mean((X @ w + b - y) ** 2)
    print(f"w={np.round(w, 3)} b={b:.3f} J={mse:.4f}")
`;

export const logregPython = (alpha: number, l2: number) => `import numpy as np

# Logistic Regression (binary) via gradient descent — mirrors the lab
ALPHA  = ${alpha}
LAMBDA = ${l2}      # L2 regularisation strength (0 = unpenalised)
EPOCHS = 400

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def fit(X, y, alpha=ALPHA, lam=LAMBDA, epochs=EPOCHS):
    w = np.zeros(X.shape[1]); b = 0.0
    n = len(y)
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        grad_w = X.T @ (p - y) / n + lam * w   # cross-entropy grad + L2 weight decay
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

export const kmeansPython = (k: number, init: 'random' | 'kpp' | 'ff') => `import numpy as np

# k-Means clustering — mirrors the lab (init = ${init})
K = ${k}
INIT = "${init}"   # random | kpp (k-means++) | ff (farthest-first)

def init_centroids(X, k, method=INIT):
    if method == "random":
        return X[np.random.choice(len(X), k, replace=False)].copy()
    if method == "ff":
        # farthest-first: each new centre is the point maximally far from the chosen set
        centers = [X[np.random.randint(len(X))]]
        for _ in range(1, k):
            d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
            centers.append(X[int(np.argmax(d2))])
        return np.array(centers)
    # k-means++  (probability ∝ squared distance)
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

export const pcaPython = (whiten: boolean, threshold: number) => `import numpy as np

# Principal Component Analysis (2-D) — mirrors the lab
WHITEN    = ${whiten ? 'True' : 'False'}  # rescale each component to unit variance
THRESHOLD = ${threshold.toFixed(2)}      # keep enough PCs to capture this much variance

def pca(X, whiten=WHITEN):
    Xc = X - X.mean(axis=0)             # centre
    cov = np.cov(Xc, rowvar=False)      # covariance matrix
    vals, vecs = np.linalg.eigh(cov)    # eigen-decomposition
    order = np.argsort(vals)[::-1]      # largest variance first
    vals, vecs = vals[order], vecs[:, order]
    explained = vals / vals.sum()
    proj = Xc @ vecs
    if whiten:
        proj = proj / np.sqrt(vals + 1e-12)   # unit-variance components
    return vecs, vals, explained, proj

def n_components_for(explained, threshold=THRESHOLD):
    # smallest number of PCs whose cumulative variance >= threshold
    return int(np.searchsorted(np.cumsum(explained), threshold) + 1)

if __name__ == "__main__":
    X = np.random.randn(200, 2) @ np.array([[2.0, 0.6], [0.6, 0.5]])
    components, variances, explained, proj = pca(X)
    keep = n_components_for(explained)
    print("PC1", components[:, 0], "explains", round(float(explained[0]), 3))
    print(f"keep {keep} component(s) for {THRESHOLD:.0%} variance")
`;
