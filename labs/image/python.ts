// Runnable Python exports for the Image Classification labs (template strings —
// not LLM generated), mirroring the on-screen pipeline and parameters.

const KERNELS: Record<string, string> = {
  identity: '[[0,0,0],[0,1,0],[0,0,0]]',
  'edge-detect': '[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]',
  sharpen: '[[0,-1,0],[-1,5,-1],[0,-1,0]]',
  'box-blur': '[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]',
  'sobel-x': '[[-1,0,1],[-2,0,2],[-1,0,1]]',
  'sobel-y': '[[-1,-2,-1],[0,0,0],[1,2,1]]',
  emboss: '[[-2,-1,0],[-1,1,1],[0,1,2]]',
};

export const convPython = (kernelName: string) => `import numpy as np

# 2-D convolution with zero-padding (stride 1, "same" output) — mirrors the lab
# Kernel preset: ${kernelName}
K = np.array(${KERNELS[kernelName] ?? KERNELS.identity}, dtype=float)

def convolve2d(img, kernel):
    """Cross-correlation form used by CNNs: (I*K)(i,j) = sum_mn I(i+m,j+n)*K(m,n)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(img, dtype=float)
    H, W = img.shape
    for i in range(H):
        for j in range(W):
            region = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(region * kernel)
    return out

def normalise(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

if __name__ == "__main__":
    # a 14x14 cross glyph in [0,1]
    img = np.zeros((14, 14))
    img[6:8, 2:12] = 1.0   # horizontal bar
    img[2:12, 6:8] = 1.0   # vertical bar

    feat = convolve2d(img, K)
    print("input range :", img.min(), img.max())
    print("output range:", round(float(feat.min()), 3), round(float(feat.max()), 3))
    print("normalised feature map:\\n", np.round(normalise(feat), 2))
    # Output size for input W, kernel F, padding P, stride S:
    #   floor((W - F + 2P) / S) + 1
`;

export const featureMapsPython = () => `import numpy as np

# Tiny CNN forward pass on a small glyph — mirrors the lab.
# Pipeline: input -> conv (3 FIXED filters) -> ReLU -> 2x2 max-pool
#           -> flatten -> cosine match to class templates -> softmax.
# Honest note: filters are hand-picked and the classifier is template matching
# (no training). A real CNN learns both by backpropagation.

FILTERS = {
    "vertical-edge":   np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
    "horizontal-edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float),
    "blob":            np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=float) / 10.0,
}

def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(img, dtype=float)
    H, W = img.shape
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
    return out

def relu(x):
    return np.maximum(0.0, x)

def max_pool2x2(x):
    H, W = x.shape
    H2, W2 = H // 2, W // 2
    out = np.zeros((H2, W2))
    for i in range(H2):
        for j in range(W2):
            out[i, j] = x[2 * i:2 * i + 2, 2 * j:2 * j + 2].max()
    return out

def forward(img):
    maps = [max_pool2x2(relu(convolve2d(img, k))) for k in FILTERS.values()]
    return np.concatenate([m.ravel() for m in maps])  # flatten

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

if __name__ == "__main__":
    # three 12x12 class templates
    H = np.zeros((12, 12)); H[2:10, 2:4] = 1; H[2:10, 8:10] = 1; H[5:7, 2:10] = 1
    T = np.zeros((12, 12)); T[2:4, 2:10] = 1; T[2:10, 5:7] = 1
    O = np.zeros((12, 12)); O[2:10, 2:10] = 1; O[4:8, 4:8] = 0
    classes = {"H": H, "T": T, "O": O}

    templates = {c: forward(im) for c, im in classes.items()}
    query = forward(H)  # try classifying an 'H'
    scores = np.array([cosine(query, templates[c]) for c in classes])
    probs = softmax(scores * 8.0)
    for c, p in zip(classes, probs):
        print(f"{c}: {p:.3f}")
`;
