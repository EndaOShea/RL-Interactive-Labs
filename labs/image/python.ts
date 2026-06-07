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
  laplacian: '[[0,1,0],[1,-4,1],[0,1,0]]',
  'gaussian-blur': '[[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]',
};

// numpy `mode=` string for each border-padding choice in the lab.
const PAD_MODE: Record<string, string> = {
  zero: 'constant',
  replicate: 'edge',
  reflect: 'reflect',
};

export const convPython = (
  kernelName: string,
  padding: 'zero' | 'replicate' | 'reflect' = 'zero',
  stride = 1,
) => `import numpy as np

# 2-D convolution — mirrors the lab configuration.
# Kernel preset: ${kernelName}   padding: ${padding}   stride: ${stride}
K = np.array(${KERNELS[kernelName] ?? KERNELS.identity}, dtype=float)
PAD_MODE = "${PAD_MODE[padding] ?? 'constant'}"   # zero->constant, replicate->edge, reflect->reflect
STRIDE = ${stride}

def convolve2d(img, kernel, pad_mode=PAD_MODE, stride=STRIDE):
    """Cross-correlation form used by CNNs: (I*K)(i,j) = sum_mn I(i+m,j+n)*K(m,n).

    Output side length = floor((W - F + 2P) / S) + 1 for input W, kernel F,
    padding P = (F-1)//2, stride S.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode=pad_mode)
    H, W = img.shape
    out_h = (H - kh + 2 * ph) // stride + 1
    out_w = (W - kw + 2 * pw) // stride + 1
    out = np.zeros((out_h, out_w), dtype=float)
    for oi, i in enumerate(range(0, H, stride)):
        for oj, j in enumerate(range(0, W, stride)):
            if oi >= out_h or oj >= out_w:
                continue
            out[oi, oj] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
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
    print("input  size :", img.shape)
    print("output size :", feat.shape, "  (stride", STRIDE, "padding", PAD_MODE + ")")
    print("output range:", round(float(feat.min()), 3), round(float(feat.max()), 3))
    print("normalised feature map:\\n", np.round(normalise(feat), 2))
`;

export const featureMapsPython = (pooling: 'max' | 'avg' = 'max') => `import numpy as np

# Tiny CNN forward pass on a small glyph — mirrors the lab.
# Pipeline: input -> conv (3 FIXED filters) -> ReLU -> 2x2 ${pooling}-pool
#           -> flatten -> cosine match to class templates -> softmax.
# Honest note: filters are hand-picked and the classifier is template matching
# (no training). A real CNN learns both by backpropagation.

POOLING = "${pooling}"   # "max" keeps the strongest activation; "avg" smooths the block

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

def pool2x2(x, mode=POOLING):
    H, W = x.shape
    H2, W2 = H // 2, W // 2
    out = np.zeros((H2, W2))
    for i in range(H2):
        for j in range(W2):
            block = x[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            out[i, j] = block.max() if mode == "max" else block.mean()
    return out

def forward(img):
    maps = [pool2x2(relu(convolve2d(img, k))) for k in FILTERS.values()]
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
