// Runnable Python exports for the Neural Network labs. Each mirrors its lab
// exactly: the lab's actual data and initial weights are embedded (the labs draw
// them with Math.random, so a seed could not reproduce them), with the same
// formulas, constants, update order and stopping rules as the TypeScript.
import type { PythonSample } from '../../utils/pythonSamples';
import { MLP, makeDataset, LEAKY_ALPHA, GELU_C } from './mlp';
import type { Act, Optimizer, DatasetKind } from './mlp';
import { LEAKY_SLOPE, INPUT_PRESETS } from './backpropMath';

/** Python literal for a list of number rows, `perLine` rows per line (shortest round-trip floats). */
const pyRows = (rows: number[][], perLine: number, indent = '    ') => {
  const lines: string[] = [];
  for (let i = 0; i < rows.length; i += perLine) {
    lines.push(indent + rows.slice(i, i + perLine).map((r) => JSON.stringify(r)).join(', ') + ',');
  }
  return `[\n${lines.join('\n')}\n]`;
};
const pyList = (v: number[]) => JSON.stringify(v);

/* ---------- MLP ---------- */
export interface MlpExport {
  sizes: number[];
  act: Act;
  lr: number;
  optimizer: Optimizer;
  l2: number;
  dataset: DatasetKind;
  X: number[][];
  Y: number[];
  /** The lab's initial weights W[l] (shape out × in) and biases, as drawn when the net was built. */
  W0: number[][][];
  b0: number[][];
  /** Epochs the lab had trained when the file was downloaded. */
  epochsDone: number;
}

export const mlpPython = (e: MlpExport) => `import numpy as np

# MLP classifier with backprop (NumPy) — mirrors the lab exactly: the SAME
# training points and the SAME initial weights the lab drew (both embedded
# below), full-batch epochs with a sigmoid output + binary cross-entropy,
# mean-over-batch gradients, L2 on the weights only, and the lab's optimizer.
# If you changed alpha / optimizer / lambda mid-run in the lab, its curve mixes
# settings; this script uses the settings current at download time throughout.
SIZES = ${pyList(e.sizes)}
ACT, LR = "${e.act}", ${e.lr}
OPTIMIZER, L2 = "${e.optimizer}", ${e.l2}
LEAKY = ${LEAKY_ALPHA}                # leaky-ReLU negative slope (as in the lab)
GELU_C = ${GELU_C}           # GELU uses the tanh approximation (as in the lab)
LAB_EPOCHS = ${e.epochsDone}             # epochs the lab had run at download (3 per tick)
EPOCHS = max(LAB_EPOCHS, 1500)

DATASET = "${e.dataset}"  # the lab's ${e.X.length} points in [-1, 1]^2, classes 0/1
X = np.array(${pyRows(e.X, 4)})
y = np.array(${pyList(e.Y)}, dtype=float)

# The lab's initial weights: He sqrt(2/fan_in) for relu/leaky/gelu, LeCun
# sqrt(1/fan_in) for tanh/sigmoid. Stored (out, in) as in the lab; transposed
# to (in, out) so a layer is X @ W + b.
W = [np.array(w, dtype=float).T for w in ${JSON.stringify(e.W0)}]
b = [np.array(v, dtype=float) for v in ${JSON.stringify(e.b0)}]

K = np.sqrt(2 / np.pi)


def gelu(z):
    return 0.5 * z * (1 + np.tanh(K * (z + GELU_C * z ** 3)))


def dgelu(z):                                  # exact derivative of the tanh-approximate GELU
    th = np.tanh(K * (z + GELU_C * z ** 3))
    return 0.5 * (1 + th) + 0.5 * z * (1 - th ** 2) * K * (1 + 3 * GELU_C * z ** 2)


def act(z):
    if ACT == "relu":  return np.maximum(0, z)
    if ACT == "leaky": return np.where(z > 0, z, LEAKY * z)
    if ACT == "gelu":  return gelu(z)
    if ACT == "tanh":  return np.tanh(z)
    return 1 / (1 + np.exp(-z))                # sigmoid


def dact(z, a):                                # derivative w.r.t. the pre-activation z
    if ACT == "relu":  return (z > 0).astype(float)
    if ACT == "leaky": return np.where(z > 0, 1.0, LEAKY)
    if ACT == "gelu":  return dgelu(z)
    if ACT == "tanh":  return 1 - a ** 2
    return a * (1 - a)                         # sigmoid


sigmoid = lambda z: 1 / (1 + np.exp(-z))
mW = [np.zeros_like(w) for w in W]; vW = [np.zeros_like(w) for w in W]
mb = [np.zeros_like(x) for x in b]; vb = [np.zeros_like(x) for x in b]
t = 0


def forward(A):
    a, zs = [A], [None]
    for l in range(len(W)):
        z = a[-1] @ W[l] + b[l]; zs.append(z)
        a.append(sigmoid(z) if l == len(W) - 1 else act(z))
    return a, zs


def update(l, gW, gb):
    b1, b2, eps = 0.9, 0.999, 1e-8
    gW = gW + L2 * W[l]                        # L2 weight decay (not on the bias)
    if OPTIMIZER == "momentum":                # classic momentum: v <- b1 v + g ; W <- W - LR v
        mW[l][:] = b1 * mW[l] + gW; mb[l][:] = b1 * mb[l] + gb
        W[l] -= LR * mW[l]; b[l] -= LR * mb[l]
    elif OPTIMIZER == "adam":                  # bias-corrected Adam
        mW[l][:] = b1 * mW[l] + (1 - b1) * gW; vW[l][:] = b2 * vW[l] + (1 - b2) * gW ** 2
        mb[l][:] = b1 * mb[l] + (1 - b1) * gb; vb[l][:] = b2 * vb[l] + (1 - b2) * gb ** 2
        mWh, vWh = mW[l] / (1 - b1 ** t), vW[l] / (1 - b2 ** t)
        mbh, vbh = mb[l] / (1 - b1 ** t), vb[l] / (1 - b2 ** t)
        W[l] -= LR * mWh / (np.sqrt(vWh) + eps); b[l] -= LR * mbh / (np.sqrt(vbh) + eps)
    else:                                      # sgd
        W[l] -= LR * gW; b[l] -= LR * gb


def epoch():
    """One full-batch epoch; returns the loss measured BEFORE the update (as the lab reports it)."""
    global t
    a, zs = forward(X)
    p = np.clip(a[-1][:, 0], 1e-7, 1 - 1e-7)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * L2 * sum(float((w ** 2).sum()) for w in W)
    t += 1
    delta = (a[-1] - y[:, None]) / len(X)      # sigmoid + BCE: dL/dz_out, mean over the batch
    for l in reversed(range(len(W))):
        gW, gb = a[l].T @ delta, delta.sum(0)
        if l > 0:
            delta = (delta @ W[l].T) * dact(zs[l], a[l])
        update(l, gW, gb)
    return loss


def accuracy():
    return float(np.mean((forward(X)[0][-1][:, 0] > 0.5) == y))


if __name__ == "__main__":
    print(f"dataset {DATASET} ({len(X)} points) | net {SIZES} {ACT} | {OPTIMIZER} lr={LR} L2={L2}")
    for ep in range(1, EPOCHS + 1):
        loss = epoch()
        if ep == LAB_EPOCHS:
            print(f"epoch {ep:5d} (where the lab was): loss {loss:.4f}  acc {accuracy():.3f}")
        if ep % 300 == 0 or ep == EPOCHS:
            print(f"epoch {ep:5d}: loss {loss:.4f}  acc {accuracy():.3f}")
`;

/* ---------- Activations ---------- */
export const activationsPython = (fn: string) => `import numpy as np

# Activation "${fn}" and its ANALYTIC derivative — mirrors the lab: the same
# formulas (GELU is the tanh approximation of x*Phi(x); leaky slope ${LEAKY_ALPHA}),
# the plot's x-grid, the sweep marker's steps and the lab's "healthy gradient"
# test |f'(x)| >= 0.05.
FN = "${fn}"
LEAKY = ${LEAKY_ALPHA}
GELU_C = ${GELU_C}
K = np.sqrt(2 / np.pi)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f(x):
    if FN == "sigmoid": return sigmoid(x)
    if FN == "tanh":    return np.tanh(x)
    if FN == "relu":    return np.maximum(0, x)
    if FN == "leaky":   return np.where(x > 0, x, LEAKY * x)
    if FN == "gelu":    return 0.5 * x * (1 + np.tanh(K * (x + GELU_C * x ** 3)))
    if FN == "silu":    return x * sigmoid(x)                   # SiLU / Swish
    return np.where(x > 0, x, np.exp(x) - 1)                    # elu


def df(x):
    if FN == "sigmoid":
        s = sigmoid(x); return s * (1 - s)
    if FN == "tanh":    return 1 - np.tanh(x) ** 2
    if FN == "relu":    return (x > 0).astype(float)
    if FN == "leaky":   return np.where(x > 0, 1.0, LEAKY)
    if FN == "gelu":
        th = np.tanh(K * (x + GELU_C * x ** 3))
        return 0.5 * (1 + th) + 0.5 * x * (1 - th ** 2) * K * (1 + 3 * GELU_C * x ** 2)
    if FN == "silu":
        s = sigmoid(x); return s + x * s * (1 - s)
    return np.where(x > 0, 1.0, np.exp(x))                      # elu


if __name__ == "__main__":
    grid = np.linspace(-5, 5, 121)                  # the lab's plot grid
    smooth = grid[np.abs(grid) > 1e-9]              # skip the kink at 0 (relu / leaky / elu)
    h = 1e-6
    numeric = (f(smooth + h) - f(smooth - h)) / (2 * h)
    print(f"{FN}: max |analytic - central difference| on the grid = {np.max(np.abs(df(smooth) - numeric)):.2e}")
    print(f"f(0) = {float(f(np.array(0.0))):.4f}   f'(0) = {float(df(np.array(0.0))):.4f}")
    for xv in np.arange(-5, 5.0001, 0.25):          # the sweep marker's positions
        g = float(df(np.array(xv)))
        state = "healthy" if abs(g) >= 0.05 else "gradient ~ 0"
        print(f"x = {xv:5.2f}   f = {float(f(np.array(xv))):8.4f}   f' = {g:7.4f}   {state}")
`;

/* ---------- Perceptron ---------- */
export interface PerceptronExport {
  rule: 'perceptron' | 'pocket' | 'margin';
  eta: number;
  margin: number;
  /** The lab's points and ±1 labels, in the order the lab visits them. */
  X: number[][];
  y: number[];
  /** The lab's starting line. */
  w0: [number, number];
  b0: number;
}

export const perceptronPython = (e: PerceptronExport) => `import numpy as np

# Perceptron learning rule — mirrors the lab exactly: the lab's points (embedded),
# its starting line, one point per step in the lab's order, and the same rule.
#   perceptron: update when y(w.x+b) <= 0
#   pocket:     the same updates, but keep the best-accuracy weights seen
#   margin:     update when y(w.x+b) <= GAMMA — a FUNCTIONAL margin; the band
#               lines w.x+b = +/-GAMMA sit GAMMA/||w|| from the boundary
RULE, ETA, GAMMA = "${e.rule}", ${e.eta}, ${e.margin}
W0, B0 = np.array(${pyList(e.w0)}), ${e.b0}
MAX_PASSES = 50

X = np.array(${pyRows(e.X, 4)})
y = np.array(${pyList(e.y)})


def predict(w, b):
    return np.where(X @ w + b > 0, 1, -1)          # a score of exactly 0 counts as -1, as in the lab


def accuracy(w, b):
    return float(np.mean(predict(w, b) == y))


def train():
    w, b = W0.copy(), B0
    best = (w.copy(), b, accuracy(w, b))            # the pocket starts with the starting line
    thresh = GAMMA if RULE == "margin" else 0.0
    for p in range(1, MAX_PASSES + 1):
        misses = updates = 0
        for xi, yi in zip(X, y):
            s = w @ xi + b
            if (1 if s > 0 else -1) != yi:
                misses += 1                         # genuinely misclassified at visit time
            if yi * s <= thresh:                    # the update rule
                updates += 1
                w = w + ETA * yi * xi
                b = b + ETA * yi
                if RULE == "pocket":
                    acc = accuracy(w, b)
                    if acc > best[2]:
                        best = (w.copy(), b, acc)
        nw = np.linalg.norm(w)
        line = f"pass {p:2d}: misclassified {misses:3d}  updates {updates:3d}  acc {accuracy(w, b):.3f}"
        if RULE == "margin":
            geo = np.min(y * (X @ w + b)) / nw
            line += f"  band half-width GAMMA/||w|| {GAMMA / nw:.3f}  geometric margin {geo:.3f}"
        if RULE == "pocket":
            line += f"  pocket acc {best[2]:.3f}"
        print(line)
        if updates == 0 and RULE != "pocket":
            print("converged:", "every point is outside the band" if RULE == "margin" else "every point is correctly classified")
            break
    return best if RULE == "pocket" else (w, b, accuracy(w, b))


if __name__ == "__main__":
    w, b, acc = train()
    print("w =", np.round(w, 4), " b =", round(float(b), 4), " accuracy =", round(acc, 3))
`;

/* ---------- Backpropagation ---------- */
export const backpropPython = (activation: string, lr: number, target = 1, x: number[] = [1.0, 0.5, -0.5]) => `import numpy as np

# Backpropagation from scratch — fixed feed-forward net 3 -> 4 -> 4 -> 1.
# Mirrors the lab exactly: same hardcoded weights/biases, activation and input,
# so the printed activations / deltas / gradients match what you see on screen.
ACT, LR, TARGET = "${activation}", ${lr}, ${target}
X = np.array(${JSON.stringify(x)})
LEAKY = ${LEAKY_SLOPE}                 # this lab's leaky-ReLU slope

# Fixed initial parameters. W[l] has shape [out, in]: row j = incoming weights of unit j.
INIT_W = [
    np.array([[0.50, -0.30, 0.20], [-0.40, 0.60, 0.10], [0.30, 0.20, -0.50], [-0.20, -0.40, 0.70]]),
    np.array([[0.40, -0.50, 0.30, 0.10], [0.20, 0.30, -0.60, 0.40], [-0.30, 0.50, 0.20, -0.40], [0.60, -0.20, 0.40, 0.30]]),
    np.array([[0.50, -0.40, 0.30, 0.60]]),
]
INIT_B = [np.array([0.10, -0.20, 0.30, -0.10]), np.array([-0.10, 0.20, 0.10, -0.30]), np.array([0.10])]

def act(z):
    if ACT == "relu":  return np.maximum(0, z)
    if ACT == "leaky": return np.where(z > 0, z, LEAKY * z)
    if ACT == "tanh":  return np.tanh(z)
    return 1 / (1 + np.exp(-z))                 # sigmoid

def dact(z, a):                                 # derivative wrt pre-activation z
    if ACT == "relu":  return (z > 0).astype(float)
    if ACT == "leaky": return np.where(z > 0, 1.0, LEAKY)
    if ACT == "tanh":  return 1 - a * a
    return a * (1 - a)                          # sigmoid a(1-a)

def forward(W, B, x):
    a, z = [x.astype(float)], [x.astype(float)]   # z[0] is a placeholder (the input)
    for l in range(len(W)):
        zl = W[l] @ a[-1] + B[l]
        z.append(zl); a.append(act(zl))
    yhat = a[-1][0]
    loss = 0.5 * (yhat - TARGET) ** 2
    return a, z, yhat, loss

def backward(W, a, z, yhat):
    L = len(W)
    delta = [None] * (L + 1)
    delta[L] = np.array([(yhat - TARGET) * dact(z[L][0], a[L][0])])   # output delta
    for l in range(L - 1, 0, -1):
        delta[l] = (W[l].T @ delta[l + 1]) * dact(z[l], a[l])         # (W^T delta) ⊙ act'(z)
    gW = [np.outer(delta[l + 1], a[l]) for l in range(L)]             # dL/dW = delta a^T
    gB = [delta[l + 1] for l in range(L)]                            # dL/db = delta
    return delta, gW, gB

if __name__ == "__main__":
    W = [w.copy() for w in INIT_W]
    B = [b.copy() for b in INIT_B]

    a, z, yhat, loss = forward(W, B, X)
    print("=== FORWARD ===")
    for l in range(1, len(a)):
        print(f"layer {l}: z = {np.round(z[l], 4)}")
        print(f"         a = {np.round(a[l], 4)}")
    print(f"yhat = {yhat:.4f}   loss = {loss:.5f}")

    delta, gW, gB = backward(W, a, z, yhat)
    print("\\n=== BACKWARD (deltas) ===")
    for l in range(1, len(W) + 1):
        print(f"layer {l}: delta = {np.round(delta[l], 4)}")
    print("\\n=== GRADIENTS ===")
    for l in range(len(W)):
        print(f"dL/dW[{l}] =\\n{np.round(gW[l], 4)}")
        print(f"dL/db[{l}] = {np.round(gB[l], 4)}")

    # one gradient-descent step: W -= LR * dL/dW, b -= LR * dL/db
    for l in range(len(W)):
        W[l] -= LR * gW[l]
        B[l] -= LR * gB[l]
    _, _, yhat2, loss2 = forward(W, B, X)
    print("\\n=== AFTER ONE STEP ===")
    change = loss2 - loss
    passed = (yhat - TARGET) * (yhat2 - TARGET) < 0          # yhat jumped to the other side of the target
    if change < 0:
        verdict = f"dropped by {-change:.5f}"
    elif change > 0:
        verdict = f"ROSE by {change:.5f} — LR = {LR} is too large" + (" (yhat overshot the target)" if passed else "") + "; try a smaller learning rate"
    else:
        verdict = "unchanged (zero gradient — e.g. every unit dead)"
    print(f"yhat: {yhat:.4f} -> {yhat2:.4f}   loss: {loss:.5f} -> {loss2:.5f}   ({verdict})")
`;

/* ---------- export check: one entry per export × representative params ---------- */
function mlpSample(kind: DatasetKind, hidden: number, hlayers: number, act: Act, lr: number, optimizer: Optimizer, l2: number, epochsDone: number): string {
  const pts = makeDataset(kind, 220);
  const net = new MLP([2, ...Array<number>(hlayers).fill(hidden), 1], act);
  return mlpPython({
    sizes: net.sizes, act, lr, optimizer, l2, dataset: kind,
    X: pts.map((p) => [p.x, p.y]), Y: pts.map((p) => p.cls),
    W0: net.W.map((m) => m.map((r) => r.slice())), b0: net.b.map((r) => r.slice()), epochsDone,
  });
}

function perceptronSample(rule: PerceptronExport['rule'], eta: number, margin: number, sep: number, noise: number, perClass: number): string {
  // the lab's generator (Perceptron.tsx makeData), with a Box–Muller normal
  const randn = () => { let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const off = 0.2 + sep * 0.45, X: number[][] = [], y: number[] = [];
  for (let i = 0; i < perClass; i++) {
    X.push([-off + randn() * (0.13 + noise), -off + randn() * (0.13 + noise)]); y.push(-1);
    X.push([off + randn() * (0.13 + noise), off + randn() * (0.13 + noise)]); y.push(1);
  }
  return perceptronPython({ rule, eta, margin, X, y, w0: [0.4, -0.6], b0: 0 });
}

const ACTIVATIONS = ['sigmoid', 'tanh', 'relu', 'leaky', 'gelu', 'silu', 'elu'];

export const PYTHON_SAMPLES: PythonSample[] = [
  // MLP — default, every preset, every activation / optimizer, slider edges
  { name: 'mlp-default', code: () => mlpSample('spiral', 6, 2, 'tanh', 0.3, 'sgd', 0, 0) },
  { name: 'mlp-xor-minimal', code: () => mlpSample('xor', 4, 1, 'tanh', 0.5, 'sgd', 0, 60) },
  { name: 'mlp-spiral-relu-adam', code: () => mlpSample('spiral', 8, 2, 'relu', 0.05, 'adam', 0, 300) },
  { name: 'mlp-rings-gelu-adam', code: () => mlpSample('circles', 6, 2, 'gelu', 0.05, 'adam', 0, 90) },
  { name: 'mlp-regularised', code: () => mlpSample('spiral', 8, 2, 'tanh', 0.1, 'momentum', 0.01, 30) },
  { name: 'mlp-sigmoid', code: () => mlpSample('circles', 6, 1, 'sigmoid', 0.5, 'sgd', 0, 0) },
  { name: 'mlp-leaky-momentum', code: () => mlpSample('xor', 5, 2, 'leaky', 0.2, 'momentum', 0.002, 0) },
  { name: 'mlp-edge-deep-wide', code: () => mlpSample('spiral', 10, 3, 'relu', 1.0, 'sgd', 0.05, 0), timeoutSec: 120 },
  { name: 'mlp-edge-adam-min', code: () => mlpSample('xor', 2, 1, 'gelu', 0.001, 'adam', 0, 3) },
  // Activations — every function
  ...ACTIVATIONS.map((fn) => ({ name: `activations-${fn}`, code: () => activationsPython(fn) })),
  // Perceptron — default, every preset, every rule, slider edges
  { name: 'perceptron-default', code: () => perceptronSample('perceptron', 0.5, 0.2, 0.5, 0, 20) },
  { name: 'perceptron-separable-fast', code: () => perceptronSample('perceptron', 0.5, 0.2, 0.8, 0, 20) },
  { name: 'perceptron-overlap-pocket', code: () => perceptronSample('pocket', 0.5, 0.2, 0.1, 0.12, 24) },
  { name: 'perceptron-wide-margin', code: () => perceptronSample('margin', 0.4, 0.2, 0.6, 0.02, 20) },
  { name: 'perceptron-edge-margin-max', code: () => perceptronSample('margin', 1, 0.6, 1, 0, 50) },
  { name: 'perceptron-edge-overlap', code: () => perceptronSample('perceptron', 0.05, 0.2, 0, 0.2, 8) },
  // Backpropagation — every activation, the inputs, target and step-size edges (incl. an overshoot)
  { name: 'backprop-default', code: () => backpropPython('sigmoid', 0.5, 1, INPUT_PRESETS[0]?.x) },
  { name: 'backprop-relu-overshoot', code: () => backpropPython('relu', 3, 1, INPUT_PRESETS[0]?.x) },
  { name: 'backprop-tanh-target0', code: () => backpropPython('tanh', 0.1, 0, INPUT_PRESETS[1]?.x) },
  { name: 'backprop-leaky', code: () => backpropPython('leaky', 1.5, 0.5, INPUT_PRESETS[2]?.x) },
  { name: 'backprop-relu-dead', code: () => backpropPython('relu', 0.5, 1, INPUT_PRESETS[3]?.x) },
];
