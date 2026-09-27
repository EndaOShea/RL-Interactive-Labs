// Runnable Python exports for the Deep Learning labs. Each one mirrors its lab
// exactly: the same seeded draws (PY_RNG ports the labs' mulberry32 generator
// bit-for-bit), the same parameters (read from the live lab state), the same
// algorithm. ResNet / BatchNorm / Optimizers use PyTorch; Dropout / Transfer use
// NumPy ports of labs/deep-learning/tinyMlp.ts; the Architecture Builder emits
// Keras so model.summary() prints the per-layer params/shapes the lab shows.
import type { PythonSample } from '../../utils/pythonSamples';
import type { Layer, Shape, Mode } from './archBuilder';
import type { BnMode } from './batchNormSim';
import type { BranchScale } from './resnetSim';
import type { Optimizer, Schedule } from './optimSim';
import type { DataPoint, ToyKind } from './mlpTrainer';
import { RES_D, RES_B, RES_SEED, branchAlpha } from './resnetSim';
import { BN_D, BN_SEED, BN_EPS, BN_MOMENTUM, SAT_LEVEL } from './batchNormSim';
import { DROPOUT_DEFAULTS, DROP_CENTERS, DROP_CLASSES, DROP_TRAIN_FRAC, DROP_CLIP } from './dropoutSim';
import {
  TL_CENTERS, TL_XOR, TL_H, TL_TAGS, TL_NOISE, TL_SOURCE_PER, TL_POOL_PER, TL_VAL_PER, TL_PRE_EPOCHS, TL_EPOCHS,
  TL_LR, FT_BACKBONE_LR_SCALE, TL_TRIALS, TL_SWEEP, TL_DEFAULTS,
} from './transferSim';
import {
  START, BOX, MAX_ITERS, RMS_RHO, ADAM_B2, OPT_EPS, CONVERGED_LOSS, STEP_EVERY, STEP_GAMMA, WARMUP_ITERS, LR_PRESETS, OPTIMIZERS,
} from './optimSim';
import { LEAKY_SLOPE, BN_EPSILON, BN_MOMENTUM as MLP_BN_MOMENTUM, MLP_MAX_EPOCHS, makeData } from './mlpTrainer';

const pyBool = (b: boolean) => (b ? 'True' : 'False');
const pyNum = (v: number) => (Number.isInteger(v) ? v.toFixed(1) : String(v));

/** The labs' seeded generator (mulberry32) + Box–Muller, reproduced bit-for-bit. */
const PY_RNG = `class Mulberry32:
    """The lab's seeded generator (mulberry32), reproduced bit-for-bit.
    Its state is a Weyl sequence a_k = seed + k*0x6D2B79F5 (mod 2^32), so the
    k-th draw is a pure hash of a_k and n draws can be made at once."""
    M = 0xFFFFFFFF
    def __init__(self, seed):
        self.a = seed & self.M
    def take(self, n):
        """The next n uniforms in [0, 1), in order."""
        M = np.uint64(self.M)
        k = np.arange(1, n + 1, dtype=np.uint64)
        a = (np.uint64(self.a) + k * np.uint64(0x6D2B79F5)) & M
        if n:
            self.a = int(a[-1])
        t = ((a ^ (a >> np.uint64(15))) * (a | np.uint64(1))) & M
        t = ((t + (((t ^ (t >> np.uint64(7))) * (t | np.uint64(61))) & M)) & M) ^ t
        return ((t ^ (t >> np.uint64(14))) & M).astype(np.float64) / 4294967296.0
    def __call__(self):
        return float(self.take(1)[0])

def gauss(r, n):
    """n standard normals, Box-Muller, two uniforms per sample (u first, then v)."""
    U = r.take(2 * n)
    return np.sqrt(-2.0 * np.log(1.0 - U[0::2])) * np.cos(2.0 * np.pi * U[1::2])`;

/** NumPy port of tinyMlp.ts (ReLU/sigmoid hidden layers, BCE or one-vs-rest output, Adam, inverted dropout). */
const PY_MLP = `B1, B2, ADAM_EPS = 0.9, 0.999, 1e-8

def sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))

def bce_logit(z, y):
    return np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))

class MLP:
    """Full-batch MLP with real backprop and Adam — the lab's tinyMlp.ts, line for line.
    Hidden layers: act(W a + b) * mask; output: logits ('bce' = 1 sigmoid, 'ovr' = K sigmoid tags).
    Init: He normal W ~ N(0, 2/fan_in) drawn row-major layer by layer, b = 0."""
    def __init__(self, dims, r=None, acts=None, W=None, b=None):
        self.dims = list(dims)
        self.acts = list(acts) if acts else ["relu"] * (len(dims) - 2)
        if W is None:
            W, b = [], []
            for fi, fo in zip(dims[:-1], dims[1:]):
                W.append(gauss(r, fo * fi).reshape(fo, fi) * np.sqrt(2.0 / fi))
                b.append(np.zeros(fo))
        self.W, self.b = [w.copy() for w in W], [v.copy() for v in b]
        self.t = 0
        self.mW = [np.zeros_like(w) for w in self.W]; self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(v) for v in self.b]; self.vb = [np.zeros_like(v) for v in self.b]

    def copy(self):
        return MLP(self.dims, acts=self.acts, W=self.W, b=self.b)

    def forward(self, X, masks=None):
        a, acts, ders = X, [X], []
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l].T + self.b[l]
            if l < L - 1:
                if self.acts[l] == "sigmoid":
                    v = sigmoid(z); g = v * (1.0 - v)
                else:
                    v = np.maximum(z, 0.0); g = (z > 0).astype(np.float64)
                if masks is not None:                       # inverted dropout: 0 or 1/keep
                    v, g = v * masks[l], g * masks[l]
                a = v
                ders.append(g)
            else:
                a = z
            acts.append(a)
        return acts, ders

    def evaluate(self, X, y, loss="bce"):
        z = self.forward(X)[0][-1]
        if loss == "bce":
            z = z[:, 0]
            return float(bce_logit(z, y).mean()), float(((z >= 0) == (y == 1)).mean())
        onehot = np.eye(z.shape[1])[y]
        return float(bce_logit(z, onehot).sum(1).mean()), float((z.argmax(1) == y).mean())

    def train_step(self, X, y, loss="bce", lr=0.01, lr_scale=None, trainable=None, dropout=0.0, mask_rng=None):
        """One full-batch Adam step on the mean loss."""
        N, L = len(X), len(self.W)
        masks = None
        if dropout > 0:
            keep = 1.0 - dropout
            H = self.dims[1:-1]
            U = mask_rng.take(N * sum(H)).reshape(N, sum(H))   # per sample: layer 1's units, then layer 2's
            masks = np.split(np.where(U < keep, 1.0 / keep, 0.0), np.cumsum(H)[:-1], axis=1)
        acts, ders = self.forward(X, masks)
        z = acts[-1]
        target = y[:, None] if loss == "bce" else np.eye(z.shape[1])[y]
        delta = (sigmoid(z) - target) / N
        self.t += 1
        bc1, bc2 = 1 - B1 ** self.t, 1 - B2 ** self.t
        trainable = trainable or [True] * L
        lr_scale = lr_scale or [1.0] * L
        lowest = next((l for l in range(L) if trainable[l]), L)   # no gradient needed below this layer
        for l in range(L - 1, lowest - 1, -1):
            prev = (delta @ self.W[l]) * ders[l - 1] if l > lowest else None
            if trainable[l]:
                eta = lr * lr_scale[l]
                for p, g, m, v in ((self.W[l], delta.T @ acts[l], self.mW[l], self.vW[l]),
                                   (self.b[l], delta.sum(0), self.mb[l], self.vb[l])):
                    m[...] = B1 * m + (1 - B1) * g
                    v[...] = B2 * v + (1 - B2) * g * g
                    p -= eta * (m / bc1) / (np.sqrt(v / bc2) + ADAM_EPS)
            if prev is not None:
                delta = prev`;

/* ───────────────────────── Residual Networks ───────────────────────── */

export const resnetPython = (depth = 28, gain = 0.9, scale: BranchScale = 'invsqrt', focusResidual = true) => {
  const alpha = branchAlpha(scale, depth);
  return `"""Plain vs residual deep tanh stack: gradient norm at every layer.
Mirrors the Residual Networks lab — depth L = ${depth}, weight gain = ${gain}, residual branch
scale alpha = ${scale === 'one' ? '1' : '1/sqrt(L)'} = ${alpha.toPrecision(6)}, width d = ${RES_D}, batch ${RES_B}, seed ${RES_SEED}${focusResidual ? '' : ' (focus: plain)'}.

Both nets share the SAME weights W_0..W_{L-1} (entries ~ N(0, gain^2/d)), the same inputs
x ~ N(0, I) and the same output gradient: each sample's loss is u.h_L for a fixed unit
vector u, so dL/dh_L = u. Tensor hooks capture dL/dh_l at every layer; the lab plots the
RMS over the batch of ||dL/dh_l||.
  plain     h_{l+1} = tanh(W_l h_l)
  residual  h_{l+1} = h_l + alpha * tanh(W_l h_l)
"""
import math
import numpy as np
import torch

${PY_RNG}

L, GAIN, D, B, SEED = ${depth}, ${pyNum(gain)}, ${RES_D}, ${RES_B}, ${RES_SEED}
ALPHA = ${scale === 'one' ? '1.0' : '1.0 / math.sqrt(L)'}

# The lab's draws, in the lab's order: X (B x d), u (d), then the L base matrices.
r = Mulberry32(SEED)
X = gauss(r, B * D).reshape(B, D)
u = gauss(r, D)
u = u / np.linalg.norm(u)
W = [gauss(r, D * D).reshape(D, D) * (GAIN / math.sqrt(D)) for _ in range(L)]

class Block(torch.nn.Module):
    def __init__(self, w, residual):
        super().__init__()
        self.fc = torch.nn.Linear(D, D, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.fc.weight.copy_(torch.from_numpy(w))
        self.residual = residual

    def forward(self, h):
        y = torch.tanh(self.fc(h))
        return h + ALPHA * y if self.residual else y

def grad_profile(residual):
    """RMS over the batch of ||dL/dh_l|| for l = 0 (input) .. L (output), via tensor hooks."""
    blocks = [Block(w, residual) for w in W]
    grads = [None] * (L + 1)
    hs = [torch.tensor(X, dtype=torch.float64, requires_grad=True)]
    for blk in blocks:
        hs.append(blk(hs[-1]))
    for l, h in enumerate(hs):
        h.register_hook(lambda g, l=l: grads.__setitem__(l, g.detach()))
    (hs[-1] @ torch.from_numpy(u)).sum().backward()      # dL/dh_L = u for every sample
    return [float(g.norm() / math.sqrt(B)) for g in grads]

def verdict(g):
    if g != g: return "exploded"
    if g < 1e-2: return "vanished"
    if g < 0.1: return "shrinking"
    if g <= 10: return "healthy"
    if g <= 100: return "growing"
    return "exploded"

plain, resid = grad_profile(False), grad_profile(True)
print(" layer   plain ||dL/dh_l||   residual ||dL/dh_l||")
for l in range(L, -1, -1):
    print(f"{l:6d}   {plain[l]:18.6e}   {resid[l]:20.6e}")
print(f"\\ngradient at the input: plain {plain[0]:.4e} ({verdict(plain[0])}), residual {resid[0]:.4e} ({verdict(resid[0])})")
print(f"average factor per layer ||dL/dh_0||^(1/L): plain {plain[0] ** (1 / L):.4f}, residual {resid[0] ** (1 / L):.4f}")
`;
};

/* ───────────────────────── Batch Normalization ───────────────────────── */

export interface BatchNormExport { depth: number; gain: number; batch: number; useBN: boolean; mode: BnMode; warmup: number; seed?: number; }

export const batchNormPython = (o: BatchNormExport = { depth: 16, gain: 2.5, batch: 256, useBN: false, mode: 'train', warmup: 40 }) => {
  const evalMode = o.useBN && o.mode === 'eval';
  const seed = o.seed ?? BN_SEED;
  return `"""Deep tanh stack with / without BatchNorm: pre-activation statistics per layer.
Mirrors the Batch Normalization lab — depth ${o.depth}, init scale ${o.gain}, batch ${o.batch}, d = ${BN_D},
BatchNorm ${o.useBN ? `ON (${o.mode} mode${evalMode ? `, running statistics from ${o.warmup} warm-up batches` : ''})` : 'OFF'}, seed ${seed}.

Layer l: z = W_l a  ->  u = BatchNorm1d(z) (or z)  ->  a = tanh(u), with W_l ~ N(0, gain^2/d).
BatchNorm1d uses the lab's conventions (PyTorch defaults): batch mean + biased variance with
eps = ${BN_EPS} inside the sqrt; running_mean/var updated with momentum ${BN_MOMENTUM} (the running
variance uses the unbiased batch variance); gamma = 1, beta = 0 are left at their initial
values — nothing is trained. Reported per layer: the std of u (what BatchNorm pins to 1),
the mean tanh slope 1 - tanh(u)^2 and the fraction of |tanh(u)| > ${SAT_LEVEL}.
"""
import math
import numpy as np
import torch
import torch.nn as nn

${PY_RNG}

DEPTH, GAIN, BATCH, D, SEED = ${o.depth}, ${pyNum(o.gain)}, ${o.batch}, ${BN_D}, ${seed}
USE_BN, MODE, WARMUP = ${pyBool(o.useBN)}, "${o.mode}", ${o.warmup}
EVAL_MODE = USE_BN and MODE == "eval"

# Three independent seeded streams: weights, the displayed batch, the warm-up batches.
rW, rX, rK = Mulberry32(SEED), Mulberry32(SEED + 1), Mulberry32(SEED + 2)
X = gauss(rX, BATCH * D).reshape(BATCH, D)
warm = [gauss(rK, BATCH * D).reshape(BATCH, D) for _ in range(WARMUP)] if EVAL_MODE else []

layers = []
for _ in range(DEPTH):
    lin = nn.Linear(D, D, bias=False, dtype=torch.float64)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(gauss(rW, D * D).reshape(D, D) * (GAIN / math.sqrt(D))))
    layers.append(lin)
    if USE_BN:
        layers.append(nn.BatchNorm1d(D, eps=${BN_EPS}, momentum=${BN_MOMENTUM}, dtype=torch.float64))
    layers.append(nn.Tanh())
net = nn.Sequential(*layers)

captured = []                              # the input u of every Tanh, via forward hooks
for m in net:
    if isinstance(m, nn.Tanh):
        m.register_forward_hook(lambda mod, inp, out: captured.append(inp[0].detach()))

with torch.no_grad():
    net.train()
    for xb in warm:                        # train-mode passes update the running statistics
        net(torch.from_numpy(xb))
    if EVAL_MODE:
        net.eval()                         # normalise with running_mean / running_var
    captured.clear()
    net(torch.from_numpy(X))

def std_per_feature(t):                    # batch std per feature (biased), averaged over features
    return float(((t - t.mean(0)) ** 2).mean(0).sqrt().mean())

print(" layer   std(u)   mean tanh'   saturated")
print(f"     0   {std_per_feature(torch.from_numpy(X)):.4f}          -           -   (the input x)")
for l, u in enumerate(captured, start=1):
    t = torch.tanh(u)
    print(f"{l:6d}   {std_per_feature(u):.4f}   {float((1 - t ** 2).mean()):10.4f}   {float((t.abs() > ${SAT_LEVEL}).double().mean()) * 100:8.1f}%")
`;
};

/* ───────────────────────── Dropout ───────────────────────── */

export interface DropoutExport { p: number; perCluster: number; seed: number; }

export const dropoutPython = (o: DropoutExport = { p: 0.3, perCluster: DROPOUT_DEFAULTS.perCluster, seed: DROPOUT_DEFAULTS.seed }) => `"""Dropout, side by side: the same 2-64-64-1 network trained with p = 0 and with p = ${o.p}.
Mirrors the Dropout lab — same seeded data (seed ${o.seed}, ${o.perCluster} points per cluster),
the same initial weights for both nets, full-batch Adam (lr ${DROPOUT_DEFAULTS.lr}) for ${DROPOUT_DEFAULTS.maxEpochs} epochs, and the
same dropout masks (inverted dropout on both hidden layers, drawn from the seeded generator).
"""
import numpy as np

${PY_RNG}

${PY_MLP}

P, PER_CLUSTER, SEED = ${pyNum(o.p)}, ${o.perCluster}, ${o.seed}
NOISE, HIDDEN, LR, EPOCHS = ${DROPOUT_DEFAULTS.noise}, ${JSON.stringify(DROPOUT_DEFAULTS.hidden)}, ${DROPOUT_DEFAULTS.lr}, ${DROPOUT_DEFAULTS.maxEpochs}
CENTERS, CLASSES, TRAIN_FRAC, CLIP = ${JSON.stringify(DROP_CENTERS)}, ${JSON.stringify(DROP_CLASSES)}, ${DROP_TRAIN_FRAC}, ${JSON.stringify(DROP_CLIP)}

# Data: per cluster, per point: x-noise, y-noise, then the train/validation uniform.
r = Mulberry32(SEED)
tr, trY, va, vaY = [], [], [], []
for (cx, cy), c in zip(CENTERS, CLASSES):
    for _ in range(PER_CLUSTER):
        x = min(CLIP[1], max(CLIP[0], cx + NOISE * gauss(r, 1)[0]))
        y = min(CLIP[1], max(CLIP[0], cy + NOISE * gauss(r, 1)[0]))
        if r() < TRAIN_FRAC:
            tr.append((x, y)); trY.append(c)
        else:
            va.append((x, y)); vaY.append(c)
Xtr, Ytr, Xva, Yva = np.array(tr), np.array(trY), np.array(va), np.array(vaY)

plain = MLP([2, *HIDDEN, 1], Mulberry32(SEED + 1))   # identical initial weights ...
drop = plain.copy()                                   # ... for both nets
mask_rng = Mulberry32(SEED + 2)
hist = {"plain": [], "drop": []}
for epoch in range(1, EPOCHS + 1):
    plain.train_step(Xtr, Ytr, "bce", LR)
    drop.train_step(Xtr, Ytr, "bce", LR, dropout=P, mask_rng=mask_rng)
    for name, net in (("plain", plain), ("drop", drop)):
        tl, ta = net.evaluate(Xtr, Ytr); vl, vac = net.evaluate(Xva, Yva)
        hist[name].append((tl, vl, ta, vac))
    if epoch % 25 == 0:
        a, b = hist["plain"][-1], hist["drop"][-1]
        print(f"epoch {epoch:3d} | p=0: train loss {a[0]:.3f} val loss {a[1]:.3f} acc {a[2]:.0%}/{a[3]:.0%}"
              f" | p={P}: train loss {b[0]:.3f} val loss {b[1]:.3f} acc {b[2]:.0%}/{b[3]:.0%}")

def boundary_cells(net, G=40):                        # adjacent grid cells with different predicted class
    g = (np.arange(G) + 0.5) / G
    xx, yy = np.meshgrid(g, g)
    cls = (net.forward(np.c_[xx.ravel(), yy.ravel()])[0][-1][:, 0] >= 0).reshape(G, G)
    return int((cls[:, 1:] != cls[:, :-1]).sum() + (cls[1:, :] != cls[:-1, :]).sum())

a, b = hist["plain"][-1], hist["drop"][-1]
vl0 = [h[1] for h in hist["plain"]]
print(f"\\n{len(Xtr)} train / {len(Xva)} validation points")
print(f"p=0:   train acc {a[2]:.1%}  val acc {a[3]:.1%}  gap {a[2] - a[3]:+.1%}  val loss {a[1]:.3f} (min {min(vl0):.3f} at epoch {int(np.argmin(vl0)) + 1})")
print(f"p={P}: train acc {b[2]:.1%}  val acc {b[3]:.1%}  gap {b[2] - b[3]:+.1%}  val loss {b[1]:.3f}")
print(f"decision-boundary cells on a 40x40 grid: p=0 {boundary_cells(plain)}, p={P} {boundary_cells(drop)}")
`;

/* ───────────────────────── Transfer Learning ───────────────────────── */

export interface TransferExport { seed: number; rotationDeg: number; n: number; mode: 'frozen' | 'finetune'; }

export const transferPython = (o: TransferExport = { seed: TL_DEFAULTS.seed, rotationDeg: TL_DEFAULTS.rotationDeg, n: 16, mode: 'frozen' }) => `"""Transfer learning vs training from scratch, on few labels.
Mirrors the Transfer Learning lab — seed ${o.seed}, target domain rotated ${o.rotationDeg} degrees
(highlighted: n = ${o.n}, ${o.mode === 'frozen' ? 'frozen backbone' : 'fine-tuning'}).

SOURCE task: tag which of four clusters a point came from ("is it cluster k?", four sigmoid
outputs, binary cross-entropy) on ${4 * TL_SOURCE_PER} labelled points. The pretrained 2 -> ${TL_H} ReLU -> ${TL_TAGS} sigmoid
network is the backbone phi(x).
TARGET task: XOR of the clusters on a domain rotated by theta, with only n labels.
Three learners, all 2 -> ${TL_H} -> ${TL_TAGS} -> 1 with the same ${TL_EPOCHS} full-batch Adam steps (lr ${TL_LR}):
  scratch   - He init, every layer trained;
  frozen    - pretrained backbone fixed, only the zero-initialised head (4 weights + bias) trains;
  fine-tune - backbone trained at lr x ${FT_BACKBONE_LR_SCALE}, head at lr.
Validation accuracy on ${4 * TL_VAL_PER} held-out target points, mean of ${TL_TRIALS} random labelled subsets.
"""
import math
import numpy as np

${PY_RNG}

${PY_MLP}

SEED, THETA, N_SHOW, MODE = ${o.seed}, ${pyNum(o.rotationDeg)}, ${o.n}, "${o.mode}"
CENTERS, XOR, H, TAGS, NOISE = ${JSON.stringify(TL_CENTERS)}, ${JSON.stringify(TL_XOR)}, ${TL_H}, ${TL_TAGS}, ${TL_NOISE}
SOURCE_PER, POOL_PER, VAL_PER = ${TL_SOURCE_PER}, ${TL_POOL_PER}, ${TL_VAL_PER}
PRE_EPOCHS, EPOCHS, LR, FT_SCALE, TRIALS = ${TL_PRE_EPOCHS}, ${TL_EPOCHS}, ${TL_LR}, ${FT_BACKBONE_LR_SCALE}, ${TL_TRIALS}
SWEEP = ${JSON.stringify(TL_SWEEP)}

def cluster_set(r, per, theta_deg, label):
    """Per cluster k, per point: x-noise then y-noise; then rotate by theta about (0.5, 0.5)."""
    th = theta_deg * math.pi / 180
    c, s = math.cos(th), math.sin(th)
    X, Y = [], []
    for k, (cx, cy) in enumerate(CENTERS):
        for _ in range(per):
            px, py = cx + NOISE * gauss(r, 1)[0], cy + NOISE * gauss(r, 1)[0]
            dx, dy = px - 0.5, py - 0.5
            X.append((0.5 + c * dx - s * dy, 0.5 + s * dx + c * dy)); Y.append(label(k))
    return np.array(X), np.array(Y)

def permutation(n, r):
    """Fisher-Yates driven by the seeded generator (same order as the lab)."""
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = math.floor(r() * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    return idx

# ---- pretraining on the source task (depends on the seed only) ----
Xs, Ys = cluster_set(Mulberry32(SEED), SOURCE_PER, 0, lambda k: k)
backbone = MLP([2, H, TAGS], Mulberry32(SEED + 3))
for _ in range(PRE_EPOCHS):
    backbone.train_step(Xs, Ys, "ovr", LR)
print(f"source-task accuracy (argmax tag = true cluster): {backbone.evaluate(Xs, Ys, 'ovr')[1]:.1%}")

# ---- target data + the labelled subsets ----
Xp, Yp = cluster_set(Mulberry32(SEED + 1), POOL_PER, THETA, lambda k: XOR[k])
Xv, Yv = cluster_set(Mulberry32(SEED + 2), VAL_PER, THETA, lambda k: XOR[k])
rS = Mulberry32(SEED + 4)
subsets = [[permutation(len(Yp), rS)[:min(n, len(Yp))] for _ in range(TRIALS)] for n in SWEEP]

def transfer_net():
    """Pretrained backbone (2 -> H -> 4, tags through a sigmoid) + a zero-initialised 4 -> 1 head."""
    return MLP([2, H, TAGS, 1], acts=["relu", "sigmoid"],
               W=[backbone.W[0], backbone.W[1], np.zeros((1, TAGS))], b=[backbone.b[0], backbone.b[1], np.zeros(1)])

print(f"\\n  n   scratch   frozen   fine-tune   (target rotated {THETA} deg)")
rows = []
for i, n in enumerate(SWEEP):
    r_init = Mulberry32(SEED + 5 + 1000 * (i + 1))
    acc = np.zeros(3)
    for idx in subsets[i]:
        X, Y = Xp[idx], Yp[idx]
        scratch = MLP([2, H, TAGS, 1], r_init, acts=["relu", "sigmoid"])
        frozen = transfer_net()
        tuned = frozen.copy()
        for _ in range(EPOCHS):
            scratch.train_step(X, Y, "bce", LR)
            frozen.train_step(X, Y, "bce", LR, trainable=[False, False, True])
            tuned.train_step(X, Y, "bce", LR, lr_scale=[FT_SCALE, FT_SCALE, 1.0])
        acc += [net.evaluate(Xv, Yv)[1] for net in (scratch, frozen, tuned)]
    acc /= len(subsets[i])
    rows.append((n, *acc))
    print(f"{n:4d}   {acc[0]:7.1%}   {acc[1]:6.1%}   {acc[2]:9.1%}" + ("   <- highlighted" if n == N_SHOW else ""))

n, s, f, t = next(row for row in rows if row[0] == N_SHOW)
print(f"\\nat n = {n}: {MODE} transfer {f if MODE == 'frozen' else t:.1%} vs scratch {s:.1%}")
`;

/* ───────────────────────── Optimizers ───────────────────────── */

export interface OptimizersExport { lrs: Record<Optimizer, number>; beta: number; schedule: Schedule; }

export const optimizersPython = (o: OptimizersExport = { lrs: LR_PRESETS.moderate, beta: 0.9, schedule: 'constant' }) => `"""Optimizer race on the Rosenbrock ravine f(x, y) = (1 - x)^2 + 100 (y - x^2)^2.
Mirrors the Optimizers & LR Schedules lab: all four optimisers start at (${START[0]}, ${START[1]}) with their own
learning rates, the same schedule (${o.schedule}) and beta = ${o.beta} (momentum beta and Adam beta1).
Each run stops when loss < ${CONVERGED_LOSS} ("converged"), when the iterate leaves the plotted region
x in [${BOX.xLo}, ${BOX.xHi}], y in [${BOX.yLo}, ${BOX.yHi}] ("diverged"), or after ${MAX_ITERS} updates.
Update t (0-based) uses lr * lr_factor(t); the gradient is the lab's analytic one.
The printed outcomes match the lab. In a chaotic setting (e.g. Adam with beta = 0 at a large
lr, which bounces across the ravine walls) the path is sensitive to floating-point rounding, so
torch's CPU kernels can end at a slightly different point than the browser, with the same outcome.
"""
import math
import torch

LRS = {${OPTIMIZERS.map((k) => `"${k}": ${o.lrs[k]}`).join(', ')}}
BETA, SCHEDULE = ${pyNum(o.beta)}, "${o.schedule}"
START, MAX_ITERS, CONVERGED = (${START[0]}, ${START[1]}), ${MAX_ITERS}, ${CONVERGED_LOSS}
X_LO, X_HI, Y_LO, Y_HI = ${BOX.xLo}, ${BOX.xHi}, ${BOX.yLo}, ${BOX.yHi}
RHO, BETA2, EPS = ${RMS_RHO}, ${ADAM_B2}, ${OPT_EPS}
STEP_EVERY, STEP_GAMMA, WARMUP = ${STEP_EVERY}, ${STEP_GAMMA}, ${WARMUP_ITERS}

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2

def grad(x, y):
    return (-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x))

def lr_factor(t):
    if SCHEDULE == "step":
        return STEP_GAMMA ** (t // STEP_EVERY)
    if SCHEDULE == "cosine":
        return 0.5 * (1 + math.cos(math.pi * min(t, MAX_ITERS) / MAX_ITERS))
    if SCHEDULE == "warmup":
        if t < WARMUP:
            return (t + 1) / WARMUP
        return 0.5 * (1 + math.cos(math.pi * min(t - WARMUP, MAX_ITERS - WARMUP) / (MAX_ITERS - WARMUP)))
    return 1.0

def make_optimizer(name, p):
    lr = LRS[name]
    if name == "sgd":
        return torch.optim.SGD([p], lr=lr)
    if name == "momentum":                       # v <- beta v + g ; p <- p - lr v
        return torch.optim.SGD([p], lr=lr, momentum=BETA, dampening=0)
    if name == "rmsprop":                        # s <- rho s + (1 - rho) g^2 ; p <- p - lr g / (sqrt(s) + eps)
        return torch.optim.RMSprop([p], lr=lr, alpha=RHO, eps=EPS)
    return torch.optim.Adam([p], lr=lr, betas=(BETA, BETA2), eps=EPS)

for name in ["sgd", "momentum", "rmsprop", "adam"]:
    p = torch.tensor(START, dtype=torch.float64, requires_grad=True)
    opt = make_optimizer(name, p)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    status, t = "maxed", 0
    while t < MAX_ITERS:
        x, y = p.detach().tolist()
        opt.zero_grad()
        p.grad = torch.tensor(grad(x, y), dtype=torch.float64)
        opt.step()
        sched.step()
        t += 1
        nx, ny = p.detach().tolist()
        if not (math.isfinite(nx) and math.isfinite(ny) and X_LO <= nx <= X_HI and Y_LO <= ny <= Y_HI):
            status = "diverged"
            break
        if rosenbrock(nx, ny) < CONVERGED:
            status = "converged"
            break
    loss = rosenbrock(nx, ny) if math.isfinite(nx) and math.isfinite(ny) else float("inf")
    print(f"{name:9s} lr {LRS[name]:<7g} {status:9s} after {t:4d} updates   loss {loss:.4e}   at ({nx:.4f}, {ny:.4f})")
`;

/* ───────────────────────── Architecture Builder ───────────────────────── */

export interface ArchExport {
  mode: Mode;
  input: Shape;
  layers: Layer[];
  /** MLP mode only: the live-training setup and the lab's current toy data. */
  lr?: number;
  epochs?: number;
  dataset?: ToyKind;
  data?: DataPoint[];
}

const kerasAct = (a: Layer['activation']) => (a === 'none' || a === 'leaky' || !a ? 'None' : `"${a}"`);
const kerasInit = (a: Layer['activation']) =>
  `keras.initializers.VarianceScaling(${a === 'relu' || a === 'leaky' ? '2.0' : '1.0'}, "fan_in", "untruncated_normal")`;

export const architectureBuilderPython = (o: ArchExport) => {
  const mlp = o.mode === 'mlp';
  const lines = o.layers.map((l) => {
    switch (l.kind) {
      case 'conv': {
        const lr = l.activation === 'leaky';
        return `    layers.Conv2D(${l.filters}, ${l.kernel}, strides=${l.stride}, padding="${l.padding}", activation=${kerasAct(l.activation)}),${lr ? `\n    layers.LeakyReLU(negative_slope=${LEAKY_SLOPE}),` : ''}`;
      }
      case 'pool': return `    layers.MaxPooling2D(${l.pool}),`;
      case 'flatten': return '    layers.Flatten(),';
      case 'dense': {
        const lr = l.activation === 'leaky';
        const init = mlp ? `, kernel_initializer=${kerasInit(l.activation)}` : '';
        return `    layers.Dense(${l.units}, activation=${kerasAct(l.activation)}${init}),${lr ? `\n    layers.LeakyReLU(negative_slope=${LEAKY_SLOPE}),` : ''}`;
      }
      case 'dropout': return `    layers.Dropout(${l.rate}),`;
      case 'batchnorm': return `    layers.BatchNormalization(momentum=${MLP_BN_MOMENTUM}, epsilon=${BN_EPSILON}),`;
      default: { const _exhaustive: never = l.kind; return _exhaustive; }
    }
  }).join('\n');
  const inputShape = mlp ? `(${o.input.c},)` : `(${o.input.h}, ${o.input.w}, ${o.input.c})`;
  const header = `${mlp ? 'import numpy as np\n' : ''}import keras
from keras import layers, models

# Architecture composed in the Architecture Builder lab (${o.mode.toUpperCase()} mode). model.summary()
# prints the per-layer output shapes and parameter counts the lab shows — Total params
# includes BatchNormalization's non-trainable moving mean/variance (2*C per layer), exactly
# as the lab's trainable / non-trainable split reports.
model = models.Sequential([
    layers.Input(shape=${inputShape}),
${lines}
])
model.summary()
`;
  if (!mlp) return header;
  const pts = o.data ?? [];
  const tr = pts.filter((d) => d.train), va = pts.filter((d) => !d.train);
  const arr = (ps: DataPoint[]) => `np.array([${ps.map((d) => `[${d.x}, ${d.y}]`).join(', ')}])`;
  const lab = (ps: DataPoint[]) => `np.array([${ps.map((d) => d.label).join(', ')}])`;
  return `${header}
# ---- live training, as in the lab: full-batch gradient descent on binary cross-entropy ----
# The lab's current ${o.dataset ?? 'toy'} data (${tr.length} train / ${va.length} validation points, embedded).
# Weights are initialised like the lab (He normal for relu/leaky, LeCun normal otherwise,
# zero biases) but with Keras's own random numbers; Dropout masks are random too.
X_train = ${arr(tr)}
y_train = ${lab(tr)}
X_val = ${arr(va)}
y_val = ${lab(va)}

model.compile(optimizer=keras.optimizers.SGD(learning_rate=${o.lr ?? 0.5}), loss="binary_crossentropy", metrics=["accuracy"])
hist = model.fit(X_train, y_train, batch_size=len(X_train), epochs=${o.epochs ?? MLP_MAX_EPOCHS}, shuffle=False,
                 validation_data=(X_val, y_val), verbose=0)
h = hist.history
print(f"after {len(h['loss'])} epochs: train loss {h['loss'][-1]:.3f}  val loss {h['val_loss'][-1]:.3f}")
_, train_acc = model.evaluate(X_train, y_train, verbose=0)   # inference mode, like the lab's chips
_, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"train acc {train_acc:.1%}  val acc {val_acc:.1%}  gap {train_acc - val_acc:+.1%}")
`;
};

/* ───────────────────────── export-check samples ───────────────────────── */

const L = (kind: Layer['kind'], extra: Partial<Layer> = {}, id: string = kind): Layer => ({ id, kind, ...extra });
const CNN_DEFAULT: Layer[] = [
  L('conv', { kernel: 3, filters: 32, stride: 1, padding: 'same', activation: 'relu' }, 'c1'), L('pool', { pool: 2 }, 'p1'),
  L('flatten', {}, 'f1'), L('dense', { units: 64, activation: 'relu' }, 'd1'), L('dense', { units: 10, activation: 'none' }, 'd2'),
];
const CNN_EVERYTHING: Layer[] = [
  L('conv', { kernel: 5, filters: 16, stride: 2, padding: 'valid', activation: 'leaky' }, 'c1'), L('batchnorm', {}, 'b1'),
  L('pool', { pool: 2 }, 'p1'), L('conv', { kernel: 3, filters: 8, stride: 1, padding: 'same', activation: 'tanh' }, 'c2'),
  L('flatten', {}, 'f1'), L('dropout', { rate: 0.5 }, 'x1'), L('dense', { units: 32, activation: 'sigmoid' }, 'd1'), L('dense', { units: 10, activation: 'none' }, 'd2'),
];
const MLP_DEFAULT: Layer[] = [
  L('dense', { units: 16, activation: 'relu' }, 'd1'), L('dense', { units: 8, activation: 'relu' }, 'd2'), L('dense', { units: 1, activation: 'sigmoid' }, 'h'),
];
const MLP_EVERYTHING: Layer[] = [
  L('dense', { units: 64, activation: 'leaky' }, 'd1'), L('batchnorm', {}, 'b1'), L('dropout', { rate: 0.3 }, 'x1'),
  L('dense', { units: 32, activation: 'tanh' }, 'd2'), L('dense', { units: 8, activation: 'none' }, 'd3'), L('dense', { units: 1, activation: 'sigmoid' }, 'h'),
];

export const PYTHON_SAMPLES: PythonSample[] = [
  // Residual Networks: defaults, both branch scales, depth / gain extremes of the sliders.
  { name: 'resnet-default', code: () => resnetPython(28, 0.9, 'invsqrt', true) },
  { name: 'resnet-alpha1-plain-focus', code: () => resnetPython(28, 0.9, 'one', false) },
  { name: 'resnet-deep-explode', code: () => resnetPython(64, 1.8, 'one', true) },
  { name: 'resnet-shallow-small-gain', code: () => resnetPython(8, 0.3, 'invsqrt', false) },
  // Batch Normalization: defaults (BN off), BN train, BN eval (K = 40 and K = 0), collapse init, extremes.
  { name: 'batchnorm-default', code: () => batchNormPython({ depth: 16, gain: 2.5, batch: 256, useBN: false, mode: 'train', warmup: 40 }) },
  { name: 'batchnorm-bn-train', code: () => batchNormPython({ depth: 16, gain: 2.5, batch: 256, useBN: true, mode: 'train', warmup: 40 }) },
  { name: 'batchnorm-bn-eval', code: () => batchNormPython({ depth: 16, gain: 2.5, batch: 256, useBN: true, mode: 'eval', warmup: 40 }) },
  { name: 'batchnorm-bn-eval-k0', code: () => batchNormPython({ depth: 16, gain: 2.5, batch: 256, useBN: true, mode: 'eval', warmup: 0 }) },
  { name: 'batchnorm-collapse', code: () => batchNormPython({ depth: 30, gain: 0.6, batch: 32, useBN: false, mode: 'train', warmup: 40, seed: 12 }) },
  { name: 'batchnorm-eval-big', code: () => batchNormPython({ depth: 30, gain: 3, batch: 512, useBN: true, mode: 'eval', warmup: 60 }) },
  // Dropout: defaults, the p = 0.5 preset, data-size and rate extremes.
  { name: 'dropout-default', code: () => dropoutPython({ p: 0.3, perCluster: 28, seed: 303 }) },
  { name: 'dropout-p05', code: () => dropoutPython({ p: 0.5, perCluster: 28, seed: 303 }) },
  { name: 'dropout-small-rate-small-data', code: () => dropoutPython({ p: 0.05, perCluster: 16, seed: 304 }) },
  { name: 'dropout-max-rate-big-data', code: () => dropoutPython({ p: 0.6, perCluster: 40, seed: 305 }) },
  // Transfer Learning: defaults (frozen and fine-tune highlight), no shift, maximum shift.
  { name: 'transfer-default', code: () => transferPython({ seed: 7, rotationDeg: 20, n: 16, mode: 'frozen' }), timeoutSec: 180 },
  { name: 'transfer-finetune', code: () => transferPython({ seed: 7, rotationDeg: 20, n: 160, mode: 'finetune' }), timeoutSec: 180 },
  { name: 'transfer-no-shift', code: () => transferPython({ seed: 8, rotationDeg: 0, n: 4, mode: 'frozen' }), timeoutSec: 180 },
  { name: 'transfer-max-shift', code: () => transferPython({ seed: 7, rotationDeg: 45, n: 64, mode: 'finetune' }), timeoutSec: 180 },
  // Optimizers: both presets under every schedule, beta extremes.
  ...(['constant', 'step', 'cosine', 'warmup'] as Schedule[]).flatMap((s) => [
    { name: `optimizers-moderate-${s}`, code: () => optimizersPython({ lrs: LR_PRESETS.moderate, beta: 0.9, schedule: s }) },
    { name: `optimizers-high-${s}`, code: () => optimizersPython({ lrs: LR_PRESETS.high, beta: 0.9, schedule: s }) },
  ]),
  { name: 'optimizers-beta0', code: () => optimizersPython({ lrs: LR_PRESETS.moderate, beta: 0, schedule: 'constant' }) },
  { name: 'optimizers-beta095', code: () => optimizersPython({ lrs: LR_PRESETS.moderate, beta: 0.95, schedule: 'constant' }) },
  // Architecture Builder (Keras — not installed here, so parse/name checks only).
  { name: 'arch-cnn-default', code: () => architectureBuilderPython({ mode: 'cnn', input: { h: 32, w: 32, c: 3 }, layers: CNN_DEFAULT }), noRun: true },
  { name: 'arch-cnn-everything', code: () => architectureBuilderPython({ mode: 'cnn', input: { h: 32, w: 32, c: 3 }, layers: CNN_EVERYTHING }), noRun: true },
  { name: 'arch-mlp-default', code: () => architectureBuilderPython({ mode: 'mlp', input: { h: 1, w: 1, c: 2 }, layers: MLP_DEFAULT, lr: 0.5, epochs: MLP_MAX_EPOCHS, dataset: 'xor', data: makeData('xor') }), noRun: true },
  { name: 'arch-mlp-everything', code: () => architectureBuilderPython({ mode: 'mlp', input: { h: 1, w: 1, c: 2 }, layers: MLP_EVERYTHING, lr: 1, epochs: MLP_MAX_EPOCHS, dataset: 'spirals', data: makeData('spirals') }), noRun: true },
  { name: 'arch-mlp-logistic', code: () => architectureBuilderPython({ mode: 'mlp', input: { h: 1, w: 1, c: 2 }, layers: [MLP_DEFAULT[2]!], lr: 0.05, epochs: MLP_MAX_EPOCHS, dataset: 'circles', data: makeData('circles') }), noRun: true },
];
