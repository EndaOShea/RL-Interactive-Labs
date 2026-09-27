// Runnable Python exports for the Image Classification labs (template strings —
// not LLM generated), mirroring the on-screen pipeline and parameters exactly.
// Kernels, filters and constants are serialised from the same modules the labs
// run (convMath.ts, featureMath.ts), so the export cannot drift from them.
import type { PythonSample } from '../../utils/pythonSamples';
import { IMG_PRESETS, KERNELS, KERNEL_NAMES, N as IMG_N, PAD_MODES } from './convMath';
import type { ImgPreset, KernelName, PadMode } from './convMath';
import { CLASSES, FILTERS, GLYPH, NOISE_AMP, NOISE_SEED, PERTURBS, POOL_MODES, SOFTMAX_SCALE } from './featureMath';
import type { ClassId, Perturb, PoolMode } from './featureMath';

/** A number as a Python literal: integers plain, ninths/sixteenths as fractions, else JS's round-trip repr. */
function pyNum(v: number): string {
  if (Number.isInteger(v)) return String(v);
  for (const d of [9, 16]) {
    const n = v * d;
    if (Math.abs(n - Math.round(n)) < 1e-12) return `${Math.round(n)}/${d}`;
  }
  return String(v);
}
const pyMatrix = (m: number[][]) => `[${m.map((row) => `[${row.map(pyNum).join(', ')}]`).join(', ')}]`;

// numpy `mode=` string for each border-padding choice in the lab.
const PAD_NP: Record<PadMode, string> = { zero: 'constant', replicate: 'edge', reflect: 'reflect' };

export const convPython = (
  image: ImgPreset,
  kernelName: KernelName,
  padding: PadMode = 'zero',
  stride = 1,
  flip = false,
) => `import numpy as np

# 2-D ${flip ? 'true convolution' : 'cross-correlation (what CNN layers call "convolution")'} — mirrors the lab exactly.
# image: ${image}   kernel: ${kernelName}   padding: ${padding}   stride: ${stride}
N = ${IMG_N}
IMAGE = "${image}"
K = np.array(${pyMatrix(KERNELS[kernelName])}, dtype=float)
FLIP = ${flip ? 'True' : 'False'}   # True = true convolution: cross-correlate with K rotated 180 degrees
PAD_MODE = "${PAD_NP[padding]}"   # zero->constant, replicate->edge, reflect->reflect (P = 1)
STRIDE = ${stride}


def make_image(kind, n=N):
    """The lab's ${IMG_N}x${IMG_N} test images (1 = ink)."""
    r, c = np.indices((n, n))
    if kind == "cross":          # 2-px bars through the middle, full length
        mask = ((c >= 6) & (c < 8)) | ((r >= 6) & (r < 8))
    elif kind == "diagonal":     # 2-px-wide main diagonal
        mask = (c == r) | (c == r + 1)
    else:                        # ring of radius 5 around the centre
        cy = cx = (n - 1) / 2
        mask = np.abs(np.hypot(r - cy, c - cx) - 5) < 1.1
    return mask.astype(float)


def correlate2d(img, kernel, pad_mode=PAD_MODE, stride=STRIDE):
    """out(i,j) = sum_mn I(s*i+m, s*j+n) * K(m,n) for m,n in {-1,0,1}.

    Output side = floor((W - F + 2P) / S) + 1 with W=${IMG_N}, F=3, P=1.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode=pad_mode)
    H, W = img.shape
    out_h = (H - kh + 2 * ph) // stride + 1
    out_w = (W - kw + 2 * pw) // stride + 1
    out = np.zeros((out_h, out_w))
    for oi in range(out_h):
        for oj in range(out_w):
            i, j = oi * stride, oj * stride          # window centre in the input
            out[oi, oj] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
    return out


def reads_padding(i, j, n=N):
    """True when the 3x3 window centred on input (i, j) reads a padded pixel."""
    return i - 1 < 0 or j - 1 < 0 or i + 1 > n - 1 or j + 1 > n - 1


def peaks(out, stride=STRIDE):
    """Strongest responses ranked by |value| in sweep (row-major) order.

    Returns (overall, interior) as (value, row, col, reads_padding); ties keep the
    first cell swept. Border peaks are partly artefacts of the padding mode.
    """
    best_all = best_in = None
    for oi in range(out.shape[0]):
        for oj in range(out.shape[1]):
            v = float(out[oi, oj])
            border = reads_padding(oi * stride, oj * stride)
            if best_all is None or abs(v) > abs(best_all[0]) + 1e-12:
                best_all = (v, oi, oj, border)
            if not border and (best_in is None or abs(v) > abs(best_in[0]) + 1e-12):
                best_in = (v, oi, oj, border)
    return best_all, best_in


if __name__ == "__main__":
    img = make_image(IMAGE)
    kernel = np.rot90(K, 2) if FLIP else K   # rotating 180 deg = flipping both axes
    feat = correlate2d(img, kernel)
    M = float(np.abs(feat).max()) or 1.0     # the lab's zero-centred colour scale: [-M, +M]
    zeros = int(np.sum(np.abs(feat) < 1e-12))
    overall, interior = peaks(feat)

    print("operation   :", "true convolution" if FLIP else "cross-correlation")
    print("kernel used :\\n", kernel)
    print("output size :", feat.shape, "  (stride", STRIDE, ", padding", PAD_MODE + ")")
    print("output range: [%.3f, %.3f]   colour scale: +/-%.3f (0 = neutral)" % (feat.min(), feat.max(), M))
    print("exact zeros : %d of %d cells (flat regions)" % (zeros, feat.size))
    v, i, j, b = overall
    print("largest |out|: %+.3f at (%d,%d)%s" % (v, i, j, "  <- border: window reads padding" if b else ""))
    if interior is not None:
        v, i, j, _ = interior
        print("strongest interior: %+.3f at (%d,%d)" % (v, i, j))
    print("feature map:\\n", np.round(feat, 3))
`;

export const featureMapsPython = (
  cls: ClassId = 'H',
  perturb: Perturb = 'shift1',
  pooling: PoolMode = 'max',
) => `import numpy as np

# Tiny CNN forward pass on a perturbed glyph — mirrors the lab exactly.
# Pipeline: glyph -> perturbation -> conv (3 FIXED filters; same-size, zero-padded
#           cross-correlation) -> ReLU -> 2x2 pooling -> flatten (${FILTERS.length * (GLYPH / 2) ** 2} dims)
#           -> cosine to the CLEAN class templates -> softmax(${SOFTMAX_SCALE} * cos).
# Honest note: filters are hand-picked and the classifier is template matching
# (no training). A real CNN learns both by backpropagation.

CLASS = "${cls}"            # input glyph: ${CLASSES.join(' | ')}
PERTURB = "${perturb}"      # ${PERTURBS.join(' | ')}
POOLING = "${pooling}"        # ${POOL_MODES.join(' | ')}
SOFTMAX_SCALE = ${SOFTMAX_SCALE}
NOISE_SEED, NOISE_AMP = ${NOISE_SEED}, ${NOISE_AMP}
G = ${GLYPH}

FILTERS = {
${FILTERS.map((f) => `    "${f.id}": np.array(${pyMatrix(f.k)}, dtype=float),`).join('\n')}
}


def make_glyph(cls):
    m = np.zeros((G, G))
    if cls == "H":
        m[2:10, 2:4] = 1; m[2:10, 8:10] = 1; m[5:7, 2:10] = 1
    elif cls == "T":
        m[2:4, 2:10] = 1; m[2:10, 5:7] = 1
    else:  # O ring
        m[2:10, 2:10] = 1; m[4:8, 4:8] = 0
    return m


def mulberry32(seed):
    """Exact port of the lab's 32-bit PRNG (same stream as the browser)."""
    a = seed & 0xFFFFFFFF
    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rnd


def shift(img, dy, dx):
    """Translate by (dy, dx); pixels shifted in from outside are 0."""
    H, W = img.shape
    out = np.zeros_like(img)
    for r in range(H):
        for c in range(W):
            rr, cc = r - dy, c - dx
            if 0 <= rr < H and 0 <= cc < W:
                out[r, c] = img[rr, cc]
    return out


def dilate(img):
    """Grow every stroke by one pixel (3x3 max filter)."""
    H, W = img.shape
    out = np.zeros_like(img)
    for r in range(H):
        for c in range(W):
            out[r, c] = img[max(0, r - 1):r + 2, max(0, c - 1):c + 2].max()
    return out


def add_noise(img, seed=NOISE_SEED, amp=NOISE_AMP):
    """Seeded uniform noise in [-amp, amp], drawn in row-major order, clipped to [0, 1]."""
    rnd = mulberry32(seed)
    out = img.copy()
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            out[r, c] = min(1.0, max(0.0, img[r, c] + amp * (2.0 * rnd() - 1.0)))
    return out


def perturb(img, kind):
    if kind == "shift1":
        return shift(img, 0, 1)     # 1 px to the right
    if kind == "shift2":
        return shift(img, 0, 2)     # 2 px to the right
    if kind == "thick":
        return dilate(img)
    if kind == "noise":
        return add_noise(img)
    return img.copy()               # clean


def conv_same(img, kernel):
    """Same-size 3x3 cross-correlation with zero padding."""
    padded = np.pad(img, 1, mode="constant")
    out = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(padded[i:i + 3, j:j + 3] * kernel)
    return out


def pool2x2(x, mode):
    H2, W2 = x.shape[0] // 2, x.shape[1] // 2
    out = np.zeros((H2, W2))
    for i in range(H2):
        for j in range(W2):
            block = x[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            out[i, j] = block.max() if mode == "max" else block.mean()
    return out


def forward(img, mode):
    raw = [conv_same(img, k) for k in FILTERS.values()]
    relu = [np.maximum(0.0, z) for z in raw]
    pooled = [pool2x2(a, mode) for a in relu]
    return raw, relu, np.concatenate([p.ravel() for p in pooled])   # row-major flatten


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def classify(query, mode, classes):
    templates = {c: forward(make_glyph(c), mode)[2] for c in classes}   # CLEAN glyphs
    vec = forward(query, mode)[2]
    scores = np.array([cosine(vec, templates[c]) for c in classes])
    return scores, softmax(SOFTMAX_SCALE * scores)


if __name__ == "__main__":
    classes = ${JSON.stringify(CLASSES).replace(/,/g, ', ')}
    query = perturb(make_glyph(CLASS), PERTURB)
    raw, relu, _ = forward(query, POOLING)
    print(f"query: {CLASS} ({PERTURB}), pooling: {POOLING}")
    for name, z, a in zip(FILTERS, raw, relu):
        print(f"  {name:16s} raw [{z.min():+.2f}, {z.max():+.2f}]   ReLU peak {a.max():.2f}  total {a.sum():.2f}")

    own = classes.index(CLASS)
    for mode in (POOLING, "avg" if POOLING == "max" else "max"):
        scores, probs = classify(query, mode, classes)
        pred = classes[int(np.argmax(probs))]
        margin = scores[own] - max(s for i, s in enumerate(scores) if i != own)
        tag = "selected" if mode == POOLING else "compare "
        print(f"[{tag}] {mode}-pool: cos " + "  ".join(f"{c} {s:.3f}" for c, s in zip(classes, scores))
              + f"  -> predict {pred} (p = {probs.max():.3f}), own cos {scores[own]:.3f}, margin {margin:+.3f}")
`;

// ---------------------------------------------------------------------------
// Samples for scripts/check-python-exports.mjs — every image, kernel, padding,
// stride and operation of the Convolution lab, the lab's presets, and every
// class × perturbation × pooling of the Feature Maps lab.
// ---------------------------------------------------------------------------
const convSamples: PythonSample[] = [];
const addConv = (name: string, img: ImgPreset, k: KernelName, pad: PadMode, stride: number, flip: boolean) => {
  convSamples.push({ name: `conv-${name}`, code: () => convPython(img, k, pad, stride, flip) });
};
addConv('default', 'cross', 'edge-detect', 'zero', 1, false);
KERNEL_NAMES.forEach((k) => addConv(`kernel-${k}`, 'circle', k, 'zero', 1, false));
KERNEL_NAMES.forEach((k) => addConv(`flip-${k}`, 'cross', k, 'replicate', 1, true));
IMG_PRESETS.forEach((img) => PAD_MODES.forEach((pad) => [1, 2].forEach((s) => addConv(`${img}-${pad}-s${s}`, img, 'sobel-x', pad, s, false))));
// the lab's presets
addConv('preset-edge-hunt', 'circle', 'edge-detect', 'zero', 1, false);
addConv('preset-vertical-strokes', 'cross', 'sobel-x', 'zero', 1, false);
addConv('preset-flip-sobel', 'cross', 'sobel-x', 'zero', 1, true);
addConv('preset-soften', 'diagonal', 'gaussian-blur', 'reflect', 1, false);
addConv('preset-stride2', 'cross', 'edge-detect', 'zero', 2, false);
addConv('preset-laplacian-ridge', 'circle', 'laplacian', 'replicate', 1, false);
addConv('emboss-reflect-s2-flip', 'diagonal', 'emboss', 'reflect', 2, true);

const fmSamples: PythonSample[] = CLASSES.flatMap((c) => PERTURBS.flatMap((p) => POOL_MODES.map((pm) => ({
  name: `featuremaps-${c}-${p}-${pm}`,
  code: () => featureMapsPython(c, p, pm),
}))));

export const PYTHON_SAMPLES: PythonSample[] = [...convSamples, ...fmSamples];
