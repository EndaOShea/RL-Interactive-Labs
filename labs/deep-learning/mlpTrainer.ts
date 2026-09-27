// Real (analytic, in-browser) backprop MLP trainer for the Architecture Builder's
// MLP mode. The composed layer list is trained AS SHOWN: every hidden Dense layer
// (its units + activation), every Dropout layer at its own position with its own
// rate, and every BatchNorm layer (Keras semantics), ending in the fixed 1-unit
// sigmoid output head. Full-batch gradient descent on binary cross-entropy over
// the 2-D toy data, so overfit/underfit are EMPIRICAL (train vs validation loss),
// not just rule-flagged. Pure module, no React.
//
// Conventions (mirrored by the Keras export):
//   Dense      z = W·a + b ;  a' = f(z)   (He-normal init for relu/leaky, LeCun-normal otherwise; b = 0)
//   leaky      f(z) = z > 0 ? z : 0.01·z
//   Dropout    training: a' = a·m, m ∈ {0, 1/(1−p)} per unit and sample (inverted dropout); eval: identity
//   BatchNorm  training: batch mean + biased variance, x̂ = (a − μ)/√(σ² + ε), a' = γ·x̂ + β,
//              moving ← 0.9·moving + 0.1·batch (ε = 1e-3); eval: moving statistics
//   Head       Dense(1) → sigmoid, loss = mean BCE over the training split
//   Optimiser  plain gradient descent, one full-batch step per epoch
import type { Layer, Activation } from './archBuilder';

export type ToyKind = 'xor' | 'circles' | 'spirals';
export interface DataPoint { x: number; y: number; label: number; train: boolean; }

export const LEAKY_SLOPE = 0.01;
export const BN_EPSILON = 1e-3;
export const BN_MOMENTUM = 0.9;
export const MLP_MAX_UNITS = 64;          // Units slider cap in MLP mode (keeps live training fast)
export const MLP_MAX_EPOCHS = 250;        // live-training budget (one full-batch step per epoch)
export const TRAIN_FRACTION = 0.7;

export type OpSpec =
  | { kind: 'dense'; units: number; act: Activation; head: boolean }
  | { kind: 'dropout'; rate: number }
  | { kind: 'batchnorm' };
export interface Arch { ops: OpSpec[]; }

interface DenseL { kind: 'dense'; fanIn: number; units: number; act: Activation; head: boolean; W: Float64Array; b: Float64Array; }
interface DropL { kind: 'dropout'; rate: number; width: number; }
interface BnL { kind: 'batchnorm'; width: number; gamma: Float64Array; beta: Float64Array; mean: Float64Array; varr: Float64Array; }
type NetLayer = DenseL | DropL | BnL;
export interface Net { inputDim: number; layers: NetLayer[]; }

/* ── activations ── */
const sigm = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
const actFn = (a: Activation, z: number): number => {
  switch (a) {
    case 'relu': return z > 0 ? z : 0;
    case 'leaky': return z > 0 ? z : LEAKY_SLOPE * z;
    case 'tanh': return Math.tanh(z);
    case 'sigmoid': return sigm(z);
    default: return z; // 'none'
  }
};
// f′ evaluated from the PRE-activation z (never from a dropout-scaled output).
const actDeriv = (a: Activation, z: number): number => {
  switch (a) {
    case 'relu': return z > 0 ? 1 : 0;
    case 'leaky': return z > 0 ? 1 : LEAKY_SLOPE;
    case 'tanh': { const t = Math.tanh(z); return 1 - t * t; }
    case 'sigmoid': { const s = sigm(z); return s * (1 - s); }
    default: return 1;
  }
};

/* ── rng (this lab is not seeded — Math.random is fine here) ── */
const randn = (): number => {
  let u = 0; let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const clamp01 = (x: number) => Math.max(0.02, Math.min(0.98, x));

/* ── toy datasets in the unit square [0,1]² ── */
export function makeData(kind: ToyKind, perClass = 90, noise = 0.09): DataPoint[] {
  const pts: DataPoint[] = [];
  const push = (x: number, y: number, label: number) =>
    pts.push({ x: clamp01(x), y: clamp01(y), label, train: Math.random() < TRAIN_FRACTION });

  if (kind === 'xor') {
    const C = [{ x: 0.3, y: 0.3, l: 0 }, { x: 0.7, y: 0.3, l: 1 }, { x: 0.3, y: 0.7, l: 1 }, { x: 0.7, y: 0.7, l: 0 }];
    for (const c of C) for (let i = 0; i < Math.round(perClass / 2); i++) push(c.x + randn() * noise * 1.3, c.y + randn() * noise * 1.3, c.l);
  } else if (kind === 'circles') {
    for (let i = 0; i < perClass; i++) {
      const t = Math.random() * 2 * Math.PI;
      const r0 = 0.12 * Math.sqrt(Math.random());                 // inner disc → class 0
      push(0.5 + r0 * Math.cos(t), 0.5 + r0 * Math.sin(t) + randn() * noise * 0.4, 0);
      const r1 = 0.33 + randn() * noise * 0.5;                     // outer ring → class 1
      push(0.5 + r1 * Math.cos(t), 0.5 + r1 * Math.sin(t), 1);
    }
  } else { // spirals
    for (let i = 0; i < perClass; i++) {
      const t = (i / perClass) * 3.2;                             // radius grows with t
      const r = 0.03 + t * 0.13;
      for (const l of [0, 1]) {
        const ang = t * 2.4 + l * Math.PI + randn() * noise * 1.2;
        push(0.5 + r * Math.cos(ang), 0.5 + r * Math.sin(ang), l);
      }
    }
  }
  return pts;
}

/** Index of the output head in a layer list: the LAST Dense layer (−1 if none). */
export const headIndex = (layers: Layer[]): number => {
  for (let i = layers.length - 1; i >= 0; i--) if (layers[i]!.kind === 'dense') return i;
  return -1;
};

/**
 * The trainable architecture: every layer before the head in order (Dense /
 * Dropout / BatchNorm), then the 1-unit sigmoid head. The MLP-mode UI pins the
 * head to 1 unit · sigmoid and keeps it last, so this is exactly the listed stack.
 */
export function archFromLayers(layers: Layer[]): Arch {
  const h = headIndex(layers);
  const ops: OpSpec[] = [];
  layers.slice(0, Math.max(0, h)).forEach((l) => {
    if (l.kind === 'dense') ops.push({ kind: 'dense', units: Math.max(1, l.units ?? 16), act: l.activation ?? 'relu', head: false });
    else if (l.kind === 'dropout') ops.push({ kind: 'dropout', rate: Math.min(0.95, Math.max(0, l.rate ?? 0)) });
    else if (l.kind === 'batchnorm') ops.push({ kind: 'batchnorm' });
  });
  ops.push({ kind: 'dense', units: 1, act: 'sigmoid', head: true });
  return { ops };
}

export function initNet(arch: Arch, inputDim = 2): Net {
  const layers: NetLayer[] = [];
  let width = inputDim;
  for (const op of arch.ops) {
    if (op.kind === 'dense') {
      const relu = op.act === 'relu' || op.act === 'leaky';
      const std = Math.sqrt((relu ? 2 : 1) / width);
      const W = new Float64Array(op.units * width);
      for (let i = 0; i < W.length; i++) W[i] = randn() * std;
      layers.push({ kind: 'dense', fanIn: width, units: op.units, act: op.act, head: op.head, W, b: new Float64Array(op.units) });
      width = op.units;
    } else if (op.kind === 'dropout') {
      layers.push({ kind: 'dropout', rate: op.rate, width });
    } else {
      layers.push({ kind: 'batchnorm', width, gamma: new Float64Array(width).fill(1), beta: new Float64Array(width), mean: new Float64Array(width), varr: new Float64Array(width).fill(1) });
    }
  }
  return { inputDim, layers };
}

/* ── forward / backward over a batch (row-major N × width) ── */
interface Cache { inp: Float64Array; z?: Float64Array; mask?: Float64Array; xhat?: Float64Array; invStd?: Float64Array; }

function forward(net: Net, X: Float64Array, N: number, training: boolean): { out: Float64Array; caches: Cache[] } {
  let a = X;
  const caches: Cache[] = [];
  for (const L of net.layers) {
    if (L.kind === 'dense') {
      const z = new Float64Array(N * L.units);
      const o = new Float64Array(N * L.units);
      for (let n = 0; n < N; n++) {
        for (let j = 0; j < L.units; j++) {
          let s = L.b[j]!;
          for (let i = 0; i < L.fanIn; i++) s += L.W[j * L.fanIn + i]! * a[n * L.fanIn + i]!;
          z[n * L.units + j] = s;
          o[n * L.units + j] = L.head ? s : actFn(L.act, s);     // the head keeps its logit
        }
      }
      caches.push({ inp: a, z });
      a = o;
    } else if (L.kind === 'dropout') {
      if (!training || L.rate <= 0) { caches.push({ inp: a }); continue; }
      const keep = 1 - L.rate;
      const mask = new Float64Array(N * L.width);
      const o = new Float64Array(N * L.width);
      for (let k = 0; k < mask.length; k++) { mask[k] = Math.random() < keep ? 1 / keep : 0; o[k] = a[k]! * mask[k]!; }
      caches.push({ inp: a, mask });
      a = o;
    } else {
      const C = L.width;
      const mu = new Float64Array(C), vr = new Float64Array(C);
      if (training) {
        for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) mu[c] = mu[c]! + a[n * C + c]! / N;
        for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { const d = a[n * C + c]! - mu[c]!; vr[c] = vr[c]! + (d * d) / N; }
        for (let c = 0; c < C; c++) {
          L.mean[c] = BN_MOMENTUM * L.mean[c]! + (1 - BN_MOMENTUM) * mu[c]!;
          L.varr[c] = BN_MOMENTUM * L.varr[c]! + (1 - BN_MOMENTUM) * vr[c]!;
        }
      } else {
        mu.set(L.mean); vr.set(L.varr);
      }
      const invStd = vr.map((v) => 1 / Math.sqrt(v + BN_EPSILON));
      const xhat = new Float64Array(N * C);
      const o = new Float64Array(N * C);
      for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
        const k = n * C + c;
        xhat[k] = (a[k]! - mu[c]!) * invStd[c]!;
        o[k] = L.gamma[c]! * xhat[k]! + L.beta[c]!;
      }
      caches.push({ inp: a, xhat, invStd });
      a = o;
    }
  }
  return { out: a, caches };
}

const toMatrix = (pts: DataPoint[]): Float64Array => {
  const X = new Float64Array(pts.length * 2);
  pts.forEach((p, i) => { X[2 * i] = p.x; X[2 * i + 1] = p.y; });
  return X;
};

export const predictProb = (net: Net, x0: number, x1: number): number =>
  sigm(forward(net, Float64Array.of(x0, x1), 1, false).out[0]!);

const bceLogit = (z: number, y: number) => Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));

export function evaluate(net: Net, data: DataPoint[]): { trainLoss: number; valLoss: number; trainAcc: number; valAcc: number } {
  const out = forward(net, toMatrix(data), data.length, false).out;
  let tl = 0; let vl = 0; let ta = 0; let va = 0; let tn = 0; let vn = 0;
  data.forEach((d, i) => {
    const z = out[i]!;
    const correct = (z >= 0 ? 1 : 0) === d.label ? 1 : 0;
    if (d.train) { tl += bceLogit(z, d.label); ta += correct; tn++; } else { vl += bceLogit(z, d.label); va += correct; vn++; }
  });
  return { trainLoss: tn ? tl / tn : 0, valLoss: vn ? vl / vn : 0, trainAcc: tn ? ta / tn : 0, valAcc: vn ? va / vn : 0 };
}

/** One full-batch gradient-descent epoch over the training split (mutates net). */
export function trainEpoch(net: Net, data: DataPoint[], lr: number): void {
  const train = data.filter((d) => d.train);
  const N = train.length;
  if (!N) return;
  const { out, caches } = forward(net, toMatrix(train), N, true);
  // mean BCE through the sigmoid head: ∂L/∂z_out = (σ(z) − y)/N
  let g: Float64Array = new Float64Array(N);
  for (let n = 0; n < N; n++) g[n] = (sigm(out[n]!) - train[n]!.label) / N;
  for (let li = net.layers.length - 1; li >= 0; li--) {
    const L = net.layers[li]!;
    const c = caches[li]!;
    if (L.kind === 'dense') {
      const z = c.z!;
      // through the activation (hidden layers only), from the pre-activation z
      if (!L.head) for (let k = 0; k < g.length; k++) g[k] = g[k]! * actDeriv(L.act, z[k]!);
      const gIn = new Float64Array(N * L.fanIn);
      const gW = new Float64Array(L.W.length), gb = new Float64Array(L.units);
      for (let n = 0; n < N; n++) {
        for (let j = 0; j < L.units; j++) {
          const gj = g[n * L.units + j]!;
          if (gj === 0) continue;
          gb[j] = gb[j]! + gj;
          for (let i = 0; i < L.fanIn; i++) {
            gW[j * L.fanIn + i] = gW[j * L.fanIn + i]! + gj * c.inp[n * L.fanIn + i]!;
            gIn[n * L.fanIn + i] = gIn[n * L.fanIn + i]! + gj * L.W[j * L.fanIn + i]!;
          }
        }
      }
      for (let k = 0; k < L.W.length; k++) L.W[k] = L.W[k]! - lr * gW[k]!;
      for (let j = 0; j < L.units; j++) L.b[j] = L.b[j]! - lr * gb[j]!;
      g = gIn;
    } else if (L.kind === 'dropout') {
      if (c.mask) { const m = c.mask; g = g.map((v, k) => v * m[k]!); }
    } else {
      const C = L.width, xh = c.xhat!, inv = c.invStd!;
      const sg = new Float64Array(C), sgx = new Float64Array(C);
      for (let n = 0; n < N; n++) for (let q = 0; q < C; q++) { const k = n * C + q; sg[q] = sg[q]! + g[k]!; sgx[q] = sgx[q]! + g[k]! * xh[k]!; }
      const gIn = new Float64Array(N * C);
      for (let n = 0; n < N; n++) for (let q = 0; q < C; q++) {
        const k = n * C + q;
        gIn[k] = L.gamma[q]! * inv[q]! * (g[k]! - sg[q]! / N - (xh[k]! * sgx[q]!) / N);
      }
      for (let q = 0; q < C; q++) { L.gamma[q] = L.gamma[q]! - lr * sgx[q]!; L.beta[q] = L.beta[q]! - lr * sg[q]!; }
      g = gIn;
    }
  }
}

/** Human-readable summary, e.g. "2 → 16 relu → drop 0.3 → 8 relu → BN → 1 sigmoid". */
export const archSummary = (arch: Arch): string => {
  const parts = ['2'];
  for (const op of arch.ops) {
    if (op.kind === 'dense') parts.push(op.head ? '1 sigmoid' : `${op.units} ${op.act}`);
    else if (op.kind === 'dropout') parts.push(`drop ${op.rate}`);
    else parts.push('BN');
  }
  return parts.join(' → ');
};
