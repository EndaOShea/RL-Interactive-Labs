// Forward pass of a deep tanh stack with / without Batch Normalization, for the
// Batch Normalization lab. Pure module, no React.
//
// Layer l (1..L):  z_l = W_l a_{l-1}  →  u_l = BN(z_l) or z_l  →  a_l = tanh(u_l)
// with a_0 = x ~ N(0, I) and W_l entries ~ N(0, gain²/d). BN normalises each
// feature over the batch, x̂ = (z − μ)/√(σ² + ε), with γ = 1 and β = 0 — their
// initial values; nothing is trained here. Conventions follow torch.nn.BatchNorm1d
// so the Python export reproduces every number:
//   • train mode: batch mean + BIASED batch variance, ε = 1e-5 inside the sqrt;
//     running_mean ← (1−m)·running_mean + m·μ,
//     running_var  ← (1−m)·running_var  + m·σ²·B/(B−1)   (unbiased), m = 0.1,
//     starting from running_mean = 0, running_var = 1;
//   • eval mode: normalise with the running statistics gathered from K warm-up
//     batches run in train mode (K = 0 leaves them at 0 / 1, i.e. no-op BN).
// Recorded per layer: the pre-activation u_l (what BN pins to std 1) and two
// health signals of the tanh units it feeds.
import { mulberry32, gauss } from './seeded';
import type { Rng } from './seeded';

export const BN_D = 24;           // features per layer
export const BN_SEED = 11;        // weights use BN_SEED, the batch BN_SEED+1, warm-up batches BN_SEED+2
export const BN_EPS = 1e-5;
export const BN_MOMENTUM = 0.1;
export const SAT_LEVEL = 0.95;    // |tanh(u)| above this counts as saturated

export type BnMode = 'train' | 'eval';

export interface BnConfig {
  depth: number;
  gain: number;
  batch: number;
  useBN: boolean;
  mode: BnMode;       // only used when useBN
  warmup: number;     // K warm-up batches for the running statistics (eval mode)
  seed?: number;
}

export interface BnLayerStat {
  layer: number;        // 0 = the input batch x, 1..L = layers
  preStd: number;       // mean over features of the batch std of u_l (for layer 0: of x)
  preAbsMean: number;   // mean over features of |batch mean of u_l|
  meanDeriv: number;    // mean of tanh′(u_l) = 1 − tanh²(u_l) over batch × features (NaN for layer 0)
  sat: number;          // fraction of |tanh(u_l)| > SAT_LEVEL (NaN for layer 0)
}

const gaussMatrix = (r: Rng, n: number): Float64Array => {
  const m = new Float64Array(n);
  for (let i = 0; i < n; i++) m[i] = gauss(r);
  return m;
};

/** z = a · Wᵀ · s for a batch a (B×d, row-major) and W (d×d, [out][in]). */
function linear(a: Float64Array, W: Float64Array, s: number, B: number, d: number): Float64Array {
  const z = new Float64Array(B * d);
  for (let b = 0; b < B; b++) {
    for (let o = 0; o < d; o++) {
      let acc = 0;
      for (let i = 0; i < d; i++) acc += W[o * d + i]! * a[b * d + i]!;
      z[b * d + o] = acc * s;
    }
  }
  return z;
}

/** Per-feature batch mean and biased variance. */
function moments(z: Float64Array, B: number, d: number): { mean: Float64Array; varB: Float64Array } {
  const mean = new Float64Array(d);
  const varB = new Float64Array(d);
  for (let b = 0; b < B; b++) for (let j = 0; j < d; j++) mean[j] = mean[j]! + z[b * d + j]!;
  for (let j = 0; j < d; j++) mean[j] = mean[j]! / B;
  for (let b = 0; b < B; b++) for (let j = 0; j < d; j++) { const v = z[b * d + j]! - mean[j]!; varB[j] = varB[j]! + v * v; }
  for (let j = 0; j < d; j++) varB[j] = varB[j]! / B;
  return { mean, varB };
}

function normalise(z: Float64Array, mean: Float64Array, varr: Float64Array, B: number, d: number): Float64Array {
  const out = new Float64Array(B * d);
  for (let j = 0; j < d; j++) {
    const inv = 1 / Math.sqrt(varr[j]! + BN_EPS);
    for (let b = 0; b < B; b++) out[b * d + j] = (z[b * d + j]! - mean[j]!) * inv;
  }
  return out;
}

function statOf(u: Float64Array, layer: number, B: number, d: number, withTanh: boolean): BnLayerStat {
  const { mean, varB } = moments(u, B, d);
  let sStd = 0, sAbs = 0;
  for (let j = 0; j < d; j++) { sStd += Math.sqrt(varB[j]!); sAbs += Math.abs(mean[j]!); }
  let sDer = 0, nSat = 0;
  if (withTanh) {
    for (let i = 0; i < u.length; i++) {
      const t = Math.tanh(u[i]!);
      sDer += 1 - t * t;
      if (Math.abs(t) > SAT_LEVEL) nSat++;
    }
  }
  return {
    layer,
    preStd: sStd / d,
    preAbsMean: sAbs / d,
    meanDeriv: withTanh ? sDer / u.length : NaN,
    sat: withTanh ? nSat / u.length : NaN,
  };
}

const tanhAll = (u: Float64Array): Float64Array => u.map((v) => Math.tanh(v));

/**
 * Incremental runner: `input` is the stat of x (layer 0); each `next()` pushes
 * the batch through one more layer (drawing that layer's weights) and returns
 * its stat. Layer-by-layer so the eval-mode warm-up cost is spread over steps.
 */
export interface BnRunner { input: BnLayerStat; layer: number; next: () => BnLayerStat | null; }

export function createBnRunner(cfg: BnConfig): BnRunner {
  const { depth, gain, batch: B, useBN, mode } = cfg;
  const seed = cfg.seed ?? BN_SEED;
  const d = BN_D;
  const s = gain / Math.sqrt(d);
  const rW = mulberry32(seed), rX = mulberry32(seed + 1), rK = mulberry32(seed + 2);
  const X = gaussMatrix(rX, B * d);
  const evalMode = useBN && mode === 'eval';
  const K = evalMode ? Math.max(0, Math.round(cfg.warmup)) : 0;
  let warm: Float64Array[] = Array.from({ length: K }, () => gaussMatrix(rK, B * d));
  let a = X;
  const runner: BnRunner = {
    input: statOf(X, 0, B, d, false),
    layer: 0,
    next: () => {
      if (runner.layer >= depth) return null;
      const Wl = gaussMatrix(rW, d * d);
      const z = linear(a, Wl, s, B, d);
      let u = z;
      if (useBN) {
        if (evalMode) {
          // running statistics of this layer from the K warm-up batches (train mode)
          const rm = new Float64Array(d);
          const rv = new Float64Array(d).fill(1);
          warm = warm.map((ak) => {
            const zk = linear(ak, Wl, s, B, d);
            const { mean, varB } = moments(zk, B, d);
            for (let j = 0; j < d; j++) {
              rm[j] = (1 - BN_MOMENTUM) * rm[j]! + BN_MOMENTUM * mean[j]!;
              rv[j] = (1 - BN_MOMENTUM) * rv[j]! + BN_MOMENTUM * varB[j]! * (B / (B - 1));
            }
            return tanhAll(normalise(zk, mean, varB, B, d));
          });
          u = normalise(z, rm, rv, B, d);
        } else {
          const { mean, varB } = moments(z, B, d);
          u = normalise(z, mean, varB, B, d);
        }
      }
      runner.layer += 1;
      const st = statOf(u, runner.layer, B, d, true);
      a = tanhAll(u);
      return st;
    },
  };
  return runner;
}

/** Run the whole stack; returns the input stat (layer 0) followed by layers 1..L. */
export function runBatchNorm(cfg: BnConfig): BnLayerStat[] {
  const runner = createBnRunner(cfg);
  const stats: BnLayerStat[] = [runner.input];
  for (let st = runner.next(); st; st = runner.next()) stats.push(st);
  return stats;
}
