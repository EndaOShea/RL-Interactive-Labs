// A tiny full-batch MLP with real backprop and Adam, shared by the Dropout and
// Transfer Learning labs. Pure module, no React.
//
//   hidden layer l  a = act_l(W·a_prev + b) · m    act_l ∈ {relu, sigmoid}; m = inverted-
//                                                  dropout mask (training only)
//   output          'bce': 1 logit → sigmoid, binary cross-entropy
//                   'ce' : K logits → softmax, cross-entropy
//                   'ovr': K logits → K independent sigmoids ("is it class k?"),
//                          BCE summed over the K tags (multi-label, one-vs-rest)
//   init            He normal, W ~ N(0, 2/fan_in), b = 0 (seeded)
//   optimiser       Adam (β₁ 0.9, β₂ 0.999, ε 1e-8) on the mean loss over the batch,
//                   with a per-layer learning-rate multiplier and per-layer freezing.
// Weights are row-major W[l][out·fanIn + in]. The Python exports implement the
// same maths in NumPy with the same seeded generator.
import { gauss } from './seeded';
import type { Rng } from './seeded';

export type LossKind = 'bce' | 'ce' | 'ovr';
export type HiddenAct = 'relu' | 'sigmoid';

export interface Mlp {
  dims: number[];
  W: Float64Array[];
  b: Float64Array[];
  /** Activation of each hidden layer (length dims.length − 2); default ReLU. */
  acts?: HiddenAct[];
}

export interface Adam { mW: Float64Array[]; vW: Float64Array[]; mb: Float64Array[]; vb: Float64Array[]; t: number; }

export const ADAM_B1 = 0.9, ADAM_B2 = 0.999, ADAM_EPS = 1e-8;

/** He-normal init; draws W[0] row-major, then W[1], … from the generator. */
export function initMlp(dims: number[], r: Rng, acts?: HiddenAct[]): Mlp {
  const W: Float64Array[] = [];
  const b: Float64Array[] = [];
  for (let l = 0; l < dims.length - 1; l++) {
    const fanIn = dims[l]!, fanOut = dims[l + 1]!;
    const std = Math.sqrt(2 / fanIn);
    const w = new Float64Array(fanOut * fanIn);
    for (let i = 0; i < w.length; i++) w[i] = gauss(r) * std;
    W.push(w);
    b.push(new Float64Array(fanOut));
  }
  return { dims: dims.slice(), W, b, acts: acts?.slice() };
}

export const cloneMlp = (net: Mlp): Mlp => ({
  dims: net.dims.slice(), W: net.W.map((w) => w.slice()), b: net.b.map((v) => v.slice()), acts: net.acts?.slice(),
});

export function initAdam(net: Mlp): Adam {
  return {
    mW: net.W.map((w) => new Float64Array(w.length)), vW: net.W.map((w) => new Float64Array(w.length)),
    mb: net.b.map((v) => new Float64Array(v.length)), vb: net.b.map((v) => new Float64Array(v.length)),
    t: 0,
  };
}

const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
/** Numerically stable BCE from a logit: max(z,0) − z·y + log(1 + e^{−|z|}). */
export const bceFromLogit = (z: number, y: number) => Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));

const actOf = (net: Mlp, l: number): HiddenAct => net.acts?.[l] ?? 'relu';

interface Fwd { a: Float64Array[]; d: (Float64Array | null)[]; }

/**
 * Forward pass for N inputs (X row-major N×dims[0]). a[l] = input of layer l
 * (a[L] = output logits); d[l] = ∂a[l+1]/∂z for hidden layer l (activation
 * derivative × dropout mask), used by the backward pass.
 */
function forwardAll(net: Mlp, X: Float64Array, N: number, masks?: (Float64Array | null)[]): Fwd {
  const L = net.W.length;
  const a: Float64Array[] = [X];
  const d: (Float64Array | null)[] = [];
  let prev = X;
  for (let l = 0; l < L; l++) {
    const fanIn = net.dims[l]!, fanOut = net.dims[l + 1]!;
    const W = net.W[l]!, bl = net.b[l]!;
    const out = new Float64Array(N * fanOut);
    const hidden = l < L - 1;
    const der = hidden ? new Float64Array(N * fanOut) : null;
    const act = hidden ? actOf(net, l) : 'relu';
    const m = hidden && masks ? masks[l] ?? null : null;
    for (let n = 0; n < N; n++) {
      for (let o = 0; o < fanOut; o++) {
        let s = bl[o]!;
        for (let i = 0; i < fanIn; i++) s += W[o * fanIn + i]! * prev[n * fanIn + i]!;
        const k = n * fanOut + o;
        if (der) {
          let v: number, g: number;
          if (act === 'sigmoid') { v = sigmoid(s); g = v * (1 - v); } else { v = s > 0 ? s : 0; g = s > 0 ? 1 : 0; }
          const mk = m ? m[k]! : 1;                  // inverted dropout: 0 or 1/keep
          out[k] = v * mk;
          der[k] = g * mk;
        } else {
          out[k] = s;
        }
      }
    }
    a.push(out);
    d.push(der);
    prev = out;
  }
  return { a, d };
}

/** Output probabilities: N values ('bce'), N×K independent sigmoids ('ovr') or N×K softmax ('ce'), row-major. */
export function predict(net: Mlp, X: Float64Array, N: number, loss: LossKind): Float64Array {
  const z = forwardAll(net, X, N).a[net.W.length]!;
  if (loss !== 'ce') return z.map(sigmoid);
  const K = net.dims[net.dims.length - 1]!;
  const out = new Float64Array(N * K);
  for (let n = 0; n < N; n++) {
    let mx = -Infinity;
    for (let k = 0; k < K; k++) mx = Math.max(mx, z[n * K + k]!);
    let s = 0;
    for (let k = 0; k < K; k++) { const e = Math.exp(z[n * K + k]! - mx); out[n * K + k] = e; s += e; }
    for (let k = 0; k < K; k++) out[n * K + k] = out[n * K + k]! / s;
  }
  return out;
}

/** Mean loss + accuracy of the net (eval mode: no dropout). Y = labels (class index). */
export function evaluate(net: Mlp, X: Float64Array, Y: ArrayLike<number>, N: number, loss: LossKind): { loss: number; acc: number } {
  if (N === 0) return { loss: 0, acc: 0 };
  const z = forwardAll(net, X, N).a[net.W.length]!;
  let tot = 0, ok = 0;
  if (loss === 'bce') {
    for (let n = 0; n < N; n++) {
      const y = Y[n]!;
      tot += bceFromLogit(z[n]!, y);
      if ((z[n]! >= 0 ? 1 : 0) === y) ok++;
    }
  } else {
    const K = net.dims[net.dims.length - 1]!;
    for (let n = 0; n < N; n++) {
      let mx = -Infinity, arg = 0;
      for (let k = 0; k < K; k++) { const v = z[n * K + k]!; if (v > mx) { mx = v; arg = k; } }
      if (loss === 'ovr') {
        for (let k = 0; k < K; k++) tot += bceFromLogit(z[n * K + k]!, k === Y[n] ? 1 : 0);
      } else {
        let s = 0;
        for (let k = 0; k < K; k++) s += Math.exp(z[n * K + k]! - mx);
        tot += -(z[n * K + Y[n]!]! - mx - Math.log(s));
      }
      if (arg === Y[n]) ok++;
    }
  }
  return { loss: tot / N, acc: ok / N };
}

export interface StepOpts {
  lr: number;
  /** Per-layer LR multiplier (default 1); layer l maps dims[l] → dims[l+1]. */
  lrScale?: number[];
  /** Per-layer trainable flag (default true). Frozen layers receive no update. */
  trainable?: boolean[];
  /** Inverted-dropout rate on every hidden layer (training only). */
  dropout?: number;
  /** Mask generator — required when dropout > 0. */
  maskRng?: Rng;
}

/** Inverted-dropout masks for one full-batch step: per sample, hidden layer 1's units, then layer 2's, … */
export function drawMasks(net: Mlp, N: number, p: number, r: Rng): Float64Array[] {
  const L = net.W.length;
  const keep = 1 - p;
  const masks = Array.from({ length: L - 1 }, (_, l) => new Float64Array(N * net.dims[l + 1]!));
  for (let n = 0; n < N; n++) {
    for (let l = 0; l < L - 1; l++) {
      const H = net.dims[l + 1]!;
      const m = masks[l]!;
      for (let j = 0; j < H; j++) m[n * H + j] = r() < keep ? 1 / keep : 0;
    }
  }
  return masks;
}

/** One full-batch Adam step on the mean loss; returns the (training-mode) mean loss before the update. */
export function trainStep(net: Mlp, adam: Adam, X: Float64Array, Y: ArrayLike<number>, N: number, loss: LossKind, opts: StepOpts): number {
  const L = net.W.length;
  const p = opts.dropout ?? 0;
  const masks = p > 0 && opts.maskRng ? drawMasks(net, N, p, opts.maskRng) : null;
  const { a, d } = forwardAll(net, X, N, masks ?? undefined);
  const out = a[L]!;
  const K = net.dims[L]!;
  // dL/dz at the output (mean over the batch)
  let delta = new Float64Array(N * K);
  let tot = 0;
  for (let n = 0; n < N; n++) {
    if (loss === 'bce') {
      const z = out[n]!, y = Y[n]!;
      tot += bceFromLogit(z, y);
      delta[n] = (sigmoid(z) - y) / N;
    } else if (loss === 'ovr') {
      for (let k = 0; k < K; k++) {
        const z = out[n * K + k]!, t = k === Y[n] ? 1 : 0;
        tot += bceFromLogit(z, t);
        delta[n * K + k] = (sigmoid(z) - t) / N;
      }
    } else {
      let mx = -Infinity;
      for (let k = 0; k < K; k++) mx = Math.max(mx, out[n * K + k]!);
      let s = 0;
      for (let k = 0; k < K; k++) s += Math.exp(out[n * K + k]! - mx);
      for (let k = 0; k < K; k++) delta[n * K + k] = (Math.exp(out[n * K + k]! - mx) / s - (k === Y[n] ? 1 : 0)) / N;
      tot += -(out[n * K + Y[n]!]! - mx - Math.log(s));
    }
  }
  adam.t += 1;
  const bc1 = 1 - Math.pow(ADAM_B1, adam.t), bc2 = 1 - Math.pow(ADAM_B2, adam.t);
  // Layers below the lowest trainable one need no gradient, so backprop stops there.
  let lowest = 0;
  while (lowest < L && opts.trainable?.[lowest] === false) lowest++;
  for (let l = L - 1; l >= lowest; l--) {
    const fanIn = net.dims[l]!, fanOut = net.dims[l + 1]!;
    const W = net.W[l]!, bl = net.b[l]!;
    const aPrev = a[l]!;
    // back-propagate to layer l−1's pre-activations BEFORE updating W (uses the current weights)
    let prevDelta: Float64Array | null = null;
    const der = l > lowest ? d[l - 1] ?? null : null;
    if (der) {
      prevDelta = new Float64Array(N * fanIn);
      for (let n = 0; n < N; n++) {
        for (let i = 0; i < fanIn; i++) {
          const g = der[n * fanIn + i]!;
          if (g === 0) continue;
          let s = 0;
          for (let o = 0; o < fanOut; o++) s += W[o * fanIn + i]! * delta[n * fanOut + o]!;
          prevDelta[n * fanIn + i] = s * g;
        }
      }
    }
    if (opts.trainable?.[l] !== false) {
      const eta = opts.lr * (opts.lrScale?.[l] ?? 1);
      const mW = adam.mW[l]!, vW = adam.vW[l]!, mb = adam.mb[l]!, vb = adam.vb[l]!;
      for (let o = 0; o < fanOut; o++) {
        let gb = 0;
        for (let n = 0; n < N; n++) gb += delta[n * fanOut + o]!;
        for (let i = 0; i < fanIn; i++) {
          let g = 0;
          for (let n = 0; n < N; n++) g += delta[n * fanOut + o]! * aPrev[n * fanIn + i]!;
          const k = o * fanIn + i;
          mW[k] = ADAM_B1 * mW[k]! + (1 - ADAM_B1) * g;
          vW[k] = ADAM_B2 * vW[k]! + (1 - ADAM_B2) * g * g;
          W[k] = W[k]! - eta * (mW[k]! / bc1) / (Math.sqrt(vW[k]! / bc2) + ADAM_EPS);
        }
        mb[o] = ADAM_B1 * mb[o]! + (1 - ADAM_B1) * gb;
        vb[o] = ADAM_B2 * vb[o]! + (1 - ADAM_B2) * gb * gb;
        bl[o] = bl[o]! - eta * (mb[o]! / bc1) / (Math.sqrt(vb[o]! / bc2) + ADAM_EPS);
      }
    }
    if (prevDelta) delta = prevDelta;
  }
  return tot / N;
}
