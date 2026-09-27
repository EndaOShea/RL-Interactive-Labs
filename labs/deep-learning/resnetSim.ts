// Real forward + backward pass through a deep tanh stack, for the Residual
// Networks lab. Pure module, no React.
//
// Both nets share the SAME weights W_0..W_{L-1} (d×d, entries ~ N(0, gain²/d)),
// the same batch of inputs x ~ N(0, I) and the same output gradient: each
// sample's loss is uᵀh_L for a fixed unit vector u, so ∂L/∂h_L = u (norm 1).
//
//   plain     h_{l+1} = tanh(W_l h_l)            δ_l = W_lᵀ(δ_{l+1} ⊙ tanh′(z_l))
//   residual  h_{l+1} = h_l + α·tanh(W_l h_l)    δ_l = δ_{l+1} + α·W_lᵀ(δ_{l+1} ⊙ tanh′(z_l))
//
// with z_l = W_l h_l and δ_l = ∂L/∂h_l. The profile reported per layer is the
// RMS over the batch of ‖δ_l‖ (index 0 = input, index L = output = 1).
// α = 1 is the textbook block h = x + f(x); α = 1/√L is the standard
// residual-branch scaling that keeps the forward/backward growth bounded.
import { mulberry32, gauss } from './seeded';

export const RES_D = 32;      // width of every layer
export const RES_B = 64;      // inputs in the batch
export const RES_SEED = 7;    // generator seed (the Python export reuses it)

export type BranchScale = 'one' | 'invsqrt';
export const branchAlpha = (scale: BranchScale, depth: number): number =>
  scale === 'one' ? 1 : 1 / Math.sqrt(depth);

export interface GradProfile {
  plain: number[];      // ‖δ_l‖ for l = 0..L
  residual: number[];   // ‖δ_l‖ for l = 0..L
  alpha: number;        // residual-branch scale used
}

interface Draws { X: Float64Array; u: Float64Array; base: Float64Array[]; }

// Draw order (mirrored by the Python export): X (B×d), then u (d), then the L
// base matrices row-major [out][in]. The gain only rescales these base draws,
// so moving the gain slider rescales the same weights instead of redrawing.
function draw(depth: number, seed: number, d: number, B: number): Draws {
  const r = mulberry32(seed);
  const X = new Float64Array(B * d);
  for (let i = 0; i < X.length; i++) X[i] = gauss(r);
  const u = new Float64Array(d);
  let nu = 0;
  for (let i = 0; i < d; i++) { const v = gauss(r); u[i] = v; nu += v * v; }
  nu = Math.sqrt(nu);
  for (let i = 0; i < d; i++) u[i] = u[i]! / nu;
  const base: Float64Array[] = [];
  for (let l = 0; l < depth; l++) {
    const W = new Float64Array(d * d);
    for (let i = 0; i < W.length; i++) W[i] = gauss(r);
    base.push(W);
  }
  return { X, u, base };
}

function profile(dr: Draws, depth: number, gain: number, residual: boolean, alpha: number, d: number, B: number): number[] {
  const s = gain / Math.sqrt(d);
  // forward: keep tanh′(z_l) per layer for the backward pass
  const dAct: Float64Array[] = [];
  let h: Float64Array = Float64Array.from(dr.X);
  for (let l = 0; l < depth; l++) {
    const W = dr.base[l]!;
    const next = new Float64Array(B * d);
    const der = new Float64Array(B * d);
    for (let b = 0; b < B; b++) {
      for (let o = 0; o < d; o++) {
        let z = 0;
        for (let i = 0; i < d; i++) z += W[o * d + i]! * h[b * d + i]!;
        z *= s;
        const t = Math.tanh(z);
        der[b * d + o] = 1 - t * t;
        next[b * d + o] = residual ? h[b * d + o]! + alpha * t : t;
      }
    }
    dAct.push(der);
    h = next;
  }
  // backward from δ_L = u for every sample
  const norms = new Array<number>(depth + 1).fill(0);
  let delta = new Float64Array(B * d);
  for (let b = 0; b < B; b++) for (let i = 0; i < d; i++) delta[b * d + i] = dr.u[i]!;
  const rms = (v: Float64Array) => { let q = 0; for (let i = 0; i < v.length; i++) q += v[i]! * v[i]!; return Math.sqrt(q / B); };
  norms[depth] = rms(delta);
  for (let l = depth - 1; l >= 0; l--) {
    const W = dr.base[l]!;
    const der = dAct[l]!;
    const prev = new Float64Array(B * d);
    const scale = residual ? alpha * s : s;
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < d; i++) {
        let acc = 0;
        for (let o = 0; o < d; o++) acc += W[o * d + i]! * delta[b * d + o]! * der[b * d + o]!;
        prev[b * d + i] = (residual ? delta[b * d + i]! : 0) + scale * acc;
      }
    }
    delta = prev;
    norms[l] = rms(delta);
  }
  return norms;
}

/** Gradient-norm profiles of the plain and residual nets on identical weights. */
export function resnetGradients(depth: number, gain: number, scale: BranchScale, seed = RES_SEED, d = RES_D, B = RES_B): GradProfile {
  const dr = draw(depth, seed, d, B);
  const alpha = branchAlpha(scale, depth);
  return {
    plain: profile(dr, depth, gain, false, 1, d, B),
    residual: profile(dr, depth, gain, true, alpha, d, B),
    alpha,
  };
}

/** Shared verdict for a gradient norm reaching the input (chips, narration, text). */
export type GradVerdict = 'vanished' | 'shrinking' | 'healthy' | 'growing' | 'exploded';
export function gradVerdict(g: number): GradVerdict {
  if (Number.isNaN(g)) return 'exploded';  // overflow
  if (g < 1e-2) return 'vanished';         // below 1% of the output gradient
  if (g < 0.1) return 'shrinking';
  if (g <= 10) return 'healthy';           // within one order of magnitude of 1
  if (g <= 100) return 'growing';
  return 'exploded';
}
