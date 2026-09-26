// Pure maths for the Forward & Reverse Diffusion lab (no React, no DOM):
//   • a seeded PRNG (mulberry32 + Box–Muller) so the lab and its Python export
//     draw the same dataset, forward noise, starting noise and sampler noise;
//   • the three 2-D datasets;
//   • an ANALYTIC denoiser — the exact posterior mean E[x₀ | xₜ] when the data
//     distribution is taken to be a Gaussian kernel density over the dataset
//     (one isotropic component N(μᵢ, s²I) per training point). No network is
//     trained: this is the denoiser a perfectly trained ε-network would converge
//     to for that density;
//   • real DDPM (ancestral, σₜ² = β̃ₜ) and DDIM (η = 0, strided) samplers with
//     classifier-free guidance, both starting from fresh N(0, I) noise;
//   • sample-quality metrics (nearest-neighbour distances to the data).
// python.ts mirrors this file.

import type { Schedule } from './schedules';

/* ---------- seeded PRNG ---------- */

/** mulberry32: a 32-bit seeded uniform generator on [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard-normal draw from a uniform generator (Box–Muller, cosine branch). */
export function gauss(r: () => number): number {
  let u = 0;
  while (u === 0) u = r();
  let v = 0;
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Independent stream seeds: 1 dataset, 2 forward ε, 3 starting x_T, 4 DDPM z, 5 ideal-KDE reference. */
export const streamSeed = (seed: number, stream: number) => (seed * 1000003 + stream * 7919) >>> 0;

/* ---------- datasets ---------- */

export type Dataset = 'two-moons' | 'ring' | 'blobs';

export interface Data {
  n: number;
  x: Float64Array;
  y: Float64Array;
  cls: Int32Array;
  nClasses: number;
}

export const N_POINTS = 500;

/** The three 2-D datasets (centred, roughly within ±1.5), drawn from stream 1. */
export function makeData(kind: Dataset, n: number, seed: number): Data {
  const r = mulberry32(streamSeed(seed, 1));
  const g = () => gauss(r);
  const xs: number[] = [];
  const ys: number[] = [];
  const cs: number[] = [];
  const push = (x: number, y: number, c: number) => { xs.push(x); ys.push(y); cs.push(c); };
  let nClasses = 2;
  if (kind === 'two-moons') {
    const m = Math.floor(n / 2);
    for (let i = 0; i < m; i++) {
      const t = (i / (m - 1)) * Math.PI;
      push(Math.cos(t) - 0.5 + g() * 0.06, Math.sin(t) - 0.25 + g() * 0.06, 0);
      push(1 - Math.cos(t) - 0.5 + g() * 0.06, 0.25 - Math.sin(t) + g() * 0.06, 1);
    }
  } else if (kind === 'ring') {
    for (let i = 0; i < n; i++) {
      const t = (i / n) * 2 * Math.PI;
      const rad = 1.05 + g() * 0.05;
      push(rad * Math.cos(t), rad * Math.sin(t), t > Math.PI ? 0 : 1);
    }
  } else {
    nClasses = 3;
    const cx = [-0.85, 0.85, 0];
    const cy = [-0.85, -0.85, 0.95];
    for (let i = 0; i < n; i++) {
      const c = i % 3;
      push((cx[c] ?? 0) + g() * 0.17, (cy[c] ?? 0) + g() * 0.17, c);
    }
  }
  return { n: xs.length, x: Float64Array.from(xs), y: Float64Array.from(ys), cls: Int32Array.from(cs), nClasses };
}

/** One FIXED forward-noise vector ε per data point (stream 2). */
export function makeEps(n: number, seed: number): { ex: Float64Array; ey: Float64Array } {
  const r = mulberry32(streamSeed(seed, 2));
  const ex = new Float64Array(n);
  const ey = new Float64Array(n);
  for (let i = 0; i < n; i++) { ex[i] = gauss(r); ey[i] = gauss(r); }
  return { ex, ey };
}

/* ---------- the analytic denoiser ---------- */

/** KDE bandwidth s: each training point μᵢ is the mean of a component N(μᵢ, s²I). */
export const KDE_S = 0.05;

/** Component indices of each class (the class-conditional mixtures) and of all points. */
export interface Mixture {
  data: Data;
  s2: number;
  byClass: Int32Array[];
  all: Int32Array;
}

export function makeMixture(data: Data, s = KDE_S): Mixture {
  const byClass: Int32Array[] = [];
  for (let c = 0; c < data.nClasses; c++) {
    const idx: number[] = [];
    for (let i = 0; i < data.n; i++) if (data.cls[i] === c) idx.push(i);
    byClass.push(Int32Array.from(idx));
  }
  const all = Int32Array.from(Array.from({ length: data.n }, (_, i) => i));
  return { data, s2: s * s, byClass, all };
}

/**
 * Exact posterior means E[x₀ | xₜ] (equal component weights within a mixture):
 *   xₜ | i ~ N(√ᾱ·μᵢ, v·I),  v = ᾱ·s² + 1 − ᾱ
 *   wᵢ ∝ exp(−‖xₜ − √ᾱ·μᵢ‖² / 2v)           (log-sum-exp stabilised)
 *   E[x₀ | xₜ] = Σ wᵢ [μᵢ + (√ᾱ·s²/v)(xₜ − √ᾱ·μᵢ)] = m + k(xₜ − √ᾱ·m),  m = Σ wᵢμᵢ
 * Always computes the class-`cls` mixture (→ outC); when `outU` is given it also
 * computes the full mixture (→ outU) from the same pass of distances. `buf` is
 * scratch of length ≥ the number of points.
 */
export function posteriorMeans(
  mix: Mixture, cls: number, x: number, y: number, ab: number,
  buf: Float64Array, outC: Float64Array, outU: Float64Array | null,
): void {
  const mx = mix.data.x;
  const my = mix.data.y;
  const cl = mix.data.cls;
  const sa = Math.sqrt(ab);
  const v = ab * mix.s2 + 1 - ab;
  const inv = 1 / (2 * v);
  const k = (sa * mix.s2) / v;
  const finish = (sw: number, sx: number, sy: number, out: Float64Array) => {
    const m0 = sx / sw;
    const m1 = sy / sw;
    out[0] = m0 + k * (x - sa * m0);
    out[1] = m1 + k * (y - sa * m1);
  };
  if (!outU) {
    // class mixture only: loop over the class's own components
    const idx = mix.byClass[cls] ?? mix.all;
    let lmax = -Infinity;
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j] ?? 0;
      const dx = x - sa * (mx[i] ?? 0);
      const dy = y - sa * (my[i] ?? 0);
      const l = -(dx * dx + dy * dy) * inv;
      buf[j] = l;
      if (l > lmax) lmax = l;
    }
    let sw = 0;
    let sx = 0;
    let sy = 0;
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j] ?? 0;
      const w = Math.exp((buf[j] ?? 0) - lmax);
      sw += w;
      sx += w * (mx[i] ?? 0);
      sy += w * (my[i] ?? 0);
    }
    finish(sw, sx, sy, outC);
    return;
  }
  // both mixtures: one pass of logits over every component
  const n = mx.length;
  let lmaxA = -Infinity;
  let lmaxC = -Infinity;
  for (let i = 0; i < n; i++) {
    const dx = x - sa * (mx[i] ?? 0);
    const dy = y - sa * (my[i] ?? 0);
    const l = -(dx * dx + dy * dy) * inv;
    buf[i] = l;
    if (l > lmaxA) lmaxA = l;
    if (cl[i] === cls && l > lmaxC) lmaxC = l;
  }
  let swA = 0;
  let sxA = 0;
  let syA = 0;
  let swC = 0;
  let sxC = 0;
  let syC = 0;
  const shared = lmaxC === lmaxA;
  for (let i = 0; i < n; i++) {
    const l = buf[i] ?? 0;
    const e = Math.exp(l - lmaxA);
    const px = mx[i] ?? 0;
    const py = my[i] ?? 0;
    swA += e;
    sxA += e * px;
    syA += e * py;
    if (cl[i] === cls) {
      const ec = shared ? e : Math.exp(l - lmaxC);
      swC += ec;
      sxC += ec * px;
      syC += ec * py;
    }
  }
  finish(swC, sxC, syC, outC);
  finish(swA, sxA, syA, outU);
}

/* ---------- samplers ---------- */

export type Sampler = 'ddpm' | 'ddim';

/** DDIM's visited timesteps, descending T → 0: τ_k = round(k·T/S), S = min(steps, T). */
export function ddimTimesteps(T: number, steps: number): number[] {
  const S = Math.max(1, Math.min(steps, T));
  const ts: number[] = [];
  for (let k = S; k >= 0; k--) ts.push(Math.round((k * T) / S));
  return ts;
}

export interface RunConfig {
  mix: Mixture;
  sched: Schedule;
  sampler: Sampler;
  /** DDIM step count S (ignored by DDPM, which visits every t = T..1). */
  steps: number;
  /** Classifier-free guidance scale w: ε̂ = (1+w)·ε_cond − w·ε_uncond. */
  w: number;
  seed: number;
}

/** Per-step coefficients, for the Math panel. */
export interface StepCoeffs {
  t: number;
  tNext: number;
  abar: number;
  abarNext: number;
  beta: number;
  alpha: number;
  sigma: number;
}

/**
 * A reverse-diffusion run, computed lazily frame by frame (frames are cached, and
 * always computed in order, so the DDPM noise sequence never depends on how the
 * user scrubs). Sample i is conditioned on class data.cls[i], so the generated
 * classes have the same proportions as the data.
 */
export class ReverseRun {
  readonly ts: number[];
  readonly xs: Float64Array[] = [];
  readonly ys: Float64Array[] = [];
  private readonly cfg: RunConfig;
  private readonly zr: () => number;
  private readonly buf: Float64Array;
  private readonly ec = new Float64Array(2);
  private readonly eu = new Float64Array(2);

  constructor(cfg: RunConfig) {
    this.cfg = cfg;
    const T = cfg.sched.T;
    if (cfg.sampler === 'ddpm') {
      this.ts = [];
      for (let t = T; t >= 0; t--) this.ts.push(t);
    } else {
      this.ts = ddimTimesteps(T, cfg.steps);
    }
    const n = cfg.mix.data.n;
    const r = mulberry32(streamSeed(cfg.seed, 3));
    const x = new Float64Array(n);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = gauss(r); y[i] = gauss(r); }   // fresh x_T ~ N(0, I)
    this.xs.push(x);
    this.ys.push(y);
    this.zr = mulberry32(streamSeed(cfg.seed, 4));
    this.buf = new Float64Array(n);
  }

  /** Coefficients of the step that leaves frame k (k < ts.length − 1). */
  coeffs(k: number): StepCoeffs {
    const { sched, sampler } = this.cfg;
    const t = this.ts[k] ?? 0;
    const tNext = this.ts[k + 1] ?? 0;
    const abar = sched.abar[t] ?? 1;
    const abarNext = sched.abar[tNext] ?? 1;
    const beta = sampler === 'ddpm' ? (sched.beta[t] ?? 0) : 1 - abar / abarNext;
    const alpha = 1 - beta;
    const sigma = sampler === 'ddpm' ? Math.sqrt(((1 - abarNext) / (1 - abar)) * beta) : 0;
    return { t, tNext, abar, abarNext, beta, alpha, sigma };
  }

  /** Frame k (samples at timestep ts[k]); computes any missing frames first. */
  frame(k: number): { x: Float64Array; y: Float64Array } {
    const kk = Math.max(0, Math.min(k, this.ts.length - 1));
    while (this.xs.length <= kk) this.advance();
    return { x: this.xs[kk] as Float64Array, y: this.ys[kk] as Float64Array };
  }

  /** Index of the visited timestep nearest to t. */
  indexOf(t: number): number {
    let best = 0;
    for (let k = 1; k < this.ts.length; k++) {
      if (Math.abs((this.ts[k] ?? 0) - t) < Math.abs((this.ts[best] ?? 0) - t)) best = k;
    }
    return best;
  }

  private advance(): void {
    const { mix, sampler, w } = this.cfg;
    const k = this.xs.length - 1;
    const c = this.coeffs(k);
    const x = this.xs[k] as Float64Array;
    const y = this.ys[k] as Float64Array;
    const n = x.length;
    const nx = new Float64Array(n);
    const ny = new Float64Array(n);
    const sa = Math.sqrt(c.abar);
    const s1 = Math.sqrt(1 - c.abar);
    for (let i = 0; i < n; i++) {
      const xi = x[i] ?? 0;
      const yi = y[i] ?? 0;
      posteriorMeans(mix, mix.data.cls[i] ?? 0, xi, yi, c.abar, this.buf, this.ec, w !== 0 ? this.eu : null);
      let hx = this.ec[0] ?? 0;
      let hy = this.ec[1] ?? 0;
      if (w !== 0) {
        // CFG: ε̂ = (1+w)·ε_cond − w·ε_uncond, i.e. x̂₀ = (1+w)·E_cond − w·E_uncond
        hx = (1 + w) * hx - w * (this.eu[0] ?? 0);
        hy = (1 + w) * hy - w * (this.eu[1] ?? 0);
      }
      const epx = (xi - sa * hx) / s1;   // ε̂ = (xₜ − √ᾱₜ·x̂₀)/√(1−ᾱₜ)
      const epy = (yi - sa * hy) / s1;
      if (sampler === 'ddpm') {
        // xₜ₋₁ = (xₜ − βₜ/√(1−ᾱₜ)·ε̂)/√αₜ + σₜ·z,  σₜ² = β̃ₜ = (1−ᾱₜ₋₁)/(1−ᾱₜ)·βₜ
        const zx = gauss(this.zr);
        const zy = gauss(this.zr);
        const ra = Math.sqrt(c.alpha);
        nx[i] = (xi - (c.beta / s1) * epx) / ra + c.sigma * zx;
        ny[i] = (yi - (c.beta / s1) * epy) / ra + c.sigma * zy;
      } else {
        // DDIM (η = 0): xₜ′ = √ᾱₜ′·x̂₀ + √(1−ᾱₜ′)·ε̂
        const an = Math.sqrt(c.abarNext);
        const bn = Math.sqrt(1 - c.abarNext);
        nx[i] = an * hx + bn * epx;
        ny[i] = an * hy + bn * epy;
      }
    }
    this.xs.push(nx);
    this.ys.push(ny);
  }
}

/* ---------- metrics ---------- */

/** Mean over points of `a` of the Euclidean distance to the nearest point of `b`. */
export function meanNN(ax: Float64Array, ay: Float64Array, bx: Float64Array, by: Float64Array): number {
  let tot = 0;
  for (let i = 0; i < ax.length; i++) {
    const px = ax[i] ?? 0;
    const py = ay[i] ?? 0;
    let best = Infinity;
    for (let j = 0; j < bx.length; j++) {
      const dx = px - (bx[j] ?? 0);
      const dy = py - (by[j] ?? 0);
      const d2 = dx * dx + dy * dy;
      if (d2 < best) best = d2;
    }
    tot += Math.sqrt(best);
  }
  return tot / Math.max(1, ax.length);
}

export interface SampleMetrics {
  /** mean distance from each sample to its nearest training point (off-data error; lower = on the data) */
  toData: number;
  /** mean distance from each training point to its nearest sample (coverage gap; lower = data covered) */
  toSample: number;
}

export function sampleMetrics(data: Data, sx: Float64Array, sy: Float64Array): SampleMetrics {
  return { toData: meanNN(sx, sy, data.x, data.y), toSample: meanNN(data.x, data.y, sx, sy) };
}

/** Reference: the same metrics for exact draws from the KDE (μᵢ + s·z, stream 5). */
export function idealMetrics(data: Data, seed: number, s = KDE_S): SampleMetrics {
  const r = mulberry32(streamSeed(seed, 5));
  const kx = new Float64Array(data.n);
  const ky = new Float64Array(data.n);
  for (let i = 0; i < data.n; i++) {
    kx[i] = (data.x[i] ?? 0) + s * gauss(r);
    ky[i] = (data.y[i] ?? 0) + s * gauss(r);
  }
  return sampleMetrics(data, kx, ky);
}
