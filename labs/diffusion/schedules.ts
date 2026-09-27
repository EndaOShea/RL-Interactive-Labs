// Pure noise-schedule maths for the Diffusion labs (no React, no DOM). Shared by
// the Noise Schedules lab (four families + resolution shift) and the Forward &
// Reverse lab (linear / cosine). python.ts mirrors this file operation for
// operation, so the exported scripts print the same numbers.
//
// Every schedule is built with one recipe (improved DDPM's betas_for_alpha_bar):
//   1. a raw signal-retention curve ᾱ(t), t = 0..T, with ᾱ(0) = 1;
//   2. an optional resolution shift in log-SNR: SNR′ = SNR / shift²
//      (ᾱ′ = ᾱ / (ᾱ + shift²·(1 − ᾱ))) — shift > 1 adds noise at every t;
//   3. per-step β_t = min(0.999, 1 − ᾱ_t / ᾱ_{t−1});
//   4. ᾱ_t re-accumulated as ∏(1 − β_i), so β, α and ᾱ always agree.
// The linear family is defined by its β directly: DDPM's 1e-4 → 0.02 endpoints
// (tuned for T = 1000) scaled by 1000/T, so the same continuous-time schedule is
// discretised whatever T is.

export type ScheduleKind = 'linear' | 'cosine' | 'sigmoid' | 'edm';

export const BETA_MAX = 0.999;
export const COSINE_S = 0.008;
/** Chen (2023), "On the importance of noise scheduling", Algorithm 1 defaults. */
export const SIGMOID_START = -3;
export const SIGMOID_END = 3;
export const SIGMOID_TAU = 1;
/** Karras et al. (2022, EDM) sampling σ-spacing, mapped to VP form by ᾱ = 1/(1+σ²). */
export const EDM_SIGMA_MIN = 0.002;
export const EDM_SIGMA_MAX = 80;
export const EDM_RHO = 7;

/** A discrete schedule. Arrays are indexed by t = 0..T (index 0 is the clean data). */
export interface Schedule {
  kind: ScheduleKind;
  T: number;
  shift: number;
  /** β_t; beta[0] = 0 is a placeholder. */
  beta: number[];
  /** α_t = 1 − β_t; alpha[0] = 1. */
  alpha: number[];
  /** ᾱ_t = ∏_{i≤t} α_i; abar[0] = 1. */
  abar: number[];
  /** ln SNR_t = ln(ᾱ_t / (1 − ᾱ_t)); +Infinity at t = 0. */
  logsnr: number[];
}

const sigm = (z: number) => 1 / (1 + Math.exp(-z));

/** Linear β_i, i = 1..T: DDPM's 1e-4 → 0.02 endpoints scaled by 1000/T. */
export function linearBetas(T: number): number[] {
  const scale = 1000 / T;
  const b0 = 1e-4 * scale;
  const b1 = 0.02 * scale;
  const out: number[] = [];
  for (let i = 0; i < T; i++) out.push(T > 1 ? b0 + ((b1 - b0) * i) / (T - 1) : b0);
  return out;
}

/** Raw ᾱ(t) for t = 0..T before the β clip (ᾱ(0) = 1). */
export function rawAbar(kind: ScheduleKind, T: number): number[] {
  const out: number[] = [1];
  if (kind === 'linear') {
    let p = 1;
    for (const b of linearBetas(T)) {
      p *= 1 - b;
      out.push(p);
    }
  } else if (kind === 'cosine') {
    // ᾱ_t = f(t)/f(0),  f(t) = cos²((t/T + s)/(1 + s) · π/2)
    const f = (t: number) => {
      const c = Math.cos(((t / T + COSINE_S) / (1 + COSINE_S)) * (Math.PI / 2));
      return c * c;
    };
    const f0 = f(0);
    for (let t = 1; t <= T; t++) out.push(f(t) / f0);
  } else if (kind === 'sigmoid') {
    // Chen 2023: ᾱ(u) = (σ(end/τ) − σ((u·(end − start) + start)/τ)) / (σ(end/τ) − σ(start/τ))
    const vStart = sigm(SIGMOID_START / SIGMOID_TAU);
    const vEnd = sigm(SIGMOID_END / SIGMOID_TAU);
    for (let t = 1; t <= T; t++) {
      const u = t / T;
      out.push((vEnd - sigm((u * (SIGMOID_END - SIGMOID_START) + SIGMOID_START) / SIGMOID_TAU)) / (vEnd - vStart));
    }
  } else {
    // EDM: σ_t = (σ_min^{1/ρ} + u·(σ_max^{1/ρ} − σ_min^{1/ρ}))^ρ, u = (t−1)/(T−1); ᾱ = 1/(1+σ²)
    const a = Math.pow(EDM_SIGMA_MIN, 1 / EDM_RHO);
    const b = Math.pow(EDM_SIGMA_MAX, 1 / EDM_RHO);
    for (let t = 1; t <= T; t++) {
      const u = T > 1 ? (t - 1) / (T - 1) : 1;
      const sigma = Math.pow(a + u * (b - a), EDM_RHO);
      out.push(1 / (1 + sigma * sigma));
    }
  }
  return out;
}

/** Build the discrete schedule: raw ᾱ → optional log-SNR shift → clipped β → ᾱ = ∏(1 − β). */
export function buildSchedule(kind: ScheduleKind, T: number, shift = 1): Schedule {
  const beta: number[] = [0];
  if (kind === 'linear' && shift === 1) {
    for (const b of linearBetas(T)) beta.push(b);
  } else {
    const raw = rawAbar(kind, T);
    const k = shift * shift;
    const ab = shift === 1 ? raw : raw.map((a) => a / (a + k * (1 - a)));
    for (let t = 1; t <= T; t++) {
      const prev = ab[t - 1] ?? 1;
      const cur = ab[t] ?? 0;
      beta.push(Math.min(BETA_MAX, Math.max(0, 1 - cur / prev)));
    }
  }
  const alpha = beta.map((b) => 1 - b);
  const abar: number[] = [1];
  for (let t = 1; t <= T; t++) abar.push((abar[t - 1] ?? 1) * (alpha[t] ?? 1));
  const logsnr = abar.map((a) => Math.log(a / (1 - a)));
  return { kind, T, shift, beta, alpha, abar, logsnr };
}

/**
 * Where SNR crosses 1 (ln SNR crosses 0): the fractional t* found by linear
 * interpolation of ln SNR between the two bracketing integer steps, or null when
 * SNR never drops to 1 within the chain.
 */
export function snrCrossing(s: Schedule): number | null {
  for (let t = 1; t <= s.T; t++) {
    const cur = s.logsnr[t] ?? 0;
    if (cur <= 0) {
      const prev = s.logsnr[t - 1] ?? Infinity;
      if (!Number.isFinite(prev)) return t;
      return t - 1 + prev / (prev - cur);
    }
  }
  return null;
}

/** Share of the T steps whose ᾱ_t is below `limit` (default 0.01: under 10% signal amplitude). */
export function shareNearNoise(s: Schedule, limit = 0.01): number {
  let n = 0;
  for (let t = 1; t <= s.T; t++) if ((s.abar[t] ?? 0) < limit) n++;
  return n / s.T;
}

/** Share of the T steps whose ᾱ_t is above `limit` (default 0.99: under 10% noise amplitude). */
export function shareNearClean(s: Schedule, limit = 0.99): number {
  let n = 0;
  for (let t = 1; t <= s.T; t++) if ((s.abar[t] ?? 0) > limit) n++;
  return n / s.T;
}

/** Share of the T steps with |ln SNR| ≤ band (default 1: SNR between 1/e and e). */
export function shareCrossover(s: Schedule, band = 1): number {
  let n = 0;
  for (let t = 1; t <= s.T; t++) if (Math.abs(s.logsnr[t] ?? 0) <= band) n++;
  return n / s.T;
}
