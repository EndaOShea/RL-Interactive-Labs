// Pure maths for the HMM lab (no React): scaled forward filtering (with the log
// likelihood Σ log c_t), forward–backward smoothing, log-space Viterbi with
// backpointers, and the seeded sequence generator. Framework-free so a Node
// harness can verify the numbers and preset notes shown on screen.
import { rng } from './shared';

export interface Hmm { A: number[][]; B: number[][]; pi: number[]; }
export const SEQ_SEED = 12345;

/** The two-state casino: fair die (state 0) and a die loaded toward six (state 1). */
export function casino(stay: number, p6: number): Hmm {
  return {
    A: [[stay, 1 - stay], [1 - stay, stay]],
    B: [new Array(6).fill(1 / 6), [...new Array(5).fill((1 - p6) / 5), p6]],
    pi: [0.5, 0.5],
  };
}

/**
 * Scaled forward pass: α_t(s) ∝ B[s,o_t]·Σ_{s′} α_{t−1}(s′)A[s′,s], normalised by
 * c_t each step (so α_t is the filtered posterior). Σ_t log c_t = log p(o_1..o_T).
 */
export function forward(obs: number[], h: Hmm): { alpha: number[][]; c: number[]; logLik: number } {
  const T = obs.length, S = h.pi.length;
  const alpha: number[][] = Array.from({ length: T }, () => new Array(S).fill(0));
  const c: number[] = new Array(T).fill(0);
  for (let t = 0; t < T; t++) {
    const o = obs[t] ?? 0;
    const at = alpha[t]!;
    for (let s = 0; s < S; s++) {
      let prior: number;
      if (t === 0) prior = h.pi[s] ?? 0;
      else { prior = 0; const prev = alpha[t - 1]!; for (let sp = 0; sp < S; sp++) prior += (prev[sp] ?? 0) * (h.A[sp]?.[s] ?? 0); }
      at[s] = prior * (h.B[s]?.[o] ?? 0);
    }
    c[t] = at.reduce((a, b) => a + b, 0) || 1;
    for (let s = 0; s < S; s++) at[s] = (at[s] ?? 0) / (c[t] ?? 1);
  }
  const logLik = c.reduce((a, v) => a + Math.log(v), 0);
  return { alpha, c, logLik };
}

/** Forward–backward smoothed posterior γ_t(s) = P(state_t = s | o_1..o_T). */
export function smoothed(obs: number[], h: Hmm): number[][] {
  const T = obs.length, S = h.pi.length;
  const { alpha, c } = forward(obs, h);
  const beta: number[][] = Array.from({ length: T }, () => new Array(S).fill(0));
  for (let s = 0; s < S; s++) beta[T - 1]![s] = 1;
  for (let t = T - 2; t >= 0; t--) {
    for (let s = 0; s < S; s++) {
      let acc = 0;
      for (let sp = 0; sp < S; sp++) acc += (h.A[s]?.[sp] ?? 0) * (h.B[sp]?.[obs[t + 1] ?? 0] ?? 0) * (beta[t + 1]?.[sp] ?? 0);
      beta[t]![s] = acc / (c[t + 1] || 1);
    }
  }
  return alpha.map((a, t) => {
    const g = a.map((v, s) => v * (beta[t]?.[s] ?? 0));
    const z = g.reduce((p, q) => p + q, 0) || 1;
    return g.map((v) => v / z);
  });
}

/** Log-space Viterbi: the single most probable hidden path (max-product + backpointers). */
export function viterbi(obs: number[], h: Hmm): number[] {
  const T = obs.length, S = h.pi.length;
  const ln = (x: number) => Math.log(Math.max(x, 1e-12));
  const d: number[][] = Array.from({ length: T }, () => new Array(S).fill(0));
  const psi: number[][] = Array.from({ length: T }, () => new Array(S).fill(0));
  for (let s = 0; s < S; s++) d[0]![s] = ln(h.pi[s] ?? 0) + ln(h.B[s]?.[obs[0] ?? 0] ?? 0);
  for (let t = 1; t < T; t++) {
    for (let s = 0; s < S; s++) {
      let best = -Infinity, arg = 0;
      for (let sp = 0; sp < S; sp++) { const val = (d[t - 1]?.[sp] ?? 0) + ln(h.A[sp]?.[s] ?? 0); if (val > best) { best = val; arg = sp; } }
      d[t]![s] = best + ln(h.B[s]?.[obs[t] ?? 0] ?? 0); psi[t]![s] = arg;
    }
  }
  const path = new Array(T).fill(0);
  path[T - 1] = (d[T - 1]?.[0] ?? 0) >= (d[T - 1]?.[1] ?? 0) ? 0 : 1;
  for (let t = T - 2; t >= 0; t--) path[t] = psi[t + 1]?.[path[t + 1]] ?? 0;
  return path;
}

/** Sample a hidden-state / roll sequence from the model with a fixed seed. */
export function generate(T: number, h: Hmm, seed: number): { states: number[]; obs: number[] } {
  const r = rng(seed);
  const pick = (p: number[]) => { const u = r(); let acc = 0; for (let i = 0; i < p.length; i++) { acc += p[i] ?? 0; if (u <= acc) return i; } return p.length - 1; };
  const states: number[] = [], obs: number[] = [];
  let s = pick(h.pi);
  for (let t = 0; t < T; t++) { states.push(s); obs.push(pick(h.B[s] ?? [])); s = pick(h.A[s] ?? []); }
  return { states, obs };
}
