// Pure maths for the MCMC lab (no React): the mixture targets, the random-walk
// Metropolis–Hastings step and the acceptance-rate band that every caption,
// colour and narration line uses. Kept framework-free so a Node harness can
// verify the numbers quoted on screen.

export interface Comp { w: number; mu: number; sd: number; }
export type TargetKey = 'bimodal' | 'trimodal' | 'skewed';
export interface TargetDef {
  label: string;
  comps: Comp[];
  domain: [number, number];
  /** Chain start: deliberately out in the right tail, so burn-in is visible. */
  x0: number;
  note: string;
}

export const TARGETS: Record<TargetKey, TargetDef> = {
  bimodal: {
    label: 'bimodal (two peaks)',
    comps: [{ w: 0.5, mu: -2, sd: 0.6 }, { w: 0.5, mu: 2, sd: 0.8 }],
    domain: [-6, 6],
    x0: 5.5,
    note: 'Two well-separated modes. The chain must cross the low-density valley between them to be unbiased.',
  },
  trimodal: {
    label: 'trimodal (three peaks)',
    comps: [{ w: 0.35, mu: -3, sd: 0.5 }, { w: 0.4, mu: 0, sd: 0.7 }, { w: 0.25, mu: 3.2, sd: 0.6 }],
    domain: [-7, 7],
    x0: 6.5,
    note: 'Three uneven modes — a harder mixing test where a small σ can strand the chain on one peak.',
  },
  skewed: {
    label: 'skewed (close peaks)',
    comps: [{ w: 0.7, mu: -0.6, sd: 0.5 }, { w: 0.3, mu: 1.4, sd: 1.1 }],
    domain: [-5, 6],
    x0: 5.5,
    note: 'A heavy mode beside a broad shoulder — asymmetric, so the histogram must capture both scales.',
  },
};
export const TARGET_KEYS: TargetKey[] = ['bimodal', 'trimodal', 'skewed'];

/** Normalised Gaussian-mixture density π(x). */
export function targetPdf(x: number, comps: Comp[]): number {
  let p = 0;
  for (const c of comps) p += c.w * Math.exp(-0.5 * ((x - c.mu) / c.sd) ** 2) / (c.sd * Math.sqrt(2 * Math.PI));
  return p;
}

/**
 * One random-walk Metropolis–Hastings iteration: propose x′ = x + σ·N(0,1) and
 * accept with probability min(1, π(x′)/π(x)) (the proposal is symmetric, so the
 * Hastings correction is 1). On rejection the chain stays at x — that repeated
 * state is still the next sample.
 */
export function mhStep(
  x: number, sigma: number, comps: Comp[], randn: () => number, unif: () => number,
): { x: number; accepted: boolean } {
  const xp = x + sigma * randn();
  const ratio = targetPdf(xp, comps) / Math.max(1e-300, targetPdf(x, comps));
  if (unif() < Math.min(1, ratio)) return { x: xp, accepted: true };
  return { x, accepted: false };
}

/** The acceptance band every label uses: below 20% too low, above 50% too high. */
export const ACC_LO = 0.2;
export const ACC_HI = 0.5;
export type AccBand = 'low' | 'ok' | 'high';
export const accBand = (a: number): AccBand => (a < ACC_LO ? 'low' : a > ACC_HI ? 'high' : 'ok');

/**
 * Stationary acceptance rate of a N(0,σ²) random walk on a single Gaussian
 * target of std s: (2/π)·arctan(2s/σ) — 44% at σ ≈ 2.4·s.
 */
export const gaussAcceptance = (s: number, sigma: number) => (2 / Math.PI) * Math.atan((2 * s) / sigma);
