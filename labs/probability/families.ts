// Pure maths for the Distributions lab (no React): the eight families (pmf/pdf,
// exact samplers, mean, variance, entropy in NATS), the histogram layouts for
// single draws and for sums of k draws (the CLT view), and the running sample
// statistics. Framework-free so a Node harness can verify every number.
import { lgamma, digamma, normCdf, betaInc } from './probMath';

export type Family =
  | 'bernoulli' | 'binomial' | 'poisson' | 'geometric'
  | 'uniform' | 'normal' | 'exponential' | 'beta';

export type Params = Record<string, number>;
export const num = (p: Params, k: string) => p[k] ?? 0;

/** Standard normal via Box–Muller (the same transform the lab's other samplers use). */
export function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Marsaglia–Tsang Gamma(k, 1) sampler; shapes below 1 use the U^(1/k) boost. */
export function gammaSample(k: number): number {
  if (k < 1) { const u = Math.random(); return gammaSample(1 + k) * Math.pow(u, 1 / k); }
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const x = randn(); let v = 1 + c * x; if (v <= 0) continue;
    v = v * v * v; const u = Math.random();
    if (u < 1 - 0.0331 * x ** 4) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

const lfact = (n: number) => lgamma(n + 1);
const logChoose = (n: number, k: number) => lfact(n) - lfact(k) - lfact(n - k);
const negPlnP = (p: number) => (p > 0 ? -p * Math.log(p) : 0);

export interface Knob { key: string; name: string; min: number; max: number; step: number; hint: string; }
export interface FamilyDef {
  label: string;
  kind: 'discrete' | 'continuous';
  formula: string;
  knobs: Knob[];
  /** discrete: displayed k range; continuous: plotting window */
  support: (p: Params) => [number, number];
  pmf?: (k: number, p: Params) => number;
  pdf?: (x: number, p: Params) => number;
  /** continuous families: the CDF, so histogram bins are compared with exact bin probabilities */
  cdf?: (x: number, p: Params) => number;
  sample: (p: Params) => number;
  mean: (p: Params) => number;
  variance: (p: Params) => number;
  /** Shannon entropy (discrete) or differential entropy (continuous), in nats. */
  entropy: (p: Params) => number;
  note: string;
}

export const FAMILIES: Record<Family, FamilyDef> = {
  bernoulli: {
    label: 'Bernoulli', kind: 'discrete', formula: 'P(k)=pᵏ(1−p)¹⁻ᵏ',
    knobs: [{ key: 'p', name: 'p · success prob', min: 0.01, max: 0.99, step: 0.01, hint: 'P(success)' }],
    support: () => [0, 1],
    pmf: (k, p) => (k === 1 ? num(p, 'p') : k === 0 ? 1 - num(p, 'p') : 0),
    sample: (p) => (Math.random() < num(p, 'p') ? 1 : 0),
    mean: (p) => num(p, 'p'), variance: (p) => num(p, 'p') * (1 - num(p, 'p')),
    entropy: (p) => negPlnP(num(p, 'p')) + negPlnP(1 - num(p, 'p')),
    note: 'A single yes/no trial — the atom every other count distribution is built from.',
  },
  binomial: {
    label: 'Binomial', kind: 'discrete', formula: 'P(k)=C(n,k)pᵏ(1−p)ⁿ⁻ᵏ',
    knobs: [
      { key: 'n', name: 'n · trials', min: 1, max: 40, step: 1, hint: 'number of trials' },
      { key: 'p', name: 'p · success prob', min: 0.01, max: 0.99, step: 0.01, hint: 'per-trial P(success)' },
    ],
    support: (p) => [0, num(p, 'n')],
    pmf: (k, p) => {
      const n = num(p, 'n'), q = num(p, 'p');
      return k < 0 || k > n ? 0 : Math.exp(logChoose(n, k) + k * Math.log(q) + (n - k) * Math.log(1 - q));
    },
    sample: (p) => { let s = 0; for (let i = 0; i < num(p, 'n'); i++) if (Math.random() < num(p, 'p')) s++; return s; },
    mean: (p) => num(p, 'n') * num(p, 'p'), variance: (p) => num(p, 'n') * num(p, 'p') * (1 - num(p, 'p')),
    entropy: (p) => { let h = 0; for (let k = 0; k <= num(p, 'n'); k++) h += negPlnP(FAMILIES.binomial.pmf!(k, p)); return h; },
    note: 'Successes in n independent Bernoulli trials — sum of n coin flips.',
  },
  poisson: {
    label: 'Poisson', kind: 'discrete', formula: 'P(k)=e⁻λ λᵏ/k!',
    knobs: [{ key: 'lam', name: 'λ · rate', min: 0.2, max: 18, step: 0.2, hint: 'expected count' }],
    support: (p) => [0, Math.max(8, Math.ceil(num(p, 'lam') + 4 * Math.sqrt(num(p, 'lam'))))],
    pmf: (k, p) => (k < 0 ? 0 : Math.exp(-num(p, 'lam') + k * Math.log(num(p, 'lam')) - lfact(k))),
    sample: (p) => { const L = Math.exp(-num(p, 'lam')); let k = 0, prod = 1; do { k++; prod *= Math.random(); } while (prod > L); return k - 1; },
    mean: (p) => num(p, 'lam'), variance: (p) => num(p, 'lam'),
    entropy: (p) => {
      const lam = num(p, 'lam');
      const K = Math.ceil(lam + 12 * Math.sqrt(lam) + 12);   // the tail beyond K is negligible
      let h = 0; for (let k = 0; k <= K; k++) h += negPlnP(FAMILIES.poisson.pmf!(k, p));
      return h;
    },
    note: 'Counts of rare events at rate λ — the n→∞, p→0 limit of the Binomial.',
  },
  geometric: {
    label: 'Geometric', kind: 'discrete', formula: 'P(k)=(1−p)ᵏ⁻¹p',
    knobs: [{ key: 'p', name: 'p · success prob', min: 0.05, max: 0.95, step: 0.01, hint: 'P(success) per trial' }],
    support: (p) => [1, Math.max(6, Math.ceil(Math.log(0.02) / Math.log(1 - num(p, 'p'))))],
    pmf: (k, p) => (k < 1 ? 0 : Math.pow(1 - num(p, 'p'), k - 1) * num(p, 'p')),
    sample: (p) => Math.ceil(Math.log(1 - Math.random()) / Math.log(1 - num(p, 'p'))),
    mean: (p) => 1 / num(p, 'p'), variance: (p) => (1 - num(p, 'p')) / (num(p, 'p') * num(p, 'p')),
    entropy: (p) => { const q = num(p, 'p'); return (negPlnP(1 - q) + negPlnP(q)) / q; },
    note: 'Trials until the first success — memoryless: past failures do not help.',
  },
  uniform: {
    label: 'Uniform', kind: 'continuous', formula: 'f(x)=1/(b−a)',
    knobs: [
      { key: 'a', name: 'a · lower', min: -3, max: 0, step: 0.1, hint: 'support start' },
      { key: 'b', name: 'b · upper', min: 0.5, max: 4, step: 0.1, hint: 'support end' },
    ],
    support: (p) => { const a = num(p, 'a'), b = num(p, 'b'); return [a - 0.4 * (b - a), b + 0.4 * (b - a)]; },
    pdf: (x, p) => (x >= num(p, 'a') && x <= num(p, 'b') ? 1 / (num(p, 'b') - num(p, 'a')) : 0),
    cdf: (x, p) => Math.min(1, Math.max(0, (x - num(p, 'a')) / (num(p, 'b') - num(p, 'a')))),
    sample: (p) => num(p, 'a') + (num(p, 'b') - num(p, 'a')) * Math.random(),
    mean: (p) => (num(p, 'a') + num(p, 'b')) / 2, variance: (p) => (num(p, 'b') - num(p, 'a')) ** 2 / 12,
    entropy: (p) => Math.log(num(p, 'b') - num(p, 'a')),
    note: 'No preference on [a,b] — maximum entropy given a bounded support.',
  },
  normal: {
    label: 'Normal', kind: 'continuous', formula: 'f(x)=e^(−z²/2)/(σ√2π)',
    knobs: [
      { key: 'mu', name: 'μ · mean', min: -3, max: 3, step: 0.1, hint: 'centre' },
      { key: 'sigma', name: 'σ · std-dev', min: 0.2, max: 2.5, step: 0.05, hint: 'spread' },
    ],
    support: (p) => [num(p, 'mu') - 4 * num(p, 'sigma'), num(p, 'mu') + 4 * num(p, 'sigma')],
    pdf: (x, p) => normalPdf(x, num(p, 'mu'), num(p, 'sigma')),
    cdf: (x, p) => normCdf((x - num(p, 'mu')) / num(p, 'sigma')),
    sample: (p) => num(p, 'mu') + num(p, 'sigma') * randn(),
    mean: (p) => num(p, 'mu'), variance: (p) => num(p, 'sigma') ** 2,
    entropy: (p) => 0.5 * Math.log(2 * Math.PI * Math.E * num(p, 'sigma') ** 2),
    note: 'The bell curve — the CLT attractor for sums of many small effects.',
  },
  exponential: {
    label: 'Exponential', kind: 'continuous', formula: 'f(x)=λe^(−λx)',
    knobs: [{ key: 'lam', name: 'λ · rate', min: 0.3, max: 4, step: 0.05, hint: 'rate (mean=1/λ)' }],
    support: (p) => [0, Math.max(3, 6 / num(p, 'lam'))],
    pdf: (x, p) => (x >= 0 ? num(p, 'lam') * Math.exp(-num(p, 'lam') * x) : 0),
    cdf: (x, p) => (x > 0 ? 1 - Math.exp(-num(p, 'lam') * x) : 0),
    sample: (p) => -Math.log(1 - Math.random()) / num(p, 'lam'),
    mean: (p) => 1 / num(p, 'lam'), variance: (p) => 1 / num(p, 'lam') ** 2,
    entropy: (p) => 1 - Math.log(num(p, 'lam')),
    note: 'Waiting time between Poisson events — the continuous memoryless law.',
  },
  beta: {
    label: 'Beta', kind: 'continuous', formula: 'f(x)∝x^(α−1)·(1−x)^(β−1)',
    knobs: [
      { key: 'a', name: 'α · shape', min: 0.5, max: 8, step: 0.1, hint: 'pulls mass toward 1' },
      { key: 'b', name: 'β · shape', min: 0.5, max: 8, step: 0.1, hint: 'pulls mass toward 0' },
    ],
    support: () => [0, 1],
    pdf: (x, p) => {
      if (x <= 0 || x >= 1) return 0;
      const a = num(p, 'a'), b = num(p, 'b');
      const logB = lgamma(a) + lgamma(b) - lgamma(a + b);
      return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB);
    },
    cdf: (x, p) => betaInc(x, num(p, 'a'), num(p, 'b')),
    // ratio of two Gamma draws (Marsaglia–Tsang) → Beta(α, β)
    sample: (p) => { const ga = gammaSample(num(p, 'a')), gb = gammaSample(num(p, 'b')); return ga / (ga + gb); },
    mean: (p) => num(p, 'a') / (num(p, 'a') + num(p, 'b')),
    variance: (p) => { const a = num(p, 'a'), b = num(p, 'b'); return (a * b) / ((a + b) ** 2 * (a + b + 1)); },
    entropy: (p) => {
      const a = num(p, 'a'), b = num(p, 'b');
      return lgamma(a) + lgamma(b) - lgamma(a + b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b);
    },
    note: 'A flexible law over a probability in [0,1] — the Bernoulli’s conjugate prior.',
  },
};

export function normalPdf(x: number, mu: number, sd: number): number {
  const z = (x - mu) / sd;
  return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI));
}

export const DEFAULTS: Record<Family, Params> = {
  bernoulli: { p: 0.4 },
  binomial: { n: 20, p: 0.4 },
  poisson: { lam: 4 },
  geometric: { p: 0.35 },
  uniform: { a: 0, b: 1 },
  normal: { mu: 0, sigma: 1 },
  exponential: { lam: 1 },
  beta: { a: 2, b: 5 },
};
export const ORDER: Family[] = ['bernoulli', 'binomial', 'poisson', 'geometric', 'uniform', 'normal', 'exponential', 'beta'];
export const N_BINS = 36;      // continuous histogram bins
export const BATCH = 40;       // samples drawn per animation tick

/**
 * Histogram layout. `integer`: one unit-width bin per integer in [lo, hi];
 * `bins`: N_BINS equal bins over [lo, hi). Anything outside is counted as
 * out-of-window (kept in n, never folded into an edge bin). `mass[b]` is the
 * exact reference probability of bin b — the pmf for single discrete draws, a
 * CDF difference for single continuous draws — and for sums of k draws the CLT
 * normal N(kμ, kσ²) (continuity-corrected ±½ for integer sums).
 */
export interface Layout {
  type: 'integer' | 'bins';
  lo: number;
  hi: number;
  nBins: number;
  binW: number;
  key: (x: number) => number | null;
  center: (b: number) => number;
  mass: number[];
  /** reference density curve for the plot (continuous single draws and all sums) */
  density: ((x: number) => number) | null;
  /** reference probability outside the window */
  outMass: number;
}

export function makeLayout(def: FamilyDef, p: Params, k: number): Layout {
  const mu = def.mean(p), sd = Math.sqrt(def.variance(p));
  const binsKey = (lo: number, hi: number, binW: number) => (x: number) =>
    (x >= lo && x < hi ? Math.min(N_BINS - 1, Math.floor((x - lo) / binW)) : null);
  if (k <= 1 && def.kind === 'discrete') {
    const [lo, hi] = def.support(p);
    const mass = Array.from({ length: hi - lo + 1 }, (_, b) => def.pmf!(lo + b, p));
    return {
      type: 'integer', lo, hi, nBins: mass.length, binW: 1,
      key: (x) => (x >= lo && x <= hi ? Math.round(x) - lo : null),
      center: (b) => lo + b,
      mass, density: null,
      outMass: Math.max(0, 1 - mass.reduce((a, v) => a + v, 0)),
    };
  }
  if (k <= 1) {
    const [lo, hi] = def.support(p);
    const binW = (hi - lo) / N_BINS;
    const F = (x: number) => def.cdf!(x, p);
    return {
      type: 'bins', lo, hi, nBins: N_BINS, binW,
      key: binsKey(lo, hi, binW),
      center: (b) => lo + (b + 0.5) * binW,
      mass: Array.from({ length: N_BINS }, (_, b) => F(lo + (b + 1) * binW) - F(lo + b * binW)),
      density: (x) => def.pdf!(x, p),
      outMass: Math.max(0, 1 - (F(hi) - F(lo))),
    };
  }
  const m = k * mu, s = Math.sqrt(k) * sd;
  const Phi = (x: number) => normCdf((x - m) / s);
  const density = (x: number) => normalPdf(x, m, s);
  if (def.kind === 'discrete') {
    const lo = Math.floor(m - 4 * s), hi = Math.ceil(m + 4 * s);
    return {
      type: 'integer', lo, hi, nBins: hi - lo + 1, binW: 1,
      key: (x) => (x >= lo && x <= hi ? Math.round(x) - lo : null),
      center: (b) => lo + b,
      mass: Array.from({ length: hi - lo + 1 }, (_, b) => Phi(lo + b + 0.5) - Phi(lo + b - 0.5)),
      density,
      outMass: Math.max(0, 1 - (Phi(hi + 0.5) - Phi(lo - 0.5))),
    };
  }
  const lo = m - 4 * s, hi = m + 4 * s, binW = (hi - lo) / N_BINS;
  return {
    type: 'bins', lo, hi, nBins: N_BINS, binW,
    key: binsKey(lo, hi, binW),
    center: (b) => lo + (b + 0.5) * binW,
    mass: Array.from({ length: N_BINS }, (_, b) => Phi(lo + (b + 1) * binW) - Phi(lo + b * binW)),
    density,
    outMass: Math.max(0, 1 - (Phi(hi) - Phi(lo))),
  };
}

/** Running sample statistics: histogram counts, out-of-window count and Welford moments. */
export interface Accum { n: number; counts: number[]; out: number; mean: number; m2: number; }
export const newAccum = (nBins: number): Accum => ({ n: 0, counts: new Array(nBins).fill(0), out: 0, mean: 0, m2: 0 });

/** One draw of the lab's random variable: X (k = 1) or the sum of k draws. */
export function drawValue(def: FamilyDef, p: Params, k: number): number {
  if (k <= 1) return def.sample(p);
  let s = 0; for (let j = 0; j < k; j++) s += def.sample(p);
  return s;
}

export function addSample(acc: Accum, layout: Layout, x: number): void {
  acc.n += 1;
  const d = x - acc.mean;
  acc.mean += d / acc.n;
  acc.m2 += d * (x - acc.mean);
  const b = layout.key(x);
  if (b == null) acc.out += 1;
  else acc.counts[b] = (acc.counts[b] ?? 0) + 1;
}

/** Empirical variance (population form, matching NumPy's default var()). */
export const empVar = (acc: Accum) => (acc.n > 0 ? acc.m2 / acc.n : 0);

/** L1 distance between the binned empirical law and the reference: Σ|emp−ref| over bins + |out−outMass|. */
export function l1Error(acc: Accum, layout: Layout): number {
  if (acc.n === 0) return 0;
  let err = 0;
  for (let b = 0; b < layout.nBins; b++) err += Math.abs((acc.counts[b] ?? 0) / acc.n - (layout.mass[b] ?? 0));
  return err + Math.abs(acc.out / acc.n - layout.outMass);
}
