// Pure maths shared by the Probability labs (no React): log-gamma, digamma, the
// Beta density and its credible interval, and the diagnostic-test (Bayes' rule)
// arithmetic. Framework-free so a Node harness can verify every quoted number.

/** log Γ(x) — Lanczos approximation (g = 7, 9 coefficients), reflection below ½. */
export function lgamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = c[0] ?? 1;
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += (c[i] ?? 0) / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Digamma ψ(x) for x > 0: recurrence up to x ≥ 6, then the asymptotic series. */
export function digamma(x: number): number {
  let r = 0;
  while (x < 6) { r -= 1 / x; x += 1; }
  const f = 1 / (x * x);
  return r + Math.log(x) - 0.5 / x
    - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))));
}

/** Standard normal CDF Φ(z) via the Abramowitz–Stegun 7.1.26 erf (|error| ≤ 1.5e-7). */
export function normCdf(z: number): number {
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - poly * Math.exp(-x * x);
  return z >= 0 ? 0.5 * (1 + erf) : 0.5 * (1 - erf);
}

/** Continued fraction for the incomplete beta (modified Lentz, Numerical Recipes betacf). */
function betaCf(a: number, b: number, x: number): number {
  const FPMIN = 1e-300;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-14) break;
  }
  return h;
}

/** Regularised incomplete beta I_x(a, b) — the Beta(a, b) CDF. */
export function betaInc(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betaCf(a, b, x)) / a : 1 - (bt * betaCf(b, a, 1 - x)) / b;
}

/** Beta(a, b) density. */
export function betaPdf(x: number, a: number, b: number): number {
  if (x <= 0 || x >= 1) return 0;
  const logB = lgamma(a) + lgamma(b) - lgamma(a + b);
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB);
}

/** Equal-tailed 90% credible interval of Beta(a, b) from an 801-point grid CDF. */
export function betaCI90(a: number, b: number): { lo: number; hi: number } {
  const G = 801;
  const xs: number[] = [], pdf: number[] = [];
  for (let i = 0; i < G; i++) { const x = i / (G - 1); xs.push(x); pdf.push(betaPdf(x, a, b)); }
  let total = 0; for (const v of pdf) total += v;
  const target = (q: number) => {
    let acc = 0;
    for (let i = 0; i < G; i++) { acc += pdf[i] ?? 0; if (acc / total >= q) return xs[i] ?? 1; }
    return 1;
  };
  return { lo: target(0.05), hi: target(0.95) };
}

/** Bayes' rule for a diagnostic test: P(+), P(−), P(D|+) and P(D|−). */
export function diagnostic(prev: number, sens: number, spec: number) {
  const pPos = sens * prev + (1 - spec) * (1 - prev);
  const pNeg = (1 - sens) * prev + spec * (1 - prev);
  return {
    pPos, pNeg,
    postPos: pPos > 0 ? (sens * prev) / pPos : 0,
    postNeg: pNeg > 0 ? ((1 - sens) * prev) / pNeg : 0,
  };
}

export interface Counts { tp: number; fn: number; fp: number; tn: number; }

/** Exact expected confusion counts in a population of N (fractional people). */
export function expectedCounts(N: number, prev: number, sens: number, spec: number): Counts {
  const sick = N * prev, healthy = N - sick;
  return { tp: sick * sens, fn: sick * (1 - sens), fp: healthy * (1 - spec), tn: healthy * spec };
}

/**
 * The same counts rounded to whole people for an icon array of N: the sick count
 * is rounded first, then true positives within the sick and false positives
 * within the healthy, so the four cells always sum to N.
 */
export function roundedCounts(N: number, prev: number, sens: number, spec: number): Counts {
  const sick = Math.round(prev * N);
  const tp = Math.round(sick * sens);
  const healthy = N - sick;
  const fp = Math.round(healthy * (1 - spec));
  return { tp, fn: sick - tp, fp, tn: healthy - fp };
}
