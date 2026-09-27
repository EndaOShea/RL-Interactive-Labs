// Seeded datasets for the Supervised Learning labs. Every point is drawn from a
// seeded mulberry32 stream and rounded to 4 decimals, so a dataset is fully
// reproducible and the Python exports (which embed the exact points) train on
// the very same numbers the lab shows. Pure module, no React.

export type Rng = () => number;

/** mulberry32: a small 32-bit PRNG returning uniforms in [0, 1). */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal via Box–Muller (two uniforms per sample). */
export function gauss(r: Rng): number {
  const u = 1 - r();           // (0, 1] — never log(0)
  const v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Round to 4 decimals — the stored (and exported) precision of every point. */
export const r4 = (v: number) => Math.round(v * 1e4) / 1e4;
const clampUnit = (v: number) => Math.max(0.03, Math.min(0.97, v));

/** Fisher–Yates shuffle of 0..n-1 driven by the seeded generator. */
export function permutation(n: number, r: Rng): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    const t = idx[i]!; idx[i] = idx[j]!; idx[j] = t;
  }
  return idx;
}

/** Offset between the training stream and the held-out test stream of a seed. */
export const TEST_SEED_OFFSET = 7919;

/* ─────────────── XOR clusters (Decision Tree + Gradient Boosting) ─────────────── */

export interface XorPt { x: number; y: number; cls: number; }
/** Cluster centres in [0,1]²: bottom-left, bottom-right, top-left, top-right. */
export const XOR_CENTERS: readonly { x: number; y: number }[] = [
  { x: 0.28, y: 0.3 }, { x: 0.72, y: 0.3 }, { x: 0.28, y: 0.72 }, { x: 0.72, y: 0.72 },
];
/** XOR labelling: BL and TR are class 0, BR and TL are class 1. */
export const XOR_CLS: readonly number[] = [0, 1, 1, 0];

/**
 * Decision-tree layouts. 'uneven': big clusters (3 parts each) on the bottom row,
 * small ones (1 part) on the top row, the big ones nearer the centre line x₁ = 0.5
 * — the centre cut then has a clear Gini gain and no stray point of the small
 * clusters can make an off-centre root cut better. 'balanced': the classic equal
 * XOR, where the centre cut leaves both halves 50/50 (zero gain).
 */
export type XorLayout = 'uneven' | 'balanced';
export const XOR_LAYOUTS: Record<XorLayout, { centers: readonly { x: number; y: number }[]; weights: readonly number[] }> = {
  uneven: { centers: [{ x: 0.3, y: 0.3 }, { x: 0.7, y: 0.3 }, { x: 0.2, y: 0.72 }, { x: 0.8, y: 0.72 }], weights: [3, 3, 1, 1] },
  balanced: { centers: XOR_CENTERS, weights: [1, 1, 1, 1] },
};
/** Cluster sizes for `total` points split by the layout's weights. */
export const xorSizes = (layout: XorLayout, total: number): number[] => {
  const w = XOR_LAYOUTS[layout].weights;
  const s = w.reduce((a, b) => a + b, 0);
  return w.map((k) => Math.round((total * k) / s));
};
/** Cluster spread of the decision-tree data. */
export const DT_STD = 0.07;
/** Gradient-boosting data: the balanced XOR with this spread. */
export const GBM_STD = 0.085;

/**
 * Four Gaussian clusters with the XOR labelling. `sizes[k]` points in cluster k,
 * isotropic spread `std`, clamped to [0.03, 0.97]. `noise` flips the labels of
 * exactly round(noise·n) points chosen by a seeded shuffle (label noise).
 */
export function makeXorData(sizes: readonly number[], std: number, seed: number, noise = 0, centers: readonly { x: number; y: number }[] = XOR_CENTERS): XorPt[] {
  const r = mulberry32(seed);
  const pts: XorPt[] = [];
  centers.forEach((c, k) => {
    const cnt = sizes[k] ?? 0;
    for (let i = 0; i < cnt; i++) {
      const x = r4(clampUnit(c.x + gauss(r) * std));
      const y = r4(clampUnit(c.y + gauss(r) * std));
      pts.push({ x, y, cls: XOR_CLS[k] ?? 0 });
    }
  });
  const flips = Math.round(noise * pts.length);
  if (flips > 0) {
    const order = permutation(pts.length, r);
    for (let k = 0; k < flips; k++) {
      const p = pts[order[k]!];
      if (p) p.cls = 1 - p.cls;
    }
  }
  return pts;
}

/* ─────────────────────────────── SVM datasets ─────────────────────────────── */

export type SvmShape = 'blobs' | 'moons' | 'rings';
export interface SvmPt { x: number; y: number; yy: number; } // yy ∈ {−1, +1}

/**
 * Two-class data in [−1.2, 1.2]² for the SVM lab. `sep` ∈ [0, 1] widens the class
 * gap (blobs) or lowers the noise (moons, rings). Points alternate −1, +1.
 */
export function makeSvmData(perClass: number, sep: number, shape: SvmShape, seed: number): SvmPt[] {
  const r = mulberry32(seed);
  const out: SvmPt[] = [];
  const push = (x: number, y: number, yy: number) => out.push({ x: r4(x), y: r4(y), yy });
  if (shape === 'blobs') {
    const off = 0.22 + sep * 0.3;
    for (let i = 0; i < perClass; i++) {
      push(-off + gauss(r) * 0.16, -off + gauss(r) * 0.16, -1);
      push(off + gauss(r) * 0.16, off + gauss(r) * 0.16, 1);
    }
  } else if (shape === 'moons') {
    const noise = 0.12 - sep * 0.06;
    for (let i = 0; i < perClass; i++) {
      const t = Math.PI * (i / perClass);
      push(Math.cos(t) * 0.6 - 0.25 + gauss(r) * noise, Math.sin(t) * 0.6 - 0.18 + gauss(r) * noise, -1);
      push(Math.cos(t) * 0.6 + 0.25 + gauss(r) * noise, -Math.sin(t) * 0.6 + 0.18 + gauss(r) * noise, 1);
    }
  } else {
    const noise = 0.1 - sep * 0.05;
    for (let i = 0; i < perClass; i++) {
      const a = 2 * Math.PI * r();
      push(Math.cos(a) * 0.32 + gauss(r) * noise, Math.sin(a) * 0.32 + gauss(r) * noise, -1);
      const b = 2 * Math.PI * r();
      push(Math.cos(b) * 0.88 + gauss(r) * noise, Math.sin(b) * 0.88 + gauss(r) * noise, 1);
    }
  }
  return out;
}

/* ───────────────────────────── Naive Bayes blobs ───────────────────────────── */

export interface NbPt { x: number; y: number; cls: number; }
export const NB_CENTERS: readonly { x: number; y: number }[] = [
  { x: 0.3, y: 0.32 }, { x: 0.7, y: 0.34 }, { x: 0.5, y: 0.72 },
];
/**
 * Per-class spread multipliers (σx, σy) for the "unequal shapes" option: class 0
 * tight and round, class 1 wide along x, class 2 tall along y. "Equal" uses 1.
 */
export const NB_SHAPES: readonly { sx: number; sy: number }[] = [
  { sx: 0.6, sy: 0.6 }, { sx: 1.7, sy: 0.8 }, { sx: 0.8, sy: 1.5 },
];

/** Class counts: class 1 gets round(imbalance·perClass), classes 0 and 2 get perClass. */
export const nbCounts = (perClass: number, imbalance: number): number[] =>
  [perClass, Math.round(imbalance * perClass), perClass];

export function makeNbData(perClass: number, spread: number, unequal: boolean, imbalance: number, seed: number): NbPt[] {
  const r = mulberry32(seed);
  const counts = nbCounts(perClass, imbalance);
  const pts: NbPt[] = [];
  NB_CENTERS.forEach((c, k) => {
    const m = unequal ? NB_SHAPES[k]! : { sx: 1, sy: 1 };
    for (let i = 0; i < (counts[k] ?? 0); i++) {
      const x = r4(clampUnit(c.x + gauss(r) * spread * m.sx));
      const y = r4(clampUnit(c.y + gauss(r) * spread * m.sy));
      pts.push({ x, y, cls: k });
    }
  });
  return pts;
}
