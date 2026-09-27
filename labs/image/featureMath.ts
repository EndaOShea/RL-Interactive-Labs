// Pure maths for the CNN Feature Maps lab (no React, no imports) so the lab, its
// Python export and verification harnesses all share one implementation.
//
// Pipeline: glyph → optional perturbation → 3 fixed 3×3 filters (same-size,
// zero-padded cross-correlation) → ReLU → 2×2 pooling (max | avg) → flatten →
// cosine similarity to CLEAN class templates → softmax(8·cos).
//
// The templates are the pipeline vectors of the three clean glyphs. Classifying a
// clean glyph therefore always scores cos = 1 for its own class; the perturbations
// (shifts, thicker strokes, noise) are what make the match — and pooling's
// tolerance to them — an actual measurement.

export const GLYPH = 12; // glyph side (px)

export type ClassId = 'H' | 'T' | 'O';
export const CLASSES: ClassId[] = ['H', 'T', 'O'];

export type PoolMode = 'max' | 'avg';
export const POOL_MODES: PoolMode[] = ['max', 'avg'];

/** Softmax temperature: p = softmax(SOFTMAX_SCALE · cos). */
export const SOFTMAX_SCALE = 8;

const at =(m: number[][], r: number, c: number): number => m[r]?.[c] ?? 0;

/** The three clean 12×12 binary glyphs (1 = ink). */
export function makeGlyph(cls: ClassId): number[][] {
  const inRect = (r: number, c: number, r0: number, r1: number, c0: number, c1: number) => r >= r0 && r < r1 && c >= c0 && c < c1;
  return Array.from({ length: GLYPH }, (_, r) => Array.from({ length: GLYPH }, (_, c) => {
    if (cls === 'H') return inRect(r, c, 2, 10, 2, 4) || inRect(r, c, 2, 10, 8, 10) || inRect(r, c, 5, 7, 2, 10) ? 1 : 0;
    if (cls === 'T') return inRect(r, c, 2, 4, 2, 10) || inRect(r, c, 2, 10, 5, 7) ? 1 : 0;
    return inRect(r, c, 2, 10, 2, 10) && !inRect(r, c, 4, 8, 4, 8) ? 1 : 0; // O ring
  }));
}

export interface Filter { id: string; name: string; k: number[][]; }

// Hand-picked filters (NOT trained). The two edge filters are signed; "blur" is
// a 3×3 weighted average (weights sum to 1) — it responds to ink mass, not to a
// particular shape.
export const FILTERS: Filter[] = [
  { id: 'vertical-edge', name: 'vertical edge', k: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] },
  { id: 'horizontal-edge', name: 'horizontal edge', k: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] },
  { id: 'blur', name: 'blur (3×3 average)', k: [[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]] },
];

/** Same-size 3×3 cross-correlation with zero padding: out(i,j) = Σ I(i+m,j+n)·K(m,n). */
export function conv(img: number[][], k: number[][]): number[][] {
  const H = img.length, W = img[0]?.length ?? 0;
  return Array.from({ length: H }, (_, i) => Array.from({ length: W }, (_, j) => {
    let s = 0;
    for (let m = -1; m <= 1; m++) for (let n = -1; n <= 1; n++) {
      const r = i + m, c = j + n;
      const v = r >= 0 && r < H && c >= 0 && c < W ? at(img, r, c) : 0;
      s += v * at(k, m + 1, n + 1);
    }
    return s;
  }));
}

export const relu = (m: number[][]): number[][] => m.map((row) => row.map((v) => Math.max(0, v)));

/** 2×2 pooling, stride 2 (12×12 → 6×6). */
export function pool2(m: number[][], mode: PoolMode): number[][] {
  const H2 = m.length >> 1, W2 = (m[0]?.length ?? 0) >> 1;
  return Array.from({ length: H2 }, (_, i) => Array.from({ length: W2 }, (_, j) => {
    const a = at(m, 2 * i, 2 * j), b = at(m, 2 * i, 2 * j + 1), c = at(m, 2 * i + 1, 2 * j), d = at(m, 2 * i + 1, 2 * j + 1);
    return mode === 'max' ? Math.max(a, b, c, d) : (a + b + c + d) / 4;
  }));
}

/** Row-major flatten of each map, maps concatenated in FILTERS order. */
export const flatten = (mats: number[][][]): number[] => mats.flatMap((m) => m.flat());

export function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0, y = b[i] ?? 0;
    dot += x * y; na += x * x; nb += y * y;
  }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

export function softmax(z: number[]): number[] {
  const mx = Math.max(...z);
  const e = z.map((v) => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export interface Pipeline {
  raw: number[][][];    // signed conv maps, one per filter
  relued: number[][][]; // post-ReLU maps
  pooled: number[][][]; // pooled maps (6×6)
  vec: number[];        // flattened feature vector (3·36 = 108 dims)
}

export function pipeline(img: number[][], pool: PoolMode): Pipeline {
  const raw = FILTERS.map((f) => conv(img, f.k));
  const relued = raw.map(relu);
  const pooled = relued.map((m) => pool2(m, pool));
  return { raw, relued, pooled, vec: flatten(pooled) };
}

// ---------------------------------------------------------------------------
// Perturbations — applied to the QUERY glyph only; the templates stay clean.
// ---------------------------------------------------------------------------
export type Perturb = 'clean' | 'shift1' | 'shift2' | 'thick' | 'noise';
export const PERTURBS: Perturb[] = ['clean', 'shift1', 'shift2', 'thick', 'noise'];

/** Seed and amplitude of the additive noise perturbation (uniform ±NOISE_AMP, clipped to [0,1]). */
export const NOISE_SEED = 7;
export const NOISE_AMP = 0.5;

/** Deterministic 32-bit PRNG (mulberry32) — ported exactly to the Python export. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Translate by (dy, dx) pixels; pixels shifted in from outside are 0. */
export function shiftImg(img: number[][], dy: number, dx: number): number[][] {
  const H = img.length, W = img[0]?.length ?? 0;
  return Array.from({ length: H }, (_, r) => Array.from({ length: W }, (_, c) => {
    const rr = r - dy, cc = c - dx;
    return rr >= 0 && rr < H && cc >= 0 && cc < W ? at(img, rr, cc) : 0;
  }));
}

/** Grow every stroke by one pixel (3×3 max filter = binary dilation). */
export function dilate(img: number[][]): number[][] {
  const H = img.length, W = img[0]?.length ?? 0;
  return Array.from({ length: H }, (_, r) => Array.from({ length: W }, (_, c) => {
    let mx = 0;
    for (let m = -1; m <= 1; m++) for (let n = -1; n <= 1; n++) {
      const rr = r + m, cc = c + n;
      if (rr >= 0 && rr < H && cc >= 0 && cc < W) mx = Math.max(mx, at(img, rr, cc));
    }
    return mx;
  }));
}

/** Add seeded uniform noise in [−amp, amp] (row-major draw order), clip to [0, 1]. */
export function addNoise(img: number[][], seed = NOISE_SEED, amp = NOISE_AMP): number[][] {
  const rnd = mulberry32(seed);
  return img.map((row) => row.map((v) => Math.min(1, Math.max(0, v + amp * (2 * rnd() - 1)))));
}

export function perturb(img: number[][], p: Perturb): number[][] {
  switch (p) {
    case 'clean': return img.map((row) => row.slice());
    case 'shift1': return shiftImg(img, 0, 1); // 1 px to the right
    case 'shift2': return shiftImg(img, 0, 2); // 2 px to the right
    case 'thick': return dilate(img);
    case 'noise': return addNoise(img);
  }
}

export interface PoolResult {
  pool: PoolMode;
  scores: number[];
  probs: number[];
  predIdx: number;
  ownCos: number;  // cosine to the input's own clean template
  margin: number;  // ownCos − best other-class cosine (< 0 ⇒ misclassified)
}

/** Classify one (possibly perturbed) query under a pooling mode against clean templates. */
export function evaluate(query: number[][], cls: ClassId, pool: PoolMode, tpl: Record<ClassId, number[]> = templates(pool)): PoolResult {
  const r = classifyVec(pipeline(query, pool).vec, tpl);
  const own = CLASSES.indexOf(cls);
  const ownCos = r.scores[own] ?? 0;
  const others = r.scores.filter((_, i) => i !== own);
  return { pool, ...r, ownCos, margin: ownCos - Math.max(...others) };
}

/** Clean-glyph templates for one pooling mode. */
export function templates(pool: PoolMode): Record<ClassId, number[]> {
  return {
    H: pipeline(makeGlyph('H'), pool).vec,
    T: pipeline(makeGlyph('T'), pool).vec,
    O: pipeline(makeGlyph('O'), pool).vec,
  };
}

/** Per-filter summary of a post-ReLU map stack: peak and total activation. */
export function filterStats(maps: number[][][]): { peak: number; total: number }[] {
  return maps.map((m) => ({ peak: matMax(m), total: matSum(m) }));
}

export interface Classification {
  scores: number[]; // cosine to each class template, CLASSES order
  probs: number[];  // softmax(8·cos)
  predIdx: number;
}

export function classifyVec(vec: number[], tpl: Record<ClassId, number[]>): Classification {
  const scores = CLASSES.map((c) => cosine(vec, tpl[c]));
  const probs = softmax(scores.map((s) => s * SOFTMAX_SCALE));
  let predIdx = 0;
  probs.forEach((p, i) => { if (p > (probs[predIdx] ?? -Infinity)) predIdx = i; });
  return { scores, probs, predIdx };
}

/** Max of a matrix (−Infinity for an empty one). */
export const matMax = (m: number[][]): number => m.reduce((mx, row) => row.reduce((a, v) => Math.max(a, v), mx), -Infinity);
/** Sum of a matrix. */
export const matSum = (m: number[][]): number => m.reduce((s, row) => row.reduce((a, v) => a + v, s), 0);
