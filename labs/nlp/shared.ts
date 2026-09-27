// Deterministic, client-side maths shared by the NLP labs: vector ops, the
// tokenizers, the stop-word / negation lists, the seeded PRNG, and a symmetric
// eigen-solver used for PCA. No servers, no TF.js. The lab-specific cores live
// alongside this file:
//   embeddingTable.ts — the shared, hand-built word-vector table + analogies
//   ngramCore.ts      — add-k n-gram LM, perplexity, seeded sampling
//   tfidfCore.ts      — TF-IDF variants, cosine matrix, per-term contributions
//   nerCore.ts        — BIO tagger: emission features, transitions, Viterbi
//   searchCore.ts     — text → vector retrieval + the TF-IDF keyword baseline
//   sentimentCore.ts  — review features, PCA-2, regularised logistic regression

export type Vec = number[];

/* ---------- vector ops ---------- */
export const addV = (a: Vec, b: Vec): Vec => a.map((x, i) => x + (b[i] ?? 0));
export const subV = (a: Vec, b: Vec): Vec => a.map((x, i) => x - (b[i] ?? 0));
export const scaleV = (a: Vec, s: number): Vec => a.map((x) => x * s);
export const dot = (a: Vec, b: Vec): number => a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0);
export const norm = (a: Vec): number => Math.sqrt(dot(a, a));

/** Cosine similarity in [-1, 1]; 0 for a zero vector. */
export const cosine = (a: Vec, b: Vec): number => {
  const d = norm(a) * norm(b);
  return d < 1e-9 ? 0 : dot(a, b) / d;
};

/** a / |a| (the zero vector stays zero). */
export const unitV = (a: Vec): Vec => {
  const n = norm(a);
  return n < 1e-12 ? a.map(() => 0) : a.map((x) => x / n);
};

/** Element-wise mean of equal-length rows (summed in row order). */
export function meanV(rows: Vec[], dim: number): Vec {
  const m = new Array<number>(dim).fill(0);
  for (const r of rows) for (let j = 0; j < dim; j++) m[j] = (m[j] ?? 0) + (r[j] ?? 0);
  return rows.length ? m.map((x) => x / rows.length) : m;
}

/* ---------- tokenizers + word lists ---------- */
export const tokenize = (s: string): string[] =>
  s.toLowerCase().match(/[a-z']+/g) ?? [];

/** Like tokenize, but keeps . , ; : ! ? as tokens (they end a negation scope). */
export const tokenizeWithPunct = (s: string): string[] =>
  s.toLowerCase().match(/[a-z']+|[.,;:!?]/g) ?? [];

export const isPunct = (t: string): boolean => /^[.,;:!?]$/.test(t);

/** The one stop-word list used by the TF-IDF toggle, search and classification. */
export const STOP_WORDS: readonly string[] = [
  'a', 'after', 'all', 'an', 'and', 'around', 'as', 'at', 'be', 'but', 'by', 'every',
  'for', 'from', 'in', 'into', 'is', 'it', 'its', 'of', 'on', 'or', 'so', 'than',
  'that', 'the', 'their', 'then', 'there', 'these', 'they', 'this', 'those', 'to',
  'was', 'were', 'with',
];
const STOP_SET = new Set(STOP_WORDS);
export const isStopWord = (t: string): boolean => STOP_SET.has(t);

/** Negation cues: not / no / never / nor / hardly, and any "…n't" contraction. */
export const NEGATORS: readonly string[] = ['not', 'no', 'never', 'nor', 'hardly'];
const NEG_SET = new Set(NEGATORS);
export const isNegator = (t: string): boolean => NEG_SET.has(t) || t.endsWith("n't");

/* ---------- seeded PRNG ---------- */
/** mulberry32: a 32-bit seeded PRNG, uniform on [0, 1). Ported bit-for-bit to the Python exports. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- symmetric eigen-decomposition (cyclic Jacobi) + PCA ---------- */
/** Eigen-decomposition of a symmetric matrix. Returns eigenvalues (descending) and
 *  unit eigenvectors (vectors[i] belongs to values[i]), each signed so that its
 *  largest-magnitude entry is positive (a deterministic convention; numpy's eigh
 *  with the same rule gives the same vectors). */
export function symEig(A: number[][]): { values: number[]; vectors: Vec[] } {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  const at = (i: number, j: number) => a[i]?.[j] ?? 0;
  const set = (m: number[][], i: number, j: number, v: number) => { const r = m[i]; if (r) r[j] = v; };
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += at(p, q) ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = at(p, q);
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (at(q, q) - at(p, p)) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { // A ← A·J (columns p, q)
          const akp = at(k, p), akq = at(k, q);
          set(a, k, p, c * akp - s * akq); set(a, k, q, s * akp + c * akq);
        }
        for (let k = 0; k < n; k++) { // A ← Jᵀ·A (rows p, q)
          const apk = at(p, k), aqk = at(q, k);
          set(a, p, k, c * apk - s * aqk); set(a, q, k, s * apk + c * aqk);
        }
        for (let k = 0; k < n; k++) { // V ← V·J
          const vkp = V[k]?.[p] ?? 0, vkq = V[k]?.[q] ?? 0;
          set(V, k, p, c * vkp - s * vkq); set(V, k, q, s * vkp + c * vkq);
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => at(j, j) - at(i, i));
  const values = order.map((i) => at(i, i));
  const vectors = order.map((i) => {
    const v = V.map((row) => row[i] ?? 0);
    let big = 0;
    v.forEach((x, j) => { if (Math.abs(x) > Math.abs(v[big] ?? 0)) big = j; });
    return (v[big] ?? 0) < 0 ? v.map((x) => -x) : v;
  });
  return { values, vectors };
}

export interface PcaResult {
  mean: Vec;
  /** top-k principal directions (unit vectors in the input space) */
  components: Vec[];
  /** their variances (eigenvalues of the sample covariance, n − 1 denominator) */
  variances: number[];
  /** variances / total variance */
  explained: number[];
}

/** PCA of the rows (n × d): centre, sample covariance, eigen-decompose, keep k. */
export function pca(rows: Vec[], k: number): PcaResult {
  const d = rows[0]?.length ?? 0;
  const mean = meanV(rows, d);
  const X = rows.map((r) => subV(r, mean));
  const denom = Math.max(1, rows.length - 1);
  const C: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (const x of X) {
    for (let i = 0; i < d; i++) {
      const xi = x[i] ?? 0;
      const Ci = C[i];
      if (!Ci) continue;
      for (let j = 0; j < d; j++) Ci[j] = (Ci[j] ?? 0) + xi * (x[j] ?? 0);
    }
  }
  for (const row of C) for (let j = 0; j < d; j++) row[j] = (row[j] ?? 0) / denom;
  const { values, vectors } = symEig(C);
  const total = values.reduce((s, v) => s + Math.max(0, v), 0) || 1;
  return {
    mean,
    components: vectors.slice(0, k),
    variances: values.slice(0, k),
    explained: values.slice(0, k).map((v) => Math.max(0, v) / total),
  };
}

/** Coordinates of v on the PCA components (after subtracting the PCA mean). */
export const projectPca = (p: PcaResult, v: Vec): number[] => {
  const c = subV(v, p.mean);
  return p.components.map((u) => dot(c, u));
};
