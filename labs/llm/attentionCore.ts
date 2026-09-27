// Pure core of the Self-Attention lab — mirrored exactly by attentionPython
// (labs/llm/python.ts). Everything below is fixed and disclosed.
//
// INPUT  x_i = e(token_i) + PE(i)          (d_model = 8)
//   dims 0–3 : hand-set token embedding, readable features
//              [determiner, noun, animate, verb]  ("The" and "the" share one row)
//   dims 4–7 : sinusoidal positional encoding  PE(pos) = [sin ω₀pos, cos ω₀pos,
//              sin ω₁pos, cos ω₁pos],  ω_k = BASE^(−2k/4)  (zero when PE is off)
// PROJECTIONS  Q = X·W_Q,  K = X·W_K,  V = X·W_V   (8×8 each, row-vector convention)
//   Q/K columns 0–3 are "role" slots: a noun's query looks for a determiner, a
//   verb's query looks for an animate noun (its agent), a determiner's query looks
//   for a noun; keys advertise which slot a token fills.
//   Q/K columns 4–7 act on the positional dims: W_K copies PE(j), W_Q ROTATES each
//   (sin, cos) pair by −ω_k, so q_i·k_j = Σ c_k cos(ω_k (i−1−j)) peaks at j = i−1
//   (a previous-token relation — possible because PE(i−1) is a linear map of PE(i)).
//   W_V copies the 4 content features into BOTH halves of V, so every head passes
//   on token content.
// HEADS  h ∈ {1, 2, 4, 8}: head m uses columns [m·d_h, (m+1)·d_h) of Q, K and V
//   (d_h = 8/h). A_m = softmax(Q_m K_mᵀ / (√d_h · s) [+ causal −1e9 mask]),
//   O_m = A_m V_m, and the concatenation [O_1 … O_h] is the N×8 output (a real
//   layer would then apply an output projection W_O, omitted here).

export const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
export const D_MODEL = 8;
export const HEAD_OPTIONS = [1, 2, 4, 8];
export const PE_BASE = 10;
export const FEATURES = ['det', 'noun', 'animate', 'verb'];

/** Token embeddings: content in dims 0–3, zeros in the positional dims 4–7. */
export const EMB: number[][] = [
  [1, 0, 0, 0, 0, 0, 0, 0],   // The
  [0, 1, 1, 0, 0, 0, 0, 0],   // cat
  [0, 0, 0, 1, 0, 0, 0, 0],   // sat
  [0, 0, 0, 0, 0, 0, 0, 0],   // on   (a function word: no content features here)
  [1, 0, 0, 0, 0, 0, 0, 0],   // the  (same token as "The")
  [0, 1, 0, 0, 0, 0, 0, 0],   // mat
];

export const OMEGA = [0, 1].map((k) => Math.pow(PE_BASE, (-2 * k) / 4));

export function positionalEncoding(pos: number): number[] {
  const out: number[] = [];
  for (const w of OMEGA) out.push(Math.sin(w * pos), Math.cos(w * pos));
  return out;
}

// Strengths of the hand-set relations.
const A_ROLE = 4;     // noun → determiner, verb → animate noun
const B_ROLE = 2;     // determiner → noun
const C_POS = [6, 6]; // previous-token rotation weight per frequency pair

/** W_Q (8×8, rows = input dims, cols = query dims). */
export const W_Q: number[][] = (() => {
  const W = Array.from({ length: 8 }, () => new Array(8).fill(0));
  W[0]![2] = B_ROLE;   // determiner looks for a noun         (query slot 2)
  W[1]![0] = A_ROLE;   // noun looks for a determiner         (query slot 0)
  W[3]![1] = A_ROLE;   // verb looks for an animate noun      (query slot 1)
  OMEGA.forEach((w, k) => {           // rotate (sin, cos) by −ω: PE(i) → PE(i−1)
    const r = 4 + 2 * k, c = C_POS[k]!;
    W[r]![r] = c * Math.cos(w); W[r]![r + 1] = c * Math.sin(w);
    W[r + 1]![r] = -c * Math.sin(w); W[r + 1]![r + 1] = c * Math.cos(w);
  });
  return W;
})();

/** W_K (8×8): keys advertise the slot a token fills; positional dims copied. */
export const W_K: number[][] = (() => {
  const W = Array.from({ length: 8 }, () => new Array(8).fill(0));
  W[0]![0] = 1;        // determiner fills slot 0
  W[2]![1] = 1;        // animate fills slot 1 (agent)
  W[1]![2] = 1;        // noun fills slot 2
  W[3]![3] = 1;        // verb fills slot 3
  for (let r = 4; r < 8; r++) W[r]![r] = 1;
  return W;
})();

/** W_V (8×8): content features copied into both halves; positional dims dropped. */
export const W_V: number[][] = (() => {
  const W = Array.from({ length: 8 }, () => new Array(8).fill(0));
  for (let f = 0; f < 4; f++) { W[f]![f] = 1; W[f]![f + 4] = 1; }
  return W;
})();

export interface AttnConfig { heads: number; scale: number; causal: boolean; pe: boolean; }

const matmul = (X: number[][], W: number[][]) =>
  X.map((row) => W[0]!.map((_, c) => row.reduce((s, v, r) => s + v * W[r]![c]!, 0)));

export function inputs(pe: boolean): number[][] {
  return EMB.map((e, i) => {
    const p = positionalEncoding(i);
    return e.map((v, d) => v + (pe && d >= 4 ? p[d - 4]! : 0));
  });
}

export interface HeadResult {
  cols: [number, number];      // column range [lo, hi) of Q/K/V this head uses
  scores: number[][];          // Q_m K_mᵀ / (√d_h · s), masked entries −1e9
  A: number[][];               // attention weights (rows sum to 1)
  O: number[][];               // A · V_m
}

export interface AttnResult { X: number[][]; Q: number[][]; K: number[][]; V: number[][]; dh: number; heads: HeadResult[]; out: number[][]; }

const softmax = (z: number[]) => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

export function attention(cfg: AttnConfig): AttnResult {
  const X = inputs(cfg.pe);
  const Q = matmul(X, W_Q), K = matmul(X, W_K), V = matmul(X, W_V);
  const N = X.length;
  const dh = D_MODEL / cfg.heads;
  const denom = Math.sqrt(dh) * cfg.scale;
  const heads: HeadResult[] = [];
  for (let m = 0; m < cfg.heads; m++) {
    const lo = m * dh, hi = lo + dh;
    const scores = Q.map((q, i) => K.map((k, j) => {
      if (cfg.causal && j > i) return -1e9;
      let s = 0;
      for (let d = lo; d < hi; d++) s += q[d]! * k[d]!;
      return s / denom;
    }));
    const A = scores.map(softmax);
    const O = A.map((row) => Array.from({ length: dh }, (_, c) => row.reduce((s, w, j) => s + w * V[j]![lo + c]!, 0)));
    heads.push({ cols: [lo, hi], scores, A, O });
  }
  const out = Array.from({ length: N }, (_, i) => heads.flatMap((h) => h.O[i]!));
  return { X, Q, K, V, dh, heads, out };
}

/** Computed summary of a head's pattern (used for honest captions). */
export function headStats(A: number[][]) {
  const N = A.length;
  let prev = 0, self = 0;
  for (let i = 0; i < N; i++) { self += A[i]![i]!; if (i > 0) prev += A[i]![i - 1]!; }
  const argmax = A.map((row) => row.indexOf(Math.max(...row)));
  const maxW = A.map((row) => Math.max(...row));
  // a row is "flat" when every key it may see gets the same weight
  const flat = A.map((row) => {
    const vis = row.filter((v) => v > 1e-12);
    return Math.max(...vis) - Math.min(...vis) < 1e-9;
  });
  return { meanPrev: prev / (N - 1), meanSelf: self / N, argmax, maxW, flat };
}

/** One-line, fully computed description of what a head's weights show. */
export function describeHead(A: number[][]): string {
  const st = headStats(A);
  const N = A.length;
  if (st.flat.every(Boolean)) return 'flat — every visible key gets equal weight (all scores in this head are equal)';
  const links = A.map((row, i) => {
    const w = st.maxW[i]!;
    const js = row.map((v, j) => (Math.abs(v - w) < 1e-6 ? j : -1)).filter((j) => j >= 0 && j !== i);
    return { i, js, w };
  }).filter((r) => !st.flat[r.i] && r.js.length > 0 && r.w >= 0.3)
    .sort((a, b) => (Math.round(b.w * 1e6) - Math.round(a.w * 1e6)) || (a.i - b.i)).slice(0, 3)
    .map((r) => `${TOKENS[r.i]}→${r.js.map((j) => TOKENS[j]).join('/')} ${r.w.toFixed(2)}`);
  const top = links.length ? `strongest links: ${links.join(', ')}` : 'no row puts ≥ 30% of its weight on another token';
  if (st.meanPrev >= 0.5) return `previous-token pattern — rows 2–${N} put ${Math.round(st.meanPrev * 100)}% of their weight on the token just before them on average; ${top}`;
  return links.length ? `content pattern — ${top}` : `diffuse — ${top}`;
}
