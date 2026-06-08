// Shared, deterministic small-matrix maths for the Sequence-Models labs
// (RNN, LSTM, seq2seq). Everything here is analytic and client-side — tiny
// hand-rolled linear algebra so the three labs reuse the SAME cell maths and
// the same small-weight initialiser (parameterised by a spectral scale). Hidden
// sizes are ~4–8 and sequence lengths ~8–20, kept small so the on-screen maths
// stays correct and legible.

export type Vec = number[];
export type Mat = number[][];

/* ---------- elementwise activations ---------- */
export const tanh = (x: number) => Math.tanh(x);
export const dtanh = (x: number) => 1 - Math.tanh(x) ** 2;       // tanh′ ∈ (0, 1]
export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export const vtanh = (v: Vec): Vec => v.map(tanh);
export const vsigmoid = (v: Vec): Vec => v.map(sigmoid);

/* ---------- tiny vector / matrix ops ---------- */
export const zeros = (n: number): Vec => new Array(n).fill(0);
export const add = (a: Vec, b: Vec): Vec => a.map((x, i) => x + b[i]);
export const hadamard = (a: Vec, b: Vec): Vec => a.map((x, i) => x * b[i]);
export const scale = (a: Vec, s: number): Vec => a.map((x) => x * s);
export const l2 = (a: Vec): number => Math.sqrt(a.reduce((s, x) => s + x * x, 0));

/** M·v for an (out×in) matrix M and length-in vector v. */
export const matVec = (M: Mat, v: Vec): Vec =>
  M.map((row) => row.reduce((s, w, j) => s + w * v[j], 0));

/* ---------- deterministic small PRNG (mulberry32) ---------- */
// A seeded generator so every render/Python export reproduces the SAME weights —
// no Math.random(), so the visualisations and exported NumPy line up exactly.
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Uniform in [-1, 1) from a seeded generator. */
const u11 = (r: () => number) => r() * 2 - 1;

/**
 * A square recurrent matrix W_hh whose spectral radius is set to `scale`.
 * We build a random matrix, then renormalise its rows so the (Gershgorin-bounded)
 * operator norm is ≈ scale. This is the SINGLE knob behind vanishing (<1) vs
 * exploding (>1) gradients in the RNN lab — and reused by the encoder in seq2seq.
 */
export function recurrentMatrix(h: number, scale: number, seed = 1): Mat {
  const r = rng(seed);
  const M: Mat = Array.from({ length: h }, () => Array.from({ length: h }, () => u11(r)));
  // Estimate the spectral norm via power iteration on MᵀM, then rescale.
  let v = Array.from({ length: h }, () => r());
  for (let it = 0; it < 24; it++) {
    const Mv = matVec(M, v);
    // t = Mᵀ(Mv) — one power-iteration step on MᵀM to find the top singular vector.
    const t = zeros(h);
    for (let i = 0; i < h; i++) for (let j = 0; j < h; j++) t[j] += M[i][j] * Mv[i];
    const n = l2(t) || 1;
    v = t.map((x) => x / n);
  }
  const Mv = matVec(M, v);
  const sigma = l2(Mv) || 1;                 // current largest singular value
  const factor = scale / sigma;
  return M.map((row) => row.map((w) => w * factor));
}

/** Random (out×in) input-projection matrix W_xh, small fixed magnitude. */
export function inputMatrix(out: number, inp: number, seed = 7, mag = 0.6): Mat {
  const r = rng(seed);
  return Array.from({ length: out }, () => Array.from({ length: inp }, () => u11(r) * mag));
}

/** A length-n bias vector (deterministic, small) plus an optional constant offset. */
export function biasVector(n: number, seed = 13, offset = 0, mag = 0.3): Vec {
  const r = rng(seed);
  return Array.from({ length: n }, () => u11(r) * mag + offset);
}

/* ---------- the two cell steps (the heart of all three labs) ---------- */

/** Vanilla RNN cell: h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b). */
export function rnnStep(Whh: Mat, Wxh: Mat, b: Vec, hPrev: Vec, x: Vec): Vec {
  const pre = add(add(matVec(Whh, hPrev), matVec(Wxh, x)), b);
  return vtanh(pre);
}

export interface LstmGates { f: Vec; i: Vec; o: Vec; g: Vec; }
export interface LstmState { c: Vec; h: Vec; gates: LstmGates; }

/**
 * LSTM cell (one timestep). Gate pre-activations are formed from the input and
 * the previous hidden state; we expose all four gates so the lab can visualise
 * forget/input/output and the candidate g.
 *   f = σ(W_f·[x,h] + b_f)   i = σ(W_i·[x,h] + b_i)
 *   o = σ(W_o·[x,h] + b_o)   g = tanh(W_g·[x,h] + b_g)
 *   c_t = f⊙c_{t-1} + i⊙g    h_t = o⊙tanh(c_t)
 * `forgetBias` is added to the forget pre-activation to push f toward 1 (the
 * "gradient highway" knob in the LSTM lab).
 */
export function lstmStep(
  W: { f: Mat; i: Mat; o: Mat; g: Mat },
  bias: { f: Vec; i: Vec; o: Vec; g: Vec },
  cPrev: Vec, hPrev: Vec, x: Vec, forgetBias = 0,
): LstmState {
  const xh = [...x, ...hPrev];
  const f = vsigmoid(add(matVec(W.f, xh), bias.f).map((z) => z + forgetBias));
  const i = vsigmoid(add(matVec(W.i, xh), bias.i));
  const o = vsigmoid(add(matVec(W.o, xh), bias.o));
  const g = vtanh(add(matVec(W.g, xh), bias.g));
  const c = add(hadamard(f, cPrev), hadamard(i, g));
  const h = hadamard(o, vtanh(c));
  return { c, h, gates: { f, i, o, g } };
}

/** Build a full set of LSTM gate weights (each (h × (inp+h))) deterministically. */
export function lstmWeights(h: number, inp: number) {
  const d = inp + h;
  return {
    W: {
      f: inputMatrix(h, d, 21, 0.4),
      i: inputMatrix(h, d, 22, 0.4),
      o: inputMatrix(h, d, 23, 0.4),
      g: inputMatrix(h, d, 24, 0.6),
    },
    bias: {
      f: biasVector(h, 31, 0, 0.2),
      i: biasVector(h, 32, 0, 0.2),
      o: biasVector(h, 33, 0, 0.2),
      g: biasVector(h, 34, 0, 0.2),
    },
  };
}

/* ---------- analytic gradient-through-time ---------- */

/**
 * Vanilla-RNN gradient norm vs lag. The backprop-through-time Jacobian is
 *   ∂h_t/∂h_{t-k} = Π_{j} diag(tanh′(pre_j))·W_hh.
 * For a clean teaching curve we use the average tanh′ factor ḡ together with the
 * spectral scale ρ of W_hh, giving ‖∂h_t/∂h_{t-k}‖ ≈ (ḡ·ρ)^k. Returns the norm
 * at each lag 0..maxLag (normalised so lag 0 = 1).
 */
export function rnnGradNorm(scale: number, maxLag: number, tanhFactor = 0.72): number[] {
  const per = tanhFactor * scale;             // effective per-step multiplier
  const out: number[] = [];
  for (let k = 0; k <= maxLag; k++) out.push(Math.pow(per, k));
  return out;
}

/**
 * LSTM cell-state gradient norm vs lag. The cell carry c_t = f⊙c_{t-1} + i⊙g has
 * Jacobian ∂c_t/∂c_{t-1} ≈ diag(f), so over k steps the factor is ≈ f̄^k. When the
 * forget gate f̄ → 1 the product stays ≈ 1 (the constant error carousel), instead
 * of decaying like the RNN. `fbar` is the mean forget-gate activation.
 */
export function lstmGradNorm(fbar: number, maxLag: number): number[] {
  const out: number[] = [];
  for (let k = 0; k <= maxLag; k++) out.push(Math.pow(fbar, k));
  return out;
}

/** Mean forget-gate activation implied by a forget-gate bias (σ of a typical pre-act + bias). */
export const meanForget = (forgetBias: number, basePre = -0.2) => sigmoid(basePre + forgetBias);

/* ---------- seq2seq context-bottleneck model (analytic) ---------- */

/**
 * Per-position reconstruction fidelity for a fixed d-dim context vector that must
 * encode an input of length L over a vocabulary of size V. This is an ANALYTIC
 * illustration, not a trained model: a single d-dim real vector carries a finite
 * capacity (≈ d·bitsPerDim bits); the demand is L·log2(V) bits. When demand
 * exceeds capacity the encoder must drop information, and because the context is
 * the encoder's LAST hidden state, the EARLY tokens (seen first, overwritten most)
 * fade fastest. We model fidelity for position p (0 = first token) as a logistic
 * of the per-token bit budget, with a recency advantage for later positions.
 */
export function reconstructionAccuracy(
  pos: number, L: number, dim: number, vocab: number,
  bitsPerDim = 2.2,
): number {
  const capacity = dim * bitsPerDim;           // bits the context vector can hold
  const demand = L * Math.log2(vocab);          // bits the input carries
  const budgetRatio = capacity / Math.max(1e-6, demand);   // >1 ⇒ roomy, <1 ⇒ squeezed
  // Recency: the last hidden state remembers recent tokens better. position p∈[0,L-1].
  const recency = L > 1 ? pos / (L - 1) : 1;    // 0 (first/oldest) → 1 (last/newest)
  // Effective per-position budget combines the global budget with a recency bonus.
  const eff = budgetRatio * (0.55 + 0.9 * recency);
  const chance = 1 / vocab;
  // Logistic squashing of the log-budget, floored at chance level.
  const acc = 1 / (1 + Math.exp(-3.0 * (Math.log(eff))));
  return chance + (1 - chance) * acc;
}

/** Overall mean reconstruction accuracy across all L positions. */
export function meanReconstruction(L: number, dim: number, vocab: number): number {
  let s = 0;
  for (let p = 0; p < L; p++) s += reconstructionAccuracy(p, L, dim, vocab);
  return s / Math.max(1, L);
}
