// Shared, deterministic small-matrix maths for the Sequence-Models labs
// (RNN, LSTM, seq2seq). Everything is computed client-side with tiny hand-rolled
// linear algebra: the three labs reuse the SAME cell maths, the same seeded
// initialisers, and EXACT Jacobian / spectral-norm computations — no canned
// curves. The Python exports (./python.ts) port the same PRNG and algorithms,
// so they reproduce these numbers.

export type Vec = number[];
export type Mat = number[][];

/* ---------- elementwise activations ---------- */
export const tanh = (x: number) => Math.tanh(x);
export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export const vtanh = (v: Vec): Vec => v.map(tanh);
export const vsigmoid = (v: Vec): Vec => v.map(sigmoid);

/* ---------- tiny vector / matrix ops ---------- */
export const zeros = (n: number): Vec => new Array<number>(n).fill(0);
export const add = (a: Vec, b: Vec): Vec => a.map((x, i) => x + (b[i] ?? 0));
export const sub = (a: Vec, b: Vec): Vec => a.map((x, i) => x - (b[i] ?? 0));
export const hadamard = (a: Vec, b: Vec): Vec => a.map((x, i) => x * (b[i] ?? 0));
export const dot = (a: Vec, b: Vec): number => a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0);
export const l2 = (a: Vec): number => Math.sqrt(a.reduce((s, x) => s + x * x, 0));
export const mean = (a: Vec): number => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);

/** Compact number for chips and text: fixed decimals in [0.01, 100), scientific otherwise. */
export const fmtVal = (v: number, d = 3): string =>
  !Number.isFinite(v) ? '—'
    : v === 0 ? '0'
      : Math.abs(v) >= 0.01 && Math.abs(v) < 100 ? v.toFixed(d) : v.toExponential(2);

/** M·v for an (out×in) matrix M and length-in vector v. */
export const matVec = (M: Mat, v: Vec): Vec => M.map((row) => dot(row, v));

export const transpose = (A: Mat): Mat =>
  (A[0] ?? []).map((_, j) => A.map((row) => row[j] ?? 0));

/** A·B for (n×k)·(k×m). */
export function matMul(A: Mat, B: Mat): Mat {
  const m = B[0]?.length ?? 0;
  return A.map((row) => {
    const out = zeros(m);
    row.forEach((a, k) => {
      const Bk = B[k];
      if (!Bk || a === 0) return;
      for (let j = 0; j < m; j++) out[j] = (out[j] ?? 0) + a * (Bk[j] ?? 0);
    });
    return out;
  });
}

export const identity = (n: number): Mat =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

/* ---------- exact symmetric eigenvalues + spectral norm ---------- */

/**
 * Eigenvalues of a symmetric matrix by cyclic Jacobi rotations (converges to
 * machine precision for the ≤ 12×12 matrices used here).
 */
export function symmetricEigenvalues(S: Mat): number[] {
  const n = S.length;
  const A = S.map((r) => r.slice());
  let total = 0;
  for (const r of A) for (const v of r) total += v * v;
  if (total === 0) return zeros(n);
  for (let sweep = 0; sweep < 64; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p]![q]! * A[p]![q]!;
    if (off <= 1e-30 * total) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p]![q]!;
        if (apq === 0) continue;
        const theta = (A[q]![q]! - A[p]![p]!) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {          // A ← A·J (columns p, q)
          const Ak = A[k]!;
          const akp = Ak[p]!, akq = Ak[q]!;
          Ak[p] = c * akp - s * akq;
          Ak[q] = s * akp + c * akq;
        }
        const Ap = A[p]!, Aq = A[q]!;
        for (let k = 0; k < n; k++) {          // A ← Jᵀ·A (rows p, q)
          const apk = Ap[k]!, aqk = Aq[k]!;
          Ap[k] = c * apk - s * aqk;
          Aq[k] = s * apk + c * aqk;
        }
      }
    }
  }
  return A.map((r, i) => r[i] ?? 0);
}

/**
 * Spectral norm ‖M‖₂ = largest singular value = √λ_max(MᵀM), computed exactly
 * (the matrix is rescaled first so tiny or huge Jacobian products stay accurate).
 */
export function spectralNorm(M: Mat): number {
  let s = 0;
  for (const r of M) for (const v of r) s = Math.max(s, Math.abs(v));
  if (s === 0) return 0;
  if (!Number.isFinite(s)) return Infinity;
  const N = M.map((r) => r.map((v) => v / s));
  const rows = N.length, cols = N[0]?.length ?? 0;
  const G = rows <= cols ? matMul(N, transpose(N)) : matMul(transpose(N), N);
  return s * Math.sqrt(Math.max(0, ...symmetricEigenvalues(G)));
}

/* ---------- deterministic small PRNG (mulberry32) ---------- */
// A seeded generator so every render reproduces the SAME weights and samples —
// no Math.random(). The Python exports port this exact generator (same seed →
// same numbers, same order), so the downloaded NumPy reproduces the lab.
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

/** (rows×cols) matrix of seeded uniform [-1, 1) entries, filled row by row. */
function uniformMatrix(rows: number, cols: number, seed: number): Mat {
  const r = rng(seed);
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => u11(r)));
}

/**
 * Seeded random recurrent matrix rescaled so its LARGEST SINGULAR VALUE
 * σ_max = ‖W‖₂ equals `sigmaMax` exactly. The matrix is non-normal, so its
 * spectral radius is smaller than σ_max. Used by the seq2seq encoder.
 */
export function recurrentMatrix(h: number, sigmaMax: number, seed = 1): Mat {
  const M = uniformMatrix(h, h, seed);
  const factor = sigmaMax / spectralNorm(M);
  return M.map((row) => row.map((w) => w * factor));
}

/**
 * Scaled orthogonal recurrent matrix W_hh = ρ·Q, where Q comes from modified
 * Gram–Schmidt on the rows of a seeded uniform matrix. Every singular value of
 * W_hh equals ρ and every eigenvalue has modulus ρ, so ρ is BOTH the spectral
 * radius and ‖W_hh‖₂ — the single knob of the RNN lab (and its LSTM baseline).
 */
export function orthogonalRecurrent(h: number, rho: number, seed = 1): Mat {
  const Q: Vec[] = [];
  for (const row of uniformMatrix(h, h, seed)) {
    let v = row.slice();
    for (const q of Q) {
      const d = dot(v, q);
      v = v.map((x, k) => x - d * (q[k] ?? 0));
    }
    const n = l2(v);
    Q.push(v.map((x) => x / n));
  }
  return Q.map((r) => r.map((w) => w * rho));
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

/** Run the RNN from h_0 = 0 over a sequence; returns h after each input. */
export function runRnn(Whh: Mat, Wxh: Mat, b: Vec, xs: Vec[]): Vec[] {
  let h = zeros(Whh.length);
  return xs.map((x) => (h = rnnStep(Whh, Wxh, b, h, x)));
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

export type LstmWeights = ReturnType<typeof lstmWeights>;

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

/** Run the LSTM from c_0 = h_0 = 0 over a sequence; returns the state after each input. */
export function runLstm(wts: LstmWeights, xs: Vec[], forgetBias: number): LstmState[] {
  const H = wts.bias.f.length;
  let c = zeros(H), h = zeros(H);
  return xs.map((x) => {
    const s = lstmStep(wts.W, wts.bias, c, h, x, forgetBias);
    c = s.c; h = s.h;
    return s;
  });
}

/* ---------- exact gradient-through-time ---------- */

/**
 * Exact BPTT Jacobian norms of a vanilla tanh RNN. With hs[i] the hidden state
 * after input i (i = 0..t−1), returns ‖∂h_{t−1}/∂h_{t−1−k}‖₂ for k = 0..t−1:
 *   ∂h_{t−1}/∂h_{t−1−k} = Π_{m=0}^{k−1} diag(1 − h_{t−1−m}²)·W_hh
 * (tanh′(pre) = 1 − tanh(pre)² = 1 − h², so the stored states give it exactly).
 */
export function rnnJacobianNorms(hs: Vec[], Whh: Mat): number[] {
  const t = hs.length;
  if (t === 0) return [];
  let J = identity(Whh.length);
  const out = [1];
  for (let k = 1; k < t; k++) {
    const h = hs[t - k] ?? [];
    const DW = Whh.map((row, i) => {
      const hi = h[i] ?? 0;
      const g = 1 - hi * hi;
      return row.map((w) => g * w);
    });
    J = matMul(J, DW);
    out.push(spectralNorm(J));
  }
  return out;
}

/**
 * Rigorous upper bound on the same norms by sub-multiplicativity:
 *   ‖J_k‖ ≤ Π_{m=0}^{k−1} max_i(1 − h_{t−1−m,i}²) · ‖W_hh‖₂^k.
 */
export function rnnJacobianBound(hs: Vec[], wNorm: number): number[] {
  const t = hs.length;
  if (t === 0) return [];
  const out = [1];
  let b = 1;
  for (let k = 1; k < t; k++) {
    const h = hs[t - k] ?? [];
    b *= Math.max(...h.map((v) => 1 - v * v)) * wNorm;
    out.push(b);
  }
  return out;
}

/** Mean of tanh′ = 1 − h² over every unit of every state. */
export const meanTanhSlope = (hs: Vec[]): number =>
  mean(hs.map((h) => mean(h.map((v) => 1 - v * v))));

export type GradRegime = 'vanishing' | 'near-critical' | 'exploding';

/** Per-step factor within 1 ± REGIME_BAND counts as near-critical. */
export const REGIME_BAND = 0.1;

export interface GradVerdict { regime: GradRegime; perStep: number; far: number; lag: number; }

/**
 * One shared regime classification from a REAL Jacobian-norm curve. The
 * per-step factor is the geometric mean (‖J_K‖)^(1/K) over the furthest lag K;
 * below 1 − band the gradient vanishes, above 1 + band it explodes.
 */
export function classifyGradient(norms: number[]): GradVerdict | null {
  const lag = norms.length - 1;
  if (lag < 1) return null;
  const far = norms[lag] ?? 0;
  const perStep = far > 0 ? Math.pow(far, 1 / lag) : 0;
  const regime: GradRegime = perStep < 1 - REGIME_BAND
    ? 'vanishing'
    : perStep > 1 + REGIME_BAND ? 'exploding' : 'near-critical';
  return { regime, perStep, far, lag };
}

/**
 * LSTM direct cell path (the constant error carousel): the Jacobian of
 * c_{t−1} w.r.t. c_{t−1−k} ALONG THE CARRY ONLY is Π_{m=0}^{k−1} diag(f_{t−1−m}).
 * It is diagonal, so its spectral norm is the max over units; `mean` averages
 * the per-unit products. Paths through h → gates are not included.
 */
export function lstmCellPath(states: LstmState[]): { max: number[]; mean: number[] } {
  const t = states.length;
  if (t === 0) return { max: [], mean: [] };
  let prod = (states[0]?.gates.f ?? []).map(() => 1);
  const mx = [1], mn = [1];
  for (let k = 1; k < t; k++) {
    const f = states[t - k]?.gates.f ?? [];
    prod = prod.map((p, u) => p * (f[u] ?? 0));
    mx.push(Math.max(...prod));
    mn.push(mean(prod));
  }
  return { max: mx, mean: mn };
}

export type CarouselVerdict = 'open' | 'leaky' | 'closed';
/** Direct cell-path gain at the furthest lag: ≥ 0.5 open, ≥ 0.01 leaky, else closed. */
export const classifyCarousel = (pathFar: number): CarouselVerdict =>
  pathFar >= 0.5 ? 'open' : pathFar >= 0.01 ? 'leaky' : 'closed';

export type CarryVerdict = 'carried' | 'faded' | 'lost';
/** Injected-value influence ‖Δh_T‖ / ‖Δh_1‖: ≥ 0.5 carried, ≥ 0.01 faded, else lost. */
export const classifyCarry = (ratio: number): CarryVerdict =>
  ratio >= 0.5 ? 'carried' : ratio >= 0.01 ? 'faded' : 'lost';

/** ‖a_t − b_t‖ at every step, for two runs of the same length. */
export const traceDiff = (a: Vec[], b: Vec[]): number[] =>
  a.map((v, i) => l2(sub(v, b[i] ?? [])));

/* ---------- seq2seq: measured context bottleneck ---------- */

/** Solve A·X = B (A n×n, B n×m) by Gaussian elimination with partial pivoting. */
export function solveLinear(A: Mat, B: Mat): Mat {
  const n = A.length;
  const m = B[0]?.length ?? 0;
  const M = A.map((r, i) => [...r, ...(B[i] ?? zeros(m))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[p]![c]!)) p = r;
    const tmp = M[c]!; M[c] = M[p]!; M[p] = tmp;
    const Mc = M[c]!;
    const piv = Mc[c]!;
    for (let r = c + 1; r < n; r++) {
      const Mr = M[r]!;
      const f = Mr[c]! / piv;
      if (f === 0) continue;
      for (let k = c; k < n + m; k++) Mr[k] = Mr[k]! - f * Mc[k]!;
    }
  }
  const X: Mat = Array.from({ length: n }, () => zeros(m));
  for (let r = n - 1; r >= 0; r--) {
    const Mr = M[r]!;
    for (let j = 0; j < m; j++) {
      let s = Mr[n + j]!;
      for (let k = r + 1; k < n; k++) s -= Mr[k]! * X[k]![j]!;
      X[r]![j] = s / Mr[r]!;
    }
  }
  return X;
}

/**
 * Ridge least-squares linear readout from features (each row already has a
 * trailing 1 for the bias) to one-hot tokens: X = (FᵀF + λI)⁻¹ FᵀY, shape D×V.
 */
export function fitReadout(F: Vec[], tokens: number[], V: number, ridge: number): Mat {
  const D = F[0]?.length ?? 0;
  const A: Mat = Array.from({ length: D }, () => zeros(D));
  const B: Mat = Array.from({ length: D }, () => zeros(V));
  F.forEach((f, n) => {
    const tok = tokens[n] ?? 0;
    for (let i = 0; i < D; i++) {
      const fi = f[i] ?? 0;
      const Ai = A[i]!;
      for (let j = 0; j < D; j++) Ai[j] = Ai[j]! + fi * (f[j] ?? 0);
      B[i]![tok] = B[i]![tok]! + fi;
    }
  });
  for (let i = 0; i < D; i++) A[i]![i] = A[i]![i]! + ridge;
  return solveLinear(A, B);
}

/** Readout scores fᵀX and the arg-max token (first max wins ties, like numpy). */
export function readoutScores(f: Vec, X: Mat): Vec {
  const V = X[0]?.length ?? 0;
  const s = zeros(V);
  f.forEach((fi, i) => { const Xi = X[i]; if (Xi) for (let k = 0; k < V; k++) s[k] = s[k]! + fi * Xi[k]!; });
  return s;
}
export const argmax = (v: Vec): number => v.reduce((bi, x, i) => (x > (v[bi] ?? -Infinity) ? i : bi), 0);

export const oneHot = (tok: number, V: number): Vec =>
  Array.from({ length: V }, (_, k) => (k === tok ? 1 : 0));

export interface Encoder { Whh: Mat; Wxh: Mat; b: Vec; }

/** Encoder states h_1..h_L for a token sequence (one-hot inputs, h_0 = 0). */
export const encodeTokens = (enc: Encoder, tokens: number[], V: number): Vec[] =>
  runRnn(enc.Whh, enc.Wxh, enc.b, tokens.map((t) => oneHot(t, V)));

/**
 * ‖∂h_L/∂x_p‖₂ for every position p of one encoded sequence:
 *   ∂h_L/∂x_p = [Π_{j=p+1}^{L} diag(1 − h_j²)·W_hh] · diag(1 − h_p²)·W_xh.
 */
export function inputSensitivities(enc: Encoder, states: Vec[]): number[] {
  const L = states.length;
  const d = enc.Whh.length;
  const out = zeros(L);
  let G = identity(d);                              // Π_{j>p} diag(1 − h_j²)·W_hh
  for (let p = L - 1; p >= 0; p--) {
    const h = states[p] ?? [];
    const slope = (i: number) => 1 - (h[i] ?? 0) ** 2;
    const DU = enc.Wxh.map((row, i) => row.map((w) => slope(i) * w));
    out[p] = spectralNorm(matMul(G, DU));
    G = matMul(G, enc.Whh.map((row, i) => row.map((w) => slope(i) * w)));
  }
  return out;
}

export interface BottleneckOpts { nTrain: number; nTest: number; seed: number; ridge: number; }

export interface BottleneckResult {
  /** Held-out accuracy of the linear probe reading ONLY the context h_L, per position. */
  ctxAcc: number[];
  /** Held-out accuracy of the probe reading h_p (ideal-alignment attention), per position. */
  attAcc: number[];
  /** Mean over held-out sequences of ‖∂h_L/∂x_p‖₂, per position. */
  sens: number[];
  /** Fitted readouts (D×V, D = d + 1) per position: context path and attention path. */
  ctxW: Mat[];
  attW: Mat[];
  /** The demo sequence (the first held-out sequence) and its encoder states. */
  demo: number[];
  demoStates: Vec[];
}

/**
 * A real capacity measurement of the context vector: sample random token
 * sequences (seeded), encode them with the deterministic encoder, fit a ridge
 * least-squares linear readout PER POSITION from the context h_L (and, for the
 * attention path, from that position's own state h_p) to the one-hot token, and
 * score it on held-out sequences.
 */
export function measureBottleneck(L: number, V: number, enc: Encoder, opts: BottleneckOpts): BottleneckResult {
  const r = rng(opts.seed);
  const draw = (n: number) => Array.from({ length: n }, () => Array.from({ length: L }, () => Math.floor(r() * V)));
  const train = draw(opts.nTrain);
  const test = draw(opts.nTest);
  const trainH = train.map((s) => encodeTokens(enc, s, V));
  const testH = test.map((s) => encodeTokens(enc, s, V));
  const feat = (h: Vec | undefined) => [...(h ?? []), 1];

  const ctxAcc: number[] = [], attAcc: number[] = [], ctxW: Mat[] = [], attW: Mat[] = [];
  for (let p = 0; p < L; p++) {
    const targets = train.map((s) => s[p] ?? 0);
    const Wc = fitReadout(trainH.map((hs) => feat(hs[L - 1])), targets, V, opts.ridge);
    const Wa = fitReadout(trainH.map((hs) => feat(hs[p])), targets, V, opts.ridge);
    let okC = 0, okA = 0;
    test.forEach((s, n) => {
      const hs = testH[n] ?? [];
      if (argmax(readoutScores(feat(hs[L - 1]), Wc)) === s[p]) okC++;
      if (argmax(readoutScores(feat(hs[p]), Wa)) === s[p]) okA++;
    });
    ctxAcc.push(okC / Math.max(1, test.length));
    attAcc.push(okA / Math.max(1, test.length));
    ctxW.push(Wc); attW.push(Wa);
  }

  const sens = zeros(L);
  testH.forEach((hs) => inputSensitivities(enc, hs).forEach((v, p) => { sens[p] = sens[p]! + v; }));
  const sensMean = sens.map((v) => v / Math.max(1, testH.length));

  return { ctxAcc, attAcc, sens: sensMean, ctxW, attW, demo: test[0] ?? [], demoStates: testH[0] ?? [] };
}
