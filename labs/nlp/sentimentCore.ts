// Sentiment classification: embed each review from its text with the shared
// word-vector table (mean of content-word vectors, optional negation flip),
// reduce to 2-D with PCA fitted on the TRAINING reviews, standardise the two
// coordinates, and fit L2-regularised logistic regression by gradient descent.
// A Newton (IRLS) solve of the same objective gives the exact optimum for reference.
import { dot, pca, projectPca } from './shared';
import type { Vec, PcaResult } from './shared';
import { embedText } from './embeddingTable';
import type { EmbeddedText } from './embeddingTable';

export interface Review { text: string; label: 0 | 1; } // 0 negative, 1 positive

export const TRAIN_REVIEWS: Review[] = [
  { text: 'a wonderful delightful movie', label: 1 },
  { text: 'loved every brilliant minute', label: 1 },
  { text: 'great fun and very enjoyable', label: 1 },
  { text: 'a pleasant charming surprise', label: 1 },
  { text: 'best film of the year', label: 1 },
  { text: 'terrible boring waste of time', label: 0 },
  { text: 'awful and painfully dull', label: 0 },
  { text: 'a disappointing weak script', label: 0 },
  { text: 'hated the clumsy ending', label: 0 },
  { text: 'worst movie in ages', label: 0 },
];

/** Held-out reviews with gold labels; three contain a negation cue (not / never). */
export const TEST_REVIEWS: Review[] = [
  { text: 'absolutely loved every moment', label: 1 },
  { text: 'rather dull and forgettable', label: 0 },
  { text: 'not bad, fairly enjoyable', label: 1 },
  { text: 'not good and not funny', label: 0 },
  { text: 'great script and a wonderful ending', label: 1 },
  { text: 'a boring film, never funny', label: 0 },
];

export interface Featurised { emb: EmbeddedText; z: [number, number]; }
export interface FeatureSpace {
  pca: PcaResult;
  /** std-dev of each PCA coordinate over the training reviews (their mean is 0) */
  std: [number, number];
  train: Featurised[];
  test: Featurised[];
}

/** Embed → PCA-2 (fit on the training reviews only) → standardise. */
export function buildFeatures(negation: boolean): FeatureSpace {
  const trEmb = TRAIN_REVIEWS.map((r) => embedText(r.text, negation));
  const p = pca(trEmb.map((e) => e.vec), 2);
  const std: [number, number] = [Math.sqrt(Math.max(1e-12, p.variances[0] ?? 1)), Math.sqrt(Math.max(1e-12, p.variances[1] ?? 1))];
  const toZ = (e: EmbeddedText): Featurised => {
    const c = projectPca(p, e.vec);
    return { emb: e, z: [(c[0] ?? 0) / std[0], (c[1] ?? 0) / std[1]] };
  };
  return {
    pca: p, std,
    train: trEmb.map(toZ),
    test: TEST_REVIEWS.map((r) => toZ(embedText(r.text, negation))),
  };
}

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

export interface LogReg { w: [number, number]; b: number; }
export const ZERO_MODEL: LogReg = { w: [0, 0], b: 0 };
export const predict = (m: LogReg, z: Vec): number => sigmoid(dot(m.w, z) + m.b);

/** J = mean cross-entropy + (λ/2)|w|² (bias unpenalised) and its gradient:
 *  ∂J/∂w = mean((σ − y)·x) + λw,  ∂J/∂b = mean(σ − y). */
export function objective(m: LogReg, X: Vec[], y: number[], lambda: number): { loss: number; gw: [number, number]; gb: number } {
  const n = X.length || 1;
  let loss = 0, g0 = 0, g1 = 0, gb = 0;
  X.forEach((x, i) => {
    const s = dot(m.w, x) + m.b;
    const t = y[i] ?? 0;
    // stable log(1 + e^s) − t·s
    loss += (s > 0 ? s + Math.log1p(Math.exp(-s)) : Math.log1p(Math.exp(s))) - t * s;
    const e = sigmoid(s) - t;
    g0 += e * (x[0] ?? 0); g1 += e * (x[1] ?? 0); gb += e;
  });
  const [w0, w1] = m.w;
  return {
    loss: loss / n + 0.5 * lambda * (w0 * w0 + w1 * w1),
    gw: [g0 / n + lambda * w0, g1 / n + lambda * w1],
    gb: gb / n,
  };
}

export const gradNorm = (g: { gw: [number, number]; gb: number }): number =>
  Math.sqrt(g.gw[0] ** 2 + g.gw[1] ** 2 + g.gb ** 2);

/** One full-batch gradient-descent step. */
export function gdStep(m: LogReg, X: Vec[], y: number[], lambda: number, lr: number): LogReg {
  const g = objective(m, X, y, lambda);
  return { w: [m.w[0] - lr * g.gw[0], m.w[1] - lr * g.gw[1]], b: m.b - lr * g.gb };
}

/** Newton / IRLS on the same objective — the exact minimiser (λ > 0 makes it unique). */
export function newtonFit(X: Vec[], y: number[], lambda: number, iters = 50): LogReg {
  let w0 = 0, w1 = 0, b = 0;
  const n = X.length || 1;
  for (let it = 0; it < iters; it++) {
    // gradient g and Hessian H over θ = (w0, w1, b)
    let g0 = 0, g1 = 0, g2 = 0;
    let h00 = 0, h01 = 0, h02 = 0, h11 = 0, h12 = 0, h22 = 0;
    X.forEach((x, i) => {
      const x0 = x[0] ?? 0, x1 = x[1] ?? 0;
      const p = sigmoid(w0 * x0 + w1 * x1 + b);
      const e = p - (y[i] ?? 0), r = p * (1 - p);
      g0 += e * x0; g1 += e * x1; g2 += e;
      h00 += r * x0 * x0; h01 += r * x0 * x1; h02 += r * x0; h11 += r * x1 * x1; h12 += r * x1; h22 += r;
    });
    g0 = g0 / n + lambda * w0; g1 = g1 / n + lambda * w1; g2 /= n;
    h00 = h00 / n + lambda; h01 /= n; h02 /= n; h11 = h11 / n + lambda; h12 /= n; h22 /= n;
    // solve H·δ = g (3×3, Cramer's rule)
    const det = h00 * (h11 * h22 - h12 * h12) - h01 * (h01 * h22 - h12 * h02) + h02 * (h01 * h12 - h11 * h02);
    if (Math.abs(det) < 1e-18) break;
    const d0 = (g0 * (h11 * h22 - h12 * h12) - h01 * (g1 * h22 - h12 * g2) + h02 * (g1 * h12 - h11 * g2)) / det;
    const d1 = (h00 * (g1 * h22 - h12 * g2) - g0 * (h01 * h22 - h12 * h02) + h02 * (h01 * g2 - g1 * h02)) / det;
    const d2 = (h00 * (h11 * g2 - g1 * h12) - h01 * (h01 * g2 - g1 * h02) + g0 * (h01 * h12 - h11 * h02)) / det;
    w0 -= d0; w1 -= d1; b -= d2;
    if (Math.abs(d0) + Math.abs(d1) + Math.abs(d2) < 1e-13) break;
  }
  return { w: [w0, w1], b };
}

/** Weights re-expressed on the unstandardised PCA coordinates c = z·std: w_c = w / std (b unchanged, the coordinates are centred). */
export const weightsInPcaUnits = (m: LogReg, std: [number, number]): [number, number] => [m.w[0] / std[0], m.w[1] / std[1]];

export const accuracy = (m: LogReg, rows: Featurised[], labels: number[]): number =>
  rows.length ? rows.filter((r, i) => (predict(m, r.z) > 0.5 ? 1 : 0) === labels[i]).length / rows.length : 0;
