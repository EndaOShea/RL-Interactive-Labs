// Real (analytic, in-browser) backprop MLP trainer for the Architecture Builder's
// MLP mode — increment 2. The user's composed Dense layers become the hidden
// stack; the trainer appends a 1-unit sigmoid output head and trains on 2-D toy
// data with binary cross-entropy + SGD, so overfit/underfit become EMPIRICAL
// (train vs validation loss), not just rule-flagged. Pure module, no React.
import { Layer, Activation } from './archBuilder';

export type ToyKind = 'xor' | 'circles' | 'spirals';
export interface DataPoint { x: number; y: number; label: number; train: boolean; }

export interface HiddenSpec { units: number; act: Activation; }
export interface Arch { hidden: HiddenSpec[]; dropout: number; }

export interface Net { dims: number[]; W: number[][][]; b: number[][]; act: Activation[]; }
export interface Metrics { epoch: number; trainLoss: number; valLoss: number; trainAcc: number; valAcc: number; }

/* ── activations ── */
const actFn = (a: Activation, z: number): number => {
  switch (a) {
    case 'relu': return z > 0 ? z : 0;
    case 'leaky': return z > 0 ? z : 0.01 * z;
    case 'tanh': return Math.tanh(z);
    case 'sigmoid': return 1 / (1 + Math.exp(-z));
    default: return z; // 'none'
  }
};
// derivative expressed via the activation output value `v` (and z where needed)
const actDeriv = (a: Activation, v: number, z: number): number => {
  switch (a) {
    case 'relu': return z > 0 ? 1 : 0;
    case 'leaky': return z > 0 ? 1 : 0.01;
    case 'tanh': return 1 - v * v;
    case 'sigmoid': return v * (1 - v);
    default: return 1;
  }
};

/* ── rng (browser runtime — Math.random is fine here) ── */
const randn = (): number => {
  let u = 0; let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const clamp01 = (x: number) => Math.max(0.02, Math.min(0.98, x));

/* ── toy datasets in the unit square [0,1]² ── */
export function makeData(kind: ToyKind, perClass = 90, noise = 0.09): DataPoint[] {
  const pts: DataPoint[] = [];
  const push = (x: number, y: number, label: number) =>
    pts.push({ x: clamp01(x), y: clamp01(y), label, train: Math.random() < 0.7 });

  if (kind === 'xor') {
    const C = [{ x: 0.3, y: 0.3, l: 0 }, { x: 0.7, y: 0.3, l: 1 }, { x: 0.3, y: 0.7, l: 1 }, { x: 0.7, y: 0.7, l: 0 }];
    for (const c of C) for (let i = 0; i < Math.round(perClass / 2); i++) push(c.x + randn() * noise * 1.3, c.y + randn() * noise * 1.3, c.l);
  } else if (kind === 'circles') {
    for (let i = 0; i < perClass; i++) {
      const t = Math.random() * 2 * Math.PI;
      const r0 = 0.12 * Math.sqrt(Math.random());                 // inner disc → class 0
      push(0.5 + r0 * Math.cos(t), 0.5 + r0 * Math.sin(t) + randn() * noise * 0.4, 0);
      const r1 = 0.33 + randn() * noise * 0.5;                     // outer ring → class 1
      push(0.5 + r1 * Math.cos(t), 0.5 + r1 * Math.sin(t), 1);
    }
  } else { // spirals
    for (let i = 0; i < perClass; i++) {
      const t = (i / perClass) * 3.2;                             // radius grows with t
      const r = 0.03 + t * 0.13;
      for (const l of [0, 1]) {
        const ang = t * 2.4 + l * Math.PI + randn() * noise * 1.2;
        push(0.5 + r * Math.cos(ang), 0.5 + r * Math.sin(ang), l);
      }
    }
  }
  return pts;
}

/* ── build the trainable architecture from the user's layers ──
 * earlier Dense layers = hidden (their units + activation); the LAST Dense is
 * the output slot, replaced by a 1-unit sigmoid head for binary classification.
 * Dropout layers contribute a single effective rate (max) applied to hidden
 * activations during training. BatchNorm is ignored by the trainer. */
export function archFromLayers(layers: Layer[]): Arch {
  const denses = layers.filter((l) => l.kind === 'dense');
  const hidden: HiddenSpec[] = denses.slice(0, Math.max(0, denses.length - 1)).map((d) => ({
    units: Math.max(1, Math.min(64, d.units ?? 16)),
    act: (d.activation ?? 'relu'),
  }));
  const rates = layers.filter((l) => l.kind === 'dropout').map((l) => l.rate ?? 0);
  const dropout = rates.length ? Math.max(...rates) : 0;
  return { hidden, dropout };
}

export function initNet(arch: Arch, inputDim = 2): Net {
  const dims = [inputDim, ...arch.hidden.map((h) => h.units), 1];
  const act: Activation[] = [...arch.hidden.map((h) => h.act), 'sigmoid'];
  const W: number[][][] = [];
  const b: number[][] = [];
  for (let l = 0; l < dims.length - 1; l++) {
    const fanIn = dims[l];
    const relu = act[l] === 'relu' || act[l] === 'leaky';
    const std = Math.sqrt((relu ? 2 : 1) / fanIn);
    W.push(Array.from({ length: dims[l + 1] }, () => Array.from({ length: fanIn }, () => randn() * std)));
    b.push(Array.from({ length: dims[l + 1] }, () => 0));
  }
  return { dims, W, b, act };
}

interface FwdCache { a: number[][]; z: number[][]; mask: number[][]; }
function forward(net: Net, x0: number, x1: number, dropout: number, training: boolean): FwdCache {
  const L = net.W.length;
  const a: number[][] = [[x0, x1]];
  const z: number[][] = [[x0, x1]];
  const mask: number[][] = [[1, 1]];
  for (let l = 0; l < L; l++) {
    const zl: number[] = []; const al: number[] = []; const ml: number[] = [];
    const isHidden = l < L - 1;
    const keep = 1 - dropout;
    for (let j = 0; j < net.dims[l + 1]; j++) {
      let s = net.b[l][j];
      for (let i = 0; i < net.dims[l]; i++) s += net.W[l][j][i] * a[l][i];
      let av = actFn(net.act[l], s);
      let m = 1;
      if (training && isHidden && dropout > 0) { m = Math.random() < keep ? 1 / keep : 0; av *= m; }
      zl.push(s); al.push(av); ml.push(m);
    }
    z.push(zl); a.push(al); mask.push(ml);
  }
  return { a, z, mask };
}

export const predictProb = (net: Net, x0: number, x1: number): number =>
  forward(net, x0, x1, 0, false).a[net.W.length][0];

const bce = (p: number, y: number): number => {
  const q = Math.max(1e-7, Math.min(1 - 1e-7, p));
  return -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
};

export function evaluate(net: Net, data: DataPoint[]): { trainLoss: number; valLoss: number; trainAcc: number; valAcc: number } {
  let tl = 0; let vl = 0; let ta = 0; let va = 0; let tn = 0; let vn = 0;
  for (const d of data) {
    const p = predictProb(net, d.x, d.y);
    const correct = (p >= 0.5 ? 1 : 0) === d.label ? 1 : 0;
    if (d.train) { tl += bce(p, d.label); ta += correct; tn++; } else { vl += bce(p, d.label); va += correct; vn++; }
  }
  return { trainLoss: tn ? tl / tn : 0, valLoss: vn ? vl / vn : 0, trainAcc: tn ? ta / tn : 0, valAcc: vn ? va / vn : 0 };
}

/** One full-batch gradient-descent epoch over the training split (mutates net). */
export function trainEpoch(net: Net, data: DataPoint[], lr: number, dropout: number): void {
  const L = net.W.length;
  const gW = net.W.map((Wl) => Wl.map((row) => row.map(() => 0)));
  const gb = net.b.map((bl) => bl.map(() => 0));
  const train = data.filter((d) => d.train);
  if (!train.length) return;

  for (const d of train) {
    const { a, z, mask } = forward(net, d.x, d.y, dropout, true);
    // output delta for sigmoid + BCE: dL/dz_out = p - y
    const delta: number[][] = net.dims.map(() => []);
    delta[L] = [a[L][0] - d.label];
    for (let l = L - 1; l >= 0; l--) {
      for (let j = 0; j < net.dims[l + 1]; j++) {
        const dj = delta[l + 1][j];
        for (let i = 0; i < net.dims[l]; i++) gW[l][j][i] += dj * a[l][i];
        gb[l][j] += dj;
      }
      if (l > 0) {
        const dl: number[] = [];
        for (let i = 0; i < net.dims[l]; i++) {
          let s = 0;
          for (let j = 0; j < net.dims[l + 1]; j++) s += net.W[l][j][i] * delta[l + 1][j];
          // backprop through the activation, respecting the dropout mask of this unit
          dl.push(s * actDeriv(net.act[l - 1], a[l][i], z[l][i]) * mask[l][i]);
        }
        delta[l] = dl;
      }
    }
  }
  const n = train.length;
  for (let l = 0; l < L; l++) {
    for (let j = 0; j < net.dims[l + 1]; j++) {
      for (let i = 0; i < net.dims[l]; i++) net.W[l][j][i] -= lr * gW[l][j][i] / n;
      net.b[l][j] -= lr * gb[l][j] / n;
    }
  }
}

/** Human-readable architecture summary, e.g. "2 → 16 → 1 · ReLU · dropout 0.3". */
export const archSummary = (arch: Arch): string => {
  const dims = [2, ...arch.hidden.map((h) => h.units), 1].join(' → ');
  const acts = arch.hidden.length ? ` · ${arch.hidden.map((h) => h.act).join('/')}` : ' · logistic';
  return `${dims}${acts}${arch.dropout > 0 ? ` · dropout ${arch.dropout}` : ''}`;
};
