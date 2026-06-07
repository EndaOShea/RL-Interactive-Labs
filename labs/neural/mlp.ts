// A small multilayer perceptron (forward + backprop) for the Neural Network
// labs. Binary classification: hidden layers use the chosen activation, the
// output neuron uses sigmoid with binary cross-entropy loss.
import { randn } from '../classic-ml/shared';

export type Act = 'tanh' | 'relu' | 'sigmoid' | 'leaky' | 'gelu';
export type Optimizer = 'sgd' | 'momentum' | 'adam';

// Activations keyed on the *post*-activation value `y` where cheap (tanh/sigmoid/
// relu), and on the *pre*-activation `z` otherwise (leaky/gelu need the raw input
// for their derivative). We therefore cache z per layer and pass it to the
// derivative — see `dAct` below.
const geluF = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
const ACT: Record<Act, { f: (x: number) => number }> = {
  tanh: { f: (x: number) => Math.tanh(x) },
  relu: { f: (x: number) => Math.max(0, x) },
  sigmoid: { f: (x: number) => 1 / (1 + Math.exp(-x)) },
  leaky: { f: (x: number) => (x > 0 ? x : 0.1 * x) },
  gelu: { f: geluF },
};
// derivative as a function of the pre-activation z (and the post-activation y)
const dAct = (act: Act, z: number, y: number): number => {
  switch (act) {
    case 'tanh': return 1 - y * y;
    case 'relu': return z > 0 ? 1 : 0;
    case 'sigmoid': return y * (1 - y);
    case 'leaky': return z > 0 ? 1 : 0.1;
    case 'gelu': return (geluF(z + 1e-3) - geluF(z - 1e-3)) / 2e-3;
  }
};
const sig = (x: number) => 1 / (1 + Math.exp(-x));

export interface TrainOpts { lr: number; optimizer?: Optimizer; l2?: number; }

export class MLP {
  sizes: number[];
  act: Act;
  W: number[][][] = [];
  b: number[][] = [];
  // optimizer state (lazy, same shape as W/b)
  private vW: number[][][] = [];   // momentum / Adam first moment
  private vb: number[][] = [];
  private sW: number[][][] = [];   // Adam second moment
  private sb: number[][] = [];
  private t = 0;                   // Adam timestep

  constructor(sizes: number[], act: Act) {
    this.sizes = sizes; this.act = act;
    const reluLike = act === 'relu' || act === 'leaky' || act === 'gelu';
    for (let l = 0; l < sizes.length - 1; l++) {
      const fan = sizes[l];
      const scale = reluLike ? Math.sqrt(2 / fan) : Math.sqrt(1 / fan);
      this.W.push(Array.from({ length: sizes[l + 1] }, () => Array.from({ length: sizes[l] }, () => randn() * scale)));
      this.b.push(Array.from({ length: sizes[l + 1] }, () => 0));
      this.vW.push(this.W[l].map((r) => r.map(() => 0)));
      this.vb.push(this.b[l].map(() => 0));
      this.sW.push(this.W[l].map((r) => r.map(() => 0)));
      this.sb.push(this.b[l].map(() => 0));
    }
  }

  // forward returns both the activations `a` and the pre-activations `z`
  // (z[0] is a placeholder so indices line up with a).
  private forwardFull(x: number[]): { a: number[][]; z: number[][] } {
    const a: number[][] = [x];
    const z: number[][] = [x];
    let cur = x;
    for (let l = 0; l < this.W.length; l++) {
      const isLast = l === this.W.length - 1;
      const zl = this.W[l].map((row, i) => row.reduce((s, w, j) => s + w * cur[j], 0) + this.b[l][i]);
      cur = zl.map((v) => (isLast ? sig(v) : ACT[this.act].f(v)));
      z.push(zl); a.push(cur);
    }
    return { a, z };
  }

  forward(x: number[]): number[][] { return this.forwardFull(x).a; }

  predict(x: number[]): number { const a = this.forward(x); return a[a.length - 1][0]; }

  trainEpoch(X: number[][], Y: number[], lrOrOpts: number | TrainOpts): number {
    const opts: TrainOpts = typeof lrOrOpts === 'number' ? { lr: lrOrOpts } : lrOrOpts;
    const lr = opts.lr, optimizer = opts.optimizer ?? 'sgd', l2 = opts.l2 ?? 0;
    const L = this.W.length;
    const gW = this.W.map((m) => m.map((r) => r.map(() => 0)));
    const gb = this.b.map((r) => r.map(() => 0));
    let loss = 0;
    for (let n = 0; n < X.length; n++) {
      const { a, z } = this.forwardFull(X[n]);
      const out = a[L][0], y = Y[n];
      const pc = Math.min(1 - 1e-7, Math.max(1e-7, out));
      loss += -(y * Math.log(pc) + (1 - y) * Math.log(1 - pc));
      let delta = [out - y]; // sigmoid + BCE
      for (let l = L - 1; l >= 0; l--) {
        for (let i = 0; i < this.W[l].length; i++) {
          for (let j = 0; j < this.W[l][i].length; j++) gW[l][i][j] += delta[i] * a[l][j];
          gb[l][i] += delta[i];
        }
        if (l > 0) {
          const nd = new Array(this.sizes[l]).fill(0);
          for (let j = 0; j < this.sizes[l]; j++) {
            let s = 0; for (let i = 0; i < this.W[l].length; i++) s += this.W[l][i][j] * delta[i];
            nd[j] = s * dAct(this.act, z[l][j], a[l][j]);
          }
          delta = nd;
        }
      }
    }
    const m = X.length || 1;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    this.t += 1;
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < this.W[l].length; i++) {
        for (let j = 0; j < this.W[l][i].length; j++) {
          let g = gW[l][i][j] / m + l2 * this.W[l][i][j];   // L2 weight decay (not on bias)
          this.W[l][i][j] -= this.optStep('W', l, i, j, g, lr, optimizer, b1, b2, eps);
        }
        const gbi = gb[l][i] / m;
        this.b[l][i] -= this.optStep('b', l, i, 0, gbi, lr, optimizer, b1, b2, eps);
      }
    }
    if (l2 > 0) {
      let sq = 0; this.W.forEach((mm) => mm.forEach((r) => r.forEach((w) => { sq += w * w; })));
      loss += 0.5 * l2 * sq * m;   // report the penalised loss (scaled back per-sample below)
    }
    return loss / m;
  }

  // returns the actual parameter delta to subtract
  private optStep(kind: 'W' | 'b', l: number, i: number, j: number, g: number, lr: number, opt: Optimizer, b1: number, b2: number, eps: number): number {
    if (opt === 'sgd') return lr * g;
    const v = kind === 'W' ? this.vW[l][i] : this.vb[l];
    const s = kind === 'W' ? this.sW[l][i] : this.sb[l];
    const k = kind === 'W' ? j : i;
    if (opt === 'momentum') {
      v[k] = b1 * v[k] + g;              // classic momentum
      return lr * v[k];
    }
    // adam
    v[k] = b1 * v[k] + (1 - b1) * g;
    s[k] = b2 * s[k] + (1 - b2) * g * g;
    const vh = v[k] / (1 - Math.pow(b1, this.t));
    const sh = s[k] / (1 - Math.pow(b2, this.t));
    return lr * vh / (Math.sqrt(sh) + eps);
  }

  accuracy(X: number[][], Y: number[]): number {
    let ok = 0; for (let n = 0; n < X.length; n++) if ((this.predict(X[n]) > 0.5 ? 1 : 0) === Y[n]) ok++;
    return ok / (X.length || 1);
  }
}

/* ─────────────────────────── datasets (domain [-1,1]²) ─────────────────────────── */
export type DatasetKind = 'xor' | 'circles' | 'spiral';
const rnd = (a: number, b: number) => a + Math.random() * (b - a);

export function makeDataset(kind: DatasetKind, n: number): { x: number; y: number; cls: number }[] {
  const out: { x: number; y: number; cls: number }[] = [];
  if (kind === 'xor') {
    for (let i = 0; i < n; i++) { const x = rnd(-0.9, 0.9), y = rnd(-0.9, 0.9); out.push({ x, y, cls: (x > 0) !== (y > 0) ? 1 : 0 }); }
  } else if (kind === 'circles') {
    for (let i = 0; i < n; i++) {
      const inner = i % 2 === 0;
      const r = inner ? rnd(0, 0.38) : rnd(0.62, 0.95);
      const t = rnd(0, Math.PI * 2);
      out.push({ x: r * Math.cos(t), y: r * Math.sin(t), cls: inner ? 0 : 1 });
    }
  } else {
    const half = Math.floor(n / 2);
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < half; i++) {
        const t = (i / half) * 3.2 + (Math.random() * 0.2);
        const r = 0.1 + t * 0.26;
        const ang = t * 1.7 + c * Math.PI;
        out.push({ x: r * Math.cos(ang) + rnd(-0.04, 0.04), y: r * Math.sin(ang) + rnd(-0.04, 0.04), cls: c });
      }
    }
  }
  return out;
}
