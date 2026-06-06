// A small multilayer perceptron (forward + backprop) for the Neural Network
// labs. Binary classification: hidden layers use the chosen activation, the
// output neuron uses sigmoid with binary cross-entropy loss.
import { randn } from '../classic-ml/shared';

export type Act = 'tanh' | 'relu' | 'sigmoid';
const ACT = {
  tanh: { f: (x: number) => Math.tanh(x), d: (y: number) => 1 - y * y },
  relu: { f: (x: number) => Math.max(0, x), d: (y: number) => (y > 0 ? 1 : 0) },
  sigmoid: { f: (x: number) => 1 / (1 + Math.exp(-x)), d: (y: number) => y * (1 - y) },
};
const sig = (x: number) => 1 / (1 + Math.exp(-x));

export class MLP {
  sizes: number[];
  act: Act;
  W: number[][][] = [];
  b: number[][] = [];

  constructor(sizes: number[], act: Act) {
    this.sizes = sizes; this.act = act;
    for (let l = 0; l < sizes.length - 1; l++) {
      const fan = sizes[l];
      const scale = act === 'relu' ? Math.sqrt(2 / fan) : Math.sqrt(1 / fan);
      this.W.push(Array.from({ length: sizes[l + 1] }, () => Array.from({ length: sizes[l] }, () => randn() * scale)));
      this.b.push(Array.from({ length: sizes[l + 1] }, () => 0));
    }
  }

  forward(x: number[]): number[][] {
    const a: number[][] = [x];
    let cur = x;
    for (let l = 0; l < this.W.length; l++) {
      const isLast = l === this.W.length - 1;
      const z = this.W[l].map((row, i) => row.reduce((s, w, j) => s + w * cur[j], 0) + this.b[l][i]);
      cur = z.map((v) => (isLast ? sig(v) : ACT[this.act].f(v)));
      a.push(cur);
    }
    return a;
  }

  predict(x: number[]): number { const a = this.forward(x); return a[a.length - 1][0]; }

  trainEpoch(X: number[][], Y: number[], lr: number): number {
    const L = this.W.length;
    const gW = this.W.map((m) => m.map((r) => r.map(() => 0)));
    const gb = this.b.map((r) => r.map(() => 0));
    let loss = 0;
    for (let n = 0; n < X.length; n++) {
      const a = this.forward(X[n]);
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
          const prev = a[l];
          const nd = new Array(this.sizes[l]).fill(0);
          for (let j = 0; j < this.sizes[l]; j++) {
            let s = 0; for (let i = 0; i < this.W[l].length; i++) s += this.W[l][i][j] * delta[i];
            nd[j] = s * ACT[this.act].d(prev[j]);
          }
          delta = nd;
        }
      }
    }
    const m = X.length || 1;
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < this.W[l].length; i++) {
        for (let j = 0; j < this.W[l][i].length; j++) this.W[l][i][j] -= lr * gW[l][i][j] / m;
        this.b[l][i] -= lr * gb[l][i] / m;
      }
    }
    return loss / m;
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
