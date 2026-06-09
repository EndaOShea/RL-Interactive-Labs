// Self-contained backprop maths for a FIXED feed-forward net 3 -> 4 -> 4 -> 1.
// All weights/biases are hardcoded deterministic constants (NOT randomised at
// module load), so the forward/backward numbers are identical every run and the
// Math tab can show exact, internally-consistent chain-rule values.

export type ActName = 'sigmoid' | 'relu' | 'tanh' | 'leaky';
export const LEAKY_SLOPE = 0.01;

// Layer sizes: input(3) -> hidden(4) -> hidden(4) -> output(1). Three weight
// matrices live between the four layers (l = 0,1,2).
export const SIZES = [3, 4, 4, 1] as const;

// ---- Fixed initial parameters (small, hand-picked, deterministic) ----------
// W[l] has shape [out][in]: row j is the incoming weights of unit j in layer l+1.
export const INIT_W: number[][][] = [
  // layer 0:  3 -> 4
  [
    [0.50, -0.30, 0.20],
    [-0.40, 0.60, 0.10],
    [0.30, 0.20, -0.50],
    [-0.20, -0.40, 0.70],
  ],
  // layer 1:  4 -> 4
  [
    [0.40, -0.50, 0.30, 0.10],
    [0.20, 0.30, -0.60, 0.40],
    [-0.30, 0.50, 0.20, -0.40],
    [0.60, -0.20, 0.40, 0.30],
  ],
  // layer 2:  4 -> 1
  [
    [0.50, -0.40, 0.30, 0.60],
  ],
];

export const INIT_B: number[][] = [
  [0.10, -0.20, 0.30, -0.10], // layer 0 biases (4 units)
  [-0.10, 0.20, 0.10, -0.30], // layer 1 biases (4 units)
  [0.10],                     // output bias
];

// ---- Activations + derivatives (derivative expressed wrt pre-activation z) --
export function act(name: ActName, z: number): number {
  switch (name) {
    case 'relu': return Math.max(0, z);
    case 'leaky': return z > 0 ? z : LEAKY_SLOPE * z;
    case 'tanh': return Math.tanh(z);
    default: return 1 / (1 + Math.exp(-z)); // sigmoid
  }
}

// derivative as a function of the activation value a (cheap, exact for sigmoid/tanh)
export function dactFromA(name: ActName, a: number, z: number): number {
  switch (name) {
    case 'relu': return z > 0 ? 1 : 0;
    case 'leaky': return z > 0 ? 1 : LEAKY_SLOPE;
    case 'tanh': return 1 - a * a;
    default: return a * (1 - a); // sigmoid a(1-a)
  }
}

export interface ForwardResult {
  // a[l] = activation vector at layer l (l = 0 is the input, l = 3 is output)
  a: number[][];
  // z[l] = pre-activations feeding layer l (z[0] is undefined-placeholder = input copy)
  z: number[][];
  yhat: number;
  loss: number;
}

/** Forward pass: z[l] = W[l-1] a[l-1] + b[l-1]; a[l] = act(z[l]). Output uses the
 *  same activation as the hidden layers (matches the lab + Python). */
export function forward(
  W: number[][][], B: number[][], x: number[], name: ActName, y: number,
): ForwardResult {
  const a: number[][] = [x.slice()];
  const z: number[][] = [x.slice()]; // placeholder for the input layer
  for (let l = 0; l < W.length; l++) {
    const zl: number[] = [];
    const al: number[] = [];
    for (let j = 0; j < W[l].length; j++) {
      let s = B[l][j];
      for (let i = 0; i < W[l][j].length; i++) s += W[l][j][i] * a[l][i];
      zl.push(s);
      al.push(act(name, s));
    }
    z.push(zl);
    a.push(al);
  }
  const yhat = a[a.length - 1][0];
  const loss = 0.5 * (yhat - y) * (yhat - y);
  return { a, z, yhat, loss };
}

export interface BackwardResult {
  // delta[l] aligns with layers 1..L (delta[0] is an empty placeholder for input)
  delta: number[][];
  // gradients per weight matrix, same shape as W
  gW: number[][][];
  gB: number[][];
}

/** Backward pass.
 *  delta_out = (yhat - y) * act'(z_out)
 *  delta[l]  = (W[l+1]^T delta[l+1]) ⊙ act'(z[l])
 *  dL/dW[l]  = delta[l+1] (a[l])^T ;  dL/db[l] = delta[l+1]
 */
export function backward(
  W: number[][][], fwd: ForwardResult, name: ActName, y: number,
): BackwardResult {
  const L = W.length;            // number of weight layers (3)
  const a = fwd.a, z = fwd.z;
  // delta indexed by activation layer 1..L (delta[L] = output). delta[0] unused.
  const delta: number[][] = a.map(() => []);

  // output delta (layer index L = 3)
  const outAct = a[L][0];
  delta[L] = [(fwd.yhat - y) * dactFromA(name, outAct, z[L][0])];

  // hidden deltas, going backward
  for (let l = L - 1; l >= 1; l--) {
    const dl: number[] = [];
    const Wnext = W[l]; // weights from layer l -> layer l+1, shape [out][in=units in l]
    for (let i = 0; i < a[l].length; i++) {
      let s = 0;
      for (let j = 0; j < Wnext.length; j++) s += Wnext[j][i] * delta[l + 1][j];
      dl.push(s * dactFromA(name, a[l][i], z[l][i]));
    }
    delta[l] = dl;
  }

  // gradients
  const gW: number[][][] = W.map((Wl, l) =>
    Wl.map((row, j) => row.map((_, i) => delta[l + 1][j] * a[l][i])));
  const gB: number[][] = W.map((Wl, l) => Wl.map((_, j) => delta[l + 1][j]));

  return { delta, gW, gB };
}

/** One gradient-descent step: W -= lr * gW, b -= lr * gB (returns fresh copies). */
export function applyStep(
  W: number[][][], B: number[][], grad: BackwardResult, lr: number,
): { W: number[][][]; B: number[][] } {
  const nW = W.map((Wl, l) => Wl.map((row, j) => row.map((w, i) => w - lr * grad.gW[l][j][i])));
  const nB = B.map((bl, l) => bl.map((b, j) => b - lr * grad.gB[l][j]));
  return { W: nW, B: nB };
}

export const cloneW = (W: number[][][]) => W.map((Wl) => Wl.map((row) => row.slice()));
export const cloneB = (B: number[][]) => B.map((bl) => bl.slice());

// Preset input examples (3-vectors).
export const INPUT_PRESETS: { label: string; x: number[] }[] = [
  { label: 'A · [1.0, 0.5, -0.5]', x: [1.0, 0.5, -0.5] },
  { label: 'B · [-1.0, 0.8, 0.3]', x: [-1.0, 0.8, 0.3] },
  { label: 'C · [0.6, -0.9, 1.2]', x: [0.6, -0.9, 1.2] },
  { label: 'D · [0.2, 0.2, 0.2]', x: [0.2, 0.2, 0.2] },
];
