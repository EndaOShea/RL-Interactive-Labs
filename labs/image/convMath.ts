// Pure maths for the Convolution & Filters lab (no React, no imports) so the lab,
// its Python export and verification harnesses share one implementation.
//
// The core op is CROSS-CORRELATION — what CNN libraries call "convolution":
//   (I⋆K)(i,j) = Σₘ Σₙ I(s·i+m, s·j+n)·K(m,n),  m,n ∈ {−1,0,1}
// True convolution flips the kernel first: (I∗K)(i,j) = Σ I(s·i−m, s·j−n)·K(m,n),
// i.e. cross-correlation with the kernel rotated 180°.

export const N = 14; // image side

export type ImgPreset = 'cross' | 'diagonal' | 'circle';
export const IMG_PRESETS: ImgPreset[] = ['cross', 'diagonal', 'circle'];
export type KernelName = 'identity' | 'edge-detect' | 'sharpen' | 'box-blur' | 'sobel-x' | 'sobel-y' | 'emboss' | 'laplacian' | 'gaussian-blur';
export type PadMode = 'zero' | 'replicate' | 'reflect';
export const PAD_MODES: PadMode[] = ['zero', 'replicate', 'reflect'];

export const KERNELS: Record<KernelName, number[][]> = {
  identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  'edge-detect': [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
  sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  'box-blur': [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
  'sobel-x': [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  'sobel-y': [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
  emboss: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
  laplacian: [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
  'gaussian-blur': [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]],
};

export const KERNEL_NAMES: KernelName[] = ['identity', 'edge-detect', 'sharpen', 'box-blur', 'sobel-x', 'sobel-y', 'emboss', 'laplacian', 'gaussian-blur'];

const at = (m: number[][], r: number, c: number): number => m[r]?.[c] ?? 0;

export function makeImage(preset: ImgPreset): number[][] {
  const cx = (N - 1) / 2, cy = (N - 1) / 2, R = 5;
  return Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => {
    if (preset === 'cross') return (c >= 6 && c < 8) || (r >= 6 && r < 8) ? 1 : 0;   // 2-px bars, full length
    if (preset === 'diagonal') return c === r || c === r + 1 ? 1 : 0;                 // 2-px-wide diagonal
    return Math.abs(Math.hypot(r - cy, c - cx) - R) < 1.1 ? 1 : 0;                      // ring, radius 5
  }));
}

/** Read a pixel with a 1-px border under the chosen padding mode. */
export function padRead(img: number[][], r: number, c: number, pad: PadMode): number {
  if (r >= 0 && r < N && c >= 0 && c < N) return at(img, r, c);
  if (pad === 'zero') return 0;
  if (pad === 'replicate') { // clamp to the nearest edge pixel (numpy 'edge')
    return at(img, Math.max(0, Math.min(N - 1, r)), Math.max(0, Math.min(N - 1, c)));
  }
  // reflect without repeating the edge (numpy 'reflect'): −1 → 1, N → N−2
  const refl = (x: number) => (x < 0 ? -x : x >= N ? 2 * (N - 1) - x : x);
  return at(img, refl(r), refl(c));
}

/** The (N+2)×(N+2) padded image the 3×3 window actually reads. */
export function paddedImage(img: number[][], pad: PadMode): number[][] {
  return Array.from({ length: N + 2 }, (_, r) => Array.from({ length: N + 2 }, (_, c) => padRead(img, r - 1, c - 1, pad)));
}

/** Rotate a kernel by 180° (flip both axes) — turns cross-correlation into true convolution. */
export const flipKernel = (k: number[][]): number[][] => k.slice().reverse().map((row) => row.slice().reverse());

/** The weights actually multiplied onto the window: K itself, or K rotated 180° for true convolution. */
export const appliedKernel = (k: number[][], flip: boolean): number[][] => (flip ? flipKernel(k) : k);

/** One output pixel: 3×3 cross-correlation of `k` centred at input (ci, cj). */
export function convAt(img: number[][], k: number[][], ci: number, cj: number, pad: PadMode): number {
  let s = 0;
  for (let m = -1; m <= 1; m++) for (let n = -1; n <= 1; n++) {
    s += padRead(img, ci + m, cj + n, pad) * at(k, m + 1, n + 1);
  }
  return s;
}

/** Output side for P = 1, F = 3: ⌊(N − 3 + 2)/S⌋ + 1 (14 at stride 1, 7 at stride 2). */
export const outSide = (stride: number): number => Math.floor((N - 3 + 2) / stride) + 1;

/** Full feature map. Output cell (oi, oj) is centred on input (oi·s, oj·s). */
export function fullConv(img: number[][], k: number[][], pad: PadMode, stride: number): number[][] {
  const O = outSide(stride);
  return Array.from({ length: O }, (_, oi) => Array.from({ length: O }, (_, oj) => convAt(img, k, oi * stride, oj * stride, pad)));
}

/** True when the 3×3 window centred at input (ci, cj) reads at least one padded pixel. */
export const readsPadding = (ci: number, cj: number): boolean => ci - 1 < 0 || cj - 1 < 0 || ci + 1 > N - 1 || cj + 1 > N - 1;

export interface Peak { v: number; i: number; j: number; border: boolean; }

/**
 * Strongest responses ranked by |value| over the first `count` output cells in
 * sweep order (row-major): the overall peak (flagged when its window reads
 * padding) and the strongest interior peak (window entirely inside the image).
 * Ties keep the first cell in sweep order. Returns null peaks when nothing has
 * been computed / no interior cell exists yet.
 */
export function peaks(full: number[][], stride: number, count: number): { all: Peak | null; interior: Peak | null } {
  const O = full.length;
  let all: Peak | null = null, interior: Peak | null = null;
  const lim = Math.min(count, O * O);
  for (let p = 0; p < lim; p++) {
    const i = Math.floor(p / O), j = p % O;
    const v = at(full, i, j);
    const border = readsPadding(i * stride, j * stride);
    if (!all || Math.abs(v) > Math.abs(all.v) + 1e-12) all = { v, i, j, border };
    if (!border && (!interior || Math.abs(v) > Math.abs(interior.v) + 1e-12)) interior = { v, i, j, border };
  }
  return { all, interior };
}

/** Largest |value| in a matrix (0 for an all-zero map). */
export const absMax = (m: number[][]): number => m.reduce((mx, row) => row.reduce((a, v) => Math.max(a, Math.abs(v)), mx), 0);
