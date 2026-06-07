// Area-local helpers for the Audio & Speech labs (Fourier synthesis + STFT
// spectrogram). Kept separate from the shared viz primitives so the two labs
// can share window functions, the mel warp, and curated presets without
// touching anything outside labs/audio/.

// ---------------------------------------------------------------------------
// Window functions (tapers). Each maps a sample index n in [0, N-1] to a gain.
// Tapering the frame edges suppresses spectral leakage from the abrupt cut at
// the frame boundary. Rectangular = no taper (sharpest main lobe, worst leak).
// ---------------------------------------------------------------------------
export type WindowKind = 'rectangular' | 'hann' | 'hamming' | 'blackman';

export const WINDOW_KINDS: WindowKind[] = ['rectangular', 'hann', 'hamming', 'blackman'];

export function windowGain(kind: WindowKind, n: number, N: number): number {
  if (N <= 1) return 1;
  const r = n / (N - 1); // 0..1 across the frame
  switch (kind) {
    case 'rectangular':
      return 1;
    case 'hann':
      return 0.5 - 0.5 * Math.cos(2 * Math.PI * r);
    case 'hamming':
      return 0.54 - 0.46 * Math.cos(2 * Math.PI * r);
    case 'blackman':
      return 0.42 - 0.5 * Math.cos(2 * Math.PI * r) + 0.08 * Math.cos(4 * Math.PI * r);
  }
}

// Coherent gain (mean of the window) — used to compensate amplitude so a tapered
// frame's magnitudes stay comparable to the rectangular one.
export function windowCoherentGain(kind: WindowKind, N: number): number {
  let s = 0;
  for (let n = 0; n < N; n++) s += windowGain(kind, n, N);
  return s / Math.max(1, N);
}

export const WINDOW_LABEL: Record<WindowKind, string> = {
  rectangular: 'Rectangular (boxcar)',
  hann: 'Hann',
  hamming: 'Hamming',
  blackman: 'Blackman',
};

// ---------------------------------------------------------------------------
// Mel scale — a perceptual, roughly-log frequency warp. The ear resolves low
// frequencies finely and high frequencies coarsely; mel bins mirror that. We
// use it to (a) warp the Fourier amplitude spectrum and (b) optionally remap
// the spectrogram's frequency axis.
// ---------------------------------------------------------------------------
export const hzToMel = (f: number): number => 2595 * Math.log10(1 + f / 700);
export const melToHz = (m: number): number => 700 * (Math.pow(10, m / 2595) - 1);

/**
 * Build `nMel` triangular mel-filter weights over `nLin` linear bins spanning
 * [0, fMax] Hz, then apply them to a linear magnitude vector. Returns the mel
 * energies and the centre frequency (Hz) of each mel band (for labels).
 */
export function melFilterbank(nLin: number, nMel: number, fMax: number) {
  const melMax = hzToMel(fMax);
  // nMel triangles need nMel+2 edge points in mel space.
  const edges: number[] = [];
  for (let i = 0; i < nMel + 2; i++) edges.push(melToHz((i / (nMel + 1)) * melMax));
  const centers = edges.slice(1, nMel + 1);
  const binHz = (k: number) => (k / Math.max(1, nLin - 1)) * fMax;
  const filters: number[][] = [];
  for (let m = 0; m < nMel; m++) {
    const lo = edges[m], ce = edges[m + 1], hi = edges[m + 2];
    const w: number[] = [];
    for (let k = 0; k < nLin; k++) {
      const f = binHz(k);
      let g = 0;
      if (f >= lo && f <= ce) g = ce > lo ? (f - lo) / (ce - lo) : 0;
      else if (f > ce && f <= hi) g = hi > ce ? (hi - f) / (hi - ce) : 0;
      w.push(g);
    }
    filters.push(w);
  }
  return { filters, centers };
}

export function applyMel(mag: number[], filters: number[][]): number[] {
  return filters.map((w) => {
    let s = 0;
    for (let k = 0; k < mag.length; k++) s += w[k] * mag[k];
    return s;
  });
}

// ---------------------------------------------------------------------------
// Fourier curated presets (amplitude recipes for harmonics k = 1..K). The lab
// keeps its original four; these are the extra "voices" / challenges. Each is a
// length-5 amplitude vector to match the five sliders.
// ---------------------------------------------------------------------------
export interface FourierPreset {
  id: string;
  amps: number[];
  blurb: string; // short "try this" guidance, spoken + shown
}

export const FOURIER_EXTRA_PRESETS: FourierPreset[] = [
  { id: 'pulse', amps: [1, 0.9, 0.75, 0.55, 0.35], blurb: 'Narrow pulse: every harmonic present and strong — a bright, buzzy click.' },
  { id: 'organ', amps: [1, 0.5, 0.0, 0.25, 0.0], blurb: 'Hollow organ tone: fundamental + octave + a touch of the 4th harmonic.' },
  { id: 'clarinet', amps: [1, 0.04, 0.6, 0.05, 0.4], blurb: 'Clarinet-like: odd harmonics dominate, even ones nearly absent.' },
];

// ---------------------------------------------------------------------------
// Spectrogram curated signal challenges (named, with guidance text).
// ---------------------------------------------------------------------------
export interface SignalChallenge {
  id: string;        // matches the lab's Signal union OR a guided window setting
  window: number;    // suggested window size to reveal the structure
  blurb: string;
}
