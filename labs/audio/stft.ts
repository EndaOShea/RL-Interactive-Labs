// Pure maths for the Spectrogram (STFT) lab — no React — shared by the lab, its
// Python export and verification harnesses.
//
// Realistic telephone-band audio: 8 kHz sample rate, 0.5 s signals. Each frame
// of N samples (hop N/2) is tapered by a window w[n] and transformed with a
// direct DFT evaluated only at the unique bins k = 0..N/2 (f_k = k·SR/N, up to
// Nyquist SR/2). Magnitudes are one-sided amplitudes normalised by the window's
// coherent gain, so a sinusoid of amplitude A centred on a bin reads A:
//   |X(k,τ)| = c_k/(N·c_g) · |Σₙ w[n]·x[τ·H + n]·e^(−j2πkn/N)|,  c_k = 2 (1 at k = 0, N/2)
import { WindowKind, windowGain, windowCoherentGain, melFilterbank, applyMel, hzToMel, melToHz } from './shared';

export const SR = 8000;                 // samples per second
export const DUR = 0.5;                 // seconds
export const L = Math.round(SR * DUR);  // 4000 samples
export const NYQUIST = SR / 2;
export const WINDOW_SIZES = [64, 128, 256, 512]; // 8, 16, 32, 64 ms
export const N_MEL = 20;                // mel bands (0 … 4 kHz)
export const DB_RANGE = 80;             // dB shown below the loudest cell
export const AMP_FLOOR = 1e-10;         // amplitude floor before the log (−200 dB)

export type Signal = 'chirp' | 'two-tone' | 'tone' | 'tone+noise' | 'clicks' | 'vowel';
export const SIGNALS: Signal[] = ['chirp', 'two-tone', 'tone', 'tone+noise', 'clicks', 'vowel'];

// Signal recipes (all amplitudes are peak amplitudes of each component).
export const CHIRP = { f0: 200, f1: 3600 };                    // linear sweep over DUR
export const TWO_TONE = { f1: 1000, f2: 1150, a2: 0.7 };
export const TONE_HZ = 4000 / 3;                               // 1333.3 Hz: 1/3 bin off the DFT grid at every N offered
export const NOISY_TONE = { f: 1000, amp: 0.6, seed: 7 };      // 1 kHz tone + uniform noise in ±0.6
export const CLICK_SAMPLES = [1000, 2000, 2080];               // unit impulses at 125, 250 and 260 ms
export const VOWEL = { f0: 125, formants: [300, 870, 2240], bandwidths: [60, 80, 120] }; // /u/ ("oo"), male voice

/** Deterministic 32-bit PRNG (mulberry32) — ported exactly to the Python export. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Magnitude of a cascade of 2-pole resonators (unity at DC, ≈ F/B at each formant F). */
export function formantGain(f: number): number {
  let g = 1;
  VOWEL.formants.forEach((F, i) => {
    const B = VOWEL.bandwidths[i] ?? 100;
    g *= (F * F) / Math.sqrt((F * F - f * f) ** 2 + (B * f) ** 2);
  });
  return g;
}

/** Harmonic amplitudes of the vowel: a 1/h glottal source tilt × the formant filter, max normalised to 1. */
export function vowelAmps(): number[] {
  const H = Math.floor((NYQUIST - 1) / VOWEL.f0); // harmonics below Nyquist (31)
  const raw = Array.from({ length: H }, (_, i) => formantGain((i + 1) * VOWEL.f0) / (i + 1));
  const mx = Math.max(...raw);
  return raw.map((a) => a / mx);
}

export function makeSignal(kind: Signal): number[] {
  const x: number[] = new Array<number>(L).fill(0);
  if (kind === 'clicks') {
    CLICK_SAMPLES.forEach((n) => { x[n] = 1; });
    return x;
  }
  const rnd = mulberry32(NOISY_TONE.seed);
  const amps = kind === 'vowel' ? vowelAmps() : [];
  for (let n = 0; n < L; n++) {
    const t = n / SR;
    if (kind === 'chirp') {
      x[n] = Math.sin(2 * Math.PI * (CHIRP.f0 * t + (0.5 * (CHIRP.f1 - CHIRP.f0) * t * t) / DUR));
    } else if (kind === 'two-tone') {
      x[n] = Math.sin(2 * Math.PI * TWO_TONE.f1 * t) + TWO_TONE.a2 * Math.sin(2 * Math.PI * TWO_TONE.f2 * t);
    } else if (kind === 'tone') {
      x[n] = Math.sin(2 * Math.PI * TONE_HZ * t);
    } else if (kind === 'tone+noise') {
      x[n] = Math.sin(2 * Math.PI * NOISY_TONE.f * t) + NOISY_TONE.amp * (2 * rnd() - 1);
    } else { // vowel: harmonics in cosine phase (glottal pulses) shaped by the formants
      let s = 0;
      amps.forEach((a, i) => { s += a * Math.cos(2 * Math.PI * (i + 1) * VOWEL.f0 * t); });
      x[n] = s;
    }
  }
  return x;
}

export const hopOf = (N: number): number => N / 2;
export const frameCount = (N: number): number => Math.floor((L - N) / hopOf(N)) + 1;
export const binHz = (k: number, N: number): number => (k * SR) / N;
export const bins = (N: number): number[] => Array.from({ length: N / 2 + 1 }, (_, k) => binHz(k, N));

/** One-sided, coherent-gain-normalised amplitude spectrum of one frame (bins 0..N/2). */
export function frameSpectrum(frame: number[], win: WindowKind): number[] {
  const N = frame.length;
  const cg = Math.max(1e-6, windowCoherentGain(win, N));
  const w = Array.from({ length: N }, (_, n) => windowGain(win, n, N));
  const out: number[] = [];
  for (let k = 0; k <= N / 2; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const s = (frame[n] ?? 0) * (w[n] ?? 0);
      const ang = (-2 * Math.PI * k * n) / N;
      re += s * Math.cos(ang);
      im += s * Math.sin(ang);
    }
    const ck = k === 0 || k === N / 2 ? 1 : 2;
    out.push((ck * Math.sqrt(re * re + im * im)) / (N * cg));
  }
  return out;
}

export interface Spec {
  N: number;
  hop: number;
  frames: number;
  mel: boolean;
  cols: number[][];   // [frame][row] amplitude (linear bins or mel bands), low → high frequency
  rowHz: number[];    // centre frequency of each row (Hz)
  rowLoHz: number[];  // lower edge of each row's display band (Hz)
  rowHiHz: number[];  // upper edge of each row's display band (Hz)
}

/** Full STFT: every frame's amplitude spectrum, optionally pooled into mel bands. */
export function stft(x: number[], N: number, win: WindowKind, mel: boolean): Spec {
  const hop = hopOf(N), frames = frameCount(N), fk = bins(N);
  const bank = mel ? melFilterbank(fk, N_MEL, NYQUIST) : null;
  const cols: number[][] = [];
  for (let fr = 0; fr < frames; fr++) {
    const mag = frameSpectrum(x.slice(fr * hop, fr * hop + N), win);
    cols.push(bank ? applyMel(mag, bank.filters) : mag);
  }
  let rowHz: number[], rowLoHz: number[], rowHiHz: number[];
  if (bank) {
    // Each mel band is drawn over ±half a band spacing (in mel) around its centre.
    const dMel = hzToMel(NYQUIST) / (N_MEL + 1);
    rowHz = bank.centers;
    rowLoHz = bank.centers.map((c) => melToHz(hzToMel(c) - dMel / 2));
    rowHiHz = bank.centers.map((c) => melToHz(hzToMel(c) + dMel / 2));
  } else {
    const df = SR / N;
    rowHz = fk;
    rowLoHz = fk.map((f) => Math.max(0, f - df / 2));
    rowHiHz = fk.map((f) => Math.min(NYQUIST, f + df / 2));
  }
  return { N, hop, frames, mel, cols, rowHz, rowLoHz, rowHiHz };
}

/** 20·log10 of an amplitude (floored at AMP_FLOOR). */
export const toDb = (a: number): number => 20 * Math.log10(Math.max(a, AMP_FLOOR));

/** Loudest cell of the whole spectrogram (amplitude). */
export const specMax = (s: Spec): number => s.cols.reduce((m, c) => c.reduce((a, v) => Math.max(a, v), m), 0);

/** Frame centre time (s). */
export const frameTime = (fr: number, N: number): number => (fr * hopOf(N) + N / 2) / SR;

/** Row index of the loudest cell in a frame (−1 for an all-zero frame). */
export function peakRow(col: number[]): number {
  let best = -1, bv = 0;
  col.forEach((v, i) => { if (v > bv) { bv = v; best = i; } });
  return best;
}

/** Time-average of the columns (amplitude). */
export function meanColumn(s: Spec): number[] {
  const rows = s.cols[0]?.length ?? 0;
  const out = new Array<number>(rows).fill(0);
  s.cols.forEach((c) => c.forEach((v, i) => { out[i] = (out[i] ?? 0) + v / s.frames; }));
  return out;
}

/** Index of the row whose centre frequency is closest to f. */
export function nearestRow(s: Spec, f: number): number {
  let best = 0;
  s.rowHz.forEach((h, i) => { if (Math.abs(h - f) < Math.abs((s.rowHz[best] ?? 0) - f)) best = i; });
  return best;
}

/** Top of the dB colour scale: the loudest cell rounded UP to a whole dB (the floor is DB_RANGE below). */
export const dbTop = (s: Spec): number => Math.ceil(toDb(specMax(s)) - 1e-9);

const median = (v: number[]): number => {
  const a = [...v].sort((x, y) => x - y);
  return a[Math.floor(a.length / 2)] ?? NaN;
};

/**
 * What the finished spectrogram shows, measured — one record per signal. Every
 * number the lab narrates about a demo comes from here (and is mirrored by the
 * Python export).
 */
export type Measurement =
  | { kind: 'chirp'; firstHz: number; lastHz: number; monotone: boolean }
  | { kind: 'two-tone'; resolved: boolean; dipDb: number | null }
  | { kind: 'tone'; medianDb: number; visible: number; farRows: number }
  | { kind: 'tone+noise'; toneDb: number; noiseDb: number; snrDb: number }
  | { kind: 'clicks'; events: { startMs: number; endMs: number }[]; pairSeparated: boolean }
  | { kind: 'vowel'; peaksHz: number[] };

export function measure(signal: Signal, s: Spec): Measurement {
  const mean = meanColumn(s);
  const db = mean.map(toDb);
  if (signal === 'chirp') {
    const pk = s.cols.map((c) => s.rowHz[peakRow(c)] ?? 0);
    return { kind: 'chirp', firstHz: pk[0] ?? 0, lastHz: pk[pk.length - 1] ?? 0, monotone: pk.every((v, i) => i === 0 || v >= (pk[i - 1] ?? 0)) };
  }
  if (signal === 'two-tone') {
    // Resolved = the time-averaged spectrum dips ≥ 3 dB below the weaker tone somewhere between them.
    const ia = nearestRow(s, TWO_TONE.f1), ib = nearestRow(s, TWO_TONE.f2);
    if (ib - ia < 2) return { kind: 'two-tone', resolved: false, dipDb: null };
    const peakNear = (i: number) => Math.max(db[i - 1] ?? -Infinity, db[i] ?? -Infinity, db[i + 1] ?? -Infinity);
    let mn = Infinity;
    for (let i = ia + 1; i < ib; i++) mn = Math.min(mn, db[i] ?? Infinity);
    const dip = Math.min(peakNear(ia), peakNear(ib)) - mn;
    return { kind: 'two-tone', resolved: dip >= 3, dipDb: dip };
  }
  if (signal === 'tone') {
    // Leakage: every row more than 500 Hz from the tone — its level relative to the
    // loudest row, and whether it clears the display floor (dbTop − DB_RANGE).
    const peak = Math.max(...db), floor = dbTop(s) - DB_RANGE;
    const far = s.rowHz.map((f, i) => ({ f, d: db[i] ?? -Infinity })).filter((o) => Math.abs(o.f - TONE_HZ) > 500).map((o) => o.d);
    return { kind: 'tone', medianDb: median(far.map((d) => d - peak)), visible: far.filter((d) => d > floor).length, farRows: far.length };
  }
  if (signal === 'tone+noise') {
    const toneDb = db[nearestRow(s, NOISY_TONE.f)] ?? -Infinity;
    const noise = s.rowHz.map((f, i) => ({ f, d: db[i] ?? -Infinity })).filter((o) => Math.abs(o.f - NOISY_TONE.f) > 300 && o.f > 0 && o.f < NYQUIST).map((o) => o.d);
    const noiseDb = median(noise);
    return { kind: 'tone+noise', toneDb, noiseDb, snrDb: toneDb - noiseDb };
  }
  if (signal === 'clicks') {
    // Events = runs of consecutive columns whose loudest cell clears the display floor (dbTop − DB_RANGE).
    const floor = dbTop(s) - DB_RANGE;
    const on = s.cols.map((c) => toDb(Math.max(...c)) > floor);
    const events: { startMs: number; endMs: number }[] = [];
    let start = -1;
    on.forEach((v, i) => {
      if (v && start < 0) start = i;
      const last = i === on.length - 1;
      if (start >= 0 && (!v || last)) {
        const end = v && last ? i : i - 1;
        events.push({ startMs: frameTime(start, s.N) * 1000, endMs: frameTime(end, s.N) * 1000 });
        start = -1;
      }
    });
    return { kind: 'clicks', events, pairSeparated: events.length >= CLICK_SAMPLES.length };
  }
  // vowel: local maxima of the averaged spectrum; with harmonics resolved (rows
  // finer than half the pitch), look only at the harmonic frequencies h·f0.
  const resolved = !s.mel && SR / s.N <= VOWEL.f0 / 2;
  const idx = resolved
    ? Array.from({ length: Math.floor((NYQUIST - 1) / VOWEL.f0) }, (_, h) => nearestRow(s, (h + 1) * VOWEL.f0))
    : s.rowHz.map((_, i) => i);
  const vals = idx.map((i) => db[i] ?? -Infinity);
  const peaks = vals.map((v, j) => (j > 0 && j < vals.length - 1 && v > (vals[j - 1] ?? -Infinity) && v >= (vals[j + 1] ?? -Infinity) ? j : -1)).filter((j) => j >= 0);
  const top3 = [...peaks].sort((a, b) => (vals[b] ?? 0) - (vals[a] ?? 0)).slice(0, 3).sort((a, b) => a - b);
  return { kind: 'vowel', peaksHz: top3.map((j) => s.rowHz[idx[j] ?? 0] ?? 0) };
}

/** Width in Hz of the lowest and highest mel triangles (edge to edge). */
export function melBandWidths(N: number): { lowHz: number; highHz: number } {
  const e = melFilterbank(bins(N), N_MEL, NYQUIST).edges;
  return { lowHz: (e[2] ?? 0) - (e[0] ?? 0), highHz: (e[N_MEL + 1] ?? 0) - (e[N_MEL - 1] ?? 0) };
}
