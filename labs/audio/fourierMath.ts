// Pure maths for the Fourier Synthesis lab (no React) — shared by the lab, its
// Python export and verification harnesses.
//
// x(t) = Σₖ aₖ·sin(2π·k·f₀·t + φ), k = 1…K, with φ = 0 (sine phase) or 90° (cosine
// phase). Time is measured in periods u = f₀·t. The spectrum shown by the lab is
// COMPUTED: a DFT of NS samples of one period, amplitude 2|X_k|/NS and phase ∠X_k.
import { hzToMel } from './shared';

export const K = 5;            // harmonics = amplitude sliders
export const NS = 256;         // DFT samples per period (≫ 2K, so no aliasing)
export const FINE = 2048;      // grid for the peak / overshoot metrics (per period)
export const K_SHOW = 8;       // DFT bins shown: k = 1…8 (6–8 are computed, and ≈ 0)
export const F0_CHOICES = [110, 220, 440, 880]; // Hz (A2 … A5)

export type PhaseMode = 'sin' | 'cos';
export type IdealKind = 'sine' | 'square' | 'sawtooth' | 'triangle';
export const IDEAL_KINDS: IdealKind[] = ['sine', 'square', 'sawtooth', 'triangle'];

// Truncated Fourier-series recipes (sine phase): the first K terms of each ideal wave.
export const BASE_PRESETS: Record<IdealKind, number[]> = {
  sine: [1, 0, 0, 0, 0],
  square: [1, 0, 1 / 3, 0, 1 / 5],          // odd harmonics ∝ 1/k      → (π/4)·sgn(sin 2πu)
  sawtooth: [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5], // all harmonics ∝ 1/k      → (π/2)(1 − 2u), 0 < u < 1
  triangle: [1, 0, -1 / 9, 0, 1 / 25],       // odd, alternating ∝ 1/k²  → (π²/8)·tri(u)
};

/** x at u periods. */
export function synth(amps: number[], phase: PhaseMode, u: number): number {
  const ph = phase === 'cos' ? Math.PI / 2 : 0;
  let s = 0;
  amps.forEach((a, i) => { s += a * Math.sin(2 * Math.PI * (i + 1) * u + ph); });
  return s;
}

/** One harmonic's contribution at u periods. */
export const component = (a: number, k: number, phase: PhaseMode, u: number): number =>
  a * Math.sin(2 * Math.PI * k * u + (phase === 'cos' ? Math.PI / 2 : 0));

/** n samples of one period starting at u = shift: x[i] = x(i/n + shift). */
export const samplePeriod = (amps: number[], phase: PhaseMode, shift: number, n = NS): number[] =>
  Array.from({ length: n }, (_, i) => synth(amps, phase, i / n + shift));

export interface Bin { k: number; amp: number; phaseDeg: number; }

/** DFT bins k = 1…kMax of x: amplitude 2|X_k|/n and phase ∠X_k (0 when the amplitude is ~0). */
export function dft(x: number[], kMax: number): Bin[] {
  const n = x.length;
  return Array.from({ length: kMax }, (_, j) => {
    const k = j + 1;
    let re = 0, im = 0;
    x.forEach((v, i) => {
      const ang = (-2 * Math.PI * k * i) / n;
      re += v * Math.cos(ang);
      im += v * Math.sin(ang);
    });
    const amp = (2 * Math.hypot(re, im)) / n;
    return { k, amp, phaseDeg: amp > 1e-9 ? (Math.atan2(im, re) * 180) / Math.PI : 0 };
  });
}

/** The ideal (infinite-series) waveform each sine-phase preset truncates, at u periods. */
export function ideal(kind: IdealKind, u: number): number {
  const w = u - Math.floor(u); // position within the period, [0, 1)
  switch (kind) {
    case 'sine': return Math.sin(2 * Math.PI * u);
    case 'square': return w === 0 || w === 0.5 ? 0 : w < 0.5 ? Math.PI / 4 : -Math.PI / 4;
    case 'sawtooth': return w === 0 ? 0 : (Math.PI / 2) * (1 - 2 * w);
    case 'triangle': {
      const tri = w < 0.25 ? 4 * w : w < 0.75 ? 2 - 4 * w : 4 * w - 4;
      return ((Math.PI * Math.PI) / 8) * tri;
    }
  }
}

/** Plateau (sup) of each ideal wave and the size of its jump (null = continuous). */
export const IDEAL_PLATEAU: Record<IdealKind, number> = { sine: 1, square: Math.PI / 4, sawtooth: Math.PI / 2, triangle: (Math.PI * Math.PI) / 8 };
export const IDEAL_JUMP: Record<IdealKind, number | null> = { sine: null, square: Math.PI / 2, sawtooth: Math.PI, triangle: null };

export interface IdealFit {
  kind: IdealKind;
  peak: number;               // max of the truncated sum over one period
  plateau: number;            // max of the ideal wave
  ratio: number;              // peak / plateau
  overshootPct: number | null; // (peak − plateau) / jump · 100 — the Gibbs overshoot (discontinuous waves only)
  rms: number;                // RMS difference between sum and ideal over one period
}

/** Compare the truncated sum with its ideal wave on the FINE grid (shift-invariant, so measured at shift 0). */
export function idealFit(kind: IdealKind, amps: number[]): IdealFit {
  let peak = -Infinity, se = 0;
  for (let i = 0; i < FINE; i++) {
    const u = i / FINE;
    const x = synth(amps, 'sin', u);
    peak = Math.max(peak, x);
    se += (x - ideal(kind, u)) ** 2;
  }
  const plateau = IDEAL_PLATEAU[kind], jump = IDEAL_JUMP[kind];
  return { kind, peak, plateau, ratio: peak / plateau, overshootPct: jump === null ? null : (100 * (peak - plateau)) / jump, rms: Math.sqrt(se / FINE) };
}

/** Peak |x| of the sum over one period (FINE grid). */
export function peakAbs(amps: number[], phase: PhaseMode): number {
  let p = 0;
  for (let i = 0; i < FINE; i++) p = Math.max(p, Math.abs(synth(amps, phase, i / FINE)));
  return p;
}

/** Axis position of harmonic k: k·f₀ in Hz, or mel(k·f₀). */
export const harmonicPos = (k: number, f0: number, mel: boolean): number => (mel ? hzToMel(k * f0) : k * f0);
