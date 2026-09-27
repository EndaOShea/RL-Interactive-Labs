// Seeded 2-D datasets for the Classic ML labs. Pure module. Coordinates are
// rounded to 4 decimals (q4) so the Python exports embed the lab's exact data.
import type { Rng } from './rng';
import { gauss, q4 } from './rng';

export interface LPt { x: number; y: number; cls: number; }

/** Same view clamp as shared.tsx's clamp01: keeps points inside the plot. */
export const clampView = (v: number) => Math.max(0.03, Math.min(0.97, v));

/** Gaussian blobs, one class per centre (the recipe of shared.tsx's makeBlobs, seeded). */
export function seededBlobs(r: Rng, centers: { x: number; y: number }[], spread: number, perClass: number): LPt[] {
  const pts: LPt[] = [];
  centers.forEach((c, cls) => {
    for (let i = 0; i < perClass; i++) {
      pts.push({ x: q4(clampView(c.x + gauss(r) * spread)), y: q4(clampView(c.y + gauss(r) * spread)), cls });
    }
  });
  return pts;
}

/** Two-class data: class 0 around (0.5−o, 0.5−o), class 1 around (0.5+o, 0.5+o), σ = TWO_CLASS_SIGMA per axis. */
export const TWO_CLASS_SIGMA = 0.12;
/** Offset o = TWO_CLASS_SPAN × separation, so separation 0 puts both classes on the same centre. */
export const TWO_CLASS_SPAN = 0.34;

export function seededTwoClass(r: Rng, perClass: number, separation: number): LPt[] {
  const pts: LPt[] = [];
  const off = TWO_CLASS_SPAN * separation;
  for (let i = 0; i < perClass; i++) {
    pts.push({ x: q4(clampView(0.5 - off + gauss(r) * TWO_CLASS_SIGMA)), y: q4(clampView(0.5 - off + gauss(r) * TWO_CLASS_SIGMA)), cls: 0 });
    pts.push({ x: q4(clampView(0.5 + off + gauss(r) * TWO_CLASS_SIGMA)), y: q4(clampView(0.5 + off + gauss(r) * TWO_CLASS_SIGMA)), cls: 1 });
  }
  return pts;
}

/** Bayes error of the two-class generator (ignoring the view clamp): Φ(−o·√2/σ). */
export function twoClassBayesError(separation: number) {
  const z = (TWO_CLASS_SPAN * separation * Math.SQRT2) / TWO_CLASS_SIGMA;
  // Φ(−z) via the complementary error function (Abramowitz–Stegun 7.1.26)
  const t = 1 / (1 + 0.3275911 * (z / Math.SQRT2));
  const erfc = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-(z * z) / 2);
  return 0.5 * erfc;
}
