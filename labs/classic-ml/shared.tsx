import React from 'react';

// Shared helpers for the Classic ML labs: 2-D synthetic data generation + the
// right-column parameter wrappers (same look as the RL labs' params section).

export interface Pt { x: number; y: number; cls: number; }

export const clamp01 = (v: number) => Math.max(0.03, Math.min(0.97, v));

/** Standard normal sample (Box–Muller). */
export function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Gaussian blobs, one class per centre. */
export function makeBlobs(centers: { x: number; y: number }[], spread: number, perClass: number): Pt[] {
  const pts: Pt[] = [];
  centers.forEach((c, cls) => {
    for (let i = 0; i < perClass; i++) {
      pts.push({ x: clamp01(c.x + randn() * spread), y: clamp01(c.y + randn() * spread), cls });
    }
  });
  return pts;
}

/** Linearly separable-ish 2-class data around a diagonal, with noise. */
export function makeTwoClass(perClass: number, separation: number): Pt[] {
  const pts: Pt[] = [];
  const off = 0.16 + separation * 0.18;
  for (let i = 0; i < perClass; i++) {
    pts.push({ x: clamp01(0.5 - off + randn() * 0.12), y: clamp01(0.5 - off + randn() * 0.12), cls: 0 });
    pts.push({ x: clamp01(0.5 + off + randn() * 0.12), y: clamp01(0.5 + off + randn() * 0.12), cls: 1 });
  }
  return pts;
}

export const ParamsWrap: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>{children}</div>
);

export const ParamsHead: React.FC<{ title: string; hint: string }> = ({ title, hint }) => (
  <div>
    <h3 style={{ fontFamily: 'var(--disp)', fontSize: 17, color: 'var(--t0)', margin: '0 0 4px' }}>{title}</h3>
    <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', margin: 0, letterSpacing: '.03em' }}>{hint}</p>
  </div>
);
