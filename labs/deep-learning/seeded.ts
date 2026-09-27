// Seeded randomness for the Deep Learning labs. Every lab draws its data,
// weights and dropout masks from this generator, so a run is reproducible and
// the Python exports (python.ts ports the same generator, PY_RNG) rebuild the
// exact same numbers from the same seed. Pure module, no React.

export type Rng = () => number;

/** mulberry32: a small 32-bit PRNG returning uniforms in [0, 1). */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal via Box–Muller: consumes exactly two uniforms per sample. */
export function gauss(r: Rng): number {
  const u = 1 - r();           // (0, 1] — never log(0)
  const v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Fisher–Yates shuffle of 0..n-1 driven by the seeded generator. */
export function permutation(n: number, r: Rng): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    const t = idx[i]!; idx[i] = idx[j]!; idx[j] = t;
  }
  return idx;
}
