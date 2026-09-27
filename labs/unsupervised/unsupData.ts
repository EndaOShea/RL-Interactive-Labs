// Seeded synthetic datasets for the Unsupervised Learning labs. Pure module.
// Every coordinate is rounded to 4 decimals (q4) so the Python exports embed the
// lab's exact points. `truth` is the generating component of each point (−1 =
// background noise / bridge points); the labs never cluster with it — it exists
// so the claims about each dataset can be checked.
import { mulberry32, gauss, q4 } from '../classic-ml/rng';
import type { Rng } from '../classic-ml/rng';
import type { UPt } from './shared';

export interface Dataset { pts: UPt[]; truth: number[] }

const clampView = (v: number) => Math.max(0.03, Math.min(0.97, v));

/** Round Gaussian blobs (same recipe as the classic-ml makeBlobs: clamped to [0.03, 0.97]). */
function roundBlobs(r: Rng, centers: UPt[], spread: number, perBlob: number, out: Dataset) {
  centers.forEach((c, k) => {
    for (let i = 0; i < perBlob; i++) {
      out.pts.push({ x: q4(clampView(c.x + gauss(r) * spread)), y: q4(clampView(c.y + gauss(r) * spread)) });
      out.truth.push(k);
    }
  });
}

/* ---------------- DBSCAN / OPTICS datasets ---------------- */

export type DensityKind = 'blobs' | 'mixed' | 'moons' | 'rings';

export const DENSITY_BLOB_CENTERS: UPt[] = [{ x: 0.28, y: 0.3 }, { x: 0.72, y: 0.32 }, { x: 0.5, y: 0.72 }];
export const DENSITY_BLOB_SIGMA = 0.06;
/** Mixed density: two tight blobs side by side + one sparse blob (+ a little background noise). */
export const MIXED = {
  tight: [{ x: 0.224, y: 0.72 }, { x: 0.296, y: 0.72 }] as UPt[],
  tightSigma: 0.012,
  tightShare: 0.24,
  sparse: { x: 0.62, y: 0.36 } as UPt,
  sparseSigma: 0.1,
  sparseShare: 0.47,
  noiseShare: 0.05,
};

function uniformNoise(r: Rng, count: number, out: Dataset) {
  for (let i = 0; i < count; i++) {
    out.pts.push({ x: q4(r() * 0.9 + 0.05), y: q4(r() * 0.9 + 0.05) });
    out.truth.push(-1);
  }
}

export function densityData(kind: DensityKind, n: number, seed: number): Dataset {
  const r = mulberry32(seed);
  const out: Dataset = { pts: [], truth: [] };
  if (kind === 'blobs') {
    // three σ = 0.06 blobs (28% of n each) + 12% uniform background noise
    roundBlobs(r, DENSITY_BLOB_CENTERS, DENSITY_BLOB_SIGMA, Math.max(4, Math.round(n * 0.28)), out);
    uniformNoise(r, Math.round(n * 0.12), out);
  } else if (kind === 'mixed') {
    // two tight blobs close together + one sparse blob + a little background noise
    roundBlobs(r, MIXED.tight, MIXED.tightSigma, Math.max(4, Math.round(n * MIXED.tightShare)), out);
    const sparse = Math.max(4, Math.round(n * MIXED.sparseShare));
    for (let i = 0; i < sparse; i++) {
      out.pts.push({ x: q4(clampView(MIXED.sparse.x + gauss(r) * MIXED.sparseSigma)), y: q4(clampView(MIXED.sparse.y + gauss(r) * MIXED.sparseSigma)) });
      out.truth.push(2);
    }
    uniformNoise(r, Math.round(n * MIXED.noiseShare), out);
  } else if (kind === 'moons') {
    // two interleaving half-moons (47% of n each; stratified along the arc, jitter σ 0.015) + 6% noise
    const per = Math.max(4, Math.round(n * 0.47));
    const s = 0.8 / 3;
    for (let m = 0; m < 2; m++) {
      for (let i = 0; i < per; i++) {
        const t = (Math.PI * (i + r())) / per;
        const x = m === 0 ? Math.cos(t) : 1 - Math.cos(t);
        const y = m === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
        out.pts.push({ x: q4(0.1 + (x + 1) * s + gauss(r) * 0.015), y: q4(0.3 + (y + 0.5) * s + gauss(r) * 0.015) });
        out.truth.push(m);
      }
    }
    uniformNoise(r, Math.round(n * 0.06), out);
  } else {
    // two concentric rings (radii 0.14 / 0.34; 30% / 64% of n, stratified, jitter σ 0.015) + 6% noise
    const rings: [number, number][] = [[0.14, Math.max(4, Math.round(n * 0.3))], [0.34, Math.max(4, Math.round(n * 0.64))]];
    rings.forEach(([rad, count], m) => {
      for (let i = 0; i < count; i++) {
        const t = (2 * Math.PI * (i + r())) / count;
        out.pts.push({ x: q4(0.5 + rad * Math.cos(t) + gauss(r) * 0.015), y: q4(0.5 + rad * Math.sin(t) + gauss(r) * 0.015) });
        out.truth.push(m);
      }
    });
    uniformNoise(r, Math.round(n * 0.06), out);
  }
  return out;
}

/* ---------------- Gaussian-mixture data ---------------- */

/** Three tilted / elongated Gaussian components: a genuinely full-covariance mixture. */
export const GMM_TRUTH = [
  { mx: 0.37, my: 0.6, s1: 0.12, s2: 0.038, angleDeg: 55, weight: 0.36 },
  { mx: 0.63, my: 0.6, s1: 0.12, s2: 0.038, angleDeg: -55, weight: 0.36 },
  { mx: 0.5, my: 0.24, s1: 0.13, s2: 0.035, angleDeg: 0, weight: 0.28 },
];

export function gmmData(n: number, seed: number): Dataset {
  const r = mulberry32(seed);
  const out: Dataset = { pts: [], truth: [] };
  GMM_TRUTH.forEach((c, k) => {
    const count = Math.max(3, Math.round(n * c.weight));
    const a = (c.angleDeg * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
    for (let i = 0; i < count; i++) {
      // redraw the rare point that would fall outside the view (a truncated Gaussian)
      for (let tries = 0; tries < 50; tries++) {
        const u = gauss(r) * c.s1, v = gauss(r) * c.s2;
        const x = c.mx + u * ca - v * sa, y = c.my + u * sa + v * ca;
        if (x >= 0.02 && x <= 0.98 && y >= 0.02 && y <= 0.98) { out.pts.push({ x: q4(x), y: q4(y) }); out.truth.push(k); break; }
      }
    }
  });
  return out;
}

/* ---------------- Hierarchical-clustering data ---------------- */

export type HierKind = 'blobs' | 'bridge';
export const HIER_BLOB_CENTERS: UPt[] = [{ x: 0.28, y: 0.3 }, { x: 0.72, y: 0.3 }, { x: 0.3, y: 0.72 }, { x: 0.72, y: 0.72 }];

/** 'bridge' recipe: two compact blobs A, B joined by a chain of bridge points, a long thin cluster C, one outlier. */
export const BRIDGE = {
  A: { x: 0.2, y: 0.74 } as UPt,
  B: { x: 0.62, y: 0.74 } as UPt,
  blobSigma: 0.035,
  blobShare: 0.28,
  bridge: 5,
  bridgeJitter: 0.006,
  L0: { x: 0.1, y: 0.2 } as UPt,
  L1: { x: 0.9, y: 0.3 } as UPt,
  lineJitter: 0.01,
  outlier: { x: 0.9, y: 0.93 } as UPt | null,
};

/**
 * 'blobs'  — four round σ = 0.06 blobs.
 * 'bridge' — two compact blobs joined by a sparse chain of bridge points, one
 *            long thin cluster and one far outlier: shapes on which linkages
 *            genuinely disagree (single chains through the bridge and isolates
 *            the outlier; complete / Ward prefer compact groups and cut the long one).
 */
export function hierData(kind: HierKind, n: number, seed: number): Dataset {
  const r = mulberry32(seed);
  const out: Dataset = { pts: [], truth: [] };
  if (kind === 'blobs') {
    roundBlobs(r, HIER_BLOB_CENTERS, 0.06, Math.max(2, Math.round(n / 4)), out);
    return out;
  }
  const { A, B, L0, L1 } = BRIDGE;
  const extra = BRIDGE.bridge + (BRIDGE.outlier ? 1 : 0);
  const blob = Math.max(3, Math.round((n - extra) * BRIDGE.blobShare));
  const line = Math.max(4, n - extra - 2 * blob);
  roundBlobs(r, [A, B], BRIDGE.blobSigma, blob, out);
  for (let i = 1; i <= BRIDGE.bridge; i++) {
    const t = i / (BRIDGE.bridge + 1);
    out.pts.push({ x: q4(A.x + (B.x - A.x) * t + gauss(r) * BRIDGE.bridgeJitter), y: q4(A.y + (B.y - A.y) * t + gauss(r) * BRIDGE.bridgeJitter) });
    out.truth.push(-1);
  }
  for (let i = 0; i < line; i++) {
    const t = (i + 0.5) / line + (r() - 0.5) * (0.6 / line);
    out.pts.push({ x: q4(L0.x + (L1.x - L0.x) * t + gauss(r) * BRIDGE.lineJitter), y: q4(L0.y + (L1.y - L0.y) * t + gauss(r) * BRIDGE.lineJitter) });
    out.truth.push(2);
  }
  if (BRIDGE.outlier) { out.pts.push({ x: BRIDGE.outlier.x, y: BRIDGE.outlier.y }); out.truth.push(-1); }
  return out;
}
