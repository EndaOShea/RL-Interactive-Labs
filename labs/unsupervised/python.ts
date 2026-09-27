// Runnable Python exports for the Unsupervised Learning labs. Each one embeds
// the lab's exact points (seeded and rounded to 4 decimals on screen) and the
// parameters read from the live lab state, runs the same algorithm with the
// same conventions, and prints whether it reproduces the lab's result.
import type { PythonSample } from '../../utils/pythonSamples';
import { densityData } from './unsupData';
import type { DensityKind } from './unsupData';
import { gmmData, hierData } from './unsupData';
import type { HierKind } from './unsupData';
import { dbscan, optics, extractDbscan, extractXi } from './density';
import { initGmm, REG_COVAR, TOL, STARVED, INIT_VAR, PARAMS_PER_COMP } from './gmmCore';
import type { CovType, GmmParams } from './gmmCore';
import { agglomerate, flatClusters, autoCut, largestGapCut } from './agglomerative';
import type { Linkage } from './agglomerative';
import { mulberry32 } from '../classic-ml/rng';

const pyRows = (rows: number[][], perLine = 6) => {
  const lines: string[] = [];
  for (let i = 0; i < rows.length; i += perLine) lines.push('    ' + rows.slice(i, i + perLine).map((r) => `[${r.join(', ')}]`).join(', ') + ',');
  return `[\n${lines.join('\n')}\n]`;
};
const pyList = (v: number[]) => `[${v.join(', ')}]`;
/** Shortest round-trip literal: the script uses exactly the lab's double. */
const num = (v: number) => (Number.isFinite(v) ? String(v) : 'float("inf")');

/* ---------------- DBSCAN / OPTICS ---------------- */

export interface DensityExport {
  dataset: DensityKind;
  seed: number;
  points: number[][];
  mode: 'dbscan' | 'optics';
  eps: number;
  minPts: number;
  extract: 'cut' | 'xi';
  epsPrime: number;
  xi: number;
  minCluster: number;
  /** The lab's final labels (−1 = noise), to check the script reproduces them. */
  labels: number[];
}

export const dbscanPython = (e: DensityExport) => {
  const head = `import numpy as np

# The lab's ${e.points.length} points ("${e.dataset}" dataset, seed ${e.seed}; coordinates as shown on screen).
X = np.array(${pyRows(e.points)})
LAB_LABELS = np.array(${pyList(e.labels)})  # the lab's result (-1 = noise)
`;
  if (e.mode === 'dbscan') {
    return `${head}
# DBSCAN — mirrors the lab: eps = ${num(e.eps)}, min_samples = ${e.minPts}. A core point has at
# least min_samples points within eps COUNTING ITSELF (the lab's and scikit-learn's convention);
# a border point joins the first cluster that reaches it; clusters are numbered in
# order of their lowest-index core point, exactly as the lab numbers them.
from sklearn.cluster import DBSCAN

EPS, MIN_SAMPLES = ${num(e.eps)}, ${e.minPts}
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X)
labels = db.labels_
core = np.zeros(len(X), dtype=bool)
core[db.core_sample_indices_] = True
border = (labels >= 0) & ~core
print(f"clusters: {labels.max() + 1}   noise: {int((labels == -1).sum())}   core: {int(core.sum())}   border: {int(border.sum())}")
print("same labels as the lab:", bool(np.array_equal(labels, LAB_LABELS)))
`;
  }
  const common = `
# OPTICS — mirrors the lab: min_samples = ${e.minPts}, max_eps = ${num(e.eps)} (search radius; core and
# reachability distances beyond it are infinite). The lab's ordering follows scikit-learn's
# conventions exactly (core distance counts the point itself; next point = smallest
# reachability, ties to the lowest index; distances rounded to 15 decimals).
from sklearn.cluster import OPTICS

MIN_SAMPLES, MAX_EPS = ${e.minPts}, ${num(e.eps)}
`;
  if (e.extract === 'cut') {
    return `${head}${common}EPS_PRIME = ${num(e.epsPrime)}  # the flat cut on the reachability plot (ExtractDBSCAN)

opt = OPTICS(min_samples=MIN_SAMPLES, max_eps=MAX_EPS, cluster_method="dbscan", eps=EPS_PRIME).fit(X)
order = opt.ordering_
reach = opt.reachability_[order]          # the reachability plot (inf = start of a new component)
core_d = opt.core_distances_[order]


def extract_dbscan(reach, core_d, eps_prime):
    """ExtractDBSCAN (Ankerst et al.): reach > eps' starts a cluster if core <= eps', else noise;
    reach <= eps' joins the current cluster."""
    out, c = np.full(len(reach), -1), -1
    for k, (r, cd) in enumerate(zip(reach, core_d)):
        if r > eps_prime:
            if cd <= eps_prime:
                c += 1
                out[k] = c
        else:
            out[k] = c
    return out


mine = np.empty(len(X), dtype=int)
mine[order] = extract_dbscan(reach, core_d, EPS_PRIME)
labels = opt.labels_
print(f"clusters: {labels.max() + 1}   noise: {int((labels == -1).sum())}")
print("hand-written ExtractDBSCAN == scikit-learn:", bool(np.array_equal(mine, labels)))
print("same labels as the lab:", bool(np.array_equal(labels, LAB_LABELS)))
`;
  }
  return `${head}${common}XI, MIN_CLUSTER_SIZE = ${num(e.xi)}, ${e.minCluster}  # xi-steep extraction (Ankerst et al.)

opt = OPTICS(min_samples=MIN_SAMPLES, max_eps=MAX_EPS, cluster_method="xi", xi=XI,
             min_cluster_size=MIN_CLUSTER_SIZE).fit(X)
order = opt.ordering_
reach = opt.reachability_[order]          # the reachability plot (valleys = clusters)
labels = opt.labels_                      # leaf clusters of the xi hierarchy; -1 = unassigned
print("xi clusters as [start, end] positions in the ordering:", opt.cluster_hierarchy_.tolist())
print(f"clusters: {labels.max() + 1}   unassigned: {int((labels == -1).sum())}")
print("same labels as the lab:", bool(np.array_equal(labels, LAB_LABELS)))
`;
};

/* ---------------- Gaussian mixture ---------------- */

export interface GmmExport {
  K: number;
  covType: CovType;
  points: number[][];
  /** The lab's initial parameters (k-means++ means, Σ₀ = INIT_VAR·I, π₀ = 1/K). */
  init: GmmParams;
  seed: number;
  /** EM iterations the lab had run when the file was downloaded. */
  labIters: number;
}

export const gmmPython = (e: GmmExport) => {
  const covs = e.init.covs.map(([a, b, c]) => [[a, b], [b, c]]);
  return `import numpy as np

# Gaussian mixture fitted by EM — mirrors the lab exactly: its ${e.points.length} points (seed ${e.seed}),
# its initial parameters (k-means++ means, Sigma0 = ${INIT_VAR}*I, pi0 = 1/K), the same E-step (log-sum-exp),
# M-step, covariance constraint + ${REG_COVAR} on the diagonal, starved-component rule and stopping rule.
# (The lab had run ${e.labIters} EM iteration${e.labIters === 1 ? '' : 's'} when this was downloaded.)
K, COV = ${e.K}, "${e.covType}"       # full: tilted ellipses · diag: axis-aligned · spherical: circles
REG, TOL, STARVED = ${REG_COVAR}, ${TOL}, ${STARVED}
PARAMS_PER_COMP = {"full": ${PARAMS_PER_COMP.full}, "diag": ${PARAMS_PER_COMP.diag}, "spherical": ${PARAMS_PER_COMP.spherical}}   # mean (2) + covariance

X = np.array(${pyRows(e.points)})
MEANS0 = np.array(${JSON.stringify(e.init.means.map((m) => [m.x, m.y]))})
COVS0 = np.array(${JSON.stringify(covs)})
WEIGHTS0 = np.array(${JSON.stringify(e.init.weights)})


def log_gauss(X, mu, S):
    det = S[0, 0] * S[1, 1] - S[0, 1] ** 2
    d = X - mu
    m = (S[1, 1] * d[:, 0] ** 2 - 2 * S[0, 1] * d[:, 0] * d[:, 1] + S[0, 0] * d[:, 1] ** 2) / det
    return -0.5 * m - np.log(2 * np.pi) - 0.5 * np.log(det)


def e_step(X, w, mus, Ss):
    with np.errstate(divide="ignore"):
        L = np.stack([np.log(w[k]) + log_gauss(X, mus[k], Ss[k]) for k in range(len(w))], axis=1)
    m = L.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(L - m).sum(axis=1))      # log-sum-exp
    return np.exp(L - lse[:, None]), lse.sum()               # responsibilities, total log-likelihood


def constrain(a, b, c):
    if COV == "spherical":
        s = (a + c) / 2 + REG
        return np.array([[s, 0.0], [0.0, s]])
    if COV == "diag":
        return np.array([[a + REG, 0.0], [0.0, c + REG]])
    return np.array([[a + REG, b], [b, c + REG]])


def m_step(X, R, mus, Ss):
    n = len(X)
    w, new_mu, new_S = np.zeros(K), mus.copy(), Ss.copy()
    for k in range(K):
        r = R[:, k]
        Nk = r.sum()
        w[k] = Nk / n
        if Nk < STARVED:                  # a starved component keeps its previous mean / covariance
            continue
        mu = (r[:, None] * X).sum(axis=0) / Nk
        d = X - mu
        a, b, c = (r * d[:, 0] ** 2).sum() / Nk, (r * d[:, 0] * d[:, 1]).sum() / Nk, (r * d[:, 1] ** 2).sum() / Nk
        new_mu[k], new_S[k] = mu, constrain(a, b, c)
    return w, new_mu, new_S


w, mus, Ss = WEIGHTS0.copy(), MEANS0.copy(), COVS0.copy()
prev, it = -np.inf, 0
while it < 1000:
    R, ll = e_step(X, w, mus, Ss)
    w, mus, Ss = m_step(X, R, mus, Ss)
    it += 1
    if abs(ll - prev) < TOL:              # |delta log-likelihood| of the total, as in the lab
        break
    prev = ll
R, ll = e_step(X, w, mus, Ss)
n = len(X)
p = (K - 1) + K * PARAMS_PER_COMP[COV]
bic = -2 * ll + p * np.log(n)
print(f"converged after {it} iterations")
print("weights:", np.round(w, 4))
print("means:\\n", np.round(mus, 4))
print(f"log-likelihood: {ll:.3f}   free parameters: {p}   BIC: {bic:.2f}")
print("points with max responsibility < 0.9:", int((R.max(axis=1) < 0.9).sum()))

# Cross-check with scikit-learn from the SAME start (its tol is per sample, hence TOL / n).
from sklearn.mixture import GaussianMixture

if COV == "full":
    prec0 = np.linalg.inv(COVS0)
elif COV == "diag":
    prec0 = 1.0 / np.stack([np.diag(S) for S in COVS0])
else:
    prec0 = 1.0 / COVS0[:, 0, 0]
gm = GaussianMixture(n_components=K, covariance_type=COV, weights_init=WEIGHTS0, means_init=MEANS0,
                     precisions_init=prec0, reg_covar=REG, tol=TOL / n, max_iter=1000).fit(X)
print(f"scikit-learn from the same start: BIC {gm.bic(X):.2f}   log-likelihood {gm.score(X) * n:.3f}")
`;
};

/* ---------------- Hierarchical ---------------- */

export interface HierExport {
  linkage: Linkage;
  points: number[][];
  /** The lab's cut height (lab units: Ward = √ΔSSE). */
  cut: number;
  /** Merges performed in the lab at download time. */
  done: number;
  /** The lab's merge heights in merge order. */
  labHeights: number[];
  /** Clusters the lab showed at this cut. */
  labK: number;
}

export const hierarchicalPython = (e: HierExport) => `import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

# Agglomerative clustering — mirrors the lab (linkage = "${e.linkage}"): the lab's ${e.points.length} points,
# its cut height and the number of merges it had performed.
#   single / complete / average: min / max / mean pairwise distance
#   ward: the lab's height is sqrt(dSSE), the increase in within-cluster sum of squares;
#         SciPy reports sqrt(2*dSSE), so its heights are divided by sqrt(2) below
#   centroid: distance between centroids (can produce inversions)
METHOD = "${e.linkage}"
CUT = ${num(e.cut)}          # lab units
DONE = ${e.done}              # merges performed in the lab (of ${e.points.length - 1})
LAB_K = ${e.labK}
LAB_HEIGHTS = np.array(${pyList(e.labHeights)})

X = np.array(${pyRows(e.points)})
n = len(X)

Z = linkage(X, method=METHOD)
if METHOD == "ward":
    Z[:, 2] /= np.sqrt(2)

# The highest merge inside each subtree: fcluster(criterion="distance") keeps a subtree
# as one flat cluster iff this is <= the cut (matters only for centroid inversions).
max_h = np.zeros(n - 1)
for s, (a, b, h, _) in enumerate(Z):
    max_h[s] = max([h] + [max_h[int(c) - n] for c in (a, b) if c >= n])


def flat_clusters(done, cut):
    """Clusters after the first 'done' merges, cut at 'cut' — the lab's rule."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    members = [[i] for i in range(n)]
    for s in range(len(Z)):
        a, b = int(Z[s, 0]), int(Z[s, 1])
        members.append(members[a] + members[b])
        if s < done and max_h[s] <= cut:
            r0 = find(members[-1][0])
            for i in members[-1]:
                parent[find(i)] = r0
    roots = [find(i) for i in range(n)]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots])


labels = flat_clusters(DONE, CUT)
k = len(set(labels))
print(f"{k} clusters at cut {CUT:.4f} after {DONE} merges; sizes {np.bincount(labels).tolist()}  (lab showed {LAB_K})")
inversions = int((Z[:, 2] < max_h).sum())
print("inversions (merges lower than a merge inside them):", inversions)
print("max |merge height - lab| over all merges:", float(np.max(np.abs(np.sort(Z[:, 2]) - np.sort(LAB_HEIGHTS)))))
if DONE == n - 1:
    ref = fcluster(Z, t=CUT, criterion="distance")
    same = len(set(zip(ref, labels))) == len(set(ref)) == len(set(labels))
    print("same partition as scipy fcluster(criterion='distance'):", same)
# from scipy.cluster.hierarchy import dendrogram; dendrogram(Z)   # plot with matplotlib
`;

/* ---------------- samples for scripts/check-python-exports.mjs ---------------- */

function densitySample(dataset: DensityKind, n: number, seed: number, mode: 'dbscan' | 'optics', eps: number, minPts: number, extract: 'cut' | 'xi', epsPrime: number, xi: number, minCluster: number) {
  const pts = densityData(dataset, n, seed).pts;
  const cut = Math.min(epsPrime, eps);
  const mc = Math.max(2, Math.min(minCluster, pts.length));
  let labels: number[];
  if (mode === 'dbscan') labels = dbscan(pts, eps, minPts).labels;
  else {
    const o = optics(pts, eps, minPts);
    labels = extract === 'cut' ? extractDbscan(o, cut).labels : extractXi(o, minPts, xi, mc).labels;
  }
  return dbscanPython({ dataset, seed, points: pts.map((p) => [p.x, p.y]), mode, eps, minPts, extract, epsPrime: cut, xi, minCluster: mc, labels });
}

function gmmSample(n: number, seed: number, K: number, covType: CovType) {
  const pts = gmmData(n, seed).pts;
  const init = initGmm(pts, K, mulberry32(7919 * seed + 31 * K + 1));
  return gmmPython({ K, covType, points: pts.map((p) => [p.x, p.y]), init, seed, labIters: 0 });
}

function hierSample(kind: HierKind, n: number, seed: number, linkage: Linkage, doneFrac: number, cutMode: 'auto' | 'gap' | number) {
  const pts = hierData(kind, n, seed).pts;
  const a = agglomerate(pts, linkage);
  const done = Math.round(doneFrac * a.merges.length);
  const cut = cutMode === 'auto' ? autoCut(a.merges, done, a.maxHeight) : cutMode === 'gap' ? largestGapCut(a.merges) : cutMode;
  const labK = flatClusters(pts.length, a.merges, done, cut).k;
  return hierarchicalPython({ linkage, points: pts.map((p) => [p.x, p.y]), cut, done, labHeights: a.merges.map((g) => g.height), labK });
}

export const PYTHON_SAMPLES: PythonSample[] = [
  // DBSCAN / OPTICS — defaults, every preset, every dataset / extraction, slider edges
  { name: 'dbscan-default', code: () => densitySample('blobs', 120, 1, 'dbscan', 0.07, 4, 'cut', 0.07, 0.1, 10) },
  { name: 'dbscan-tight-eps', code: () => densitySample('blobs', 120, 1, 'dbscan', 0.04, 4, 'cut', 0.04, 0.1, 10) },
  { name: 'dbscan-greedy-eps', code: () => densitySample('blobs', 120, 1, 'dbscan', 0.25, 4, 'cut', 0.07, 0.1, 10) },
  { name: 'dbscan-moons', code: () => densitySample('moons', 160, 1, 'dbscan', 0.06, 5, 'cut', 0.06, 0.1, 10) },
  { name: 'dbscan-rings', code: () => densitySample('rings', 160, 1, 'dbscan', 0.08, 5, 'cut', 0.08, 0.1, 10) },
  { name: 'dbscan-mixed', code: () => densitySample('mixed', 160, 1, 'dbscan', 0.08, 10, 'cut', 0.08, 0.1, 34) },
  { name: 'optics-mixed-xi', code: () => densitySample('mixed', 160, 1, 'optics', 0.3, 10, 'xi', 0.08, 0.1, 34) },
  { name: 'optics-blobs-cut', code: () => densitySample('blobs', 120, 1, 'optics', 0.2, 4, 'cut', 0.07, 0.1, 10) },
  { name: 'optics-moons-xi-small', code: () => densitySample('moons', 60, 3, 'optics', 0.12, 3, 'xi', 0.05, 0.01, 2) },
  { name: 'optics-rings-cut-max', code: () => densitySample('rings', 220, 2, 'optics', 0.3, 12, 'cut', 0.3, 0.3, 60) },
  { name: 'dbscan-edge-min', code: () => densitySample('blobs', 60, 5, 'dbscan', 0.02, 2, 'cut', 0.02, 0.1, 10) },
  { name: 'dbscan-edge-max', code: () => densitySample('mixed', 220, 7, 'dbscan', 0.3, 12, 'cut', 0.3, 0.1, 10) },
  // GMM — every preset, every covariance family, K and size edges
  { name: 'gmm-default-full3', code: () => gmmSample(160, 1, 3, 'full') },
  { name: 'gmm-diag3', code: () => gmmSample(160, 1, 3, 'diag') },
  { name: 'gmm-spherical3', code: () => gmmSample(160, 1, 3, 'spherical') },
  { name: 'gmm-underfit-k2', code: () => gmmSample(160, 1, 2, 'full') },
  { name: 'gmm-overfit-k5', code: () => gmmSample(160, 1, 5, 'full') },
  { name: 'gmm-edge-small-diag5', code: () => gmmSample(80, 4, 5, 'diag') },
  { name: 'gmm-edge-large-sph2', code: () => gmmSample(280, 9, 2, 'spherical') },
  // Hierarchical — every linkage on both datasets, mid-build cuts, manual cuts, size edges
  ...(['single', 'complete', 'average', 'ward', 'centroid'] as Linkage[]).flatMap((l) => [
    { name: `hier-blobs-${l}-done`, code: () => hierSample('blobs', 32, 1, l, 1, 'auto') },
    { name: `hier-bridge-${l}-gap`, code: () => hierSample('bridge', 32, 1, l, 1, 'gap') },
  ]),
  { name: 'hier-centroid-preset', code: () => hierSample('blobs', 28, 1, 'centroid', 1, 'gap') },
  { name: 'hier-midbuild-auto', code: () => hierSample('bridge', 32, 2, 'average', 0.5, 'auto') },
  { name: 'hier-start', code: () => hierSample('blobs', 16, 3, 'single', 0, 'auto') },
  { name: 'hier-manual-cut-ward', code: () => hierSample('blobs', 40, 4, 'ward', 1, 0.08) },
  { name: 'hier-manual-cut-partial', code: () => hierSample('bridge', 40, 5, 'complete', 0.8, 0.2) },
];
