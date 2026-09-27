// Runnable Python exports for the Classic ML labs (template strings — not LLM
// generated). Each embeds the lab's exact data (seeded and rounded to 4
// decimals on screen, plus any points you painted), reads its parameters from
// the live lab state, and runs the same algorithm with the same conventions.
import type { PythonSample } from '../../utils/pythonSamples';
import { mulberry32 } from './rng';
import { seededBlobs, seededTwoClass } from './datasets';
import type { Metric, LP } from './knnCore';
import { makeRegData } from './linregCore';
import type { XY } from './linregCore';
import { initCentroids, KMEANS_TOL } from './kmeansCore';
import type { Init, P } from './kmeansCore';
import { makeCloud } from './pcaCore';

const rows = (r: number[][], perLine = 6) => {
  const lines: string[] = [];
  for (let i = 0; i < r.length; i += perLine) lines.push('    ' + r.slice(i, i + perLine).map((x) => `[${x.join(', ')}]`).join(', ') + ',');
  return `[\n${lines.join('\n')}\n]`;
};
const list = (v: number[]) => `[${v.join(', ')}]`;
/** Shortest round-trip literal, so the script uses exactly the lab's double. */
const num = (v: number) => String(v);

/* ---------------- k-NN ---------------- */

export interface KnnExport { points: LP[]; k: number; metric: Metric; weighted: boolean; query: { x: number; y: number } }

export const knnPython = (e: KnnExport) => `import numpy as np

# k-Nearest Neighbours (from scratch) — mirrors the lab exactly: the lab's ${e.points.length} labelled points
# (including any you painted), its k, metric and vote rule, classified at the lab's current query point.
K = ${e.k}                  # neighbours polled (the lab caps k at the number of points)
METRIC = "${e.metric}"          # l1 = Manhattan, l2 = Euclidean, cheb = Chebyshev (L-infinity)
WEIGHTED = ${e.weighted ? 'True' : 'False'}         # vote weight 1/(d + 1e-9) instead of 1
QUERY = np.array([${num(e.query.x)}, ${num(e.query.y)}])

X = np.array(${rows(e.points.map((p) => [p.x, p.y]))})
y = np.array(${list(e.points.map((p) => p.cls))})


def distances(X, q, metric=METRIC):
    d = X - q
    if metric == "l1":
        return np.abs(d[:, 0]) + np.abs(d[:, 1])
    if metric == "cheb":
        return np.maximum(np.abs(d[:, 0]), np.abs(d[:, 1]))
    return np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])


def predict(X, y, q, k=K, weighted=WEIGHTED):
    d = distances(X, q)
    idx = np.argsort(d, kind="stable")[:k]      # nearest first; equal distances -> lower index first
    votes = {}
    for i in idx:
        votes[y[i]] = votes.get(y[i], 0.0) + (1.0 / (d[i] + 1e-9) if weighted else 1.0)
    best = max(votes.values())
    tied = [c for c, v in votes.items() if v == best]
    winner = next(y[i] for i in idx if y[i] in tied)   # a tie goes to the nearest tied neighbour's class
    return int(winner), best / sum(votes.values()), idx, votes


cls, conf, idx, votes = predict(X, y, QUERY)
print("query", QUERY, "-> class", cls, f"(vote share {conf:.2f})")
print("neighbours (index, class, distance):", [(int(i), int(y[i]), round(float(distances(X, QUERY)[i]), 4)) for i in idx])
print("votes per class:", {int(c): round(float(v), 4) for c, v in votes.items()})
`;

/* ---------------- Linear / polynomial / ridge regression ---------------- */

export interface LinregExport { train: XY[]; test: XY[]; degree: number; alpha: number; ridge: number; noise: number; labEpochs: number }

export const linregPython = (e: LinregExport) => `import numpy as np

# Polynomial / ridge regression by batch gradient descent — mirrors the lab exactly: its ${e.train.length}
# training and ${e.test.length} held-out points, the Legendre basis P1..Pd on x in [-1, 1], the loss
# J = 1/2 * mean((yhat - y)^2), the ridge penalty 1/2 * lambda * ||w||^2 (bias not penalised),
# zero initial weights, and the lab's stopping rules.
DEGREE = ${e.degree}
ALPHA = ${num(e.alpha)}
LAMBDA = ${num(e.ridge)}
NOISE = ${num(e.noise)}            # std of the noise the data were generated with
LAB_EPOCHS = ${e.labEpochs}         # epochs the lab had run at download

train = np.array(${rows(e.train.map((p) => [p.x, p.y]))})
test = np.array(${rows(e.test.map((p) => [p.x, p.y]))})
x, y = train[:, 0], train[:, 1]
xt, yt = test[:, 0], test[:, 1]


def truth(x):    # the lab's true curve: 0.1 + 0.5*P1 - 0.45*P2 + 0.35*P3
    return 0.1 + 0.5 * x + (-0.45) * (3 * x * x - 1) / 2 + 0.35 * (5 * x ** 3 - 3 * x) / 2


def features(x, d=DEGREE):
    """Legendre polynomials P1..Pd (Bonnet recurrence, same operation order as the lab)."""
    cols, p0, p1 = [], np.ones_like(x), x.copy()
    for j in range(1, d + 1):
        cols.append(p1)
        p0, p1 = p1, ((2 * j + 1) * x * p1 - j * p0) / (j + 1)
    return np.stack(cols, axis=1)


Phi, Phit = features(x), features(xt)
half_mse = lambda P, t, w, b: 0.5 * np.mean((P @ w + b - t) ** 2)
objective = lambda w, b: half_mse(Phi, y, w, b) + 0.5 * LAMBDA * (w @ w)

# Closed-form optimum of the same objective (normal equations; the bias is not penalised).
A = np.hstack([np.ones((len(x), 1)), Phi])
H = A.T @ A / len(x) + LAMBDA * np.diag([0.0] + [1.0] * DEGREE)
theta = np.linalg.lstsq(H, A.T @ y / len(x), rcond=None)[0]
b_star, w_star = theta[0], theta[1:]
J_star = objective(w_star, b_star)
lam_max = np.linalg.eigvalsh(H).max()
print(f"stable step sizes: alpha < 2/lambda_max = {2 / lam_max:.3f}   (alpha = {ALPHA})")

w, b, epoch, state = np.zeros(DEGREE), 0.0, 0, "max epochs"
with np.errstate(over="ignore", invalid="ignore"):
    while epoch < 20000:
        err = Phi @ w + b - y
        dw = Phi.T @ err / len(x) + LAMBDA * w       # ridge = weight decay
        db = err.mean()
        w, b, epoch = w - ALPHA * dw, b - ALPHA * db, epoch + 1
        obj = objective(w, b)
        if not np.isfinite(obj) or obj > 1e6:
            state = "diverged"
            break
        if obj - J_star < 1e-6:
            state = "converged"
            break
floor = 0.5 * NOISE ** 2
grid = np.linspace(-1, 1, 201)
Pg = features(grid)
print(f"{state} after {epoch} epochs (the lab had run {LAB_EPOCHS})")
if state != "diverged":
    print(f"train J = {half_mse(Phi, y, w, b):.5f}   test J = {half_mse(Phit, yt, w, b):.5f}   noise floor 1/2*sigma^2 = {floor:.5f}")
    print(f"distance from the true curve 1/2*mean((yhat - f)^2) = {0.5 * np.mean((Pg @ w + b - truth(grid)) ** 2):.5f}")
print(f"closed form: train J = {half_mse(Phi, y, w_star, b_star):.5f}   test J = {half_mse(Phit, yt, w_star, b_star):.5f}")
`;

/* ---------------- Logistic regression ---------------- */

export interface LogregExport { data: LP[]; alpha: number; l2: number; separation: number; labEpochs: number }

export const logregPython = (e: LogregExport) => `import numpy as np

# Logistic regression by batch gradient descent — mirrors the lab exactly: its ${e.data.length} points
# (class separation ${e.separation}), zero initial weights, mean cross-entropy gradient, L2 on the weights only.
ALPHA = ${num(e.alpha)}
LAMBDA = ${num(e.l2)}        # L2 penalty (0 = unpenalised)
LAB_EPOCHS = ${e.labEpochs}
EPOCHS = max(LAB_EPOCHS, 600)

D = np.array(${rows(e.data.map((p) => [p.x, p.y, p.cls]))})
X, y = D[:, :2], D[:, 2]
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))


def metrics(w, b):
    p = sigmoid(X @ w + b)
    c = np.clip(p, 1e-9, 1 - 1e-9)
    bce = -np.mean(y * np.log(c) + (1 - y) * np.log(1 - c))
    return bce, np.mean((p > 0.5) == y)


w, b = np.zeros(2), 0.0
for epoch in range(1, EPOCHS + 1):
    p = sigmoid(X @ w + b)
    grad_w = X.T @ (p - y) / len(y) + LAMBDA * w     # cross-entropy gradient + L2 weight decay
    grad_b = np.mean(p - y)
    w, b = w - ALPHA * grad_w, b - ALPHA * grad_b
    if epoch in (100, 500, 1000, 2000, EPOCHS):
        bce, acc = metrics(w, b)
        print(f"epoch {epoch:5d}: BCE {bce:.4f}  accuracy {acc:.3f}  ||w|| {np.linalg.norm(w):.2f}")
print("decision boundary: w1*x1 + w2*x2 + b = 0 with w =", np.round(w, 3), "b =", round(float(b), 3))
`;

/* ---------------- k-means ---------------- */

export interface KmeansExport { points: P[]; k: number; init: Init; seeds: P[] }

export const kmeansPython = (e: KmeansExport) => `import numpy as np

# k-Means (Lloyd's algorithm) — mirrors the lab exactly: its ${e.points.length} points and the ${e.k} seed
# centroids it drew with "${e.init}" initialisation (embedded, so this run retraces the lab's).
K = ${e.k}
INIT = "${e.init}"      # random | kpp (k-means++) | ff (farthest-first)
TOL = ${KMEANS_TOL}         # stop when the centroids move less than this in total

X = np.array(${rows(e.points.map((p) => [p.x, p.y]))})
LAB_SEEDS = np.array(${JSON.stringify(e.seeds.map((c) => [c.x, c.y]))})


def init_centroids(X, k, method, rng):
    """How the seeds are drawn (the lab uses its own seeded generator, so LAB_SEEDS are embedded)."""
    if method == "random":
        return X[rng.permutation(len(X))[:k]].copy()
    if method == "ff":
        # deterministic farthest-first: start at the point farthest from the mean,
        # then repeatedly take the point farthest from every chosen seed
        idx = [int(np.argmax(((X - X.mean(0)) ** 2).sum(1)))]
        while len(idx) < k:
            d2 = np.min([((X - X[i]) ** 2).sum(1) for i in idx], axis=0)
            idx.append(int(np.argmax(d2)))
        return X[idx].copy()
    idx = [int(rng.integers(len(X)))]             # k-means++: sample proportional to squared distance
    while len(idx) < k:
        d2 = np.min([((X - X[i]) ** 2).sum(1) for i in idx], axis=0)
        idx.append(int(rng.choice(len(X), p=d2 / d2.sum())))
    return X[idx].copy()


C = LAB_SEEDS.copy()        # or: init_centroids(X, K, INIT, np.random.default_rng(0))
for it in range(1, 101):
    labels = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), axis=1)        # assignment step
    inertia = ((X - C[labels]) ** 2).sum()
    newC = np.array([X[labels == j].mean(0) if (labels == j).any() else C[j] for j in range(K)])  # update step
    moved = np.sqrt(((newC - C) ** 2).sum(1)).sum()
    print(f"iteration {it}: inertia before update {inertia:.4f}, centroids moved {moved:.5f}")
    C = newC
    if moved < TOL:
        break
labels = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), axis=1)
print("final inertia:", round(float(((X - C[labels]) ** 2).sum()), 4), " cluster sizes:", np.bincount(labels, minlength=K).tolist())
if INIT == "ff":
    print("farthest-first is deterministic — recomputed seeds equal the lab's:", bool(np.allclose(init_centroids(X, K, "ff", None), LAB_SEEDS)))
`;

/* ---------------- PCA ---------------- */

export interface PcaExport { base: [number, number][]; angle: number; elong: number; threshold: number; whiten: boolean; project: boolean }

export const pcaPython = (e: PcaExport) => `import numpy as np

# Principal Component Analysis (2-D) — mirrors the lab exactly: the same ${e.base.length} base normals,
# stretched by (0.98, elongation) and rotated by the lab's angle; population covariance (1/n).
ANGLE = ${num(e.angle)}      # radians
ELONGATION = ${num(e.elong)}
THRESHOLD = ${num(e.threshold)}   # keep the fewest components whose cumulative variance reaches this
WHITEN = ${e.whiten ? 'True' : 'False'}       # the lab's whitened view: z = Lambda^(-1/2) V^T (x - mu)
PROJECT = ${e.project ? 'True' : 'False'}      # the lab's rank-1 projection onto PC1

UV = np.array(${rows(e.base, 8)})
c, s = np.cos(ANGLE), np.sin(ANGLE)
a, b = UV[:, 0] * 0.98, UV[:, 1] * ELONGATION
X = np.stack([0.5 + (a * c - b * s) * 0.13, 0.5 + (a * s + b * c) * 0.13], axis=1)

mu = X.mean(axis=0)
Xc = X - mu
cov = np.cov(Xc, rowvar=False, bias=True)        # 1/n, as in the lab
vals, vecs = np.linalg.eigh(cov)
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]
if vecs[:, 0] @ np.array([c, s]) < 0:            # sign convention: PC1 points along the generator's axis
    vecs[:, 0] *= -1
vecs[:, 1] = [-vecs[1, 0], vecs[0, 0]]           # PC2 = PC1 rotated +90 degrees
explained = vals / vals.sum()
keep = int(np.searchsorted(np.cumsum(explained), THRESHOLD) + 1)
print("PC1", np.round(vecs[:, 0], 4), f"angle {np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])):.1f} deg")
print("eigenvalues", vals, " explained", np.round(explained, 4))
print(f"keep {min(keep, 2)} component(s) for {THRESHOLD:.0%} of the variance")

T = Xc @ vecs                                    # scores along PC1, PC2
Xhat = mu + np.outer(T[:, 0], vecs[:, 0])        # rank-1 reconstruction
print(f"rank-1 reconstruction MSE {np.mean(((X - Xhat) ** 2).sum(1)):.3e} = lambda2 {vals[1]:.3e}")
Z = T / np.sqrt(vals)                            # whitened scores
print("covariance of the whitened scores (identity):\\n", np.round(np.cov(Z, rowvar=False, bias=True), 10))
if PROJECT:
    print("projected (1-D) coordinates, first five:", np.round((Z[:, 0] if WHITEN else T[:, 0])[:5], 4))
`;

/* ---------------- samples for scripts/check-python-exports.mjs ---------------- */

const KNN_CENTERS = [{ x: 0.25, y: 0.3 }, { x: 0.72, y: 0.35 }, { x: 0.5, y: 0.75 }];
const knnSample = (perClass: number, k: number, metric: Metric, weighted: boolean, q = { x: 0.5, y: 0.5 }) => {
  const pts = seededBlobs(mulberry32(1), KNN_CENTERS, 0.1, perClass);
  return knnPython({ points: pts, k: Math.min(k, pts.length), metric, weighted, query: q });
};
const linregSample = (degree: number, alpha: number, ridge: number, n: number, noise: number, labEpochs = 0) => {
  const { train, test } = makeRegData(mulberry32(1), n, 60, noise);
  return linregPython({ train, test, degree, alpha, ridge, noise, labEpochs });
};
const logregSample = (separation: number, alpha: number, l2: number, perClass = 40, labEpochs = 0) =>
  logregPython({ data: seededTwoClass(mulberry32(1), perClass, separation), alpha, l2, separation, labEpochs });
const KM_CENTERS = [{ x: 0.25, y: 0.3 }, { x: 0.75, y: 0.3 }, { x: 0.3, y: 0.74 }, { x: 0.72, y: 0.72 }];
const kmeansSample = (total: number, k: number, init: Init, initSeed = 1) => {
  const pts = seededBlobs(mulberry32(1), KM_CENTERS, 0.075, Math.max(1, Math.round(total / 4))).map((p) => ({ x: p.x, y: p.y }));
  // the lab's seeding stream for (data seed 1, initialisation seed)
  return kmeansPython({ points: pts, k, init, seeds: initCentroids(pts, k, init, mulberry32(104729 * 1 + initSeed)) });
};
const pcaSample = (n: number, angle: number, elong: number, threshold: number, whiten: boolean, project: boolean) =>
  pcaPython({ base: makeCloud(mulberry32(1), n), angle, elong, threshold, whiten, project });

export const PYTHON_SAMPLES: PythonSample[] = [
  // k-NN — default, every preset, a painted-in-a-tie query, slider edges
  { name: 'knn-default', code: () => knnSample(14, 5, 'l2', false) },
  { name: 'knn-overfit-k1', code: () => knnSample(14, 1, 'l2', false, { x: 0.31, y: 0.62 }) },
  { name: 'knn-smooth-k19', code: () => knnSample(14, 19, 'l2', false) },
  { name: 'knn-manhattan', code: () => knnSample(14, 7, 'l1', false, { x: 0.4, y: 0.45 }) },
  { name: 'knn-weighted', code: () => knnSample(14, 13, 'l2', true) },
  { name: 'knn-chebyshev', code: () => knnSample(14, 6, 'cheb', false, { x: 0.6, y: 0.55 }) },
  { name: 'knn-edge-k-equals-n', code: () => knnSample(5, 25, 'l2', false) },
  { name: 'knn-edge-large', code: () => knnSample(30, 25, 'l1', true, { x: 0.97, y: 0.03 }) },
  // Linear regression — every preset + edges (degree 1..12, zero noise, largest alpha)
  { name: 'linreg-line', code: () => linregSample(1, 0.5, 0, 45, 0.1) },
  { name: 'linreg-cubic', code: () => linregSample(3, 1.0, 0, 45, 0.1) },
  { name: 'linreg-overfit-deg12', code: () => linregSample(12, 1.5, 0, 15, 0.1, 1200) },
  { name: 'linreg-ridge-deg12', code: () => linregSample(12, 1.5, 0.01, 15, 0.1) },
  { name: 'linreg-diverge', code: () => linregSample(3, 2.4, 0, 45, 0.1) },
  { name: 'linreg-edge-noiseless', code: () => linregSample(2, 0.01, 0.2, 10, 0) },
  { name: 'linreg-edge-alpha-max', code: () => linregSample(8, 2.5, 0.05, 80, 0.3) },
  // Logistic regression — every preset + edges (identical classes, max alpha)
  { name: 'logreg-default', code: () => logregSample(0.6, 0.5, 0) },
  { name: 'logreg-easy', code: () => logregSample(0.9, 0.6, 0) },
  { name: 'logreg-overlap', code: () => logregSample(0.3, 0.5, 0) },
  { name: 'logreg-blowup', code: () => logregSample(0.9, 1.2, 0, 40, 2000) },
  { name: 'logreg-l2', code: () => logregSample(0.9, 1.2, 0.05) },
  { name: 'logreg-edge-no-separation', code: () => logregSample(0, 2, 0.2, 10) },
  { name: 'logreg-edge-large', code: () => logregSample(1, 0.05, 0, 80) },
  // k-means — every initialisation and preset, k edges
  { name: 'kmeans-kpp', code: () => kmeansSample(160, 4, 'kpp') },
  { name: 'kmeans-random-stuck', code: () => kmeansSample(160, 4, 'random', 2) },
  { name: 'kmeans-ff', code: () => kmeansSample(160, 4, 'ff') },
  { name: 'kmeans-toomany', code: () => kmeansSample(160, 6, 'kpp') },
  { name: 'kmeans-edge-k2-small', code: () => kmeansSample(60, 2, 'random') },
  { name: 'kmeans-edge-k6-large', code: () => kmeansSample(280, 6, 'ff') },
  // PCA — default, every preset / view, slider edges
  { name: 'pca-default', code: () => pcaSample(180, 0.5, 0.25, 0.9, false, false) },
  { name: 'pca-thin', code: () => pcaSample(180, 0.5, 0.18, 0.9, false, false) },
  { name: 'pca-round', code: () => pcaSample(180, 0.5, 0.75, 0.9, false, false) },
  { name: 'pca-whitened', code: () => pcaSample(180, 1.3, 0.34, 0.95, true, false) },
  { name: 'pca-strict', code: () => pcaSample(180, 0.5, 0.34, 0.99, false, true) },
  { name: 'pca-whiten-project', code: () => pcaSample(60, 6.28, 0.1, 0.5, true, true) },
  { name: 'pca-edge-large', code: () => pcaSample(320, 0, 0.95, 0.99, false, true) },
];
