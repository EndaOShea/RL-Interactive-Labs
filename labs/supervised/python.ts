// Runnable Python exports for the Supervised Learning labs. Each script embeds
// the exact points the lab trained on (every point is stored at 4 decimals, so
// the numbers are identical), ports the lab's algorithm line for line (same
// loops, tie-breaking and parameters, read from the live lab state), prints the
// same quantities the lab shows, and then cross-checks against scikit-learn
// where an equivalent estimator exists.
import type { PythonSample } from '../../utils/pythonSamples';
import type { XorPt, SvmPt, NbPt, XorLayout, SvmShape } from './supData';
import type { Crit } from './treeCore';
import type { Kernel } from './svmSolver';
import type { BoostCfg, GPt } from './boostCore';
import type { NbVariant } from './nbCore';
import { makeXorData, makeSvmData, makeNbData, xorSizes, XOR_LAYOUTS, DT_STD, GBM_STD, TEST_SEED_OFFSET } from './supData';
import { MIN_SPLIT } from './treeCore';
import { SMO_TOL } from './svmSolver';
import { NB_BINS, VAR_FLOOR } from './nbCore';

const num = (v: number) => String(v);
const pyBool = (b: boolean) => (b ? 'True' : 'False');

/** An (n, 2) float array of points, 5 per line. */
function pyXY(name: string, pts: readonly { x: number; y: number }[]): string {
  if (!pts.length) return `${name} = np.zeros((0, 2))`;
  const rows: string[] = [];
  for (let i = 0; i < pts.length; i += 5) rows.push('    ' + pts.slice(i, i + 5).map((p) => `[${num(p.x)}, ${num(p.y)}]`).join(', '));
  return `${name} = np.array([\n${rows.join(',\n')},\n], dtype=float)`;
}
/** A 1-D integer array, 30 per line. */
function pyInts(name: string, v: readonly number[]): string {
  const rows: string[] = [];
  for (let i = 0; i < v.length; i += 30) rows.push('    ' + v.slice(i, i + 30).join(', '));
  return `${name} = np.array([\n${rows.join(',\n')},\n])`;
}

/* ─────────────────────────────── Decision tree ─────────────────────────────── */

export interface DtExport { data: readonly XorPt[]; test: readonly XorPt[]; depth: number; crit: Crit; minLeaf: number; layout: XorLayout; noise: number; seed: number; }

export const decisionTreePython = (o: DtExport) => `import math
import numpy as np

# Decision tree (CART) — a line-for-line port of the lab (labs/supervised/treeCore.ts),
# trained on the exact points the lab shows.
# Data: ${o.layout} XOR, ${o.data.length} training points (${Math.round(o.noise * 100)}% of labels flipped, seed ${o.seed}),
#       ${o.test.length} clean held-out test points (seed ${o.seed + TEST_SEED_OFFSET}).
CRITERION = "${o.crit}"
MAX_DEPTH = ${o.depth}
MIN_LEAF = ${o.minLeaf}
MIN_SPLIT = ${MIN_SPLIT}   # a node with fewer than max(MIN_SPLIT, 2 * MIN_LEAF) points is never split

${pyXY('X_train', o.data)}
${pyInts('y_train', o.data.map((p) => p.cls))}
${pyXY('X_test', o.test)}
${pyInts('y_test', o.test.map((p) => p.cls))}


def impurity(c0, c1):
    """Gini 1 - sum p^2 or entropy -sum p log2 p of the class counts (class 0 first)."""
    n = c0 + c1
    if n == 0:
        return 0.0
    v = 1.0 if CRITERION == "gini" else 0.0
    for k in (c0, c1):
        if k == 0:
            continue
        p = k / n
        v = v - p * p if CRITERION == "gini" else v - p * math.log2(p)
    return v


def best_split(X, y):
    """Scan midpoints between consecutive distinct values; x <= thr goes left; first best wins."""
    n = len(y)
    labels = y.tolist()
    best = None
    for feat in (0, 1):
        col = X[:, feat].tolist()
        vals = sorted(set(col))
        for i in range(len(vals) - 1):
            thr = (vals[i] + vals[i + 1]) / 2
            l0 = l1 = r0 = r1 = 0
            for v, c in zip(col, labels):
                if v <= thr:
                    if c == 0:
                        l0 += 1
                    else:
                        l1 += 1
                elif c == 0:
                    r0 += 1
                else:
                    r1 += 1
            nl, nr = l0 + l1, r0 + r1
            if nl < MIN_LEAF or nr < MIN_LEAF:
                continue
            score = (nl * impurity(l0, l1) + nr * impurity(r0, r1)) / n
            if best is None or score < best[2]:
                best = (feat, thr, score)
    return best


def build(X, y, depth=0):
    n = len(y)
    c1 = int(y.sum())
    c0 = n - c1
    imp = impurity(c0, c1)
    leaf = {"leaf": True, "cls": 1 if c1 > c0 else 0, "n": n, "counts": (c0, c1)}
    if depth >= MAX_DEPTH or imp < 1e-9 or n < max(MIN_SPLIT, 2 * MIN_LEAF):
        return leaf
    best = best_split(X, y)
    if best is None or best[2] >= imp - 1e-9:
        return leaf           # no cut lowers the impurity
    feat, thr, score = best
    left = X[:, feat] <= thr
    return {"leaf": False, "feat": feat, "thr": thr, "gain": imp - score, "n": n,
            "left": build(X[left], y[left], depth + 1), "right": build(X[~left], y[~left], depth + 1)}


def predict_one(node, x):
    while not node["leaf"]:
        node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
    return node["cls"]


def predict(node, X):
    return np.array([predict_one(node, x) for x in X])


def show(node, indent=""):
    if node["leaf"]:
        print(f"{indent}class {node['cls']}   n={node['n']} counts={node['counts']}")
        return
    name = "x1" if node["feat"] == 0 else "x2"
    print(f"{indent}{name} <= {node['thr']:.4f}   (gain {node['gain']:.4f}, n={node['n']})")
    show(node["left"], indent + "|   ")
    show(node["right"], indent + "|   ")


def count(node):
    if node["leaf"]:
        return 1, 1, 0
    ln, ll, ld = count(node["left"])
    rn, rl, rd = count(node["right"])
    return 1 + ln + rn, ll + rl, 1 + max(ld, rd)


tree = build(X_train, y_train)
nodes, leaves, depth = count(tree)
show(tree)
print(f"nodes {nodes}, leaves {leaves}, depth {depth}")
print(f"train acc {(predict(tree, X_train) == y_train).mean():.3f}   test acc {(predict(tree, X_test) == y_test).mean():.3f}")

# Cross-check with scikit-learn's CART (same criterion, depth and leaf rules). Ties between
# equally good cuts may be broken differently, so a few predictions can differ.
if MAX_DEPTH >= 1:
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion=CRITERION, max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
                                 min_samples_split=max(MIN_SPLIT, 2 * MIN_LEAF), random_state=0).fit(X_train, y_train)
    agree = (clf.predict(X_test) == predict(tree, X_test)).mean()
    print(f"sklearn: {clf.get_n_leaves()} leaves, test acc {clf.score(X_test, y_test):.3f}, agrees with the port on {agree:.1%} of test points")
else:
    print("max depth 0: the tree is a single leaf (scikit-learn needs max_depth >= 1)")
`;

/* ─────────────────────────────────── SVM ─────────────────────────────────── */

export interface SvmExport { data: readonly SvmPt[]; test: readonly SvmPt[]; C: number; kernel: Kernel; gamma: number; degree: number; shape: SvmShape; sep: number; seed: number; }

export const svmPython = (o: SvmExport) => `import math
import numpy as np

# Soft-margin SVM trained by SMO — a line-for-line port of the lab (labs/supervised/svmSolver.ts):
# LIBSVM's second-order working-set selection, closed-form pair update clipped to [0, C],
# stopping when the KKT gap m(a) - M(a) < TOL, bias from LIBSVM's calculate_rho.
# Data: ${o.shape}, separation ${o.sep}, ${o.data.length} training points (seed ${o.seed}),
#       ${o.test.length} held-out test points (seed ${o.seed + TEST_SEED_OFFSET}).
KERNEL = "${o.kernel}"
C = ${num(o.C)}
GAMMA = ${num(o.gamma)}     # RBF: K = exp(-GAMMA * |a - b|^2)
DEGREE = ${o.degree}        # poly: K = (1 + a.b)^DEGREE
TOL = ${SMO_TOL}
TAU = 1e-12

${pyXY('X', o.data)}
${pyInts('y', o.data.map((p) => p.yy))}
${pyXY('X_test', o.test)}
${pyInts('y_test', o.test.map((p) => p.yy))}
y = y.astype(float)


def kernel(A, B):
    """Kernel matrix K[i, j] = K(A[i], B[j]), evaluated in the lab's operation order."""
    ax, ay = A[:, 0:1], A[:, 1:2]
    bx, by = B[:, 0], B[:, 1]
    if KERNEL == "linear":
        return ax * bx + ay * by
    if KERNEL == "poly":
        base = 1.0 + ax * bx + ay * by
        p = np.ones_like(base)
        for _ in range(DEGREE):
            p = p * base
        return p
    dx, dy = ax - bx, ay - by
    return np.exp(-GAMMA * (dx * dx + dy * dy))


n = len(y)
K = kernel(X, X)
alpha = np.zeros(n)
G = -np.ones(n)          # gradient of the dual objective, G = Q alpha - 1


def upper(a):
    return a >= C


def lower(a):
    return a <= 0


def select_pair():
    gmax, gmax2, i, j, obj_min = -math.inf, -math.inf, -1, -1, math.inf
    for t in range(n):
        if y[t] == 1:
            if not upper(alpha[t]) and -G[t] >= gmax:
                gmax, i = -G[t], t
        elif not lower(alpha[t]) and G[t] >= gmax:
            gmax, i = G[t], t
    if i < 0:
        return -1, -1, 0.0
    for t in range(n):
        quad = K[i, i] + K[t, t] - 2 * K[i, t]
        if y[t] == 1:
            if not lower(alpha[t]):
                gd = gmax + G[t]
                if G[t] >= gmax2:
                    gmax2 = G[t]
                if gd > 0:
                    obj = -(gd * gd) / (quad if quad > 0 else TAU)
                    if obj <= obj_min:
                        j, obj_min = t, obj
        elif not upper(alpha[t]):
            gd = gmax - G[t]
            if -G[t] >= gmax2:
                gmax2 = -G[t]
            if gd > 0:
                obj = -(gd * gd) / (quad if quad > 0 else TAU)
                if obj <= obj_min:
                    j, obj_min = t, obj
    return i, j, gmax + gmax2


iters = 0
while True:
    i, j, gap = select_pair()
    if i < 0 or j < 0 or gap < TOL:
        break
    oi, oj = alpha[i], alpha[j]
    quad = K[i, i] + K[j, j] - 2 * K[i, j]
    if quad <= 0:
        quad = TAU
    ai, aj = oi, oj
    if y[i] != y[j]:
        delta = (-G[i] - G[j]) / quad
        diff = oi - oj
        ai += delta
        aj += delta
        if diff > 0:
            if aj < 0:
                aj, ai = 0.0, diff
        elif ai < 0:
            ai, aj = 0.0, -diff
        if diff > 0:
            if ai > C:
                ai, aj = C, C - diff
        elif aj > C:
            aj, ai = C, C + diff
    else:
        delta = (G[i] - G[j]) / quad
        total = oi + oj
        ai -= delta
        aj += delta
        if total > C:
            if ai > C:
                ai, aj = C, total - C
        elif aj < 0:
            aj, ai = 0.0, total
        if total > C:
            if aj > C:
                aj, ai = C, total - C
        elif ai < 0:
            ai, aj = 0.0, total
    alpha[i], alpha[j] = ai, aj
    G = G + y[i] * y * K[i] * (ai - oi)
    G = G + y[j] * y * K[j] * (aj - oj)
    iters += 1

# Bias (LIBSVM calculate_rho): b = -rho, rho = mean of y_i G_i over free SVs (0 < a_i < C).
ub, lb, free_sum, n_free = math.inf, -math.inf, 0.0, 0
for t in range(n):
    yg = y[t] * G[t]
    if upper(alpha[t]):
        if y[t] == -1:
            ub = min(ub, yg)
        else:
            lb = max(lb, yg)
    elif lower(alpha[t]):
        if y[t] == 1:
            ub = min(ub, yg)
        else:
            lb = max(lb, yg)
    else:
        n_free += 1
        free_sum += yg
rho = free_sum / n_free if n_free > 0 else ((ub + lb) / 2 if math.isfinite(ub + lb) else 0.0)
b = -rho


def decision(P):
    sv = alpha > 0
    return (alpha[sv] * y[sv]) @ kernel(X[sv], P) + b


def label(P):
    return np.where(decision(P) > 0, 1, -1)       # f(x) > 0 -> +1, otherwise -1 (as in the lab)


sv = int((alpha > 0).sum())
at_c = int((alpha >= C).sum())
quad_form = float((alpha * (G + 1)).sum())          # alpha^T Q alpha
margins = G + 1 + y * b                              # y_i f(x_i)
dual = alpha.sum() - 0.5 * quad_form
primal = 0.5 * quad_form + C * np.maximum(0.0, 1 - margins).sum()
print(f"SMO: {iters} pair updates, KKT gap {max(0.0, gap):.2e} (< {TOL})")
print(f"support vectors: {sv} ({sv - at_c} on the margin with 0 < a < C, {at_c} at a = C)")
print(f"b = {b:.4f}   dual D = {dual:.4f}   primal P = {primal:.4f}   P - D = {primal - dual:.2e}")
print(f"train acc {(label(X) == y).mean():.3f}   test acc {(label(X_test) == y_test).mean():.3f}")
if KERNEL == "linear":
    w = (alpha * y) @ X
    print(f"w = {w}   margin 2/|w| = {2 / np.linalg.norm(w):.4f}")

# Cross-check with scikit-learn's SVC (LIBSVM) using the same kernel conventions:
# poly -> gamma=1, coef0=1 so K = (1 + a.b)^d; rbf -> the same gamma.
from sklearn.svm import SVC
params = dict(C=C, tol=TOL, shrinking=False)
if KERNEL == "linear":
    clf = SVC(kernel="linear", **params)
elif KERNEL == "poly":
    clf = SVC(kernel="poly", degree=DEGREE, gamma=1.0, coef0=1.0, **params)
else:
    clf = SVC(kernel="rbf", gamma=GAMMA, **params)
clf.fit(X, y)
agree = (clf.predict(X_test) == label(X_test)).mean()
print(f"sklearn SVC: {len(clf.support_)} SVs, b = {clf.intercept_[0]:.4f}, test acc {clf.score(X_test, y_test):.3f}, agrees on {agree:.1%} of test points")
`;

/* ───────────────────────────── Gradient boosting ───────────────────────────── */

export interface GbmExport { data: readonly GPt[]; test: readonly GPt[]; cfg: BoostCfg; rounds: number; noise: number; seed: number; }

export const gradientBoostingPython = (o: GbmExport) => {
  const c = o.cfg;
  const growth = c.variant === 'xgboost' ? 'XGBoost-style level-wise growth' : c.variant === 'lightgbm' ? 'LightGBM-style leaf-wise growth with histogram bins' : `CatBoost-style symmetric trees, ${c.catMode} boosting`;
  const lib = c.variant === 'xgboost'
    ? `xgb.XGBClassifier(n_estimators=ROUNDS, learning_rate=LR, max_depth=MAX_DEPTH, reg_lambda=LAMBDA, gamma=0,\n#                   min_child_weight=0, base_score=0.5, tree_method="exact")`
    : c.variant === 'lightgbm'
      ? `lgb.LGBMClassifier(n_estimators=ROUNDS, learning_rate=LR, num_leaves=NUM_LEAVES, max_bin=MAX_BIN, reg_lambda=LAMBDA,\n#                   min_child_samples=MIN_LEAF, min_child_weight=0, min_split_gain=0, max_depth=-1)`
      : `CatBoostClassifier(iterations=ROUNDS, learning_rate=LR, depth=MAX_DEPTH, l2_leaf_reg=LAMBDA,\n#                   boosting_type="${c.catMode === 'ordered' ? 'Ordered' : 'Plain'}", leaf_estimation_method="Newton")`;
  return `import math

# Gradient boosting (${growth}) — a line-for-line port of the lab (labs/supervised/boostCore.ts):
# logistic loss, g = p - y, h = max(1e-6, p(1-p)), Newton leaf w* = -sum G / (sum H + lambda)
# (0 for an empty leaf), split gain 1/2[GL^2/(HL+l) + GR^2/(HR+l) - G^2/(H+l)] - gamma,
# score F = LR * sum(trees), starting from F = 0.
# Data: balanced XOR, ${o.data.length} training points (${Math.round(o.noise * 100)}% of labels flipped, seed ${o.seed}),
#       ${o.test.length} clean held-out test points (seed ${o.seed + TEST_SEED_OFFSET}).
# The closest library call (its binning / defaults differ in detail, so numbers can differ):
#   ${lib}
VARIANT = "${c.variant}"
LR = ${num(c.lr)}
MAX_DEPTH = ${c.maxDepth}
NUM_LEAVES = ${c.numLeaves}
MAX_BIN = ${c.maxBin}
LAMBDA = ${num(c.lambda)}
GAMMA = ${num(c.gamma)}
MIN_LEAF = ${c.minLeaf}
CAT_MODE = "${c.catMode}"
PERM_SEED = ${c.permSeed}
ROUNDS = ${o.rounds}
HMIN = 1e-6

TRAIN = [
${o.data.map((p) => `    (${num(p.x)}, ${num(p.y)}, ${p.y01}),`).join('\n')}
]
TEST = [
${o.test.map((p) => `    (${num(p.x)}, ${num(p.y)}, ${p.y01}),`).join('\n')}
]


def sigmoid(z):
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0


def grad_of(score, y):
    p = sigmoid(score)
    return (p - y, max(HMIN, p * (1 - p)))


def logloss(y, p):
    return -(y * math.log(p + 1e-9) + (1 - y) * math.log(1 - p + 1e-9))


def leaf_weight(idx, gr):
    if not idx:
        return 0.0
    G = H = 0.0
    for i in idx:
        G += gr[i][0]
        H += gr[i][1]
    return -G / (H + LAMBDA)


def split_gain(L, R, gr):
    GL = HL = GR = HR = 0.0
    for i in L:
        GL += gr[i][0]
        HL += gr[i][1]
    for i in R:
        GR += gr[i][0]
        HR += gr[i][1]
    G, H = GL + GR, HL + HR
    return 0.5 * (GL * GL / (HL + LAMBDA) + GR * GR / (HR + LAMBDA) - G * G / (H + LAMBDA)) - GAMMA


def exact_thresholds(vals):
    d = sorted(set(vals))
    return [(d[i] + d[i + 1]) / 2 for i in range(len(d) - 1)]


def js_round(v):
    return math.floor(v + 0.5)


def bin_edges(vals, max_bin):
    """<= max_bin equal-frequency bins; split thresholds may only be these edges."""
    d = sorted(set(vals))
    if len(d) <= max_bin:
        return exact_thresholds(d)
    s = sorted(vals)
    n = len(s)
    edges = []
    for k in range(1, max_bin):
        i = js_round(k * n / max_bin)
        if i < 1 or i >= n or s[i] <= s[i - 1]:
            continue
        e = (s[i - 1] + s[i]) / 2
        if not edges or e > edges[-1]:
            edges.append(e)
    return edges


def best_split(pts, idx, gr, cands):
    best = None
    for feat in (0, 1):
        thrs = cands[feat] if cands else exact_thresholds([pts[i][feat] for i in idx])
        for thr in thrs:
            L = [i for i in idx if pts[i][feat] <= thr]
            R = [i for i in idx if pts[i][feat] > thr]
            if len(L) < MIN_LEAF or len(R) < MIN_LEAF:
                continue
            gain = split_gain(L, R, gr)
            if best is None or gain > best[4]:
                best = (feat, thr, L, R, gain)
    return best


def leaf(idx, gr):
    return {"leaf": True, "val": leaf_weight(idx, gr), "n": len(idx)}


def build_level_wise(pts, idx, gr, depth):
    if depth >= MAX_DEPTH or len(idx) < 2 * MIN_LEAF:
        return leaf(idx, gr)
    s = best_split(pts, idx, gr, None)
    if s is None or s[4] <= 0:
        return leaf(idx, gr)
    feat, thr, L, R, gain = s
    return {"leaf": False, "feat": feat, "thr": thr, "gain": gain, "n": len(idx),
            "left": build_level_wise(pts, L, gr, depth + 1), "right": build_level_wise(pts, R, gr, depth + 1)}


def build_leaf_wise(pts, idx, gr, cands):
    holder = {"root": leaf(idx, gr)}
    opened = [(idx, best_split(pts, idx, gr, cands), (holder, "root"))]
    for _ in range(1, NUM_LEAVES):
        bi, bg = -1, 0.0
        for k, (_, s, _) in enumerate(opened):
            if s is not None and s[4] > bg:
                bg, bi = s[4], k
        if bi < 0:
            break
        node_idx, s, (parent, key) = opened.pop(bi)
        feat, thr, L, R, gain = s
        branch = {"leaf": False, "feat": feat, "thr": thr, "gain": gain, "n": len(node_idx), "left": leaf(L, gr), "right": leaf(R, gr)}
        parent[key] = branch
        opened.append((L, best_split(pts, L, gr, cands), (branch, "left")))
        opened.append((R, best_split(pts, R, gr, cands), (branch, "right")))
    return holder["root"]


def choose_oblivious_tests(pts, gr):
    groups = [list(range(len(pts)))]
    tests = []
    thr = (exact_thresholds([p[0] for p in pts]), exact_thresholds([p[1] for p in pts]))
    for _ in range(MAX_DEPTH):
        best = None
        for feat in (0, 1):
            for t in thr[feat]:
                total, ok = 0.0, False
                for grp in groups:
                    L = [i for i in grp if pts[i][feat] <= t]
                    R = [i for i in grp if pts[i][feat] > t]
                    if len(L) >= MIN_LEAF and len(R) >= MIN_LEAF:
                        total += split_gain(L, R, gr)
                        ok = True
                if ok and (best is None or total > best[2]):
                    best = (feat, t, total)
        if best is None or best[2] <= 0:
            break
        tests.append(best)
        f, t, _ = best
        groups = [part for grp in groups for part in ([i for i in grp if pts[i][f] <= t], [i for i in grp if pts[i][f] > t])]
    return tests


def oblivious_leaf(tests, x, y):
    k = 0
    for f, t, _ in tests:
        k = 2 * k + (0 if (x if f == 0 else y) <= t else 1)
    return k


def oblivious_tree(tests, vals, counts, level=0, k=0):
    if level >= len(tests):
        return {"leaf": True, "val": vals[k], "n": counts[k]}
    f, t, g = tests[level]
    left = oblivious_tree(tests, vals, counts, level + 1, 2 * k)
    right = oblivious_tree(tests, vals, counts, level + 1, 2 * k + 1)
    return {"leaf": False, "feat": f, "thr": t, "gain": g, "n": left["n"] + right["n"], "left": left, "right": right}


def predict_tree(node, x, y):
    while not node["leaf"]:
        node = node["left"] if (x if node["feat"] == 0 else y) <= node["thr"] else node["right"]
    return node["val"]


def mulberry32(seed):
    """The lab's seeded generator, reproduced bit-for-bit."""
    state = [seed & 0xFFFFFFFF]

    def rnd():
        a = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        state[0] = a
        t = ((a ^ (a >> 15)) * (a | 1)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rnd


def permutation(n, rnd):
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = math.floor(rnd() * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def tree_shape(node):
    if node["leaf"]:
        return 0, 1
    ld, ll = tree_shape(node["left"])
    rd, rl = tree_shape(node["right"])
    return 1 + max(ld, rd), ll + rl


pts = [(x, y) for x, y, _ in TRAIN]
labels = [c for _, _, c in TRAIN]
n = len(pts)
F = [0.0] * n
trees = []
bins = (bin_edges([p[0] for p in pts], MAX_BIN), bin_edges([p[1] for p in pts], MAX_BIN)) if VARIANT == "lightgbm" else None
ordered = VARIANT == "catboost" and CAT_MODE == "ordered"
if ordered:
    perm = permutation(n, mulberry32(PERM_SEED))
    prefix = [[0.0] * (j + 1) for j in range(n)]   # prefix[j][p] = M_j(x of sample perm[p]), p <= j
if bins:
    print(f"histogram bins: {len(bins[0])} + {len(bins[1])} candidate thresholds")

for r in range(ROUNDS):
    plain = [grad_of(F[i], labels[i]) for i in range(n)]
    if VARIANT == "catboost":
        if ordered:
            scores = [0.0] * n
            for k in range(n):
                scores[perm[k]] = prefix[k][k]      # sample perm[k] scored by M_k (never saw it)
            struct = [grad_of(scores[i], labels[i]) for i in range(n)]
        else:
            struct = plain
        tests = choose_oblivious_tests(pts, struct)
        L = 1 << len(tests)
        leaf_of = [oblivious_leaf(tests, x, y) for x, y in pts]
        members = [[] for _ in range(L)]
        for i, k in enumerate(leaf_of):
            members[k].append(i)
        tree = oblivious_tree(tests, [leaf_weight(m, plain) for m in members], [len(m) for m in members])
        if ordered:
            for j in range(n):
                Gs, Hs = [0.0] * L, [0.0] * L
                row = prefix[j]
                for p in range(j):
                    i = perm[p]
                    g, h = grad_of(row[p], labels[i])
                    k = leaf_of[i]
                    Gs[k] += g
                    Hs[k] += h
                for p in range(j + 1):
                    k = leaf_of[perm[p]]
                    v = -Gs[k] / (Hs[k] + LAMBDA) if Hs[k] > 0 else 0.0
                    row[p] = row[p] + LR * v
    elif VARIANT == "lightgbm":
        tree = build_leaf_wise(pts, list(range(n)), plain, bins)
    else:
        tree = build_level_wise(pts, list(range(n)), plain, 0)
    trees.append(tree)
    for i in range(n):
        F[i] = F[i] + LR * predict_tree(tree, pts[i][0], pts[i][1])
    loss = sum(logloss(labels[i], sigmoid(F[i])) for i in range(n)) / n
    acc = sum(1 for i in range(n) if (1 if sigmoid(F[i]) >= 0.5 else 0) == labels[i]) / n
    test_scores = [LR * sum(predict_tree(t, x, y) for t in trees) for x, y, _ in TEST]
    test_acc = sum(1 for s, (_, _, c) in zip(test_scores, TEST) if (1 if sigmoid(s) >= 0.5 else 0) == c) / len(TEST)
    depth, leaves = tree_shape(tree)
    line = f"round {r + 1:2d}: tree depth {depth}, {leaves:2d} leaves | train loss {loss:.4f} acc {acc:.3f} | test acc {test_acc:.3f}"
    if ordered:
        scores = [0.0] * n
        for k in range(n):
            scores[perm[k]] = prefix[k][k]
        line += f" | ordered loss {sum(logloss(labels[i], sigmoid(scores[i])) for i in range(n)) / n:.4f}"
    print(line)
`;
};

/* ─────────────────────────────── Naive Bayes ─────────────────────────────── */

export interface NbExport { data: readonly NbPt[]; variant: NbVariant; alpha: number; learnedPrior: boolean; query: { x: number; y: number }; spread: number; unequal: boolean; imbalance: number; seed: number; }

export const naiveBayesPython = (o: NbExport) => `import math
import numpy as np

# ${o.variant === 'gaussian' ? 'Gaussian' : 'Multinomial'} Naive Bayes — a line-for-line port of the lab (labs/supervised/nbCore.ts)
# on the exact points it shows: 3 classes, base spread ${o.spread}, ${o.unequal ? 'unequal per-class shapes' : 'equal round shapes'},
# class 1 x${o.imbalance} (seed ${o.seed}).
VARIANT = "${o.variant}"
ALPHA = ${num(o.alpha)}           # Laplace smoothing (multinomial)
BINS = ${NB_BINS}                  # cells per axis on [0, 1] (multinomial)
VAR_FLOOR = ${VAR_FLOOR}          # Gaussian variance floor
LEARNED_PRIOR = ${pyBool(o.learnedPrior)}   # False = uniform prior 1/3
QUERY = (${num(o.query.x)}, ${num(o.query.y)})

${pyXY('X', o.data)}
${pyInts('y', o.data.map((p) => p.cls))}
K = 3


def bin_of(v):
    return min(BINS - 1, max(0, math.floor(v * BINS)))


# ---- fit (maximum likelihood) ----
counts = np.array([(y == c).sum() for c in range(K)])
mu = np.array([X[y == c].mean(axis=0) for c in range(K)])
var = np.array([np.maximum(VAR_FLOOR, ((X[y == c] - mu[c]) ** 2).sum(axis=0) / max(1, counts[c])) for c in range(K)])
hist = np.zeros((K, 2, BINS))
for (a, b), c in zip(X, y):
    hist[c, 0, bin_of(a)] += 1
    hist[c, 1, bin_of(b)] += 1
prior = counts / counts.sum() if LEARNED_PRIOR else np.full(K, 1 / K)


def log_post(x1, x2, c):
    lp = math.log(prior[c])
    if VARIANT == "gaussian":
        for f, v in ((0, x1), (1, x2)):
            lp += -0.5 * math.log(2 * math.pi * var[c, f]) - (v - mu[c, f]) ** 2 / (2 * var[c, f])
    else:
        for f, v in ((0, x1), (1, x2)):
            lp += math.log((hist[c, f, bin_of(v)] + ALPHA) / (counts[c] + ALPHA * BINS))
    return lp


def posterior(x1, x2):
    lp = np.array([log_post(x1, x2, c) for c in range(K)])
    e = np.exp(lp - lp.max())
    return e / e.sum()


def predict(P):
    return np.array([int(np.argmax([log_post(a, b, c) for c in range(K)])) for a, b in P])


print("class counts:", counts.tolist(), " priors P(c):", np.round(prior, 4).tolist())
if VARIANT == "gaussian":
    print("means:\\n", np.round(mu, 4), "\\nvariances:\\n", np.round(var, 5))
print(f"train acc {(predict(X) == y).mean():.3f}")
post = posterior(*QUERY)
print(f"query {QUERY}: posterior {np.round(post, 4).tolist()} -> class {int(post.argmax())}")

# Cross-check with scikit-learn. Multinomial: one-hot the fixed [0, 1] bins of both axes —
# MultinomialNB then gives exactly the same posteriors. Gaussian: GaussianNB uses its own
# tiny variance smoothing instead of the floor, so predictions agree almost everywhere.
if VARIANT == "multinomial":
    from sklearn.naive_bayes import MultinomialNB
    onehot = lambda P: np.array([np.eye(BINS)[bin_of(a)].tolist() + np.eye(BINS)[bin_of(b)].tolist() for a, b in P])
    clf = MultinomialNB(alpha=ALPHA, fit_prior=LEARNED_PRIOR, force_alpha=True).fit(onehot(X), y)
    sk_post = clf.predict_proba(onehot(np.array([QUERY])))[0]
else:
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB(priors=None if LEARNED_PRIOR else [1 / K] * K).fit(X, y)
    sk_post = clf.predict_proba(np.array([QUERY]))[0]
    onehot = lambda P: P
print(f"sklearn: train acc {clf.score(onehot(X), y):.3f}, query posterior {np.round(sk_post, 4).tolist()}")
`;

/* ─────────────────────────────── check samples ─────────────────────────────── */

const dtSample = (layout: XorLayout, noise: number, seed: number, depth: number, crit: Crit, minLeaf: number, total = 96) => {
  const sizes = xorSizes(layout, total), centers = XOR_LAYOUTS[layout].centers;
  return decisionTreePython({
    data: makeXorData(sizes, DT_STD, seed, noise, centers), test: makeXorData(sizes, DT_STD, seed + TEST_SEED_OFFSET, 0, centers),
    depth, crit, minLeaf, layout, noise, seed,
  });
};
const svmSample = (shape: SvmShape, kernel: Kernel, C: number, gamma: number, degree: number, sep = 0.5, per = 35, seed = 1) => svmPython({
  data: makeSvmData(per, sep, shape, seed), test: makeSvmData(per, sep, shape, seed + TEST_SEED_OFFSET), C, kernel, gamma, degree, shape, sep, seed,
});
const gbmSample = (cfg: Partial<BoostCfg>, noise = 0.1, per = 24, seed = 1) => {
  const toG = (d: XorPt[]): GPt[] => d.map((p) => ({ x: p.x, y: p.y, y01: p.cls }));
  const full: BoostCfg = { variant: 'xgboost', lr: 0.3, maxDepth: 3, numLeaves: 8, maxBin: 255, lambda: 1, gamma: 0, minLeaf: 1, catMode: 'ordered', permSeed: seed, ...cfg };
  return gradientBoostingPython({
    data: toG(makeXorData([per, per, per, per], GBM_STD, seed, noise)), test: toG(makeXorData([per, per, per, per], GBM_STD, seed + TEST_SEED_OFFSET, 0)),
    cfg: full, rounds: 40, noise, seed,
  });
};
const nbSample = (variant: NbVariant, spread: number, unequal: boolean, imbalance: number, alpha: number, learnedPrior = true, per = 28, seed = 1) => naiveBayesPython({
  data: makeNbData(per, spread, unequal, imbalance, seed), variant, alpha, learnedPrior, query: { x: 0.5, y: 0.5 }, spread, unequal, imbalance, seed,
});

export const PYTHON_SAMPLES: PythonSample[] = [
  // Decision tree — lab default (depth 0 → single leaf), every preset, slider extremes
  { name: 'dt-default-depth0', code: () => dtSample('uneven', 0.1, 2, 0, 'gini', 1) },
  { name: 'dt-default-depth5', code: () => dtSample('uneven', 0.1, 2, 5, 'gini', 1) },
  { name: 'dt-stump', code: () => dtSample('uneven', 0, 2, 1, 'gini', 1) },
  { name: 'dt-just-enough', code: () => dtSample('uneven', 0, 2, 2, 'gini', 1) },
  { name: 'dt-greedy-trap', code: () => dtSample('balanced', 0, 2, 2, 'gini', 1) },
  { name: 'dt-entropy', code: () => dtSample('uneven', 0.1, 2, 3, 'entropy', 1) },
  { name: 'dt-overfit', code: () => dtSample('uneven', 0.15, 2, 8, 'gini', 1) },
  { name: 'dt-pruned', code: () => dtSample('uneven', 0.15, 2, 8, 'gini', 6) },
  { name: 'dt-minleaf10-noise30-160pts', code: () => dtSample('balanced', 0.3, 7, 8, 'entropy', 10, 160) },
  { name: 'dt-48pts', code: () => dtSample('uneven', 0, 3, 4, 'gini', 1, 48) },
  // SVM — default, every preset, kernel/slider extremes
  { name: 'svm-linear-default', code: () => svmSample('blobs', 'linear', 5, 4, 3) },
  { name: 'svm-soft-c01', code: () => svmSample('blobs', 'linear', 0.1, 4, 3) },
  { name: 'svm-hard-c100', code: () => svmSample('blobs', 'linear', 100, 4, 3) },
  { name: 'svm-moons-rbf', code: () => svmSample('moons', 'rbf', 10, 4, 3) },
  { name: 'svm-rings-rbf', code: () => svmSample('rings', 'rbf', 10, 8, 3) },
  { name: 'svm-poly-d3', code: () => svmSample('moons', 'poly', 10, 4, 3), timeoutSec: 180 },
  { name: 'svm-overfit-g256', code: () => svmSample('moons', 'rbf', 50, 256, 3, 0) },
  { name: 'svm-poly-d5-rings', code: () => svmSample('rings', 'poly', 1, 4, 5, 1), timeoutSec: 180 },
  { name: 'svm-linear-moons-70', code: () => svmSample('moons', 'linear', 100, 4, 3, 0, 70), timeoutSec: 180 },
  { name: 'svm-rbf-g05-10pts', code: () => svmSample('rings', 'rbf', 0.1, 0.5, 3, 1, 10) },
  // Gradient boosting — every preset + edge cases (λ = 0 empty leaves, η = 1, 32 leaves, 40/cluster)
  { name: 'gbm-xgb-default', code: () => gbmSample({}) },
  { name: 'gbm-stumps', code: () => gbmSample({ maxDepth: 1, numLeaves: 2 }, 0) },
  { name: 'gbm-lgbm-16', code: () => gbmSample({ variant: 'lightgbm', numLeaves: 16 }) },
  { name: 'gbm-lgbm-bins4', code: () => gbmSample({ variant: 'lightgbm', numLeaves: 16, maxBin: 4 }) },
  { name: 'gbm-cat-ordered', code: () => gbmSample({ variant: 'catboost' }) },
  { name: 'gbm-cat-plain', code: () => gbmSample({ variant: 'catboost', catMode: 'plain' }) },
  { name: 'gbm-cat-lambda0-depth6', code: () => gbmSample({ variant: 'catboost', maxDepth: 6, lambda: 0 }, 0.3, 40), timeoutSec: 240 },
  { name: 'gbm-xgb-lr1-depth6', code: () => gbmSample({ lr: 1, maxDepth: 6, lambda: 0 }, 0.3, 40), timeoutSec: 180 },
  { name: 'gbm-lgbm-32-bins8', code: () => gbmSample({ variant: 'lightgbm', numLeaves: 32, maxBin: 8, lr: 0.05 }, 0, 12), timeoutSec: 180 },
  // Naive Bayes — every preset + uniform prior + extremes
  { name: 'nb-default', code: () => nbSample('gaussian', 0.1, true, 3, 1) },
  { name: 'nb-tight', code: () => nbSample('gaussian', 0.06, true, 1, 1) },
  { name: 'nb-equal', code: () => nbSample('gaussian', 0.09, false, 1, 1) },
  { name: 'nb-overlap', code: () => nbSample('gaussian', 0.14, true, 1, 1) },
  { name: 'nb-priors-uniform', code: () => nbSample('gaussian', 0.12, true, 4, 1, false) },
  { name: 'nb-multinomial', code: () => nbSample('multinomial', 0.09, true, 1, 1) },
  { name: 'nb-high-smoothing', code: () => nbSample('multinomial', 0.09, true, 1, 4) },
  { name: 'nb-no-smoothing-uniform', code: () => nbSample('multinomial', 0.09, true, 4, 0.001, false) },
  { name: 'nb-50-per-class-spread016', code: () => nbSample('gaussian', 0.16, false, 4, 1, true, 50, 9) },
  { name: 'nb-12-per-class-spread005', code: () => nbSample('multinomial', 0.05, true, 1, 5, true, 12, 4) },
];
