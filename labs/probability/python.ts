// Runnable Python exports for the Probability & Bayesian labs (template strings —
// not LLM generated). Each mirrors its lab exactly: the live parameters at
// download time, the same data (replayed coin flips, the same mixture preset and
// start point) and the same algorithms. No scipy: every pmf / pdf / CDF / sampler
// is implemented by hand, as in the lab.
import type { PythonSample } from '../../utils/pythonSamples';
import { DEFAULTS, ORDER, FAMILIES } from './families';
import type { Family, Params } from './families';
import { TARGETS, TARGET_KEYS } from './mcmcCore';
import type { Comp, TargetKey } from './mcmcCore';

export type BayesMode = 'diagnostic' | 'sequential';

/** A JS number as a Python literal (shortest round-trip repr — identical in both). */
const py = (v: number) => (Number.isFinite(v) ? String(v) : '0.0');

/** A 0/1 flip record as a Python string literal, wrapped over several lines. */
function flipsLiteral(flips: number[]): string {
  if (!flips.length) return '""';
  const s = flips.map((f) => (f ? '1' : '0')).join('');
  const lines: string[] = [];
  for (let i = 0; i < s.length; i += 100) lines.push(`    "${s.slice(i, i + 100)}"`);
  return `(\n${lines.join('\n')}\n)`;
}

export const bayesPython = (
  mode: BayesMode,
  prevalence: number,
  sensitivity: number,
  specificity: number,
  trueP: number,
  priorA = 1,
  priorB = 1,
  flips: number[] = [],
) => {
  if (mode === 'sequential') {
    return `import numpy as np
from math import lgamma

# Bayes' theorem — sequential Beta-Bernoulli updating (mirrors the lab)
# Prior:      p ~ Beta(ALPHA0, BETA0)          (the prior chosen in the lab)
# Likelihood: each coin flip is Bernoulli(p)
# Posterior:  Beta is conjugate, so heads -> alpha + 1, tails -> beta + 1.
ALPHA0, BETA0 = ${py(priorA)}, ${py(priorB)}
TRUE_P = ${py(trueP)}              # the hidden bias being estimated
# The ${flips.length} flips observed in the lab (1 = heads), replayed so this script reaches
# the same posterior. With none recorded it simulates N_SIM fresh flips instead.
FLIPS = ${flipsLiteral(flips)}
N_SIM = 200
rng = np.random.default_rng(0)

def beta_pdf(x, a, b):
    # x^(a-1) (1-x)^(b-1) / B(a, b), via log-gammas; zero outside (0, 1)
    logB = lgamma(a) + lgamma(b) - lgamma(a + b)
    out = np.zeros_like(x)
    m = (x > 0) & (x < 1)
    out[m] = np.exp((a - 1) * np.log(x[m]) + (b - 1) * np.log(1 - x[m]) - logB)
    return out

def credible_interval_90(a, b):
    # equal-tailed 90% interval from the CDF of an 801-point grid (as in the lab)
    xs = np.arange(801) / 800
    pdf = beta_pdf(xs, a, b)
    cdf = np.cumsum(pdf) / pdf.sum()
    return xs[np.argmax(cdf >= 0.05)], xs[np.argmax(cdf >= 0.95)]

if __name__ == "__main__":
    flips = [c == "1" for c in FLIPS] if FLIPS else list(rng.random(N_SIM) < TRUE_P)
    a, b = ALPHA0, BETA0
    means = []
    for heads in flips:
        if heads:
            a += 1                    # conjugate update: alpha + 1
        else:
            b += 1                    # conjugate update: beta + 1
        means.append(a / (a + b))
    lo, hi = credible_interval_90(a, b)
    n = len(flips)
    print(f"prior Beta({ALPHA0:g}, {BETA0:g}) worth {ALPHA0 + BETA0:g} pseudo-flips, mean {ALPHA0 / (ALPHA0 + BETA0):.3f}")
    print(f"after {n} flips ({sum(flips)} heads): Beta({a:g}, {b:g})")
    print(f"posterior mean = {a / (a + b):.4f}  (true p = {TRUE_P})")
    print(f"90% credible interval = [{lo:.4f}, {hi:.4f}]")
    # on average the posterior mean's bias is (a0 - p (a0 + b0)) / (a0 + b0 + n)
    s0 = ALPHA0 + BETA0
    n_close = max(0, int(np.ceil(abs(ALPHA0 - TRUE_P * s0) / 0.05 - s0)))
    print(f"flips needed on average for the mean to come within 0.05 of p: ~{n_close}")
`;
  }

  return `import math

# Bayes' theorem — diagnostic test / base-rate fallacy (mirrors the lab)
#   P(D|+) = P(+|D) P(D) / [ P(+|D) P(D) + P(+|~D) P(~D) ]
PREVALENCE  = ${py(prevalence)}     # P(D)    — the prior / base rate
SENSITIVITY = ${py(sensitivity)}     # P(+|D)  — true-positive rate
SPECIFICITY = ${py(specificity)}     # P(-|~D) — true-negative rate
GRID_N  = 1_000     # the lab's icon array: 1 square = 1 person
TABLE_N = 100_000   # the lab's exact table is quoted per 100,000

def posterior_positive(prior, sens, spec):
    p_pos = sens * prior + (1 - spec) * (1 - prior)    # total prob of a + test
    return sens * prior / p_pos                          # P(D|+)

def posterior_negative(prior, sens, spec):
    p_neg = (1 - sens) * prior + spec * (1 - prior)     # total prob of a - test
    return (1 - sens) * prior / p_neg                    # P(D|-)

def expected_counts(n, prior, sens, spec):
    sick = n * prior
    healthy = n - sick
    return sick * sens, sick * (1 - sens), healthy * (1 - spec), healthy * spec   # TP FN FP TN

def round_half_up(x):
    return math.floor(x + 0.5)          # JavaScript's Math.round

def rounded_counts(n, prior, sens, spec):
    # the icon array: round the sick count, then TP within the sick, FP within the healthy
    sick = round_half_up(prior * n)
    tp = round_half_up(sick * sens)
    healthy = n - sick
    fp = round_half_up(healthy * (1 - spec))
    return tp, sick - tp, fp, healthy - fp

if __name__ == "__main__":
    post_pos = posterior_positive(PREVALENCE, SENSITIVITY, SPECIFICITY)
    post_neg = posterior_negative(PREVALENCE, SENSITIVITY, SPECIFICITY)
    print(f"prior  P(D)   = {PREVALENCE:.4f}")
    print(f"P(D|+) = {post_pos:.4f}   P(D|-) = {post_neg:.6f}")
    tp, fn, fp, tn = expected_counts(TABLE_N, PREVALENCE, SENSITIVITY, SPECIFICITY)
    print(f"exact, per {TABLE_N:,}: TP={tp:,.1f}  FN={fn:,.1f}  FP={fp:,.1f}  TN={tn:,.1f}")
    print(f"  P(D|+) = TP / (TP + FP) = {tp / (tp + fp) * 100:.2f}%")
    gtp, gfn, gfp, gtn = rounded_counts(GRID_N, PREVALENCE, SENSITIVITY, SPECIFICITY)
    print(f"icon array of {GRID_N:,} (rounded to whole people): TP={gtp} FN={gfn} FP={gfp} TN={gtn}")
    if gtp + gfp:
        print(f"  {gtp} true + vs {gfp} false + -> {gtp / (gtp + gfp) * 100:.1f}% of the grid's positives are sick")
`;
};

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------

interface FamilyBlock { consts: string; body: string; }

// Histogram window + EXACT reference probability of every bin (as in the lab).
// Draws outside the window are counted separately, never folded into an edge bin.
const LAYOUT_DISCRETE = `def make_layout():
    # single draws: the pmf itself; sums of k draws: the CLT normal N(k mu, k var)
    # over [j - 1/2, j + 1/2] for each integer j (continuity correction)
    if K_SUM == 1:
        lo, hi = support()
        mass = [pmf(lo + i) for i in range(hi - lo + 1)]
        out_mass = max(0.0, 1 - sum(mass))
    else:
        m, s = K_SUM * MEAN, math.sqrt(K_SUM * VAR)
        phi = lambda x: norm_cdf((x - m) / s)
        lo, hi = math.floor(m - 4 * s), math.ceil(m + 4 * s)
        mass = [phi(j + 0.5) - phi(j - 0.5) for j in range(lo, hi + 1)]
        out_mass = max(0.0, 1 - (phi(hi + 0.5) - phi(lo - 0.5)))
    key = lambda x: int(round(x)) - lo if lo <= x <= hi else None
    return lo, hi, mass, key, out_mass`;

const LAYOUT_CONTINUOUS = `def make_layout():
    # N_BINS equal bins; single draws: CDF differences; sums of k draws: the CLT
    # normal N(k mu, k var) over a +-4 sd window
    if K_SUM == 1:
        lo, hi = support()
        F = cdf
    else:
        m, s = K_SUM * MEAN, math.sqrt(K_SUM * VAR)
        lo, hi = m - 4 * s, m + 4 * s
        F = lambda x: norm_cdf((x - m) / s)
    w = (hi - lo) / N_BINS
    mass = [F(lo + (i + 1) * w) - F(lo + i * w) for i in range(N_BINS)]
    key = lambda x: min(N_BINS - 1, math.floor((x - lo) / w)) if lo <= x < hi else None
    return lo, hi, mass, key, max(0.0, 1 - (F(hi) - F(lo)))`;

function familyBlock(family: Family, p: Params): FamilyBlock {
  const v = (k: string) => py(p[k] ?? 0);
  switch (family) {
    case 'bernoulli': return {
      consts: `KIND = "discrete"\nP = ${v('p')}                    # success probability`,
      body: `def pmf(k):
    return P if k == 1 else (1 - P if k == 0 else 0.0)

def sample():
    return 1 if U() < P else 0

def support():
    return 0, 1

MEAN, VAR = P, P * (1 - P)

def entropy():                   # nats
    return sum(-q * math.log(q) for q in (P, 1 - P) if q > 0)`,
    };
    case 'binomial': return {
      consts: `KIND = "discrete"\nN_TRIALS, P = ${v('n')}, ${v('p')}         # trials, success probability`,
      body: `def pmf(k):
    if k < 0 or k > N_TRIALS:
        return 0.0
    return math.exp(lgamma(N_TRIALS + 1) - lgamma(k + 1) - lgamma(N_TRIALS - k + 1)
                    + k * math.log(P) + (N_TRIALS - k) * math.log(1 - P))

def sample():                    # sum of N_TRIALS Bernoulli(P) trials
    return sum(1 for _ in range(int(N_TRIALS)) if U() < P)

def support():
    return 0, int(N_TRIALS)

MEAN, VAR = N_TRIALS * P, N_TRIALS * P * (1 - P)

def entropy():                   # nats, exact finite sum
    return sum(-q * math.log(q) for q in (pmf(k) for k in range(int(N_TRIALS) + 1)) if q > 0)`,
    };
    case 'poisson': return {
      consts: `KIND = "discrete"\nLAM = ${v('lam')}                  # rate / expected count`,
      body: `def pmf(k):
    return math.exp(-LAM + k * math.log(LAM) - lgamma(k + 1)) if k >= 0 else 0.0

def sample():                    # Knuth: multiply uniforms until the product drops below e^-lambda
    L, k, prod = math.exp(-LAM), 0, 1.0
    while True:
        k += 1
        prod *= U()
        if not prod > L:
            return k - 1

def support():
    return 0, max(8, math.ceil(LAM + 4 * math.sqrt(LAM)))

MEAN, VAR = LAM, LAM

def entropy():                   # nats, summed until the tail is negligible
    K = math.ceil(LAM + 12 * math.sqrt(LAM) + 12)
    return sum(-q * math.log(q) for q in (pmf(k) for k in range(K + 1)) if q > 0)`,
    };
    case 'geometric': return {
      consts: `KIND = "discrete"\nP = ${v('p')}                    # success prob (trials until the first success)`,
      body: `def pmf(k):                      # k = 1, 2, 3, ...
    return (1 - P) ** (k - 1) * P if k >= 1 else 0.0

def sample():                    # inverse CDF
    return math.ceil(math.log(1 - U()) / math.log(1 - P))

def support():
    return 1, max(6, math.ceil(math.log(0.02) / math.log(1 - P)))

MEAN, VAR = 1 / P, (1 - P) / P ** 2

def entropy():                   # nats, closed form
    return (-(1 - P) * math.log(1 - P) - P * math.log(P)) / P`,
    };
    case 'uniform': return {
      consts: `KIND = "continuous"\nA, B = ${v('a')}, ${v('b')}              # support [a, b]`,
      body: `def pdf(x):
    return 1.0 / (B - A) if A <= x <= B else 0.0

def cdf(x):
    return min(1.0, max(0.0, (x - A) / (B - A)))

def sample():
    return A + (B - A) * U()

def support():
    return A - 0.4 * (B - A), B + 0.4 * (B - A)

MEAN, VAR = (A + B) / 2, (B - A) ** 2 / 12

def entropy():                   # nats (differential)
    return math.log(B - A)`,
    };
    case 'normal': return {
      consts: `KIND = "continuous"\nMU, SIGMA = ${v('mu')}, ${v('sigma')}          # mean, std-dev`,
      body: `def pdf(x):
    z = (x - MU) / SIGMA
    return math.exp(-0.5 * z * z) / (SIGMA * math.sqrt(2 * math.pi))

def cdf(x):
    return norm_cdf((x - MU) / SIGMA)

def sample():
    return MU + SIGMA * randn()

def support():
    return MU - 4 * SIGMA, MU + 4 * SIGMA

MEAN, VAR = MU, SIGMA ** 2

def entropy():                   # nats (differential)
    return 0.5 * math.log(2 * math.pi * math.e * SIGMA ** 2)`,
    };
    case 'exponential': return {
      consts: `KIND = "continuous"\nLAM = ${v('lam')}                  # rate (mean = 1/lambda)`,
      body: `def pdf(x):
    return LAM * math.exp(-LAM * x) if x >= 0 else 0.0

def cdf(x):
    return 1 - math.exp(-LAM * x) if x > 0 else 0.0

def sample():                    # inverse CDF
    return -math.log(1 - U()) / LAM

def support():
    return 0.0, max(3, 6 / LAM)

MEAN, VAR = 1 / LAM, 1 / LAM ** 2

def entropy():                   # nats (differential)
    return 1 - math.log(LAM)`,
    };
    case 'beta': return {
      consts: `KIND = "continuous"\nALPHA, BETA = ${v('a')}, ${v('b')}         # shape parameters`,
      body: `def pdf(x):
    if not (0 < x < 1):
        return 0.0
    logB = lgamma(ALPHA) + lgamma(BETA) - lgamma(ALPHA + BETA)
    return math.exp((ALPHA - 1) * math.log(x) + (BETA - 1) * math.log(1 - x) - logB)

def cdf(x):
    return beta_inc(x, ALPHA, BETA)

def sample():                    # ratio of two Marsaglia-Tsang Gamma draws
    ga, gb = gamma_sample(ALPHA), gamma_sample(BETA)
    return ga / (ga + gb)

def support():
    return 0.0, 1.0

MEAN = ALPHA / (ALPHA + BETA)
VAR = ALPHA * BETA / ((ALPHA + BETA) ** 2 * (ALPHA + BETA + 1))

def entropy():                   # nats (differential)
    return (lgamma(ALPHA) + lgamma(BETA) - lgamma(ALPHA + BETA) - (ALPHA - 1) * digamma(ALPHA)
            - (BETA - 1) * digamma(BETA) + (ALPHA + BETA - 2) * digamma(ALPHA + BETA))`,
    };
  }
}

export const distributionsPython = (family: Family, params: Params = DEFAULTS[family], k = 1) => {
  const b = familyBlock(family, params);
  const label = FAMILIES[family].label;
  return `import math
import random
from math import lgamma
import numpy as np

# Probability distributions — ${label} (mirrors the lab)
${b.consts}
K_SUM = ${Math.max(1, Math.round(k))}            # 1 = single draws X; k > 1 = sums of k draws (the CLT view)
N_SAMPLES = 20_000 if K_SUM == 1 else 10_000
N_BINS = 36          # continuous histogram bins, as in the lab
U = random.Random(0).random      # uniform draws in [0, 1)

def randn():
    # Box-Muller, as in the lab
    u = 0.0
    while u == 0.0:
        u = U()
    v = 0.0
    while v == 0.0:
        v = U()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)

def gamma_sample(k):
    # Marsaglia-Tsang Gamma(k, 1); shapes below 1 use the U^(1/k) boost
    if k < 1:
        u = U()
        return gamma_sample(1 + k) * u ** (1 / k)
    d = k - 1 / 3
    c = 1 / math.sqrt(9 * d)
    while True:
        x = randn()
        v = 1 + c * x
        if v <= 0:
            continue
        v = v * v * v
        u = U()
        if u < 1 - 0.0331 * x ** 4:
            return d * v
        if math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)):
            return d * v

def digamma(x):
    # recurrence up to x >= 6, then the asymptotic series
    r = 0.0
    while x < 6:
        r -= 1 / x
        x += 1
    f = 1 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))

def norm_cdf(z):
    # Abramowitz-Stegun 7.1.26 erf (|error| <= 1.5e-7), as in the lab
    x = abs(z) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1 - poly * math.exp(-x * x)
    return 0.5 * (1 + erf) if z >= 0 else 0.5 * (1 - erf)

def beta_inc(x, a, b):
    # regularised incomplete beta I_x(a, b) by the Numerical Recipes continued fraction
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    def cf(a, b, x):
        tiny = 1e-300
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > tiny else tiny)
        h = d
        for m in range(1, 301):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; d = d if abs(d) > tiny else tiny
            c = 1 + aa / c; c = c if abs(c) > tiny else tiny
            d = 1 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; d = d if abs(d) > tiny else tiny
            c = 1 + aa / c; c = c if abs(c) > tiny else tiny
            d = 1 / d
            h *= d * c
            if abs(d * c - 1) < 1e-14:
                break
        return h
    bt = math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * cf(a, b, x) / a
    return 1 - bt * cf(b, a, 1 - x) / b

# ---- the family ----
${b.body}

def draw():
    # one value of the lab's random variable: X, or the sum of K_SUM independent draws
    return sample() if K_SUM == 1 else sum(sample() for _ in range(K_SUM))

${FAMILIES[family].kind === 'discrete' ? LAYOUT_DISCRETE : LAYOUT_CONTINUOUS}

if __name__ == "__main__":
    lo, hi, mass, key, out_mass = make_layout()
    counts, out, xs = [0] * len(mass), 0, []
    for _ in range(N_SAMPLES):
        x = draw()
        xs.append(x)
        b = key(x)
        if b is None:
            out += 1
        else:
            counts[b] += 1
    n = len(xs)
    l1 = sum(abs(c / n - q) for c, q in zip(counts, mass)) + abs(out / n - out_mass)
    xs = np.array(xs, dtype=float)
    what = "X" if K_SUM == 1 else f"S_k = sum of {K_SUM} draws"
    print(f"{what}: analytic mean = {K_SUM * MEAN:.4f}   var = {K_SUM * VAR:.4f}")
    print(f"{' ' * len(what)}  sample   mean = {xs.mean():.4f}   var = {xs.var():.4f}   (n = {n})")
    if K_SUM == 1:
        print(f"entropy = {entropy():.4f} nats")
    print(f"out of the plotted window: {out} ({out / n * 100:.2f}%), exact share {out_mass * 100:.2f}%")
    ref = "the exact law" if K_SUM == 1 else "the CLT normal N(k mu, k var)"
    print(f"L1 distance of the histogram to {ref}: {l1:.4f}")
`;
};

// ---------------------------------------------------------------------------
// MCMC
// ---------------------------------------------------------------------------

export const mcmcPython = (
  preset: TargetKey,
  comps: Comp[],
  x0: number,
  proposalSigma: number,
  burnIn: number,
) => {
  const [lo, hi] = TARGETS[preset].domain;
  return `import numpy as np

# MCMC — random-walk Metropolis-Hastings on a 1-D mixture target (mirrors the lab)
# Target preset = ${preset}. We only ever need RATIOS of pi, so the normalising
# constant cancels in the accept probability.
COMPONENTS = [${comps.map((c) => `(${py(c.w)}, ${py(c.mu)}, ${py(c.sd)})`).join(', ')}]   # (weight, mean, std)
SIGMA   = ${py(proposalSigma)}         # proposal std-dev: x' = x + Normal(0, SIGMA)
X0      = ${py(x0)}         # start far out in the right tail, as in the lab
BURN_IN = ${Math.max(0, Math.round(burnIn))}          # initial iterations discarded
N_ITERS = 20_000
DOMAIN, N_BINS = (${py(lo)}, ${py(hi)}), 60      # the lab's histogram window
ACC_LO, ACC_HI = 0.2, 0.5          # the lab's acceptance band
rng = np.random.default_rng(0)

def target(x):
    p = 0.0
    for w, mu, sd in COMPONENTS:
        p += w * np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return p

def near_a_mode(x):
    return any(abs(x - mu) < 1.5 * sd for _, mu, sd in COMPONENTS)

def metropolis_hastings(n=N_ITERS, sigma=SIGMA, x0=X0):
    x = x0
    chain = np.empty(n)
    accepted, first_hit = 0, None
    for t in range(n):
        x_prop = x + sigma * rng.standard_normal()                 # symmetric proposal
        ratio = target(x_prop) / max(1e-300, target(x))             # pi(x')/pi(x)
        if rng.random() < min(1.0, ratio):                          # accept / reject
            x = x_prop
            accepted += 1
        chain[t] = x                          # a rejection repeats the current state
        if first_hit is None and near_a_mode(x):
            first_hit = t + 1
    return chain, accepted / n, first_hit

if __name__ == "__main__":
    chain, acc, first_hit = metropolis_hastings()
    kept = chain[BURN_IN:]                                      # discard burn-in
    band = "low" if acc < ACC_LO else "high" if acc > ACC_HI else "inside the 20-50% band"
    print(f"acceptance rate = {acc:.3f}  ({band})")
    print(f"first came within 1.5 sd of a mode at iteration {first_hit}  (burn-in = {BURN_IN})")
    if first_hit is not None and first_hit > BURN_IN:
        print("  -> burn-in is shorter than the transient: tail samples leak into the kept set")
    print(f"kept samples: mean = {kept.mean():.4f}   std = {kept.std():.4f}")
    exact_mean = sum(w * mu for w, mu, _ in COMPONENTS)
    print(f"target mean  = {exact_mean:.4f}")
    # histogram of the kept states, normalised by the number of kept iterations
    lo, hi = DOMAIN
    w = (hi - lo) / N_BINS
    counts = np.zeros(N_BINS)
    for v in kept:
        if lo <= v < hi:
            counts[min(N_BINS - 1, int((v - lo) // w))] += 1
    density = counts / len(kept) / w
    centres = lo + (np.arange(N_BINS) + 0.5) * w
    print(f"max |histogram - target| over the bins = {np.max(np.abs(density - target(centres))):.4f}")
    # mixing diagnostic: lag-1 autocorrelation (lower = better mixing)
    s = kept - kept.mean()
    ac1 = float(np.sum(s[1:] * s[:-1]) / np.sum(s * s))
    print(f"lag-1 autocorr  = {ac1:.3f}  (closer to 1 = more correlated samples, slower mixing)")
`;
};

// ---------------------------------------------------------------------------
// Samples for scripts/check-python-exports.mjs
// ---------------------------------------------------------------------------

/** Deterministic pseudo-random flips for the samples (a tiny LCG, not the lab's RNG). */
function fakeFlips(n: number, p: number, seed: number): number[] {
  let s = seed >>> 0;
  const out: number[] = [];
  for (let i = 0; i < n; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; out.push(s / 4294967296 < p ? 1 : 0); }
  return out;
}

const distributionSamples: PythonSample[] = [
  ...ORDER.map((f) => ({ name: `dist-${f}-default`, code: () => distributionsPython(f, DEFAULTS[f], 1) })),
  ...ORDER.map((f) => ({ name: `dist-${f}-sum10`, code: () => distributionsPython(f, DEFAULTS[f], 10) })),
  { name: 'dist-bernoulli-p001-sum50', code: () => distributionsPython('bernoulli', { p: 0.01 }, 50) },
  { name: 'dist-binomial-n40-p099', code: () => distributionsPython('binomial', { n: 40, p: 0.99 }, 1) },
  { name: 'dist-binomial-n35-p07', code: () => distributionsPython('binomial', { n: 35, p: 0.7 }, 1) },
  { name: 'dist-poisson-lam18-sum50', code: () => distributionsPython('poisson', { lam: 18 }, 50), timeoutSec: 180 },
  { name: 'dist-poisson-lam02', code: () => distributionsPython('poisson', { lam: 0.2 }, 1) },
  { name: 'dist-geometric-p005', code: () => distributionsPython('geometric', { p: 0.05 }, 1) },
  { name: 'dist-geometric-p095-sum2', code: () => distributionsPython('geometric', { p: 0.95 }, 2) },
  { name: 'dist-uniform-wide', code: () => distributionsPython('uniform', { a: -3, b: 4 }, 1) },
  { name: 'dist-normal-narrow', code: () => distributionsPython('normal', { mu: 3, sigma: 0.2 }, 1) },
  { name: 'dist-exponential-lam03', code: () => distributionsPython('exponential', { lam: 0.3 }, 1) },
  { name: 'dist-exponential-lam4-sum2', code: () => distributionsPython('exponential', { lam: 4 }, 2) },
  { name: 'dist-beta-half-half', code: () => distributionsPython('beta', { a: 0.5, b: 0.5 }, 1) },
  { name: 'dist-beta-5-2', code: () => distributionsPython('beta', { a: 5, b: 2 }, 1) },
  { name: 'dist-beta-8-half-sum50', code: () => distributionsPython('beta', { a: 8, b: 0.5 }, 50), timeoutSec: 180 },
];

export const PYTHON_SAMPLES: PythonSample[] = [
  // Bayes — diagnostic: defaults, every preset, slider extremes
  { name: 'bayes-diag-default', code: () => bayesPython('diagnostic', 0.01, 0.99, 0.95, 0.7) },
  { name: 'bayes-diag-common', code: () => bayesPython('diagnostic', 0.3, 0.9, 0.9, 0.7) },
  { name: 'bayes-diag-spam', code: () => bayesPython('diagnostic', 0.5, 0.98, 0.97, 0.7) },
  { name: 'bayes-diag-rarest', code: () => bayesPython('diagnostic', 0.001, 0.999, 0.999, 0.7) },
  { name: 'bayes-diag-coinflip-test', code: () => bayesPython('diagnostic', 0.6, 0.5, 0.5, 0.7) },
  // Bayes — sequential: every prior, with and without recorded flips, bias extremes
  { name: 'bayes-seq-flat-noflips', code: () => bayesPython('sequential', 0.01, 0.99, 0.95, 0.7, 1, 1, []) },
  { name: 'bayes-seq-mild-40', code: () => bayesPython('sequential', 0.01, 0.99, 0.95, 0.7, 5, 5, fakeFlips(40, 0.7, 1)) },
  { name: 'bayes-seq-skeptical-1', code: () => bayesPython('sequential', 0.01, 0.99, 0.95, 0.7, 2, 8, [1]) },
  { name: 'bayes-seq-stubborn-3000', code: () => bayesPython('sequential', 0.01, 0.99, 0.95, 0.7, 20, 20, fakeFlips(3000, 0.7, 7)) },
  { name: 'bayes-seq-p002', code: () => bayesPython('sequential', 0.01, 0.99, 0.95, 0.02, 1, 1, fakeFlips(150, 0.02, 3)) },
  { name: 'bayes-seq-p098', code: () => bayesPython('sequential', 0.01, 0.99, 0.95, 0.98, 20, 20, []) },
  // Distributions — every family (single and sums) + slider edges
  ...distributionSamples,
  // MCMC — every target at the defaults, and both pathologies
  ...TARGET_KEYS.map((k) => ({ name: `mcmc-${k}-default`, code: () => mcmcPython(k, TARGETS[k].comps, TARGETS[k].x0, 3, 200) })),
  { name: 'mcmc-bimodal-tiny-sigma-noburn', code: () => mcmcPython('bimodal', TARGETS.bimodal.comps, TARGETS.bimodal.x0, 0.05, 0) },
  { name: 'mcmc-trimodal-huge-sigma', code: () => mcmcPython('trimodal', TARGETS.trimodal.comps, TARGETS.trimodal.x0, 15, 2000) },
  { name: 'mcmc-skewed-sigma01', code: () => mcmcPython('skewed', TARGETS.skewed.comps, TARGETS.skewed.x0, 0.1, 200) },
];
