// Runnable Python exports for the Probability & Bayesian labs (template strings —
// not LLM generated), mirroring the on-screen functions and parameters. No scipy:
// every pmf / pdf / sampler is implemented by hand with plain NumPy.

export type BayesMode = 'diagnostic' | 'sequential';

export const bayesPython = (
  mode: BayesMode,
  prevalence: number,
  sensitivity: number,
  specificity: number,
  trueP: number,
) => {
  if (mode === 'sequential') {
    return `import numpy as np

# Bayes' theorem — Sequential Beta-Bernoulli updating (mirrors the lab)
# Prior:      p ~ Beta(ALPHA0, BETA0)
# Likelihood: each coin flip is Bernoulli(p)
# Posterior:  Beta is conjugate, so heads -> alpha+1, tails -> beta+1.
ALPHA0, BETA0 = 1.0, 1.0     # uniform prior on p in [0, 1]
TRUE_P = ${trueP}            # the (hidden) bias we are estimating
N_FLIPS = 200
rng = np.random.default_rng(0)

def beta_pdf(x, a, b):
    # Unnormalised Beta density x^(a-1)(1-x)^(b-1) divided by the Beta function,
    # which we get from log-gammas to stay stable for large a, b.
    from math import lgamma, log, exp
    logB = lgamma(a) + lgamma(b) - lgamma(a + b)
    out = np.zeros_like(x)
    m = (x > 0) & (x < 1)
    out[m] = np.exp((a - 1) * np.log(x[m]) + (b - 1) * np.log(1 - x[m]) - logB)
    return out

def credible_interval(a, b, mass=0.90, grid=20001):
    # Equal-tailed interval from the CDF of a grid-discretised Beta.
    xs = np.linspace(0, 1, grid)
    pdf = beta_pdf(xs, a, b)
    cdf = np.cumsum(pdf); cdf /= cdf[-1]
    lo = xs[np.searchsorted(cdf, (1 - mass) / 2)]
    hi = xs[np.searchsorted(cdf, 1 - (1 - mass) / 2)]
    return lo, hi

if __name__ == "__main__":
    a, b = ALPHA0, BETA0
    for t in range(N_FLIPS):
        if rng.random() < TRUE_P:    # heads
            a += 1                    # conjugate update: alpha + 1
        else:                         # tails
            b += 1                    # conjugate update: beta + 1
    mean = a / (a + b)
    lo, hi = credible_interval(a, b)
    print(f"after {N_FLIPS} flips: Beta({a:.0f}, {b:.0f})")
    print(f"posterior mean = {mean:.4f}  (true p = {TRUE_P})")
    print(f"90% credible interval = [{lo:.4f}, {hi:.4f}]")
`;
  }

  return `import numpy as np

# Bayes' theorem — Diagnostic test / base-rate fallacy (mirrors the lab)
#   P(D|+) = P(+|D) P(D) / [ P(+|D) P(D) + P(+|~D) P(~D) ]
PREVALENCE  = ${prevalence}     # P(D)  — the prior / base rate
SENSITIVITY = ${sensitivity}     # P(+|D)  — true-positive rate
SPECIFICITY = ${specificity}     # P(-|~D) — true-negative rate

def posterior_positive(prior, sens, spec):
    p_pos = sens * prior + (1 - spec) * (1 - prior)   # total prob of a + test
    return sens * prior / p_pos                         # P(D|+)

def posterior_negative(prior, sens, spec):
    p_neg = (1 - sens) * prior + spec * (1 - prior)    # total prob of a - test
    return (1 - sens) * prior / p_neg                   # P(D|-)

if __name__ == "__main__":
    post_pos = posterior_positive(PREVALENCE, SENSITIVITY, SPECIFICITY)
    post_neg = posterior_negative(PREVALENCE, SENSITIVITY, SPECIFICITY)
    # population of 100,000 broken into the 2x2 confusion counts
    N = 100_000
    sick = PREVALENCE * N
    tp = sick * SENSITIVITY
    fn = sick - tp
    healthy = N - sick
    tn = healthy * SPECIFICITY
    fp = healthy - tn
    print(f"prior  P(D)   = {PREVALENCE:.4f}")
    print(f"P(D|+) = {post_pos:.4f}   P(D|-) = {post_neg:.6f}")
    print(f"of {N} people: TP={tp:.0f} FP={fp:.0f} TN={tn:.0f} FN={fn:.0f}")
    print(f"among the {tp+fp:.0f} positives, only {tp:.0f} are truly sick "
          f"({post_pos*100:.1f}%) — the base-rate fallacy.")
`;
};

export type Family =
  | 'bernoulli' | 'binomial' | 'poisson' | 'geometric'
  | 'uniform' | 'normal' | 'exponential' | 'beta';

export const distributionsPython = (family: Family) => {
  const blocks: Record<Family, { kind: 'pmf' | 'pdf'; params: string; def: string; stats: string; sampler: string }> = {
    bernoulli: {
      kind: 'pmf',
      params: 'P = 0.4              # success probability',
      def: `def pmf(k):
    return P if k == 1 else (1 - P)`,
      stats: `mean = P
var  = P * (1 - P)`,
      sampler: `def sample():
    return 1 if rng.random() < P else 0`,
    },
    binomial: {
      kind: 'pmf',
      params: 'N = 20\nP = 0.4             # n trials, success prob',
      def: `from math import comb
def pmf(k):
    return comb(N, k) * P ** k * (1 - P) ** (N - k)`,
      stats: `mean = N * P
var  = N * P * (1 - P)`,
      sampler: `def sample():
    return int(np.sum(rng.random(N) < P))`,
    },
    poisson: {
      kind: 'pmf',
      params: 'LAM = 4.0            # rate / expected count',
      def: `from math import exp, factorial
def pmf(k):
    return exp(-LAM) * LAM ** k / factorial(k)`,
      stats: `mean = LAM
var  = LAM`,
      sampler: `def sample():
    # Knuth's algorithm: count Poisson events by an exponential-gap product.
    L, k, p = np.exp(-LAM), 0, 1.0
    while p > L:
        k += 1; p *= rng.random()
    return k - 1`,
    },
    geometric: {
      kind: 'pmf',
      params: 'P = 0.35             # success prob (trials until first success)',
      def: `def pmf(k):                 # k = 1, 2, 3, ... (number of trials)
    return (1 - P) ** (k - 1) * P`,
      stats: `mean = 1 / P
var  = (1 - P) / P ** 2`,
      sampler: `def sample():
    # inverse-CDF: ceil(log(u)/log(1-p))
    return int(np.ceil(np.log(rng.random()) / np.log(1 - P)))`,
    },
    uniform: {
      kind: 'pdf',
      params: 'A, B = 0.0, 1.0      # support [a, b]',
      def: `def pdf(x):
    return 1.0 / (B - A) if A <= x <= B else 0.0`,
      stats: `mean = (A + B) / 2
var  = (B - A) ** 2 / 12`,
      sampler: `def sample():
    return A + (B - A) * rng.random()`,
    },
    normal: {
      kind: 'pdf',
      params: 'MU, SIGMA = 0.0, 1.0  # mean, std-dev',
      def: `def pdf(x):
    z = (x - MU) / SIGMA
    return np.exp(-0.5 * z * z) / (SIGMA * np.sqrt(2 * np.pi))`,
      stats: `mean = MU
var  = SIGMA ** 2`,
      sampler: `def sample():
    return MU + SIGMA * rng.standard_normal()   # Box-Muller under the hood`,
    },
    exponential: {
      kind: 'pdf',
      params: 'LAM = 1.0            # rate (mean = 1/lambda)',
      def: `def pdf(x):
    return LAM * np.exp(-LAM * x) if x >= 0 else 0.0`,
      stats: `mean = 1 / LAM
var  = 1 / LAM ** 2`,
      sampler: `def sample():
    return -np.log(rng.random()) / LAM          # inverse-CDF`,
    },
    beta: {
      kind: 'pdf',
      params: 'A, B = 2.0, 5.0      # shape parameters',
      def: `from math import lgamma
def pdf(x):
    if not (0 < x < 1):
        return 0.0
    logB = lgamma(A) + lgamma(B) - lgamma(A + B)
    return np.exp((A - 1) * np.log(x) + (B - 1) * np.log(1 - x) - logB)`,
      stats: `mean = A / (A + B)
var  = (A * B) / ((A + B) ** 2 * (A + B + 1))`,
      sampler: `def sample():
    # ratio of Gammas; Gammas via summed exponentials for integer-ish shapes.
    g = lambda s: np.sum(-np.log(rng.random(max(1, int(round(s))))))
    ga, gb = g(A), g(B)
    return ga / (ga + gb)`,
    },
  };

  const b = blocks[family];
  const isDiscrete = b.kind === 'pmf';

  return `import numpy as np

# Probability distributions — ${family} (mirrors the lab)
# ${isDiscrete ? 'Discrete: probability MASS function pmf(k)' : 'Continuous: probability DENSITY function pdf(x)'}
${b.params}
rng = np.random.default_rng(0)

${b.def}

${b.sampler}

def moments():
    ${b.stats.split('\n').join('\n    ')}
    return mean, var

def empirical_histogram(n_samples=100_000, bins=40):
    # Law of Large Numbers: the sample histogram converges to the true ${isDiscrete ? 'pmf' : 'pdf'}.
    xs = np.array([sample() for _ in range(n_samples)])
    hist, edges = np.histogram(xs, bins=bins, density=True)
    return hist, edges, xs.mean(), xs.var()

if __name__ == "__main__":
    mean, var = moments()
    print(f"analytic   mean = {mean:.4f}   var = {var:.4f}")
    _, _, emp_mean, emp_var = empirical_histogram()
    print(f"empirical  mean = {emp_mean:.4f}   var = {emp_var:.4f}  (LLN)")
`;
};

export const mcmcPython = (
  preset: string,
  proposalSigma: number,
) => `import numpy as np

# MCMC — Metropolis-Hastings on a 1-D multimodal target (mirrors the lab)
# Target (unnormalised): a mixture of Gaussians. We only ever need RATIOS of
# pi, so the normalising constant cancels in the accept probability.
# preset = ${preset}
SIGMA   = ${proposalSigma}      # proposal std-dev: x' = x + Normal(0, SIGMA)
N_ITERS = 20_000
BURN_IN = 2_000
rng = np.random.default_rng(0)

# mixture components: (weight, mean, std)
COMPONENTS = [(0.5, -2.0, 0.6), (0.5, 2.0, 0.8)]

def target(x):
    p = 0.0
    for w, mu, sd in COMPONENTS:
        p += w * np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return p

def metropolis_hastings(n=N_ITERS, sigma=SIGMA, x0=0.0):
    x = x0
    chain = np.empty(n)
    accepted = 0
    for t in range(n):
        x_prop = x + sigma * rng.standard_normal()           # symmetric proposal
        ratio = target(x_prop) / target(x)                    # pi(x')/pi(x)
        if rng.random() < min(1.0, ratio):                    # accept / reject
            x = x_prop
            accepted += 1
        chain[t] = x
    return chain, accepted / n

if __name__ == "__main__":
    chain, acc_rate = metropolis_hastings()
    samples = chain[BURN_IN:]                                  # discard burn-in
    print(f"acceptance rate = {acc_rate:.3f}  (sweet spot ~ 0.2-0.5)")
    print(f"posterior mean  = {samples.mean():.4f}")
    print(f"posterior std   = {samples.std():.4f}")
    # mixing diagnostic: lag-1 autocorrelation (lower = better mixing)
    s = samples - samples.mean()
    ac1 = float(np.sum(s[1:] * s[:-1]) / np.sum(s * s))
    print(f"lag-1 autocorr  = {ac1:.3f}  (sigma too small -> high autocorr, slow mixing)")
`;
