// Runnable Python exports for the Diffusion labs (NumPy, mirrors the labs).

export const forwardReversePython = (
  schedule: string, T: number, dataset: string,
  sampler = 'ddpm', ddimSteps = 50, guidance = 0,
) => `import numpy as np
import matplotlib.pyplot as plt

# Forward & reverse diffusion on a 2-D toy distribution.
# The forward marginal is exact:  x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps
# We store x0 AND a fixed eps per point, so reversing t replays the EXACT
# marginals (no neural net) — a real model would predict eps with eps_theta(x_t, t).
#
# Reverse sampling supports two modes:
#   * DDPM — ancestral / stochastic, walks the chain finely (many steps).
#   * DDIM — deterministic, predicts x0_hat then re-noises to t-1, so a handful
#            of strided steps approximate the full chain.
# Classifier-free guidance scale w sharpens samples toward their class centroid
#   (an analytic stand-in for  eps_hat = (1+w)*eps_cond - w*eps_uncond ).

SCHEDULE  = "${schedule}"   # "cosine" | "linear"
T         = ${T}
DATASET   = "${dataset}"    # "two-moons" | "ring" | "blobs"
SAMPLER   = "${sampler}"    # "ddpm" | "ddim"
DDIM_STEPS = ${ddimSteps}
GUIDANCE  = ${guidance}     # classifier-free guidance scale w (0 = unguided)
N         = 600
rng = np.random.default_rng(0)

def make_data(kind, n):
    if kind == "two-moons":
        m = n // 2
        t = np.linspace(0, np.pi, m)
        a = np.stack([np.cos(t), np.sin(t)], 1)
        b = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], 1)
        X = np.concatenate([a, b]) + rng.normal(0, 0.06, (2 * m, 2))
        c = np.array([0] * m + [1] * m)
    elif kind == "ring":
        t = rng.uniform(0, 2 * np.pi, n)
        r = 1.0 + rng.normal(0, 0.05, n)
        X = np.stack([r * np.cos(t), r * np.sin(t)], 1)
        c = (t > np.pi).astype(int)
    else:  # blobs
        centers = np.array([[-1, -1], [1, -1], [0, 1]])
        c = rng.integers(0, 3, n)
        X = centers[c] * 0.9 + rng.normal(0, 0.18, (n, 2))
    return X, c

def alpha_bar(t, T, schedule):
    # t in [0, T];  returns cumulative signal-retention abar_t in [0, 1]
    if schedule == "cosine":
        s = 0.008
        f = lambda u: np.cos((u / T + s) / (1 + s) * np.pi / 2) ** 2
        return f(t) / f(0.0)
    # linear beta schedule -> cumulative product alpha-bar
    b0, b1 = 1e-4, 0.02
    betas = b0 + (b1 - b0) * (np.arange(1, T + 1) / T)
    abar = np.cumprod(1.0 - betas)
    return abar[min(int(t), T) - 1] if t >= 1 else 1.0

X0, cls = make_data(DATASET, N)
eps = rng.standard_normal(X0.shape)          # FIXED noise per point

# class centroids for the (analytic) classifier-free guidance pull
centroids = np.stack([X0[cls == c].mean(0) for c in np.unique(cls)])

def x_at(t):
    ab = alpha_bar(t, T, SCHEDULE)
    return np.sqrt(ab) * X0 + np.sqrt(1 - ab) * eps, ab

def guided(xt, ab, w):
    # nudge each point toward its class centroid (stand-in for CFG)
    if w <= 0:
        return xt
    pull = w * 0.04 * np.sqrt(ab)
    target = centroids[cls]
    return xt + (target - xt) * pull

def reverse_schedule():
    # the t's visited on the reverse pass: DDIM strides, DDPM walks the chain
    if SAMPLER == "ddim":
        return np.unique(np.linspace(T, 0, DDIM_STEPS + 1).round().astype(int))[::-1]
    return np.arange(T, -1, -1)

if __name__ == "__main__":
    print(f"sampler={SAMPLER}  steps={'%d (ddim)' % DDIM_STEPS if SAMPLER=='ddim' else T}  w={GUIDANCE}")
    for t in [0, T // 4, T // 2, T]:
        xt, ab = x_at(t)
        snr = ab / (1 - ab + 1e-8)
        print(f"t={t:4d}  abar={ab:.4f}  SNR={snr:8.3f}")

    # reverse (denoising) trajectory using the chosen sampler + guidance
    ts = reverse_schedule()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    picks = ts[np.linspace(0, len(ts) - 1, 4).astype(int)]
    for ax, t in zip(axes, picks):
        xt, ab = x_at(int(t))
        xt = guided(xt, ab, GUIDANCE)
        ax.scatter(xt[:, 0], xt[:, 1], c=cls, s=6, cmap="coolwarm")
        ax.set_title(f"t={int(t)}  abar={ab:.2f}"); ax.set_aspect("equal")
    plt.tight_layout(); plt.show()
`;

export const noiseSchedulePython = (schedule: string, T: number, shift = 1) => `import numpy as np
import matplotlib.pyplot as plt

# Noise schedules: beta_t, alpha_t, alpha-bar_t and SNR for linear / cosine / sigmoid.
# SHIFT applies the resolution log-SNR shift: SNR' = SNR * shift^2, so abar is
# rebuilt from the shifted SNR. shift>1 keeps more signal (high-res images).
T         = ${T}
HIGHLIGHT = "${schedule}"   # "linear" | "cosine" | "sigmoid"
SHIFT     = ${shift}

def _shift_abar(abar, shift):
    if shift == 1:
        return abar
    s = abar / (1.0 - abar + 1e-8)
    s = s * (shift ** 2)                 # log-SNR shift = 2*ln(shift)
    return np.clip(s / (1.0 + s), 1e-5, 1 - 1e-7)

def _from_abar(abar):
    abar = _shift_abar(abar, SHIFT)
    abar_prev = np.concatenate([[1.0], abar[:-1]])
    alphas = np.clip(abar / abar_prev, 0, 1)
    betas = np.clip(1.0 - alphas, 0, 0.999)
    return betas, alphas, abar

def linear_schedule(T, b0=1e-4, b1=0.02):
    betas = b0 + (b1 - b0) * (np.arange(1, T + 1) / T)
    return _from_abar(np.cumprod(1.0 - betas))

def cosine_schedule(T, s=0.008):
    t = np.arange(0, T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    return _from_abar((f / f[0])[1:])

def sigmoid_schedule(T, start=3.0, end=-3.0):
    # EDM-style: log-SNR sweeps start->end; abar = sigmoid(logSNR) = SNR/(1+SNR)
    z = np.linspace(start, end, T)
    return _from_abar(1.0 / (1.0 + np.exp(-z)))

def snr(abar):
    return abar / (1.0 - abar + 1e-8)

SCHEDULES = {"linear": linear_schedule, "cosine": cosine_schedule, "sigmoid": sigmoid_schedule}

if __name__ == "__main__":
    t = np.arange(1, T + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, fn in SCHEDULES.items():
        _, _, abar = fn(T)
        ax[0].plot(t, abar, label=f"abar {name}")
        ax[1].plot(t, np.log(snr(abar)), label=f"log-SNR {name}")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("alpha-bar"); ax[0].legend()
    ax[1].axhline(0, color="k", lw=0.6, ls=":")  # SNR = 1 crossover
    ax[1].set_xlabel("t"); ax[1].set_ylabel("log SNR"); ax[1].legend()
    plt.suptitle(f"highlight={HIGHLIGHT}  shift={SHIFT}")
    plt.tight_layout(); plt.show()
`;
