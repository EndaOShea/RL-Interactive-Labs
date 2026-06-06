// Runnable Python exports for the Diffusion labs (NumPy, mirrors the labs).

export const forwardReversePython = (schedule: string, T: number, dataset: string) => `import numpy as np
import matplotlib.pyplot as plt

# Forward & reverse diffusion on a 2-D toy distribution.
# The forward marginal is exact:  x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps
# We store x0 AND a fixed eps per point, so reversing t replays the EXACT
# marginals (no neural net) — a real model would predict eps with eps_theta(x_t, t).

SCHEDULE = "${schedule}"   # "cosine" | "linear"
T        = ${T}
DATASET  = "${dataset}"    # "two-moons" | "ring" | "blobs"
N        = 600
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

def x_at(t):
    ab = alpha_bar(t, T, SCHEDULE)
    return np.sqrt(ab) * X0 + np.sqrt(1 - ab) * eps, ab

if __name__ == "__main__":
    for t in [0, T // 4, T // 2, T]:
        xt, ab = x_at(t)
        snr = ab / (1 - ab + 1e-8)
        print(f"t={t:4d}  abar={ab:.4f}  SNR={snr:8.3f}")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, t in zip(axes, [0, T // 4, T // 2, T]):
        xt, ab = x_at(t)
        ax.scatter(xt[:, 0], xt[:, 1], c=cls, s=6, cmap="coolwarm")
        ax.set_title(f"t={t}  abar={ab:.2f}"); ax.set_aspect("equal")
    plt.tight_layout(); plt.show()
`;

export const noiseSchedulePython = (schedule: string, T: number) => `import numpy as np
import matplotlib.pyplot as plt

# Noise schedules: beta_t, alpha_t, alpha-bar_t and SNR for linear vs cosine.
T = ${T}
HIGHLIGHT = "${schedule}"   # "linear" | "cosine"

def linear_schedule(T, b0=1e-4, b1=0.02):
    betas = b0 + (b1 - b0) * (np.arange(1, T + 1) / T)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    return betas, alphas, abar

def cosine_schedule(T, s=0.008):
    t = np.arange(0, T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    abar = f / f[0]
    abar = abar[1:]
    abar_prev = np.concatenate([[1.0], abar[:-1]])
    alphas = abar / abar_prev
    betas = np.clip(1.0 - alphas, 0, 0.999)
    return betas, alphas, abar

def snr(abar):
    return abar / (1.0 - abar + 1e-8)

if __name__ == "__main__":
    lb, la, lab = linear_schedule(T)
    cb, ca, cab = cosine_schedule(T)
    t = np.arange(1, T + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t, lab, label="abar linear"); ax[0].plot(t, cab, label="abar cosine")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("alpha-bar"); ax[0].legend()
    ax[1].plot(t, np.log(snr(lab)), label="log-SNR linear")
    ax[1].plot(t, np.log(snr(cab)), label="log-SNR cosine")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("log SNR"); ax[1].legend()
    plt.tight_layout(); plt.show()
`;
