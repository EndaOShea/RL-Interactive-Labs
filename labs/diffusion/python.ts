// Runnable Python exports for the Diffusion labs. Each script mirrors its lab
// exactly: the same seeded generator (a port of mulberry32 + Box–Muller) and noise
// streams, the same datasets, the same schedules (schedules.ts), the same exact
// KDE denoiser and DDPM / DDIM samplers with classifier-free guidance
// (sampler.ts), and the same metrics — so each script prints the numbers the lab
// shows for the parameters it was downloaded with.
import type { PythonSample } from '../../utils/pythonSamples';

// Shared Python: schedule construction, identical to schedules.ts.
const SCHEDULE_PY = `BETA_MAX = 0.999
COSINE_S = 0.008
SIGMOID_START, SIGMOID_END, SIGMOID_TAU = -3.0, 3.0, 1.0      # Chen 2023, Algorithm 1 defaults
EDM_SIGMA_MIN, EDM_SIGMA_MAX, EDM_RHO = 0.002, 80.0, 7.0      # Karras et al. 2022 (EDM)

def _sigm(z):
    return 1 / (1 + math.exp(-z))

def linear_betas(T):
    # DDPM's 1e-4 -> 0.02 (tuned for T = 1000) scaled by 1000/T: beta_1 = 0.1/T, beta_T = 20/T
    scale = 1000 / T
    b0 = 1e-4 * scale
    b1 = 0.02 * scale
    return [b0 + (b1 - b0) * i / (T - 1) if T > 1 else b0 for i in range(T)]

def raw_abar(kind, T):
    # raw signal-retention curve abar(t), t = 0..T, abar(0) = 1
    out = [1.0]
    if kind == "linear":
        p = 1.0
        for b in linear_betas(T):
            p *= 1 - b
            out.append(p)
    elif kind == "cosine":
        def f(t):
            c = math.cos(((t / T + COSINE_S) / (1 + COSINE_S)) * (math.pi / 2))
            return c * c
        f0 = f(0)
        out += [f(t) / f0 for t in range(1, T + 1)]
    elif kind == "sigmoid":
        v_start = _sigm(SIGMOID_START / SIGMOID_TAU)
        v_end = _sigm(SIGMOID_END / SIGMOID_TAU)
        for t in range(1, T + 1):
            u = t / T
            out.append((v_end - _sigm((u * (SIGMOID_END - SIGMOID_START) + SIGMOID_START) / SIGMOID_TAU)) / (v_end - v_start))
    else:  # edm
        a = math.pow(EDM_SIGMA_MIN, 1 / EDM_RHO)
        b = math.pow(EDM_SIGMA_MAX, 1 / EDM_RHO)
        for t in range(1, T + 1):
            u = (t - 1) / (T - 1) if T > 1 else 1.0
            sigma = math.pow(a + u * (b - a), EDM_RHO)
            out.append(1 / (1 + sigma * sigma))
    return out

def build_schedule(kind, T, shift=1):
    # raw abar -> optional log-SNR shift (SNR' = SNR / shift^2) -> beta_t = min(0.999, 1 - abar_t/abar_{t-1})
    # -> abar_t = prod(1 - beta_i). Arrays are indexed t = 0..T (index 0 = clean data).
    beta = [0.0]
    if kind == "linear" and shift == 1:
        beta += linear_betas(T)
    else:
        raw = raw_abar(kind, T)
        k = shift * shift
        ab = raw if shift == 1 else [a / (a + k * (1 - a)) for a in raw]
        for t in range(1, T + 1):
            beta.append(min(BETA_MAX, max(0.0, 1 - ab[t] / ab[t - 1])))
    alpha = [1 - b for b in beta]
    abar = [1.0]
    for t in range(1, T + 1):
        abar.append(abar[t - 1] * alpha[t])
    beta, alpha, abar = np.array(beta), np.array(alpha), np.array(abar)
    logsnr = np.full(T + 1, np.inf)
    logsnr[1:] = np.log(abar[1:] / (1 - abar[1:]))
    return beta, alpha, abar, logsnr

def snr_crossing(logsnr):
    # fractional t* where ln SNR crosses 0 (linear interpolation between steps), or None
    for t in range(1, len(logsnr)):
        if logsnr[t] <= 0:
            prev = logsnr[t - 1]
            if not math.isfinite(prev):
                return float(t)
            return t - 1 + prev / (prev - logsnr[t])
    return None
`;

export const forwardReversePython = (
  schedule: string, T: number, dataset: string,
  sampler = 'ddpm', ddimSteps = 50, guidance = 0, seed = 1,
) => `import math
import numpy as np
import matplotlib.pyplot as plt

# Forward & reverse diffusion on a 2-D toy distribution — mirrors the lab exactly.
#
# Forward:  x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps   (the exact marginal).
#           One FIXED eps per point, so every frame is an exact sample of q(x_t)
#           (consecutive frames are not one Markov sample path).
# Reverse:  starts from FRESH noise x_T ~ N(0, I). The denoiser is analytic: the exact
#           posterior mean E[x0 | x_t] when the data distribution is a Gaussian kernel
#           density over the dataset (one N(mu_i, S_KDE^2 I) component per point) --
#           what a perfectly trained eps-network would output for that density.
#           Nothing is trained.
#   DDPM:   x_{t-1} = (x_t - beta_t / sqrt(1 - abar_t) * eps_hat) / sqrt(alpha_t) + sigma_t * z
#           sigma_t^2 = beta~_t = (1 - abar_{t-1}) / (1 - abar_t) * beta_t, fresh z each step
#           (beta~_1 = 0, so the last step adds no noise). Visits all T steps.
#   DDIM:   eta = 0, visits t = round(k * T / S) for k = S..0:
#           x0_hat = (x_t - sqrt(1 - abar_t) * eps_hat) / sqrt(abar_t)
#           x_t'   = sqrt(abar_t') * x0_hat + sqrt(1 - abar_t') * eps_hat
#   CFG:    eps_hat = (1 + w) * eps_cond - w * eps_uncond. eps_cond uses the sample's own
#           class mixture, eps_uncond all points; sample i is conditioned on class c_i.
# The seeded generator is a port of the lab's (mulberry32 + Box-Muller), so the dataset,
# the noise and the samples are the ones you saw on screen.

SCHEDULE   = "${schedule}"      # "cosine" | "linear"
T          = ${T}
DATASET    = "${dataset}"   # "two-moons" | "ring" | "blobs"
SAMPLER    = "${sampler}"        # "ddpm" (all T steps) | "ddim" (DDIM_STEPS strided steps)
DDIM_STEPS = ${ddimSteps}            # S, capped at T
GUIDANCE   = ${guidance}             # classifier-free guidance scale w (0 = plain conditional)
SEED       = ${seed}             # the lab's seed ("New Map" increments it)
N          = 500
S_KDE      = 0.05          # KDE bandwidth s

# ---------- seeded generator: a port of the lab's mulberry32 + Box-Muller ----------
M32 = 0xFFFFFFFF

def mulberry32(seed):
    state = [seed & M32]
    def rand():
        a = (state[0] + 0x6D2B79F5) & M32
        state[0] = a
        t = ((a ^ (a >> 15)) * (a | 1)) & M32
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296.0
    return rand

def gauss(r):
    u = 0.0
    while u == 0.0:
        u = r()
    v = 0.0
    while v == 0.0:
        v = r()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)

def stream_seed(seed, stream):
    # independent streams: 1 dataset, 2 forward eps, 3 starting x_T, 4 DDPM z, 5 ideal-KDE reference
    return (seed * 1000003 + stream * 7919) & M32

# ---------- datasets (centred, roughly within +-1.5) ----------
def make_data(kind, n, seed):
    r = mulberry32(stream_seed(seed, 1))
    g = lambda: gauss(r)
    X, C = [], []
    if kind == "two-moons":
        m = n // 2
        for i in range(m):
            t = (i / (m - 1)) * math.pi
            X.append((math.cos(t) - 0.5 + g() * 0.06, math.sin(t) - 0.25 + g() * 0.06)); C.append(0)
            X.append((1 - math.cos(t) - 0.5 + g() * 0.06, 0.25 - math.sin(t) + g() * 0.06)); C.append(1)
        n_classes = 2
    elif kind == "ring":
        for i in range(n):
            t = (i / n) * 2 * math.pi
            rad = 1.05 + g() * 0.05
            X.append((rad * math.cos(t), rad * math.sin(t))); C.append(0 if t > math.pi else 1)
        n_classes = 2
    else:  # blobs
        cx, cy = [-0.85, 0.85, 0.0], [-0.85, -0.85, 0.95]
        for i in range(n):
            c = i % 3
            X.append((cx[c] + g() * 0.17, cy[c] + g() * 0.17)); C.append(c)
        n_classes = 3
    return np.array(X), np.array(C), n_classes

def make_eps(n, seed):
    r = mulberry32(stream_seed(seed, 2))
    return np.array([[gauss(r), gauss(r)] for _ in range(n)])   # one FIXED eps per point

# ---------- schedules (identical to the lab) ----------
${SCHEDULE_PY}
# ---------- the analytic denoiser ----------
def posterior_means(X, cls_x, mu, cls_mu, ab, s2, uncond):
    # Exact E[x0 | x_t] for each row of X under its own class's KDE mixture (and, when
    # uncond, under the full mixture). Component i: x_t | i ~ N(sqrt(ab) mu_i, v I),
    # v = ab s^2 + 1 - ab; weights w_i ~ exp(-|x_t - sqrt(ab) mu_i|^2 / 2v) (log-sum-exp);
    # E[x0 | x_t] = m + k (x_t - sqrt(ab) m), m = sum_i w_i mu_i, k = sqrt(ab) s^2 / v.
    sa = math.sqrt(ab)
    v = ab * s2 + 1 - ab
    inv = 1 / (2 * v)
    k = sa * s2 / v
    dx = X[:, 0:1] - sa * mu[None, :, 0]
    dy = X[:, 1:2] - sa * mu[None, :, 1]
    logits = -(dx * dx + dy * dy) * inv
    def mean_under(mask):
        l = np.where(mask, logits, -np.inf)
        w = np.exp(l - l.max(axis=1, keepdims=True))
        m = (w @ mu) / w.sum(axis=1, keepdims=True)
        return m + k * (X - sa * m)
    E_cond = mean_under(cls_x[:, None] == cls_mu[None, :])
    E_uncond = mean_under(np.ones(logits.shape, dtype=bool)) if uncond else None
    return E_cond, E_uncond

# ---------- samplers ----------
def ddim_timesteps(T, steps):
    S = max(1, min(steps, T))
    return [math.floor(k * T / S + 0.5) for k in range(S, -1, -1)]   # = JavaScript Math.round

def reverse_run(X0, C, beta, alpha, abar, sampler, steps, w, seed):
    n = len(X0)
    ts = list(range(T, -1, -1)) if sampler == "ddpm" else ddim_timesteps(T, steps)
    r = mulberry32(stream_seed(seed, 3))
    x = np.array([[gauss(r), gauss(r)] for _ in range(n)])     # fresh x_T ~ N(0, I)
    zr = mulberry32(stream_seed(seed, 4))
    s2 = S_KDE * S_KDE
    frames = [x]
    for j in range(len(ts) - 1):
        t, tn = ts[j], ts[j + 1]
        ab, abn = abar[t], abar[tn]
        sa, s1 = math.sqrt(ab), math.sqrt(1 - ab)
        E_cond, E_uncond = posterior_means(x, C, X0, C, ab, s2, w != 0)
        x0_hat = E_cond if w == 0 else (1 + w) * E_cond - w * E_uncond
        eps_hat = (x - sa * x0_hat) / s1
        if sampler == "ddpm":
            b = beta[t]
            sigma = math.sqrt((1 - abn) / (1 - ab) * b)
            z = np.array([[gauss(zr), gauss(zr)] for _ in range(n)])
            x = (x - (b / s1) * eps_hat) / math.sqrt(alpha[t]) + sigma * z
        else:
            x = math.sqrt(abn) * x0_hat + math.sqrt(1 - abn) * eps_hat
        frames.append(x)
    return ts, frames

# ---------- metrics ----------
def mean_nn(A, B):
    # mean over rows of A of the distance to the nearest row of B
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.min(axis=1)).mean())

def ideal_reference(X0, seed):
    # the same metrics for exact draws from the KDE (mu_i + s z, stream 5)
    r = mulberry32(stream_seed(seed, 5))
    K = np.array([[X0[i, 0] + S_KDE * gauss(r), X0[i, 1] + S_KDE * gauss(r)] for i in range(len(X0))])
    return mean_nn(K, X0), mean_nn(X0, K)

if __name__ == "__main__":
    X0, C, n_classes = make_data(DATASET, N, SEED)
    eps = make_eps(len(X0), SEED)
    beta, alpha, abar, logsnr = build_schedule(SCHEDULE, T)
    cross = snr_crossing(logsnr)
    print(f"{SCHEDULE} schedule, T={T}: abar_T={abar[T]:.3e}, SNR = 1 at "
          + (f"t~{cross:.1f}" if cross is not None else "no step (never crosses)"))
    for t in sorted({0, T // 4, T // 2, (3 * T) // 4, T}):
        xt = math.sqrt(abar[t]) * X0 + math.sqrt(1 - abar[t]) * eps
        print(f"  forward t={t:4d}  abar={abar[t]:.4e}  x_t -> data (mean NN) = {mean_nn(xt, X0):.4f}")

    w = GUIDANCE
    ts, frames = reverse_run(X0, C, beta, alpha, abar, SAMPLER, DDIM_STEPS, w, SEED)
    final = frames[-1]
    off, gap = mean_nn(final, X0), mean_nn(X0, final)
    ioff, igap = ideal_reference(X0, SEED)
    label = f"DDIM, S={len(ts) - 1}" if SAMPLER == "ddim" else f"DDPM, {T} steps"
    print(f"reverse ({label}, w={w}): {len(ts) - 1} denoiser calls from fresh N(0, I) noise")
    print(f"  off-data     (sample -> nearest data point) = {off:.4f}   exact KDE draws: {ioff:.4f}")
    print(f"  coverage gap (data point -> nearest sample) = {gap:.4f}   exact KDE draws: {igap:.4f}")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, t in zip(axes[0], [0, T // 3, (2 * T) // 3, T]):
        xt = math.sqrt(abar[t]) * X0 + math.sqrt(1 - abar[t]) * eps
        ax.scatter(xt[:, 0], xt[:, 1], c=C, s=5, cmap="viridis", vmin=0, vmax=max(1, n_classes - 1))
        ax.set_title(f"forward t={t}  abar={abar[t]:.3g}")
    picks = sorted({int(round(q * (len(ts) - 1))) for q in (0, 1 / 3, 2 / 3, 1)})
    for ax, j in zip(axes[1], picks):
        ax.scatter(X0[:, 0], X0[:, 1], c="lightgrey", s=3)
        ax.scatter(frames[j][:, 0], frames[j][:, 1], c=C, s=5, cmap="viridis", vmin=0, vmax=max(1, n_classes - 1))
        ax.set_title(f"reverse t={ts[j]}" + ("  (final)" if j == len(ts) - 1 else ""))
    for ax in axes.flat:
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    plt.suptitle(f"{DATASET} | {SCHEDULE} T={T} | {label} | w={w} | off-data {off:.3f}, coverage gap {gap:.3f}")
    plt.tight_layout(); plt.show()
`;

export const noiseSchedulePython = (schedule: string, T: number, shift = 1) => `import math
import numpy as np
import matplotlib.pyplot as plt

# Noise schedules — mirrors the lab exactly. Every schedule is built one way (the
# improved-DDPM recipe): a raw abar(t) -> optional log-SNR shift SNR' = SNR / SHIFT^2
# -> beta_t = min(0.999, 1 - abar_t / abar_{t-1}) -> abar_t = prod(1 - beta_i).
#   linear : beta from 1e-4 to 0.02 scaled by 1000/T  (beta_1 = 0.1/T, beta_T = 20/T)
#   cosine : abar_t = f(t)/f(0),  f(t) = cos^2(((t/T + s)/(1 + s)) * pi/2),  s = 0.008
#   sigmoid: Chen 2023 (start -3, end 3, tau 1): abar(u) = (sig(3) - sig(6u - 3)) / (sig(3) - sig(-3)), u = t/T
#   edm    : Karras et al. 2022 spacing, rho = 7, sigma from 0.002 to 80; abar = 1 / (1 + sigma^2)
# SHIFT > 1 divides the SNR by SHIFT^2 at every t (more noise, as higher resolutions need);
# SHIFT < 1 multiplies it (more signal).

T         = ${T}
HIGHLIGHT = "${schedule}"   # "linear" | "cosine" | "sigmoid" | "edm"
SHIFT     = ${shift}

${SCHEDULE_PY}
KINDS = ["linear", "cosine", "sigmoid", "edm"]

if __name__ == "__main__":
    scheds = {k: build_schedule(k, T, SHIFT) for k in KINDS}
    print(f"T={T}  shift={SHIFT}  (SNR' = SNR / {SHIFT * SHIFT:g})")
    print(f"{'schedule':9s} {'abar_T':>10s} {'SNR=1 at':>9s} {'abar<0.01':>10s} {'|lnSNR|<=1':>11s} {'abar>0.99':>10s}")
    for k in KINDS:
        beta, alpha, abar, logsnr = scheds[k]
        c = snr_crossing(logsnr)
        near_noise = float(np.mean(abar[1:] < 0.01))
        band = float(np.mean(np.abs(logsnr[1:]) <= 1))
        near_clean = float(np.mean(abar[1:] > 0.99))
        mark = " <- highlight" if k == HIGHLIGHT else ""
        cs = f"t~{c:.1f}" if c is not None else "none"
        print(f"{k:9s} {abar[T]:10.3e} {cs:>9s} {near_noise:10.1%} {band:11.1%} {near_clean:10.1%}{mark}")

    beta, alpha, abar, logsnr = scheds[HIGHLIGHT]
    print(f"\\n{HIGHLIGHT}: t, beta_t, alpha_t, abar_t, ln SNR")
    for t in sorted({1, T // 4, T // 2, (3 * T) // 4, T}):
        print(f"  t={t:4d}  beta={beta[t]:.5f}  alpha={alpha[t]:.5f}  abar={abar[t]:.4e}  lnSNR={logsnr[t]:+.3f}")

    t = np.arange(1, T + 1)
    k_beta = T / 40      # beta drawn x(T/40) so it fits the [0, 1] axis, as in the lab
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(t, abar[1:], label="abar_t (signal power)")
    ax[0].plot(t, np.sqrt(abar[1:]), label="sqrt(abar_t) (amplitude)")
    ax[0].plot(t, alpha[1:], label="alpha_t = 1 - beta_t")
    ax[0].plot(t, beta[1:] * k_beta, "--", label=f"beta_t x{k_beta:g}")
    ax[0].set_ylim(-0.05, 1.05); ax[0].set_xlabel("t"); ax[0].set_title(HIGHLIGHT); ax[0].legend()
    for k in KINDS:
        ax[1].plot(t, scheds[k][3][1:], lw=2.6 if k == HIGHLIGHT else 1.4, label=k)
    for y in (1, -1):
        ax[1].axhline(y, color="grey", lw=0.8, ls="--")   # |ln SNR| = 1 band
    c = snr_crossing(logsnr)
    if c is not None:
        ax[1].plot([c], [0], "o", color="k", label=f"SNR=1 ({HIGHLIGHT}) t~{c:.0f}")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("ln SNR"); ax[1].legend()
    plt.suptitle(f"T={T}  shift={SHIFT}")
    plt.tight_layout(); plt.show()
`;

export const PYTHON_SAMPLES: PythonSample[] = [
  // Forward & Reverse: defaults, every preset, sampler / guidance / T edge cases.
  { name: 'fr-default', code: () => forwardReversePython('cosine', 200, 'two-moons', 'ddpm', 50, 0, 1) },
  { name: 'fr-preset-ddpm-baseline', code: () => forwardReversePython('linear', 200, 'two-moons', 'ddpm', 50, 0, 1) },
  { name: 'fr-preset-fast-ddim', code: () => forwardReversePython('cosine', 300, 'ring', 'ddim', 20, 0, 1) },
  { name: 'fr-preset-guided-blobs', code: () => forwardReversePython('cosine', 200, 'blobs', 'ddim', 30, 3, 1) },
  { name: 'fr-preset-coarse-8', code: () => forwardReversePython('cosine', 240, 'two-moons', 'ddim', 8, 0, 1) },
  { name: 'fr-T40-ddim-steps-above-T', code: () => forwardReversePython('linear', 40, 'blobs', 'ddim', 100, 0.5, 2) },
  { name: 'fr-T500-ddpm-w6', code: () => forwardReversePython('cosine', 500, 'ring', 'ddpm', 50, 6, 7), timeoutSec: 240 },
  { name: 'fr-ddim4-linear-seed3', code: () => forwardReversePython('linear', 200, 'two-moons', 'ddim', 4, 1.5, 3) },
  // Noise schedules: defaults, every schedule / preset, shift and T edge cases.
  { name: 'ns-default', code: () => noiseSchedulePython('cosine', 200, 1) },
  { name: 'ns-preset-linear-waste', code: () => noiseSchedulePython('linear', 200, 1) },
  { name: 'ns-preset-sigmoid', code: () => noiseSchedulePython('sigmoid', 260, 1) },
  { name: 'ns-preset-edm', code: () => noiseSchedulePython('edm', 260, 1) },
  { name: 'ns-preset-hires-shift', code: () => noiseSchedulePython('cosine', 260, 2) },
  { name: 'ns-T40-shift0.25', code: () => noiseSchedulePython('linear', 40, 0.25) },
  { name: 'ns-T500-shift4', code: () => noiseSchedulePython('edm', 500, 4) },
];
