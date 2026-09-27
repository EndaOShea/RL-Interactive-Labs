// Runnable NumPy exports for the Audio & Speech labs (template strings — not LLM
// generated), mirroring the on-screen synthesis, transforms and measurements
// exactly. Constants are serialised from the modules the labs run
// (fourierMath.ts, stft.ts), so the exports cannot drift from them.
import type { PythonSample } from '../../utils/pythonSamples';
import { BASE_PRESETS, F0_CHOICES, FINE, IDEAL_KINDS, K_SHOW, NS } from './fourierMath';
import type { PhaseMode } from './fourierMath';
import { FOURIER_EXTRA_PRESETS, WINDOW_KINDS } from './shared';
import type { WindowKind } from './shared';
import {
  AMP_FLOOR, CHIRP, CLICK_SAMPLES, DB_RANGE, DUR, L, N_MEL, NOISY_TONE, SIGNALS, SR, TWO_TONE, VOWEL, WINDOW_SIZES,
} from './stft';
import type { Signal } from './stft';

const pyList = (v: number[]) => `[${v.map((x) => String(x)).join(', ')}]`;

// Exact port of the labs' 32-bit PRNG (mulberry32).
const MULBERRY_PY = `def mulberry32(seed):
    """Exact port of the lab's 32-bit PRNG (same stream as the browser)."""
    a = seed & 0xFFFFFFFF
    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rnd`;

export const fourierPython = (
  amps: number[],
  f0: number,
  preset: string,
  view: string = 'linear',
  phase: PhaseMode = 'sin',
  shift = 0,
) => `import numpy as np

# Fourier synthesis — mirrors the lab exactly.
# preset: ${preset}   phase: ${phase}   f0: ${f0} Hz   spectrum axis: ${view}
F0 = ${f0}                                   # fundamental frequency (Hz)
AMPS = np.array(${pyList(amps)})   # a_k for k = 1..K (the slider values)
PHASE = "${phase}"                           # "sin": phi = 0, "cos": phi = 90 degrees
SHIFT = ${shift}                                # time offset of the Run animation (periods)
PRESET = "${preset}"
VIEW = "${view}"
NS = ${NS}                                   # DFT samples per period
FINE = ${FINE}                                 # grid for the peak / overshoot metrics
K_SHOW = ${K_SHOW}                                # DFT bins shown: k = 1..${K_SHOW}


def synth(u, amps=AMPS, phase=PHASE):
    """x at u = f0*t periods:  x = sum_k a_k * sin(2*pi*k*u + phi)."""
    ph = np.pi / 2 if phase == "cos" else 0.0
    u = np.atleast_1d(np.asarray(u, dtype=float))
    x = np.zeros_like(u)
    for k, a in enumerate(amps, start=1):
        x += a * np.sin(2 * np.pi * k * u + ph)
    return x


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def ideal(kind, u):
    """The infinite series each sine-phase preset truncates."""
    w = u - np.floor(u)
    if kind == "sine":
        return np.sin(2 * np.pi * u)
    if kind == "square":                     # (pi/4) * sgn(sin 2 pi u)
        return np.where((w == 0) | (w == 0.5), 0.0, np.where(w < 0.5, np.pi / 4, -np.pi / 4))
    if kind == "sawtooth":                   # (pi/2) * (1 - 2u) on (0, 1)
        return np.where(w == 0, 0.0, (np.pi / 2) * (1 - 2 * w))
    tri = np.where(w < 0.25, 4 * w, np.where(w < 0.75, 2 - 4 * w, 4 * w - 4))
    return (np.pi ** 2 / 8) * tri            # (pi^2/8) * triangle


PLATEAU = {"sine": 1.0, "square": np.pi / 4, "sawtooth": np.pi / 2, "triangle": np.pi ** 2 / 8}
JUMP = {"sine": None, "square": np.pi / 2, "sawtooth": np.pi, "triangle": None}

if __name__ == "__main__":
    # The spectrum is computed, not copied from AMPS: DFT of one sampled period.
    x = synth(np.arange(NS) / NS + SHIFT)
    X = np.fft.fft(x)
    print(f"f0 = {F0} Hz   phase = {PHASE}   shift = {SHIFT} T   preset = {PRESET}")
    print(" k   f (Hz)     mel   2|X_k|/N   angle X_k (deg)")
    for k in range(1, K_SHOW + 1):
        amp = 2 * abs(X[k]) / NS
        ang = np.degrees(np.angle(X[k])) if amp > 1e-9 else 0.0
        print(f"{k:2d} {k * F0:8.0f} {hz_to_mel(k * F0):7.0f}   {amp:8.3f}   {ang:8.1f}")

    if VIEW == "mel":                        # stems at mel(k*f0), heights unchanged
        m = hz_to_mel(F0 * np.arange(1, len(AMPS) + 1))
        gaps = np.diff(m)
        print("mel gaps between harmonics:", np.round(gaps).astype(int), "  last/first = %.2f" % (gaps[-1] / gaps[0]))

    u = np.arange(FINE) / FINE
    if PRESET in PLATEAU and PHASE == "sin":
        xs, ys = synth(u), ideal(PRESET, u)
        peak, plateau = xs.max(), PLATEAU[PRESET]
        print(f"vs ideal {PRESET}: peak {peak:.4f}  plateau {plateau:.4f}  ratio {peak / plateau:.4f}  RMS error {np.sqrt(np.mean((xs - ys) ** 2)):.4f}")
        if JUMP[PRESET] is not None:
            print(f"overshoot: {100 * (peak - plateau) / JUMP[PRESET]:.2f}% of the jump (Gibbs)")
    else:
        print("peak |x| over one period: %.4f" % np.abs(synth(u)).max())
`;

const WINDOW_NP: Record<WindowKind, string> = {
  rectangular: 'np.ones(n)',
  hann: 'np.hanning(n)',
  hamming: 'np.hamming(n)',
  blackman: 'np.blackman(n)',
};

export const spectrogramPython = (
  signal: Signal,
  windowSize: number,
  window: WindowKind = 'hann',
  mel = false,
  scale: 'db' | 'linear' = 'db',
) => `import numpy as np

# Spectrogram (short-time Fourier transform) — mirrors the lab exactly.
# signal: ${signal}   window: ${windowSize} samples (${(windowSize / SR) * 1000} ms), ${window}, hop N/2   axis: ${mel ? `mel (${N_MEL} bands)` : 'linear bins'}   colour: ${scale}
SR = ${SR}                  # samples per second
DUR = ${DUR}                  # seconds
L = ${L}                    # samples
SIGNAL = "${signal}"
N = ${windowSize}                   # window length (samples)
HOP = N // 2              # 50% overlap
WINDOW = "${window}"
MEL = ${mel ? 'True' : 'False'}
SCALE = "${scale}"          # "db": 20*log10|X| over [top - DB_RANGE, top]; "linear": |X| over [0, max]
N_MEL = ${N_MEL}
DB_RANGE = ${DB_RANGE}
AMP_FLOOR = ${AMP_FLOOR}

CHIRP_F0, CHIRP_F1 = ${CHIRP.f0}.0, ${CHIRP.f1}.0              # linear sweep over DUR
TWO_TONE_F1, TWO_TONE_F2, TWO_TONE_A2 = ${TWO_TONE.f1}.0, ${TWO_TONE.f2}.0, ${TWO_TONE.a2}
TONE_HZ = 4000.0 / 3.0                       # 1/3 bin off the DFT grid for every N offered
NOISY_TONE_HZ, NOISE_AMP, NOISE_SEED = ${NOISY_TONE.f}.0, ${NOISY_TONE.amp}, ${NOISY_TONE.seed}
CLICK_SAMPLES = ${pyList(CLICK_SAMPLES)}           # unit impulses (125, 250, 260 ms)
VOWEL_F0 = ${VOWEL.f0}.0
FORMANTS = ${pyList(VOWEL.formants)}               # /u/ ("oo"): F1, F2, F3 (Hz)
BANDWIDTHS = ${pyList(VOWEL.bandwidths)}


${MULBERRY_PY}


def formant_gain(f):
    """Cascade of 2-pole resonators: unity at DC, about F/B at each formant."""
    g = 1.0
    for F, B in zip(FORMANTS, BANDWIDTHS):
        g *= (F * F) / np.sqrt((F * F - f * f) ** 2 + (B * f) ** 2)
    return g


def vowel_amps():
    H = int((SR / 2 - 1) // VOWEL_F0)                       # harmonics below Nyquist
    raw = np.array([formant_gain(h * VOWEL_F0) / h for h in range(1, H + 1)])   # 1/h source tilt
    return raw / raw.max()


def make_signal(kind=SIGNAL):
    t = np.arange(L) / SR
    if kind == "clicks":
        x = np.zeros(L)
        x[CLICK_SAMPLES] = 1.0
        return x
    if kind == "chirp":
        return np.sin(2 * np.pi * (CHIRP_F0 * t + 0.5 * (CHIRP_F1 - CHIRP_F0) * t * t / DUR))
    if kind == "two-tone":
        return np.sin(2 * np.pi * TWO_TONE_F1 * t) + TWO_TONE_A2 * np.sin(2 * np.pi * TWO_TONE_F2 * t)
    if kind == "tone":
        return np.sin(2 * np.pi * TONE_HZ * t)
    if kind == "tone+noise":
        rnd = mulberry32(NOISE_SEED)
        noise = np.array([NOISE_AMP * (2.0 * rnd() - 1.0) for _ in range(L)])
        return np.sin(2 * np.pi * NOISY_TONE_HZ * t) + noise
    x = np.zeros(L)                                         # vowel: harmonics in cosine phase
    for h, a in enumerate(vowel_amps(), start=1):
        x += a * np.cos(2 * np.pi * h * VOWEL_F0 * t)
    return x


def make_window(n=N, kind=WINDOW):
    return {"rectangular": np.ones, "hann": np.hanning, "hamming": np.hamming, "blackman": np.blackman}[kind](n)   # ${WINDOW_NP[window]}


def stft(x):
    """One-sided amplitude spectra: |X(k)| * c_k / (N * c_g), c_k = 2 (1 at DC and Nyquist)."""
    w = make_window()
    cg = max(1e-6, w.mean())                               # coherent gain
    ck = np.full(N // 2 + 1, 2.0)
    ck[0] = ck[-1] = 1.0
    cols = []
    for start in range(0, L - N + 1, HOP):                 # every frame, including the last
        X = np.fft.rfft(x[start:start + N] * w)            # k = 0..N/2, f_k = k*SR/N (up to Nyquist)
        cols.append(ck * np.abs(X) / (N * cg))
    return np.array(cols)                                  # (frames, bins)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(bin_hz, n_mel=N_MEL, f_max=SR / 2):
    """HTK triangles (peak 1) on the TRUE bin frequencies — rebuilt for each N."""
    edges = mel_to_hz(np.arange(n_mel + 2) / (n_mel + 1) * hz_to_mel(f_max))
    fb = np.zeros((n_mel, len(bin_hz)))
    for m in range(n_mel):
        lo, ce, hi = edges[m], edges[m + 1], edges[m + 2]
        fb[m] = np.maximum(0.0, np.minimum((bin_hz - lo) / (ce - lo), (hi - bin_hz) / (hi - ce)))
    return fb, edges


def to_db(a):
    return 20 * np.log10(np.maximum(a, AMP_FLOOR))


def frame_ms(fr):
    return (fr * HOP + N / 2) / SR * 1000


def nearest_row(row_hz, f):
    return int(np.argmin(np.abs(row_hz - f)))


def upper_median(v):
    return np.sort(np.asarray(v))[len(v) // 2]


def peak_row(col):
    return int(np.argmax(col)) if col.max() > 0 else -1


def jround(v):
    """Round half up, like the lab's Math.round (Python's round() is half-to-even)."""
    return int(np.floor(v + 0.5))


def measure(S, row_hz, top):
    """The lab's measurement for each signal (same rules, same thresholds)."""
    db = to_db(S.mean(axis=0))
    floor = top - DB_RANGE
    if SIGNAL == "chirp":
        pk = [row_hz[peak_row(c)] if peak_row(c) >= 0 else 0.0 for c in S]
        mono = all(b >= a for a, b in zip(pk, pk[1:]))
        return f"loudest row climbs {jround(pk[0])} -> {jround(pk[-1])} Hz ({'monotone' if mono else 'with jitter'})"
    if SIGNAL == "two-tone":
        ia, ib = nearest_row(row_hz, TWO_TONE_F1), nearest_row(row_hz, TWO_TONE_F2)
        if ib - ia < 2:
            return "merged: the two tones land in adjacent rows"
        near = lambda i: max(db[max(0, i - 1):i + 2])
        dip = min(near(ia), near(ib)) - db[ia + 1:ib].min()
        return f"{'resolved' if dip >= 3 else 'merged'}: the averaged spectrum dips {dip:.1f} dB between the tones"
    if SIGNAL == "tone":
        far = np.abs(row_hz - TONE_HZ) > 500
        med = upper_median(db[far] - db.max())
        return f"leakage: {int(np.sum(db[far] > floor))} of {int(far.sum())} rows > 500 Hz from the tone clear the floor; median {med:.1f} dB re peak"
    if SIGNAL == "tone+noise":
        tone = db[nearest_row(row_hz, NOISY_TONE_HZ)]
        keep = (np.abs(row_hz - NOISY_TONE_HZ) > 300) & (row_hz > 0) & (row_hz < SR / 2)
        noise = upper_median(db[keep])
        return f"tone {tone:.1f} dB, median noise {noise:.1f} dB -> {tone - noise:.1f} dB above the noise"
    if SIGNAL == "clicks":
        on = [to_db(c.max()) > floor for c in S]
        events, start = [], -1
        for i, v in enumerate(on):
            if v and start < 0:
                start = i
            last = i == len(on) - 1
            if start >= 0 and (not v or last):
                end = i if (v and last) else i - 1
                events.append((frame_ms(start), frame_ms(end)))
                start = -1
        pair = "separated" if len(events) >= len(CLICK_SAMPLES) else "merged"
        return f"{len(events)} events at " + ", ".join(f"{jround(a)}-{jround(b)} ms" for a, b in events) + f"; the 10 ms pair is {pair}"
    resolved = (not MEL) and SR / N <= VOWEL_F0 / 2        # vowel
    if resolved:
        idx = [nearest_row(row_hz, h * VOWEL_F0) for h in range(1, int((SR / 2 - 1) // VOWEL_F0) + 1)]
    else:
        idx = list(range(len(row_hz)))
    vals = db[idx]
    peaks = [j for j in range(1, len(vals) - 1) if vals[j] > vals[j - 1] and vals[j] >= vals[j + 1]]
    top3 = sorted(sorted(peaks, key=lambda j: -vals[j])[:3])
    return "envelope peaks near " + ", ".join(f"{jround(row_hz[idx[j]])}" for j in top3) + f" Hz (formants {FORMANTS})"


if __name__ == "__main__":
    x = make_signal()
    S = stft(x)                                            # (frames, N/2 + 1)
    row_hz = np.fft.rfftfreq(N, 1.0 / SR)                  # k * SR / N, 0 .. SR/2
    if MEL:
        fb, edges = mel_filterbank(row_hz)
        S = S @ fb.T                                       # M_m = sum_k H_m(k) |X(k)|
        row_hz = edges[1:-1]                               # band centres (Hz)
        print("mel bands: %d, triangle widths %.0f Hz (lowest) to %.0f Hz (highest)" % (N_MEL, edges[2] - edges[0], edges[-1] - edges[-3]))
    top = np.ceil(to_db(S.max()) - 1e-9)                   # colour-scale top (whole dB)
    print(f"frames: {S.shape[0]} (hop {HOP}), rows: {S.shape[1]}, bin spacing {SR / N:.2f} Hz")
    if SCALE == "db":
        print(f"colour scale: 20*log10|X| from {top - DB_RANGE:.0f} to {top:.0f} dB")
    else:
        print(f"colour scale: linear |X| from 0 to {S.max():.3f}")
    print("frame centres (ms):", np.round([frame_ms(i) for i in range(S.shape[0])], 1))
    if SIGNAL != "clicks":
        print("loudest row per frame (Hz):", np.round([row_hz[peak_row(c)] if peak_row(c) >= 0 else np.nan for c in S]).astype(float))
    print("measured:", measure(S, row_hz, top))
`;

// ---------------------------------------------------------------------------
// Samples for scripts/check-python-exports.mjs — every Fourier preset, f0,
// phase, axis and a shifted/custom case; every spectrogram signal, window
// length, taper, frequency axis and colour scale, plus the guided set-ups.
// ---------------------------------------------------------------------------
const fourierSamples: PythonSample[] = [
  ...IDEAL_KINDS.map((p) => ({ name: `fourier-${p}`, code: () => fourierPython(BASE_PRESETS[p], 440, p, 'linear', 'sin', 0) })),
  ...FOURIER_EXTRA_PRESETS.map((p) => ({ name: `fourier-${p.id}`, code: () => fourierPython(p.amps, 440, p.id, 'linear', p.phase, 0) })),
  ...F0_CHOICES.map((f) => ({ name: `fourier-square-mel-f${f}`, code: () => fourierPython(BASE_PRESETS.square, f, 'square', 'mel', 'sin', 0) })),
  { name: 'fourier-custom-cos-shift', code: () => fourierPython([0.25, -0.8, 0, 1, -0.33], 220, 'custom', 'linear', 'cos', 0.37) },
  { name: 'fourier-triangle-mel-shift', code: () => fourierPython(BASE_PRESETS.triangle, 880, 'triangle', 'mel', 'sin', 0.5) },
  { name: 'fourier-silent', code: () => fourierPython([0, 0, 0, 0, 0], 110, 'custom', 'mel', 'sin', 0) },
];

const specSamples: PythonSample[] = [];
const addSpec = (name: string, s: Signal, n: number, w: WindowKind, mel: boolean, scale: 'db' | 'linear') => {
  specSamples.push({ name: `spectrogram-${name}`, code: () => spectrogramPython(s, n, w, mel, scale) });
};
SIGNALS.forEach((s) => {
  const id = s.replace('+', '-');
  addSpec(`${id}-default`, s, 256, 'hann', false, 'db');
  addSpec(`${id}-mel`, s, 256, 'hann', true, 'db');
  addSpec(`${id}-linear-scale`, s, 128, 'hamming', false, 'linear');
});
WINDOW_KINDS.forEach((w) => addSpec(`tone-${w}`, 'tone', 256, w, false, 'db'));
WINDOW_SIZES.forEach((n) => addSpec(`two-tone-N${n}`, 'two-tone', n, 'hann', false, 'db'));
WINDOW_SIZES.forEach((n) => addSpec(`clicks-N${n}`, 'clicks', n, 'blackman', n === 512, 'db'));
// the lab's guided set-ups
addSpec('guided-clicks', 'clicks', 64, 'hann', false, 'db');
addSpec('guided-two-tone', 'two-tone', 512, 'blackman', false, 'db');
addSpec('guided-leakage', 'tone', 256, 'rectangular', false, 'db');
addSpec('guided-vowel-mel', 'vowel', 256, 'hann', true, 'db');
addSpec('guided-noise', 'tone+noise', 512, 'hann', false, 'db');
addSpec('vowel-N64-mel-linear', 'vowel', 64, 'rectangular', true, 'linear');

export const PYTHON_SAMPLES: PythonSample[] = [...fourierSamples, ...specSamples];
