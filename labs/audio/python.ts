// Runnable NumPy exports for the Audio Transcription labs (template strings —
// not LLM generated), mirroring the on-screen signal synthesis and transforms.

export const fourierPython = (amps: number[], fundamental: number, preset: string, view: string = 'linear') => `import numpy as np

# Fourier Synthesis — build a periodic signal from harmonics (mirrors the lab)
# Preset: ${preset}   |   spectrum axis: ${view}
F0 = ${fundamental}                      # fundamental frequency (Hz)
AMPS = np.array([${amps.map((a) => a.toFixed(3)).join(', ')}])   # amplitude a_k for k = 1..K
SR = 4096                                # samples per second
DUR = 1.0 / F0                           # one period

def synthesize(amps=AMPS, f0=F0, sr=SR, dur=DUR):
    t = np.arange(0, dur, 1.0 / sr)
    x = np.zeros_like(t)
    for k, a in enumerate(amps, start=1):     # x(t) = sum_k a_k sin(2*pi*k*f0*t)
        x += a * np.sin(2 * np.pi * k * f0 * t)
    return t, x

def hz_to_mel(f):                            # perceptual (roughly-log) frequency warp
    return 2595.0 * np.log10(1.0 + f / 700.0)

if __name__ == "__main__":
    t, x = synthesize()

    # Verify against the FFT: the amplitude spectrum should peak at k*F0.
    X = np.fft.rfft(x) / len(x) * 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    for k in range(1, len(AMPS) + 1):
        bin_i = np.argmin(np.abs(freqs - k * F0))
        ${view === 'mel'
          ? 'print(f"harmonic {k}: a_k={AMPS[k-1]:.3f}  fft|a_k|={np.abs(X[bin_i]):.3f}  mel={hz_to_mel(k*F0*1000):.1f}")'
          : 'print(f"harmonic {k}: target a_k={AMPS[k-1]:.3f}  fft|a_k|={np.abs(X[bin_i]):.3f}")'}
`;

const WINDOW_NP: Record<string, string> = {
  rectangular: 'np.ones(win)',
  hann: 'np.hanning(win)',
  hamming: 'np.hamming(win)',
  blackman: 'np.blackman(win)',
};

export const spectrogramPython = (signal: string, windowSize: number, window: string = 'hann', mel: boolean = false) => `import numpy as np

# Spectrogram (Short-Time Fourier Transform) — the ASR front-end (mirrors the lab)
# Signal preset: ${signal}   |   window: ${window}   |   axis: ${mel ? 'mel' : 'linear'}
SR = 2048                 # samples per second
DUR = 1.0                 # seconds
WIN = ${windowSize}                # window / frame size (samples)
HOP = WIN // 2            # 50% overlap between frames
N_MEL = 16                # mel bands (when the mel axis is on)

def make_signal(kind="${signal}", sr=SR, dur=DUR):
    t = np.arange(0, dur, 1.0 / sr)
    if kind == "chirp":                       # frequency rises linearly over time
        f0, f1 = 60.0, 600.0
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / dur * t ** 2)
        return t, np.sin(phase)
    if kind == "two-tone":                    # two steady frequencies
        return t, np.sin(2 * np.pi * 120 * t) + 0.7 * np.sin(2 * np.pi * 440 * t)
    if kind == "vowel":                       # three steady formant bands
        return t, (np.sin(2 * np.pi * 100 * t)
                   + 0.8 * np.sin(2 * np.pi * 220 * t)
                   + 0.5 * np.sin(2 * np.pi * 380 * t))
    # tone + noise
    return t, np.sin(2 * np.pi * 220 * t) + 0.6 * np.random.randn(len(t))

def make_window(win=WIN):
    w = ${WINDOW_NP[window] ?? 'np.hanning(win)'}     # taper to control spectral leakage
    return w / np.maximum(1e-6, w.mean())             # coherent-gain normalise

def hz_to_mel(f):  return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_to_hz(m):  return 700.0 * (10 ** (m / 2595.0) - 1.0)

def mel_filterbank(n_lin, n_mel, f_max):
    edges = mel_to_hz(np.linspace(0, hz_to_mel(f_max), n_mel + 2))
    bins = np.linspace(0, f_max, n_lin)
    fb = np.zeros((n_mel, n_lin))
    for m in range(n_mel):
        lo, ce, hi = edges[m], edges[m + 1], edges[m + 2]
        fb[m] = np.clip(np.minimum((bins - lo) / (ce - lo + 1e-9),
                                   (hi - bins) / (hi - ce + 1e-9)), 0, None)
    return fb

def stft(x, win=WIN, hop=HOP):
    window = make_window(win)
    spec = []
    for start in range(0, len(x) - win, hop):
        frame = x[start:start + win] * window
        mag = np.abs(np.fft.rfft(frame))      # magnitude of each frame's DFT
        spec.append(mag)
    return np.array(spec).T                    # shape: (freq_bins, n_frames)

if __name__ == "__main__":
    t, x = make_signal()
    S = stft(x)
    ${mel
      ? 'fb = mel_filterbank(S.shape[0], N_MEL, SR / 2)\n    S = fb @ S                                # pool linear bins into mel bands'
      : '# (linear frequency axis — S already holds the linear-bin magnitudes)'}
    S_db = 20 * np.log10(S + 1e-6)             # log-magnitude, like a real front-end
    print("spectrogram shape (${mel ? 'mel' : 'freq'}, time):", S_db.shape)
    # A real ASR model would consume log-mel features derived from S here.
`;
