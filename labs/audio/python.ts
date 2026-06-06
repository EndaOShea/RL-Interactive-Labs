// Runnable NumPy exports for the Audio Transcription labs (template strings —
// not LLM generated), mirroring the on-screen signal synthesis and transforms.

export const fourierPython = (amps: number[], fundamental: number, preset: string) => `import numpy as np

# Fourier Synthesis — build a periodic signal from harmonics (mirrors the lab)
# Preset: ${preset}
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

if __name__ == "__main__":
    t, x = synthesize()

    # Verify against the FFT: the amplitude spectrum should peak at k*F0.
    X = np.fft.rfft(x) / len(x) * 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    for k in range(1, len(AMPS) + 1):
        bin_i = np.argmin(np.abs(freqs - k * F0))
        print(f"harmonic {k}: target a_k={AMPS[k-1]:.3f}  fft|a_k|={np.abs(X[bin_i]):.3f}")
`;

export const spectrogramPython = (signal: string, windowSize: number) => `import numpy as np

# Spectrogram (Short-Time Fourier Transform) — the ASR front-end (mirrors the lab)
# Signal preset: ${signal}
SR = 2048                 # samples per second
DUR = 1.0                 # seconds
WIN = ${windowSize}                # window / frame size (samples)
HOP = WIN // 2            # 50% overlap between frames

def make_signal(kind="${signal}", sr=SR, dur=DUR):
    t = np.arange(0, dur, 1.0 / sr)
    if kind == "chirp":                       # frequency rises linearly over time
        f0, f1 = 60.0, 600.0
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / dur * t ** 2)
        return t, np.sin(phase)
    if kind == "two-tone":                    # two steady frequencies
        return t, np.sin(2 * np.pi * 120 * t) + 0.7 * np.sin(2 * np.pi * 440 * t)
    # tone + noise
    return t, np.sin(2 * np.pi * 220 * t) + 0.6 * np.random.randn(len(t))

def stft(x, win=WIN, hop=HOP):
    window = np.hanning(win)                   # taper to reduce spectral leakage
    frames = range(0, len(x) - win, hop)
    spec = []
    for start in frames:
        frame = x[start:start + win] * window
        mag = np.abs(np.fft.rfft(frame))      # magnitude of each frame's DFT
        spec.append(mag)
    return np.array(spec).T                    # shape: (freq_bins, n_frames)

if __name__ == "__main__":
    t, x = make_signal()
    S = stft(x)
    S_db = 20 * np.log10(S + 1e-6)             # log-magnitude, like a real front-end
    print("spectrogram shape (freq, time):", S_db.shape)
    # A real ASR model would consume log-mel features derived from S here.
`;
