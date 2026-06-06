import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Audio Transcription labs
// (rendered in each lab's Context tab via LabContext). These two labs cover the
// signal / Fourier front-end that turns a raw waveform into the spectral
// features an automatic speech recogniser (ASR) actually consumes.

export const FOURIER_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Fourier Decomposition',
      body: 'Any periodic signal can be written as a sum of sinusoids — harmonics at integer multiples of a fundamental frequency f. The Fourier series x(t) = Σ aₖ·sin(2π·k·f·t) lets us trade between the time domain (the wiggling waveform) and the frequency domain (a small list of amplitudes aₖ). A square or sawtooth wave looks complex in time but is just a tidy recipe of harmonics.',
      details: [
        { label: 'Fundamental', text: 'The lowest frequency f sets the pitch; its period 1/f is one cycle of the waveform.' },
        { label: 'Harmonics', text: 'Higher partials k·f add the fine detail (the edges and corners) — their amplitudes aₖ define the timbre.' },
        { label: 'Spectrum', text: 'The bars |aₖ| are the frequency-domain view: a complete, compact description of the same signal.' },
      ],
    },
    {
      heading: 'Time ↔ Frequency Duality',
      body: 'Sharp features in time need many high harmonics: an ideal square wave is Σ (1/k)·sin(2π·k·f·t) over odd k only, and a sawtooth uses every harmonic with amplitude 1/k. Truncating the series (a finite K) rounds off the corners — this is why band-limited audio loses crisp transients. Adding or removing a single bar in the spectrum reshapes the whole waveform.',
      details: [
        { label: 'Square wave', text: 'Odd harmonics only, amplitudes ∝ 1/k — flat tops, steep edges built from many partials.' },
        { label: 'Sawtooth', text: 'All harmonics, amplitudes ∝ 1/k — a bright, buzzy ramp.' },
        { label: 'Sine', text: 'A single harmonic (k=1) — the purest tone, one bar in the spectrum.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Phase carries information too', description: 'This lab fixes the harmonics to sines so the spectrum is just amplitudes, but real signals also carry per-harmonic phase. Two signals with identical |aₖ| but different phases sound and look different.', recommendation: 'When the relative timing of partials matters (transients, stereo imaging), keep the complex spectrum (magnitude AND phase), not just the magnitude bars.' },
    { category: 'DATA', title: 'Finite harmonics band-limit the signal', description: 'A real recording is sampled, so only frequencies below the Nyquist limit survive; truncating harmonics is the same kind of band-limiting and softens sharp edges.', recommendation: 'Choose a sample rate and harmonic count high enough to capture the bandwidth your task needs — speech needs ~8 kHz, music far more.' },
  ],
};

export const SPECTROGRAM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'The Short-Time Fourier Transform',
      body: 'A whole-signal Fourier transform tells you which frequencies are present but not WHEN. Speech is non-stationary — the frequencies change constantly — so we slide a short window across the signal and take a DFT of each frame. Stacking those per-frame magnitude spectra into a matrix (frequency × time) gives a spectrogram: a picture of how the spectrum evolves over time.',
      details: [
        { label: 'Framing', text: 'The signal is chopped into overlapping windows; each window is short enough that the spectrum is roughly stationary within it.' },
        { label: 'Per-frame DFT', text: 'Each frame is transformed to |X(f)|; one column of the spectrogram = one frame’s spectrum.' },
        { label: 'Reading it', text: 'Bright horizontal bands are steady tones; rising diagonals are chirps (sweeping pitch); broadband smear is noise.' },
      ],
    },
    {
      heading: 'Windowing & the Time–Frequency Tradeoff',
      body: 'Window length is a fundamental compromise (an uncertainty principle). A long window resolves frequency finely but blurs time; a short window pins events in time but smears frequency. There is no free lunch — you pick the window to match what you need to see, and a tapered window (Hann/Hamming) reduces spectral leakage from the hard frame edges.',
      details: [
        { label: 'Long window', text: 'Sharp frequency bins, poor time resolution — good for sustained tones, bad for fast transients.' },
        { label: 'Short window', text: 'Sharp timing, coarse frequency bins — good for clicks and onsets.' },
        { label: 'Leakage', text: 'A rectangular window’s abrupt edges spread energy across bins; tapered windows suppress this.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DEPLOYMENT', title: 'The standard ASR front-end', description: 'Almost every speech recogniser operates on spectral features (a log-mel spectrogram), not the raw waveform. The acoustic model — an RNN, CNN, or Transformer — reads these spectrogram frames and emits phoneme/character probabilities.', recommendation: 'Match the front-end (window size, hop, mel bins, normalisation) used in training and inference exactly; a mismatch silently wrecks accuracy.' },
    { category: 'METHODOLOGY', title: 'Why spectra, not raw samples', description: 'Raw audio is high-rate and phase-sensitive; the same word spoken twice gives very different sample sequences but very similar spectrograms. The spectral view is closer to how the ear and the cochlea encode sound.', recommendation: 'Use a perceptually-spaced (mel) frequency axis and log-compress magnitudes so the features emphasise the bands that carry speech.' },
  ],
};
