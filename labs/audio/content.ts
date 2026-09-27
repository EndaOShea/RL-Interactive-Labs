import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Audio Transcription labs
// (rendered in each lab's Context tab via LabContext). These two labs cover the
// signal / Fourier front-end that turns a raw waveform into the spectral
// features an automatic speech recogniser (ASR) actually consumes.

export const FOURIER_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Fourier Decomposition',
      body: 'Any periodic signal can be written as a sum of sinusoids — harmonics at integer multiples of a fundamental frequency f₀. The Fourier series x(t) = Σ aₖ·sin(2π·k·f₀·t + φ) lets us trade between the time domain (the wiggling waveform) and the frequency domain (a small list of amplitudes and phases). A square or sawtooth wave looks complex in time but is just a tidy recipe of harmonics.',
      details: [
        { label: 'Fundamental', text: 'The lowest frequency f₀ sets the pitch (110–880 Hz here); its period 1/f₀ is one cycle of the waveform.' },
        { label: 'Harmonics', text: 'Higher partials k·f₀ add the fine detail (the edges and corners) — their amplitudes aₖ define the timbre.' },
        { label: 'Spectrum', text: 'The stems are COMPUTED by a DFT of one sampled period (256 samples): 2|X_k|/N recovers each |aₖ| and ∠X_k its phase — a complete, compact description of the same signal.' },
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
    {
      heading: 'Timbre, Gibbs & the Mel Axis',
      body: 'The RELATIVE strengths of the harmonics — not the fundamental — give an instrument its timbre: a clarinet leans on odd harmonics, an organ stacks octaves, a bright pulse keeps every partial strong. Truncating the series to a finite K cannot reach a perfect edge, so the reconstruction OVERSHOOTS next to a jump (the Gibbs phenomenon): the square preset’s five-harmonic sum peaks at 0.933 against its π/4 = 0.785 plateau, 9.4% of the jump. As K grows the overshoot settles at about 9% of the jump (8.95%) — it narrows but never disappears. A continuous wave such as the triangle has no overshoot; truncation only rounds its corners. Placing the stems on a mel axis (mel(f) = 2595·log₁₀(1+f/700)) re-spaces them the way the ear hears: low harmonics spread out, high ones crowd together — at f₀ = 440 Hz the gap between harmonics 4 and 5 is half the gap between 1 and 2 — previewing the log-mel pooling used by speech front-ends.',
      details: [
        { label: 'Timbre', text: 'The amplitude pattern aₖ defines the "voice": odd-only (clarinet), octave stacks (organ), all-strong (buzzy pulse).' },
        { label: 'Gibbs ripple', text: 'Next to a discontinuity a truncated series overshoots; as harmonics are added the overshoot tends to about 9% of the jump — narrower, never gone. (With only 5 harmonics the sawtooth reaches just 0.4%: its ramp has not caught up yet.)' },
        { label: 'Mel warp', text: 'A perceptual, roughly-log frequency axis — the basis of the log-mel features every ASR system consumes.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Phase carries information too', description: 'The phase toggle keeps every |aₖ| but moves all harmonics from sine to cosine phase: the magnitude stems are identical, yet the pulse preset changes from a wiggle peaking at 2.70 (sine phase) to a 3.55 spike (cosine phase, all partials peaking together). Real signals carry a phase per harmonic.', recommendation: 'When the relative timing of partials matters (transients, stereo imaging), keep the complex spectrum (magnitude AND phase), not just the magnitude stems.' },
    { category: 'DATA', title: 'Finite harmonics band-limit the signal', description: 'A real recording is sampled, so only frequencies below the Nyquist limit survive; truncating harmonics is the same kind of band-limiting and softens sharp edges.', recommendation: 'Choose a sample rate and harmonic count high enough to capture the bandwidth your task needs — speech needs ~8 kHz, music far more.' },
    { category: 'METHODOLOGY', title: 'Why a mel-warped spectrum', description: 'Equal steps in Hz are not equal steps in perceived pitch; the cochlea resolves low frequencies finely and high ones coarsely. The mel scale linearises perceived pitch, so a handful of mel bands capture what matters for speech.', recommendation: 'For perceptual or speech features pool the linear spectrum into mel bands and log-compress; for exact analysis (tuning, partials) keep the linear |aₖ|.' },
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
        { label: 'Reading it', text: 'Bright horizontal bands are steady tones; rising diagonals are chirps (sweeping pitch); vertical lines are clicks; broadband haze is noise.' },
        { label: 'Bins', text: 'An N-sample frame at sample rate SR gives bins f_k = k·SR/N; for a real signal only k = 0…N/2 (up to the Nyquist frequency SR/2) are unique — the rest mirror them. This lab uses SR = 8 kHz, so rows run 0–4 kHz.' },
      ],
    },
    {
      heading: 'Windowing & the Time–Frequency Tradeoff',
      body: 'Window length is a fundamental compromise (an uncertainty principle). A long window resolves frequency finely but blurs time; a short window pins events in time but smears frequency. There is no free lunch — you pick the window to match what you need to see, and a tapered window (Hann/Hamming) reduces spectral leakage from the hard frame edges.',
      details: [
        { label: 'Long window', text: 'Sharp frequency bins, poor time resolution — at 64 ms (15.6 Hz bins) the 1000 and 1150 Hz tones are separated by a deep dip; at 8 ms (125 Hz bins) they merge.' },
        { label: 'Short window', text: 'Sharp timing, coarse frequency bins — an 8 ms window shows the clicks 10 ms apart as two events; 16 ms or longer merges them.' },
        { label: 'Leakage', text: 'A rectangular window’s abrupt edges spread an off-grid tone across every bin; tapered windows suppress this.' },
      ],
    },
    {
      heading: 'Window Functions & the Mel Front-End',
      body: 'The taper you multiply each frame by sets a main-lobe / side-lobe tradeoff. A rectangular (boxcar) window has the narrowest main lobe (best raw frequency resolution) but tall, slowly decaying side-lobes, so a strong tone leaks energy into far bins. Hann and Hamming taper the edges to suppress that leakage at the cost of a slightly wider main lobe; Blackman pushes side-lobes down further still. After the DFT, a real front-end POOLS the linear bins through a triangular mel filterbank — Mₘ = Σ_k Hₘ(k)·|X(k)| — so high frequencies share wide bands. This lab uses 20 bands built on the true bin frequencies: 139 Hz wide at the bottom, 780 Hz at the top (real ASR front-ends use ~40–80).',
      details: [
        { label: 'Rectangular', text: 'Narrowest main lobe, worst leakage — with a 32 ms frame, every bin more than 500 Hz from the 1333 Hz tone still reads above −80 dB (median 40 dB below the peak).' },
        { label: 'Hann / Hamming', text: 'Smooth tapers; the workhorse windows that trade a little frequency width for far less leakage (Hann: none of those far bins clears −80 dB).' },
        { label: 'Blackman', text: 'Very low side-lobes (least leakage) at the cost of the widest main lobe.' },
        { label: 'dB scale', text: 'Spectrograms are read in decibels (20·log10 of the amplitude): leakage and noise 40–100 dB below the peak vanish on a linear colour scale.' },
        { label: 'Mel pooling', text: 'Triangular filters pool linear bins into perceptual bands — the M in MFCC and log-mel features.' },
        { label: 'Formants', text: 'A voiced vowel shows steady resonant bands (formants); their pattern is the cue speech models read to tell vowels apart. The lab’s “oo” has formants at 300, 870 and 2240 Hz on a 125 Hz pitch.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DEPLOYMENT', title: 'The standard ASR front-end', description: 'Almost every speech recogniser operates on spectral features (a log-mel spectrogram), not the raw waveform. The acoustic model — an RNN, CNN, or Transformer — reads these spectrogram frames and emits phoneme/character probabilities.', recommendation: 'Match the front-end (window size, hop, mel bins, normalisation) used in training and inference exactly; a mismatch silently wrecks accuracy.' },
    { category: 'METHODOLOGY', title: 'Why spectra, not raw samples', description: 'Raw audio is high-rate and phase-sensitive; the same word spoken twice gives very different sample sequences but very similar spectrograms. The spectral view is closer to how the ear and the cochlea encode sound.', recommendation: 'Use a perceptually-spaced (mel) frequency axis and log-compress magnitudes so the features emphasise the bands that carry speech.' },
  ],
};
