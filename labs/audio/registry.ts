// Audio Transcription lab descriptors — the signal / Fourier front-end of ASR.
// Each entry lazy-loads its component (own Vite chunk) and co-locates its theory
// content. The global catalog registry spreads this in.
import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { FOURIER_CONTENT, SPECTROGRAM_CONTENT } from './content';

const ACCENT = '#fb923c';

export const AUDIO_LABS: LabDescriptor[] = [
  {
    id: 'fourier',
    category: 'audio',
    title: 'Fourier Synthesis',
    subtitle: 'Harmonic decomposition · time ↔ frequency duality',
    blurb: 'Build a waveform from harmonics — tune amplitudes or pick a square/saw preset and watch the spectrum.',
    icon: 'M3 12c1.5 0 1.5-7 3-7s1.5 14 3 14 1.5-7 3-7 1.5 7 3 7 1.5-14 3-14 1.5 7 3 7',
    accent: ACCENT,
    codeFile: 'fourier_synthesis.py',
    content: FOURIER_CONTENT,
    component: React.lazy(() => import('./Fourier')),
  },
  {
    id: 'spectrogram',
    category: 'audio',
    title: 'Spectrogram (STFT)',
    subtitle: 'Short-time Fourier transform · the ASR front-end',
    blurb: 'Slide a windowed DFT across a chirp or tone and watch the frequency-vs-time spectrogram fill in.',
    icon: 'M3 4h2v16H3zM7 8h2v12H7zM11 6h2v14h-2zM15 10h2v10h-2zM19 7h2v13h-2z',
    accent: ACCENT,
    codeFile: 'spectrogram_stft.py',
    content: SPECTROGRAM_CONTENT,
    component: React.lazy(() => import('./Spectrogram')),
  },
];
