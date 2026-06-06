import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import Heatmap from '../../components/labkit/viz/Heatmap';
import { AlgoPill, RunControls, Legend, MonoLabel, ParamSlider } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { spectrogramPython } from './python';

const ACCENT = '#fb923c';
const SR = 512;        // samples in the full signal (1 "second")
const FREQ_BINS = 24;  // displayed frequency bins (low DFT bins, which hold our content)

type Signal = 'chirp' | 'two-tone' | 'tone+noise';

// Deterministic pseudo-noise so the signal is stable across renders.
function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function makeSignal(kind: Signal): number[] {
  const x: number[] = [];
  const rnd = mulberry(7);
  for (let i = 0; i < SR; i++) {
    const t = i / SR; // 0..1
    if (kind === 'chirp') {
      const f0 = 4, f1 = 40;                          // frequency rises over time
      x.push(Math.sin(2 * Math.PI * (f0 * t + 0.5 * (f1 - f0) * t * t)));
    } else if (kind === 'two-tone') {
      x.push(Math.sin(2 * Math.PI * 6 * t) + 0.7 * Math.sin(2 * Math.PI * 22 * t));
    } else {
      x.push(Math.sin(2 * Math.PI * 14 * t) + 0.6 * (rnd() * 2 - 1));
    }
  }
  return x;
}

// Direct O(N²) DFT magnitude over the first FREQ_BINS bins of one frame.
function frameSpectrum(frame: number[], bins: number): number[] {
  const N = frame.length;
  // Hann window to reduce spectral leakage.
  const out: number[] = [];
  for (let k = 0; k < bins; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (N - 1));
      const s = frame[n] * w;
      const ang = (-2 * Math.PI * k * n) / N;
      re += s * Math.cos(ang);
      im += s * Math.sin(ang);
    }
    out.push(Math.sqrt(re * re + im * im) / N);
  }
  return out;
}

const SpectrogramLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [signal, setSignal] = useState<Signal>('chirp');
  const [windowSize, setWindowSize] = useState(64);
  const [col, setCol] = useState(0);     // next spectrogram column to fill
  const [spec, setSpec] = useState<number[][]>([]); // [freqBin][timeCol]
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const samples = useMemo(() => makeSignal(signal), [signal]);

  // Frame layout: hop = half the window (50% overlap).
  const hop = Math.max(1, Math.floor(windowSize / 2));
  const nFrames = Math.max(1, Math.floor((SR - windowSize) / hop) + 1);

  // Precompute the full spectrogram so we know the colour scale and can fill it
  // progressively. Rows are frequency bins HIGH→LOW for display.
  const fullSpec = useMemo(() => {
    const cols: number[][] = []; // [frame][freqBin]
    let peak = 1e-9;
    for (let fr = 0; fr < nFrames; fr++) {
      const start = fr * hop;
      const frame = samples.slice(start, start + windowSize);
      const mag = frameSpectrum(frame, FREQ_BINS);
      cols.push(mag);
      mag.forEach((m) => { if (m > peak) peak = m; });
    }
    return { cols, peak };
  }, [samples, windowSize, hop, nFrames]);

  const resetSpec = () => {
    setSpec(Array.from({ length: FREQ_BINS }, () => Array(nFrames).fill(0)));
    setCol(0);
    setLastLog(null);
  };

  // Reset the spectrogram whenever the layout changes.
  const layoutKey = `${signal}-${windowSize}-${nFrames}`;
  const [seenKey, setSeenKey] = useState('');
  if (layoutKey !== seenKey) {
    setSeenKey(layoutKey);
    setSpec(Array.from({ length: FREQ_BINS }, () => Array(nFrames).fill(0)));
    setCol(0);
    setLastLog(null);
  }

  const step = () => {
    if (col >= nFrames) { sim.pause(); return; }
    const mag = fullSpec.cols[col];
    // Peak bin in this frame → dominant frequency.
    let pk = 0; for (let k = 1; k < mag.length; k++) if (mag[k] > mag[pk]) pk = k;
    setSpec((prev) => {
      const next = prev.map((r) => r.slice());
      for (let k = 0; k < FREQ_BINS; k++) {
        // Display row 0 = highest frequency, so flip the bin index.
        next[FREQ_BINS - 1 - k][col] = mag[k];
      }
      return next;
    });
    const nextCol = col + 1;
    setCol(nextCol);
    setLastLog({
      algorithm: `STFT · ${signal}`,
      stepDescription: 'Take the windowed DFT of the current frame → one spectrogram column',
      formula: 'X(f,τ) = | Σₙ w[n]·x[n+τ·H]·e^(−j2πfn/N) |',
      variables: {
        'frame': `${nextCol}/${nFrames}`,
        'window': windowSize,
        'hop': hop,
        'peak bin': pk,
        '|X|max': +mag[pk].toFixed(3),
      },
      result: `frame ${nextCol}/${nFrames} · dominant bin ${pk}`,
      mathDetails: {
        params: [
          { label: 'window', info: `${windowSize} samples. Longer = finer frequency, coarser time; shorter = sharper time, blurrier frequency.` },
          { label: 'frame', info: 'Each Run tick computes one frame and paints one column of the spectrogram, left → right.' },
          { label: 'front-end', info: 'This frequency-vs-time matrix (log-mel in practice) is exactly what feeds a speech recogniser.' },
        ],
        implication: signal === 'chirp'
          ? 'Watch the bright band climb: a chirp sweeps frequency, so its peak bin rises across columns.'
          : signal === 'two-tone'
            ? 'Two steady horizontal bands appear — two constant tones at fixed frequencies.'
            : 'A clear band (the tone) sits over a broadband haze (the noise).',
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 90 });
  const reset = () => { sim.stop(); resetSpec(); };

  // Waveform plot (downsampled for clarity).
  const wave = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    const stepN = Math.max(1, Math.floor(SR / 256));
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < SR; i += stepN) { const y = samples[i]; pts.push({ x: i / SR, y }); mn = Math.min(mn, y); mx = Math.max(mx, y); }
    const span = Math.max(Math.abs(mn), Math.abs(mx), 0.5);
    return { pts, range: [-span * 1.1, span * 1.1] as [number, number] };
  }, [samples]);

  const peakBinNow = useMemo(() => {
    if (col === 0) return '—';
    const mag = fullSpec.cols[Math.min(col - 1, nFrames - 1)];
    let pk = 0; for (let k = 1; k < mag.length; k++) if (mag[k] > mag[pk]) pk = k;
    return `bin ${pk}`;
  }, [col, fullSpec, nFrames]);

  const insight = `${signal} · window ${windowSize} samples (${nFrames} frames, hop ${hop}). ` +
    'The spectrogram (frequency × time) is the standard ASR front-end: an acoustic model reads these spectral frames — not the raw waveform — and emits character/phoneme probabilities. ' +
    'Window size sets the time–frequency tradeoff: longer windows sharpen frequency but blur time.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'WINDOW', value: windowSize, color: ACCENT },
        { label: 'FRAMES', value: nFrames },
        { label: 'PEAK', value: peakBinNow },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, spectrogramPython(signal, windowSize))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'center' }}>
          <FunctionPlot
            width={560} height={170} domain={[0, 1]} range={wave.range}
            series={[{ points: wave.pts, color: ACCENT, width: 1.6 }]}
            xLabel="time" yLabel="x(t)"
          />
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>Spectrogram · freq (high→low) × time</MonoLabel>
            <Heatmap
              matrix={spec.length ? spec : [[0]]}
              mode="heat"
              min={0}
              max={fullSpec.peak}
              cell={Math.max(4, Math.min(12, Math.floor(520 / Math.max(1, nFrames))))}
              gap={1}
              accent={ACCENT}
            />
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="SPECTROGRAM" items={[
          { color: '#0c0f16', label: 'low energy' },
          { color: '#a855f7', label: 'mid' },
          { color: '#ffffff', label: 'high energy' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Spectrogram (STFT)" hint="Slide a windowed DFT across the signal." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Signal</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(['chirp', 'two-tone', 'tone+noise'] as Signal[]).map((s) => (
                <AlgoPill key={s} active={signal === s} accent={ACCENT} onClick={() => { setSignal(s); }}>{s}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider
            name="Window size"
            value={`${windowSize}`} min={16} max={128} step={16} current={windowSize}
            accent={ACCENT}
            onChange={(v) => setWindowSize(v)}
            hint="samples per frame · time–freq tradeoff"
          />
          <ParamSlider
            name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed}
            accent={ACCENT}
            onChange={sim.setSpeed} hint="per-frame interval"
          />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Run fills one spectrogram column per tick, left → right.</div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Spectrogram / STFT', signal, windowSize, frames: nFrames, hop }}
      apiPanel={apiPanel}
    />
  );
};

export default SpectrogramLab;
