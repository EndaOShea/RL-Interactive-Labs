import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import Heatmap from '../../components/labkit/viz/Heatmap';
import { AlgoPill, RunControls, Legend, MonoLabel, ParamSlider } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { spectrogramPython } from './python';
import { WindowKind, WINDOW_KINDS, WINDOW_LABEL, windowGain, windowCoherentGain, melFilterbank, applyMel } from './shared';
import SpectroOverlay from './SpectroOverlay';

const ACCENT = '#fb923c';
const TRACE = '#fde68a';
const SR = 512;        // samples in the full signal (1 "second")
const LIN_BINS = 24;   // linear DFT bins computed (low bins, which hold our content)
const MEL_BINS = 16;   // mel bands when the mel axis is on

type Signal = 'chirp' | 'two-tone' | 'tone+noise' | 'vowel';

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
    } else if (kind === 'vowel') {
      // Three steady "formants" (resonant bands) → a sustained vowel-like sound.
      x.push(
        Math.sin(2 * Math.PI * 5 * t) +
        0.8 * Math.sin(2 * Math.PI * 11 * t) +
        0.5 * Math.sin(2 * Math.PI * 19 * t),
      );
    } else {
      x.push(Math.sin(2 * Math.PI * 14 * t) + 0.6 * (rnd() * 2 - 1));
    }
  }
  return x;
}

// Direct O(N²) DFT magnitude over the first `bins` linear bins of one frame,
// with a selectable taper (window function). Coherent-gain normalised so the
// magnitude scale stays comparable across windows.
function frameSpectrum(frame: number[], bins: number, win: WindowKind): number[] {
  const N = frame.length;
  const cg = Math.max(1e-6, windowCoherentGain(win, N));
  const out: number[] = [];
  for (let k = 0; k < bins; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const s = frame[n] * windowGain(win, n, N);
      const ang = (-2 * Math.PI * k * n) / N;
      re += s * Math.cos(ang);
      im += s * Math.sin(ang);
    }
    out.push(Math.sqrt(re * re + im * im) / N / cg);
  }
  return out;
}

const SpectrogramLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [signal, setSignal] = useState<Signal>('chirp');
  const [windowSize, setWindowSize] = useState(64);
  const [win, setWin] = useState<WindowKind>('hann');
  const [mel, setMel] = useState(false);
  const [col, setCol] = useState(0);     // next spectrogram column to fill
  const [spec, setSpec] = useState<number[][]>([]); // [freqBin][timeCol]
  const [peakRows, setPeakRows] = useState<number[]>([]); // display-row of peak per col (-1 empty)
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const samples = useMemo(() => makeSignal(signal), [signal]);

  // Number of displayed frequency rows depends on the axis.
  const nBins = mel ? MEL_BINS : LIN_BINS;

  // Frame layout: hop = half the window (50% overlap).
  const hop = Math.max(1, Math.floor(windowSize / 2));
  const nFrames = Math.max(1, Math.floor((SR - windowSize) / hop) + 1);

  // Mel filterbank for the current linear-bin count (rebuilt only when needed).
  const melBank = useMemo(() => melFilterbank(LIN_BINS, MEL_BINS, LIN_BINS), []);

  // Precompute the full spectrogram so we know the colour scale and can fill it
  // progressively. Each column is the (optionally mel-warped) magnitude vector.
  const fullSpec = useMemo(() => {
    const cols: number[][] = []; // [frame][freqBin], low→high
    let peak = 1e-9;
    for (let fr = 0; fr < nFrames; fr++) {
      const start = fr * hop;
      const frame = samples.slice(start, start + windowSize);
      const magLin = frameSpectrum(frame, LIN_BINS, win);
      const mag = mel ? applyMel(magLin, melBank.filters) : magLin;
      cols.push(mag);
      mag.forEach((m) => { if (m > peak) peak = m; });
    }
    return { cols, peak };
  }, [samples, windowSize, hop, nFrames, win, mel, melBank]);

  const resetState = () => {
    setSpec(Array.from({ length: nBins }, () => Array(nFrames).fill(0)));
    setPeakRows(Array(nFrames).fill(-1));
    setCol(0);
    setLastLog(null);
  };

  // Reset the spectrogram whenever the layout changes (signal / window / axis / taper).
  const layoutKey = `${signal}-${windowSize}-${nFrames}-${mel}-${win}`;
  const [seenKey, setSeenKey] = useState('');
  if (layoutKey !== seenKey) {
    setSeenKey(layoutKey);
    setSpec(Array.from({ length: nBins }, () => Array(nFrames).fill(0)));
    setPeakRows(Array(nFrames).fill(-1));
    setCol(0);
    setLastLog(null);
  }

  const step = () => {
    if (col >= nFrames) { sim.pause(); narration.narrate('Spectrogram complete, all frames painted.', { interrupt: true }); return; }
    const mag = fullSpec.cols[col];
    // Peak bin in this frame → dominant frequency.
    let pk = 0; for (let k = 1; k < mag.length; k++) if (mag[k] > mag[pk]) pk = k;
    const displayRow = nBins - 1 - pk; // row 0 = highest frequency
    setSpec((prev) => {
      const next = prev.map((r) => r.slice());
      for (let k = 0; k < nBins; k++) {
        next[nBins - 1 - k][col] = mag[k];
      }
      return next;
    });
    setPeakRows((prev) => { const n = prev.slice(); n[col] = displayRow; return n; });
    const nextCol = col + 1;
    setCol(nextCol);

    // Track movement of the dominant band for narration variety.
    const prevPk = col > 0 ? (() => { const m = fullSpec.cols[col - 1]; let p = 0; for (let k = 1; k < m.length; k++) if (m[k] > m[p]) p = k; return p; })() : pk;
    const dir = pk > prevPk ? 'rising' : pk < prevPk ? 'falling' : 'steady';
    const axisName = mel ? 'mel band' : 'bin';

    setLastLog({
      algorithm: `STFT · ${signal} · ${WINDOW_LABEL[win]}${mel ? ' · mel' : ''}`,
      stepDescription: mel
        ? 'Windowed DFT of the frame, then pool linear bins into mel bands → one column'
        : 'Take the windowed DFT of the current frame → one spectrogram column',
      formula: mel
        ? 'M(m,τ) = Σ_f H_m(f)·|X(f,τ)|'
        : 'X(f,τ) = | Σₙ w[n]·x[n+τ·H]·e^(−j2πfn/N) |',
      variables: {
        'frame': `${nextCol}/${nFrames}`,
        'window': windowSize,
        'taper': win,
        'hop': hop,
        'peak': `${axisName} ${pk}`,
        '|X|max': +mag[pk].toFixed(3),
        'trend': dir,
      },
      result: `frame ${nextCol}/${nFrames} · dominant ${axisName} ${pk} (${dir})`,
      mathDetails: {
        params: [
          { label: 'window', info: `${windowSize} samples. Longer = finer frequency, coarser time; shorter = sharper time, blurrier frequency.` },
          { label: 'taper', info: `${WINDOW_LABEL[win]}: ${win === 'rectangular' ? 'no taper — sharpest main lobe but worst spectral leakage.' : win === 'blackman' ? 'very low side-lobes — least leakage, widest main lobe.' : 'a smooth taper that trades a little frequency width for far less leakage.'}` },
          mel
            ? { label: 'mel pooling', info: `${MEL_BINS} triangular mel filters pool the ${LIN_BINS} linear bins; high frequencies share wider bands, matching the cochlea.` }
            : { label: 'front-end', info: 'This frequency-vs-time matrix (log-mel in practice) is exactly what feeds a speech recogniser.' },
        ],
        implication: signal === 'chirp'
          ? 'Watch the bright band climb: a chirp sweeps frequency, so its peak bin rises across columns.'
          : signal === 'two-tone'
            ? 'Two steady horizontal bands appear — two constant tones at fixed frequencies.'
            : signal === 'vowel'
              ? 'Three stacked steady bands (formants) define the vowel — exactly the cue speech models read.'
              : 'A clear band (the tone) sits over a broadband haze (the noise).',
      },
    });

    // Narrate the live event: where the energy is and how it is moving.
    if (signal === 'chirp') {
      narration.narrate(`Frame ${nextCol}. Peak ${axisName} ${pk}, ${dir}.`);
    } else if (dir !== 'steady') {
      narration.narrate(`Dominant ${axisName} ${dir} to ${pk} at frame ${nextCol}.`);
    } else {
      narration.narrate(`Frame ${nextCol}, energy steady at ${axisName} ${pk}.`);
    }
  };
  const sim = useSimLoop(step, { initialSpeed: 90 });
  const reset = () => { sim.stop(); resetState(); narration.cancel(); };

  const changeSignal = (s: Signal) => { setSignal(s); narration.cancel(); narration.narrate(`Loaded ${s} signal.`, { interrupt: true }); };
  const changeWin = (w: WindowKind) => { setWin(w); narration.cancel(); narration.narrate(`${WINDOW_LABEL[w]} window selected.`, { interrupt: true }); };
  const toggleMel = () => { const v = !mel; setMel(v); narration.cancel(); narration.narrate(v ? 'Mel frequency axis on.' : 'Linear frequency axis on.', { interrupt: true }); };

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
    return `${mel ? 'mel' : 'bin'} ${pk}`;
  }, [col, fullSpec, nFrames, mel]);

  const cell = Math.max(4, Math.min(12, Math.floor(520 / Math.max(1, nFrames))));
  const gap = 1;
  // Heatmap pads by 4px on each side when there are no labels; match it.
  const HM_PAD = 4;

  const insight = `${signal} · ${WINDOW_LABEL[win]} window of ${windowSize} samples (${nFrames} frames, hop ${hop})` +
    `${mel ? ', mel frequency axis' : ''}. ` +
    'The spectrogram (frequency × time) is the standard ASR front-end: an acoustic model reads these spectral frames — not the raw waveform — and emits character/phoneme probabilities. ' +
    'Window size sets the time–frequency tradeoff, the taper controls spectral leakage, and the mel axis pools high frequencies the way the ear does. The yellow trace marks the dominant frequency over time.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'WINDOW', value: windowSize, color: ACCENT },
        { label: 'TAPER', value: win },
        { label: 'PEAK', value: peakBinNow },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, spectrogramPython(signal, windowSize, win, mel))}
      narration={narration}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'center' }}>
          <FunctionPlot
            width={560} height={170} domain={[0, 1]} range={wave.range}
            series={[{ points: wave.pts, color: ACCENT, width: 1.6 }]}
            xLabel="time" yLabel="x(t)"
          />
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>
              Spectrogram · {mel ? 'mel' : 'freq'} (high→low) × time
            </MonoLabel>
            <div style={{ position: 'relative' }}>
              <Heatmap
                matrix={spec.length ? spec : [[0]]}
                mode="heat"
                min={0}
                max={fullSpec.peak}
                cell={cell}
                gap={gap}
                accent={ACCENT}
              />
              <div style={{ position: 'absolute', left: HM_PAD, top: HM_PAD, pointerEvents: 'none' }}>
                <SpectroOverlay
                  peakRows={peakRows.length ? peakRows : [-1]}
                  nCols={nFrames}
                  nRows={nBins}
                  cell={cell}
                  gap={gap}
                  current={col}
                  color={TRACE}
                />
              </div>
            </div>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="SPECTROGRAM" items={[
          { color: '#0c0f16', label: 'low energy' },
          { color: '#a855f7', label: 'mid' },
          { color: '#ffffff', label: 'high energy' },
          { color: TRACE, label: 'peak trace' },
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
              {(['chirp', 'two-tone', 'tone+noise', 'vowel'] as Signal[]).map((s) => (
                <AlgoPill key={s} active={signal === s} accent={ACCENT} onClick={() => changeSignal(s)}>{s}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Window function (taper)</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {WINDOW_KINDS.map((w) => (
                <AlgoPill key={w} active={win === w} accent={ACCENT} onClick={() => changeWin(w)}>{w}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Frequency axis</MonoLabel>
            <AlgoPill active={mel} accent="#a855f7" onClick={toggleMel}>
              {mel ? 'mel scale (perceptual)' : 'linear → switch to mel'}
            </AlgoPill>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Try this · guided</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              <AlgoPill accent={ACCENT} onClick={() => { changeSignal('chirp'); setWindowSize(32); setWin('hann'); }}>short window → sharp time</AlgoPill>
              <AlgoPill accent={ACCENT} onClick={() => { changeSignal('two-tone'); setWindowSize(128); setWin('blackman'); }}>long window → resolve two tones</AlgoPill>
              <AlgoPill accent={ACCENT} onClick={() => { changeSignal('tone+noise'); setWin('rectangular'); }}>boxcar → see the leakage</AlgoPill>
              <AlgoPill accent={ACCENT} onClick={() => { changeSignal('vowel'); setMel(true); setWindowSize(96); }}>vowel formants on mel axis</AlgoPill>
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
      currentParams={{ topic: 'Spectrogram / STFT', signal, windowSize, window: win, mel, frames: nFrames, hop }}
      apiPanel={apiPanel}
    />
  );
};

export default SpectrogramLab;
