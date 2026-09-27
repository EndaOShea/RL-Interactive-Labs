import React, { memo, useId, useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotSeries } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, RunControls, Legend, MonoLabel, ParamSlider } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { useTheme } from '../../utils/theme';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { spectrogramPython } from './python';
import { WindowKind, WINDOW_KINDS, WINDOW_LABEL, windowGain, hzToMel } from './shared';
import SpectroOverlay, { TracePoint } from './SpectroOverlay';
import {
  SR, DUR, NYQUIST, WINDOW_SIZES, N_MEL, DB_RANGE, Signal, SIGNALS, CHIRP, TWO_TONE, TONE_HZ, NOISY_TONE, VOWEL, Measurement,
  makeSignal, stft, measure, toDb, specMax, dbTop, peakRow, frameTime, melBandWidths,
} from './stft';

const ACCENT = '#fb923c';
type Scale = 'db' | 'linear';

// Geometry: width and left/right padding equal FunctionPlot's, so the waveform
// above and the spectrogram below share one time axis (and scale together).
const PLOT_W = 560, PAD_L = 44, PAD_R = 14;
const SPEC_H = 240, PAD_T = 10, PAD_B = 32;
const INNER_W = PLOT_W - PAD_L - PAD_R, INNER_H = SPEC_H - PAD_T - PAD_B;
const MEL_TOP = hzToMel(NYQUIST);
const xOf = (tSec: number) => PAD_L + (tSec / DUR) * INNER_W;
const yOf = (hz: number, mel: boolean) => PAD_T + (1 - (mel ? hzToMel(hz) / MEL_TOP : hz / NYQUIST)) * INNER_H;
const TICKS_MS = [0, 125, 250, 375, 500];
const TICKS_HZ_LIN = [0, 1000, 2000, 3000, 4000];
const TICKS_HZ_MEL = [0, 250, 500, 1000, 2000, 4000];
const hzLabel = (f: number) => (f >= 1000 ? `${f / 1000}k` : `${f}`);

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
/** The shared Heatmap's 'heat' ramp (dark → violet → white; light: pale blue → amber → red), t ∈ [0,1]. */
function heatColor(t: number, isLight: boolean): string {
  const v = Math.max(0, Math.min(1, t));
  const rgb = (r: number, g: number, b: number) => `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
  if (isLight) {
    if (v < 0.5) { const u = v / 0.5; return rgb(lerp(230, 253, u), lerp(238, 230, u), lerp(251, 138, u)); }
    const u = (v - 0.5) / 0.5; return rgb(lerp(253, 234, u), lerp(230, 88, u), lerp(138, 12, u));
  }
  if (v < 0.5) { const u = v / 0.5; return rgb(lerp(12, 168, u), lerp(15, 85, u), lerp(22, 247, u)); }
  const u = (v - 0.5) / 0.5; return rgb(lerp(168, 255, u), lerp(85, 255, u), lerp(247, 255, u));
}

const SIGNAL_LABEL: Record<Signal, string> = {
  chirp: 'chirp 200 → 3600 Hz',
  'two-tone': 'two tones 1000 + 1150 Hz',
  tone: 'pure tone 1333 Hz (off-grid)',
  'tone+noise': '1 kHz tone + noise',
  clicks: 'clicks (125, 250, 260 ms)',
  vowel: 'vowel "oo" (formants)',
};

// Guided set-ups: each asks a question; the answer shown is the live measurement.
interface Guided { name: string; signal: Signal; N: number; win: WindowKind; mel: boolean; ask: string; }
const GUIDED: Guided[] = [
  { name: 'short window → clicks in time', signal: 'clicks', N: 64, win: 'hann', mel: false, ask: 'Can an 8 ms window separate two clicks 10 ms apart?' },
  { name: 'long window → resolve two tones', signal: 'two-tone', N: 512, win: 'blackman', mel: false, ask: '1000 and 1150 Hz with 15.6 Hz bins.' },
  { name: 'boxcar → see the leakage', signal: 'tone', N: 256, win: 'rectangular', mel: false, ask: 'An off-grid tone with no taper — then try Hann.' },
  { name: 'vowel formants on mel', signal: 'vowel', N: 256, win: 'hann', mel: true, ask: '20 mel bands over a sung "oo".' },
  { name: 'long window → tone out of noise', signal: 'tone+noise', N: 512, win: 'hann', mel: false, ask: 'Compare the tone-to-noise gap with the 64-sample window.' },
];

const r1 = (v: number) => (Math.round(v * 10) / 10).toString();

/** One sentence per signal, built only from measured values. */
function measurementText(m: Measurement, ctx: { N: number; mel: boolean; scale: Scale }): string {
  const unit = ctx.mel ? 'mel band' : 'bin';
  switch (m.kind) {
    case 'chirp':
      return `The loudest ${unit} climbed from ${Math.round(m.firstHz)} Hz to ${Math.round(m.lastHz)} Hz${m.monotone ? ', never stepping down' : ', jittering where leakage competes with the true peak'}.`;
    case 'two-tone':
      if (m.dipDb === null) return `The two tones, 150 Hz apart, land in adjacent ${ctx.mel ? 'mel bands' : `bins (${r1(SR / ctx.N)} Hz apart)`}, so they merge into one band.`;
      return m.resolved
        ? `Resolved: between the ${TWO_TONE.f1} and ${TWO_TONE.f2} Hz peaks the time-averaged spectrum dips ${r1(m.dipDb)} dB.`
        : `Merged: the time-averaged spectrum dips only ${r1(m.dipDb)} dB between the peaks (less than 3 dB), so they read as one band.`;
    case 'tone':
      return `${m.visible} of the ${m.farRows} ${unit}s more than 500 Hz from the tone sit above the ${DB_RANGE} dB floor; their median level is ${r1(-m.medianDb)} dB below the peak${ctx.scale === 'linear' ? ' — far too faint to see on a linear scale' : ''}.`;
    case 'tone+noise':
      return `The 1 kHz tone reads ${r1(m.toneDb)} dB, ${r1(m.snrDb)} dB above the median noise level per ${unit} (${r1(m.noiseDb)} dB).`;
    case 'clicks':
      return `${m.events.length} separate event${m.events.length === 1 ? '' : 's'} (${m.events.map((e) => (Math.round(e.startMs) === Math.round(e.endMs) ? `${Math.round(e.startMs)}` : `${Math.round(e.startMs)}–${Math.round(e.endMs)}`)).join(', ')} ms): ${m.pairSeparated ? 'the clicks 10 ms apart are told apart' : 'the clicks 10 ms apart merge into one event'} with a ${(ctx.N / SR) * 1000} ms window.`;
    case 'vowel':
      return `The averaged ${ctx.mel ? 'mel bands peak' : 'spectrum peaks'} near ${m.peaksHz.map((f) => Math.round(f)).join(', ')} Hz — close to the formants F1, F2, F3 = ${VOWEL.formants.join(', ')} Hz of the recipe${!ctx.mel && SR / ctx.N <= VOWEL.f0 / 2 ? `, each marked by the nearest ${VOWEL.f0} Hz harmonic` : ''}.`;
  }
}

interface ColumnProps { values: number[]; x0: number; x1: number; y0s: number[]; y1s: number[]; db: boolean; lo: number; span: number; isLight: boolean; }
// One spectrogram column; memoised so a tick re-renders only the new column.
const SpecColumn = memo(function SpecColumn({ values, x0, x1, y0s, y1s, db, lo, span, isLight }: ColumnProps) {
  return (
    <g>
      {values.map((v, r) => {
        const ya = y0s[r] ?? 0, yb = y1s[r] ?? 0;
        const t = db ? (toDb(v) - lo) / span : v / span;
        return <rect key={r} x={x0} y={ya} width={Math.max(0.5, x1 - x0)} height={Math.max(0.5, yb - ya)} fill={heatColor(t, isLight)} shapeRendering="crispEdges" />;
      })}
    </g>
  );
});

const SpectrogramLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [signal, setSignal] = useState<Signal>('chirp');
  const [N, setN] = useState(256);
  const [win, setWin] = useState<WindowKind>('hann');
  const [mel, setMel] = useState(false);
  const [scale, setScale] = useState<Scale>('db');
  const [col, setCol] = useState(0); // number of spectrogram columns computed so far
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();
  const isLight = useTheme() === 'light';
  const clipId = `spec-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const traceColor = isLight ? '#0e7490' : '#fde68a';

  const samples = useMemo(() => makeSignal(signal), [signal]);
  const spec = useMemo(() => stft(samples, N, win, mel), [samples, N, win, mel]);
  const m = useMemo(() => measure(signal, spec), [signal, spec]);
  const hop = spec.hop, frames = spec.frames;
  const binHzStep = SR / N;
  const windowMs = (N / SR) * 1000;
  const bands = useMemo(() => melBandWidths(N), [N]);

  // Colour scale: dB = 20·log10|X| over [top − DB_RANGE, top]; linear = |X| over [0, max].
  const top = useMemo(() => dbTop(spec), [spec]);
  const ampMax = useMemo(() => specMax(spec) || 1, [spec]);
  const lo = scale === 'db' ? top - DB_RANGE : 0;
  const span = scale === 'db' ? DB_RANGE : ampMax;

  // Row / column geometry in SVG pixels.
  const y0s = useMemo(() => spec.rowHiHz.map((f) => yOf(f, mel)), [spec, mel]);
  const y1s = useMemo(() => spec.rowLoHz.map((f) => yOf(f, mel)), [spec, mel]);
  const colX = useMemo(() => Array.from({ length: frames }, (_, c) => {
    const tc = frameTime(c, N), half = hop / 2 / SR;
    return [xOf(tc - half), xOf(tc + half)] as [number, number];
  }), [frames, N, hop]);

  const cur = col > 0 ? col - 1 : -1; // the column the math panel describes
  const trace: (TracePoint | null)[] = useMemo(() => (signal === 'clicks' ? [] : spec.cols.slice(0, col).map((c, i) => {
    const pr = peakRow(c);
    const xs = colX[i];
    return pr >= 0 && xs ? { x: (xs[0] + xs[1]) / 2, y: yOf(spec.rowHz[pr] ?? 0, mel) } : null;
  })), [signal, spec, col, colX, mel]);

  const restart = () => { sim.stop(); setCol(0); setLastLog(null); narration.cancel(); };
  const changeSignal = (s: Signal) => { setSignal(s); restart(); };
  const changeN = (n: number) => { setN(n); restart(); };
  const changeWin = (w: WindowKind) => { setWin(w); restart(); };
  const changeMel = (v: boolean) => { setMel(v); restart(); };
  const changeScale = (s: Scale) => { setScale(s); narration.cancel(); };
  const applyGuided = (g: Guided) => { setSignal(g.signal); setN(g.N); setWin(g.win); setMel(g.mel); setScale('db'); restart(); };
  const activeGuided = GUIDED.find((g) => g.signal === signal && g.N === N && g.win === win && g.mel === mel);

  const mText = measurementText(m, { N, mel, scale });
  const axisWord = mel
    ? `${N_MEL} mel bands (triangles ${Math.round(bands.lowHz)} Hz wide at the bottom, ${Math.round(bands.highHz)} Hz at the top)`
    : `${N / 2 + 1} bins, 0 to ${NYQUIST} Hz in steps of ${r1(binHzStep)} Hz`;

  const introNarration = () => {
    const challenge: Record<Signal, string> = {
      chirp: `The challenge here: pin down a tone whose pitch rises from ${CHIRP.f0} to ${CHIRP.f1} hertz, and show when it sits at each frequency.`,
      'two-tone': `The challenge here: pull apart two steady tones only 150 hertz apart, ${TWO_TONE.f1} and ${TWO_TONE.f2} hertz, played together.`,
      tone: `The challenge here: draw one pure ${Math.round(TONE_HZ)} hertz tone cleanly. It sits a third of a bin off the DFT grid at every window length offered, so no bin lines up with it.`,
      'tone+noise': `The challenge here: find a ${NOISY_TONE.f} hertz tone mixed with uniform noise.`,
      clicks: 'The challenge here: place three clicks in time, one at 125 milliseconds and a pair only 10 milliseconds apart at 250 and 260.',
      vowel: `The challenge here: expose the resonances of a sung oo vowel, a ${VOWEL.f0} hertz buzz whose harmonics are shaped by formants at ${VOWEL.formants.join(', ')} hertz.`,
    };
    const tape = win === 'rectangular'
      ? 'A rectangular window has no taper: the narrowest peak, but the most leakage into far bins.'
      : win === 'blackman'
        ? 'The Blackman taper has the lowest side lobes of the four, at the cost of the widest peak.'
        : `The ${WINDOW_LABEL[win]} taper fades the frame edges to cut leakage, widening the peak a little.`;
    return `${challenge[signal]} A spectrogram slides a ${N} sample window, ${windowMs} milliseconds at 8 kilohertz, across the signal in hops of ${hop} samples. Each column is the magnitude of that frame's windowed Fourier transform, scaled so a sine of amplitude A reads A, on ${mel ? `${N_MEL} mel bands that pool the bins with triangles, narrow at low frequency and wide at high frequency` : `bins ${r1(binHzStep)} hertz apart up to the ${NYQUIST} hertz Nyquist limit`}. Longer windows give finer frequency bins but coarser timing. ${tape} Colours are ${scale === 'db' ? `decibels, twenty log ten of the amplitude, over an ${DB_RANGE} dB range` : 'linear amplitude, which hides anything much quieter than the peak'}.`;
  };

  const step = () => {
    if (col >= frames) { sim.pause(); return; }
    const c = col;
    const nextCol = c + 1;
    setCol(nextCol);
    const values = spec.cols[c] ?? [];
    const pr = peakRow(values);
    const pkHz = pr >= 0 && signal !== 'clicks' ? spec.rowHz[pr] ?? 0 : null; // a click's spectrum is flat: no dominant frequency
    const pkDb = pr >= 0 ? toDb(values[pr] ?? 0) : null;
    const start = c * hop;
    const key = `${signal}:${N}:${win}:${mel}:${scale}`;
    narration.narratePhase(`run:${key}`, introNarration());
    if (nextCol >= frames) {
      narration.narratePhase(`done:${key}`, `The spectrogram is complete. ${mText} This frequency by time picture, log mel in practice, is what a speech recogniser reads instead of the raw waveform.`);
    }
    setLastLog({
      algorithm: `STFT · ${signal} · ${WINDOW_LABEL[win]}${mel ? ' · mel' : ''}`,
      stepDescription: `Frame ${nextCol}/${frames}: samples ${start}–${start + N - 1} (${r1((start / SR) * 1000)}–${r1(((start + N) / SR) * 1000)} ms) × ${win} taper → DFT at k = 0…${N / 2}${mel ? ` → ${N_MEL} mel bands` : ''} → one column`,
      formula: mel
        ? 'M(m,τ) = Σₖ Hₘ(k)·|X(k,τ)|'
        : '|X(k,τ)| = (cₖ/(N·c_g))·|Σₙ w[n]·x[τH+n]·e^(−j2πkn/N)|',
      variables: {
        frame: `${nextCol}/${frames}`,
        't (ms)': +r1(frameTime(c, N) * 1000),
        N, hop,
        'Δf (Hz)': +r1(binHzStep),
        taper: win,
        [`peak (${mel ? 'band Hz' : 'Hz'})`]: pkHz === null ? (pkDb === null ? '—' : 'flat') : Math.round(pkHz),
        'peak (dB)': pkDb === null ? '—' : +r1(pkDb),
      },
      result: pkDb === null
        ? `column ${nextCol}/${frames} · silent frame`
        : pkHz === null
          ? `column ${nextCol}/${frames} · click: flat spectrum at ${r1(pkDb)} dB`
          : `column ${nextCol}/${frames} · loudest ${Math.round(pkHz)} Hz at ${r1(pkDb)} dB`,
      mathDetails: {
        params: [
          { label: 'window', info: `N = ${N} samples = ${windowMs} ms at ${SR} Hz, hop ${hop}. Bins f_k = k·SR/N are ${r1(binHzStep)} Hz apart; only k = 0…N/2 (up to ${NYQUIST} Hz) are unique for a real signal.` },
          { label: 'scaling', info: 'cₖ = 2 for 0 < k < N/2 (1 at DC and Nyquist), c_g = mean of the window: a sine of amplitude A centred on a bin reads A (0 dB for A = 1).' },
          { label: 'taper', info: `${WINDOW_LABEL[win]}: ${win === 'rectangular' ? 'no taper — narrowest main lobe, slowly decaying side lobes (most leakage).' : win === 'blackman' ? 'lowest side lobes here, widest main lobe.' : win === 'hann' ? 'side lobes fall off fast (18 dB per octave), so far bins are far cleaner than with the boxcar.' : 'low first side lobe, but slow fall-off: far bins keep some leakage.'}` },
          mel
            ? { label: 'mel pooling', info: `${N_MEL} HTK triangles (peak 1) on the true bin frequencies, rebuilt for each N: ${Math.round(bands.lowHz)} Hz wide at the bottom, ${Math.round(bands.highHz)} Hz at the top.` }
            : { label: 'axis', info: `${N / 2 + 1} rows, 0 to ${NYQUIST} Hz.` },
          { label: 'colour', info: scale === 'db' ? `20·log10|X| (magnitude dB re amplitude 1), ${lo} to ${top} dB.` : `linear |X|, 0 to ${ampMax.toFixed(3)}.` },
        ],
        implication: mText,
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 150 });

  // Waveform (every sample) and the current frame's window drawn over it.
  const wave = useMemo(() => {
    const mx = Math.max(0.5, ...samples.map((v) => Math.abs(v)));
    return { pts: samples.map((y, n) => ({ x: (n / SR) * 1000, y })), range: [-mx * 1.12, mx * 1.12] as [number, number] };
  }, [samples]);
  const frameColor = 'var(--t0)';
  const frameSeries: PlotSeries[] = useMemo(() => {
    if (cur < 0) return [];
    const start = cur * hop;
    const [ylo, yhi] = wave.range;
    const stride = Math.max(1, Math.floor(N / 96));
    const taper: { x: number; y: number }[] = [];
    for (let n = 0; n < N; n += stride) taper.push({ x: ((start + n) / SR) * 1000, y: ylo + windowGain(win, n, N) * (yhi - ylo) });
    taper.push({ x: ((start + N - 1) / SR) * 1000, y: ylo + windowGain(win, N - 1, N) * (yhi - ylo) });
    const xa = (start / SR) * 1000, xb = ((start + N - 1) / SR) * 1000;
    return [
      { points: taper, color: frameColor, width: 1.2, area: true },
      { points: [{ x: xa, y: ylo }, { x: xa, y: yhi }], color: frameColor, width: 1, dash: true },
      { points: [{ x: xb, y: ylo }, { x: xb, y: yhi }], color: frameColor, width: 1, dash: true },
    ];
  }, [cur, hop, N, win, wave.range]);

  const peakNow = (() => {
    if (cur < 0) return '—';
    const c = spec.cols[cur] ?? [];
    const pr = peakRow(c);
    if (pr < 0) return 'silent';
    return signal === 'clicks' ? `flat · ${r1(toDb(c[pr] ?? 0))} dB` : `${Math.round(spec.rowHz[pr] ?? 0)} Hz`;
  })();

  // Not-yet-computed span (hatched) and the current column outline.
  const pendX0 = col < frames ? colX[col]?.[0] ?? PAD_L : null;
  const pendX1 = colX[frames - 1]?.[1] ?? PAD_L + INNER_W;
  const curX = cur >= 0 ? colX[cur] : undefined;
  const hatch = useMemo(() => {
    const lines: number[] = [];
    for (let x = PAD_L - INNER_H; x < PAD_L + INNER_W; x += 9) lines.push(x);
    return lines;
  }, []);

  const insight = `${SIGNAL_LABEL[signal]} at ${SR / 1000} kHz for ${DUR * 1000} ms · ${WINDOW_LABEL[win]} window of ${N} samples (${windowMs} ms, ${frames} frames, hop ${hop}) · ${axisWord} · ${scale === 'db' ? 'dB colour scale' : 'linear colour scale'}. Measured: ${mText} The spectrogram is the standard ASR front-end: an acoustic model reads these spectral frames — log-mel in practice — not the raw waveform.`;

  const barVals = Array.from({ length: 48 }, (_, i) => i / 47);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'WINDOW', value: `${N} · ${windowMs} ms`, color: ACCENT },
        { label: 'Δf', value: `${r1(binHzStep)} Hz` },
        { label: 'TAPER', value: win },
        { label: 'PEAK', value: peakNow },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, spectrogramPython(signal, N, win, mel, scale))}
      narration={narration}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'center' }}>
          <FunctionPlot
            width={PLOT_W} height={150} domain={[0, DUR * 1000]} range={wave.range}
            series={[{ points: wave.pts, color: ACCENT, width: 1 }, ...frameSeries]}
            xLabel="time (ms)" yLabel="x(t)"
          />
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>
              Spectrogram · {mel ? `${N_MEL} mel bands` : `${N / 2 + 1} bins to ${NYQUIST} Hz`} × {frames} frames
            </MonoLabel>
            <svg width={PLOT_W} height={SPEC_H} viewBox={`0 0 ${PLOT_W} ${SPEC_H}`} style={{ display: 'block', maxWidth: '100%', borderRadius: 14, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)' }}>
              <defs>
                <clipPath id={clipId}><rect x={PAD_L} y={PAD_T} width={INNER_W} height={INNER_H} /></clipPath>
              </defs>
              {spec.cols.slice(0, col).map((values, c) => {
                const xs = colX[c];
                return xs ? <SpecColumn key={c} values={values} x0={xs[0]} x1={xs[1]} y0s={y0s} y1s={y1s} db={scale === 'db'} lo={lo} span={span} isLight={isLight} /> : null;
              })}
              {pendX0 !== null && (
                <g clipPath={`url(#${clipId})`}>
                  <rect x={pendX0} y={PAD_T} width={Math.max(0, pendX1 - pendX0)} height={INNER_H} fill="var(--bg1)" fillOpacity={0.55} />
                  {hatch.filter((x) => x + INNER_H > pendX0 && x < pendX1).map((x) => (
                    <line key={x} x1={Math.max(x, pendX0)} y1={PAD_T + INNER_H - (Math.max(x, pendX0) - x)} x2={Math.min(x + INNER_H, pendX1)} y2={PAD_T + INNER_H - (Math.min(x + INNER_H, pendX1) - x)} stroke="var(--t2)" strokeOpacity={0.3} strokeWidth={1} />
                  ))}
                </g>
              )}
              <rect x={PAD_L} y={PAD_T} width={INNER_W} height={INNER_H} fill="none" stroke="var(--border)" />
              {TICKS_MS.map((ms) => (
                <g key={`x${ms}`}>
                  <line x1={xOf(ms / 1000)} y1={PAD_T + INNER_H} x2={xOf(ms / 1000)} y2={PAD_T + INNER_H + 3} stroke="var(--border)" />
                  <text x={xOf(ms / 1000)} y={PAD_T + INNER_H + 14} textAnchor="middle" fill="var(--t2)" fontSize="8.5" fontFamily="var(--mono)">{ms}</text>
                </g>
              ))}
              {(mel ? TICKS_HZ_MEL : TICKS_HZ_LIN).map((hz) => (
                <g key={`y${hz}`}>
                  <line x1={PAD_L - 3} y1={yOf(hz, mel)} x2={PAD_L} y2={yOf(hz, mel)} stroke="var(--border)" />
                  <text x={PAD_L - 6} y={yOf(hz, mel) + 3} textAnchor="end" fill="var(--t2)" fontSize="8.5" fontFamily="var(--mono)">{hzLabel(hz)}</text>
                </g>
              ))}
              <text x={PAD_L + INNER_W / 2} y={SPEC_H - 6} textAnchor="middle" fill="var(--t2)" fontSize="10" fontFamily="var(--mono)">time (ms) · each column = one frame, centred on its window</text>
              <text x={12} y={PAD_T + INNER_H / 2} textAnchor="middle" fill="var(--t2)" fontSize="10" fontFamily="var(--mono)" transform={`rotate(-90 12 ${PAD_T + INNER_H / 2})`}>{mel ? 'mel scale (Hz)' : 'frequency (Hz)'}</text>
              <SpectroOverlay
                trace={trace}
                frame={curX ? { x0: curX[0], x1: curX[1], y0: PAD_T, y1: PAD_T + INNER_H } : null}
                traceColor={traceColor}
                frameColor={frameColor}
              />
            </svg>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, justifyContent: 'center', fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)' }}>
              <span>{scale === 'db' ? `${lo} dB` : '0'}</span>
              <svg width={192} height={10} style={{ display: 'block', borderRadius: 2 }}>
                {barVals.map((t, i) => <rect key={i} x={i * 4} y={0} width={4.5} height={10} fill={heatColor(t, isLight)} shapeRendering="crispEdges" />)}
              </svg>
              <span>{scale === 'db' ? `${top} dB` : ampMax.toFixed(3)}</span>
              <span style={{ marginLeft: 6 }}>{scale === 'db' ? '20·log10|X| (magnitude dB re amplitude 1)' : '|X| linear amplitude'}</span>
            </div>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={restart} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="SPECTROGRAM" items={[
          ...(signal === 'clicks' ? [] : [{ color: traceColor, label: mel ? 'loudest band per frame' : 'loudest bin per frame' }]),
          { node: <span style={{ width: 10, height: 10, borderRadius: 2, border: `1.5px solid ${frameColor}` }} />, label: 'current frame (both plots)' },
          { node: <span style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--bg1)', border: '1px dashed var(--t2)' }} />, label: 'not computed yet' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Spectrogram (STFT)" hint="Slide a windowed DFT across 0.5 s of 8 kHz audio." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Signal</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {SIGNALS.map((s) => (
                <AlgoPill key={s} active={signal === s} accent={ACCENT} onClick={() => changeSignal(s)}>{SIGNAL_LABEL[s]}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Window length N · time–frequency tradeoff</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {WINDOW_SIZES.map((n) => (
                <AlgoPill key={n} active={N === n} accent={ACCENT} onClick={() => changeN(n)}>{`${n} · ${(n / SR) * 1000} ms`}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '7px 0 0', lineHeight: 1.45 }}>
              Bins {r1(binHzStep)} Hz apart · frames every {(hop / SR) * 1000} ms (hop N/2).
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Window function (taper)</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {WINDOW_KINDS.map((w) => (
                <AlgoPill key={w} active={win === w} accent={ACCENT} onClick={() => changeWin(w)}>{w}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Frequency axis · colour scale</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              <AlgoPill active={!mel} accent={ACCENT} onClick={() => changeMel(false)}>linear bins</AlgoPill>
              <AlgoPill active={mel} accent="#a855f7" onClick={() => changeMel(true)}>{`mel (${N_MEL} bands)`}</AlgoPill>
              <AlgoPill active={scale === 'db'} accent={ACCENT} onClick={() => changeScale('db')}>dB</AlgoPill>
              <AlgoPill active={scale === 'linear'} accent={ACCENT} onClick={() => changeScale('linear')}>linear</AlgoPill>
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Try this · guided</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {GUIDED.map((g) => (
                <AlgoPill key={g.name} active={activeGuided?.name === g.name} accent={ACCENT} onClick={() => applyGuided(g)}>{g.name}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '7px 0 0', lineHeight: 1.45 }}>
              {activeGuided ? `${activeGuided.ask} ` : ''}Measured on the full spectrogram: {mText}
            </p>
          </div>
          <ParamSlider
            name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed}
            accent={ACCENT}
            onChange={sim.setSpeed} hint="per-column interval"
          />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Run computes one spectrogram column (one frame) per tick, left → right.</div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Spectrogram / STFT', signal, sampleRate: SR, windowSize: N, windowMs, binHz: binHzStep, window: win, mel, scale, frames, hop, measured: mText }}
      apiPanel={apiPanel}
    />
  );
};

export default SpectrogramLab;
