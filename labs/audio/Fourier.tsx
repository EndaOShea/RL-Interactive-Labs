import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import DistributionBars from '../../components/labkit/viz/DistributionBars';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { fourierPython } from './python';
import { FOURIER_EXTRA_PRESETS, hzToMel } from './shared';

const ACCENT = '#fb923c';
const COMP = 'rgba(120,160,250,.45)';
const MEL_CLR = '#a855f7';
const K = 5;            // number of harmonics (= number of amplitude sliders)
const F0 = 1;          // fundamental (one period across the x-axis)
const SAMPLES = 256;   // points over one period

type Preset = 'sine' | 'square' | 'sawtooth' | 'triangle' | 'pulse' | 'organ' | 'clarinet' | 'custom';
type SpectrumView = 'linear' | 'mel';

// Amplitude presets for harmonics k = 1..K (truncated Fourier series).
const BASE_PRESETS: Record<'sine' | 'square' | 'sawtooth' | 'triangle', number[]> = {
  sine: [1, 0, 0, 0, 0],
  square: [1, 0, 1 / 3, 0, 1 / 5],          // odd harmonics ∝ 1/k
  sawtooth: [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5], // all harmonics ∝ 1/k
  triangle: [1, 0, -1 / 9, 0, 1 / 25],       // odd harmonics ∝ 1/k²
};
const PRESETS: Record<Exclude<Preset, 'custom'>, number[]> = {
  ...BASE_PRESETS,
  pulse: FOURIER_EXTRA_PRESETS[0].amps,
  organ: FOURIER_EXTRA_PRESETS[1].amps,
  clarinet: FOURIER_EXTRA_PRESETS[2].amps,
};
const EXTRA_BLURB: Record<string, string> = Object.fromEntries(FOURIER_EXTRA_PRESETS.map((p) => [p.id, p.blurb]));

const capitalise = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

const FourierLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [amps, setAmps] = useState<number[]>(PRESETS.square);
  const [preset, setPreset] = useState<Preset>('square');
  const [showComponents, setShowComponents] = useState(true);
  const [view, setView] = useState<SpectrumView>('linear');
  const [phase, setPhase] = useState(0); // animated time shift (fraction of a period)
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const setAmp = (i: number, v: number) => {
    setAmps((a) => { const n = [...a]; n[i] = v; return n; });
    setPreset('custom');
  };
  const applyPreset = (p: Exclude<Preset, 'custom'>) => {
    setAmps(PRESETS[p]); setPreset(p); setPhase(0); setLastLog(null);
    narration.cancel();
  };
  const setSpectrumView = (v: SpectrumView) => {
    setView(v);
    narration.cancel();
  };

  // Sum and per-component waveforms over one period (shifted by phase).
  const data = useMemo(() => {
    const sum: { x: number; y: number }[] = [];
    const comps: { x: number; y: number }[][] = amps.map(() => []);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < SAMPLES; i++) {
      const x = i / (SAMPLES - 1);                 // 0..1 = one period
      const t = x + phase;
      let s = 0;
      amps.forEach((a, ki) => {
        const k = ki + 1;
        const c = a * Math.sin(2 * Math.PI * k * F0 * t);
        comps[ki].push({ x, y: c });
        s += c;
      });
      sum.push({ x, y: s });
      mn = Math.min(mn, s); mx = Math.max(mx, s);
    }
    const span = Math.max(Math.abs(mn), Math.abs(mx), 0.5);
    return { sum, comps, range: [-span * 1.12, span * 1.12] as [number, number] };
  }, [amps, phase]);

  const activeHarmonics = amps.filter((a) => Math.abs(a) > 1e-4).length;
  const maxAbs = Math.max(1e-9, ...amps.map((a) => Math.abs(a)));

  // Spectrum bars: linear (one bar per harmonic) OR mel-warped (label by mel
  // band centre, so the harmonic spacing is compressed at high frequency).
  const spectrumBars = useMemo(() => {
    if (view === 'linear') {
      return amps.map((a, i) => ({
        label: `k=${i + 1}`,
        value: Math.abs(a),
        color: ACCENT,
        highlight: Math.abs(a) === maxAbs && Math.abs(a) > 1e-4,
      }));
    }
    // Mel view: place each harmonic at its mel position relative to the top
    // harmonic, label with the mel value (rounded) to show the warp.
    const melMax = hzToMel(K * F0 * 1000); // scale so the spread is visible
    return amps.map((a, i) => {
      const fHz = (i + 1) * F0 * 1000;     // treat k as kHz for a readable mel range
      const mel = hzToMel(fHz);
      return {
        label: `${Math.round(mel)}m`,
        value: Math.abs(a) * (0.55 + 0.45 * (1 - mel / melMax)), // perceptual de-emphasis of highs
        color: MEL_CLR,
        highlight: i + 1 === amps.reduce((bi, x, j, arr) => (Math.abs(x) > Math.abs(arr[bi]) ? j : bi), 0) + 1 && Math.abs(a) > 1e-4,
      };
    });
  }, [amps, view, maxAbs]);

  const dominant = amps.reduce((bi, a, i, arr) => (Math.abs(a) > Math.abs(arr[bi]) ? i : bi), 0) + 1;

  const step = () => {
    const np = (phase + 0.02) % 1;
    setPhase(np);
    // Reconstruction error vs an ideal target (Gibbs ripple proxy): how peaky
    // the running waveform is relative to its harmonic budget.
    const peakAbs = Math.max(...data.sum.map((p) => Math.abs(p.y)), 1e-9);
    const overshoot = +(peakAbs / Math.max(1e-9, maxAbs)).toFixed(2);
    setLastLog({
      algorithm: `Fourier Series · ${preset}${view === 'mel' ? ' · mel' : ''}`,
      stepDescription: view === 'mel'
        ? 'Sum harmonics, then warp the spectrum onto a perceptual mel axis'
        : 'Sum the harmonics at the current phase to reconstruct x(t)',
      formula: view === 'mel'
        ? 'mel(f) = 2595·log₁₀(1 + f/700)'
        : 'x(t) = Σₖ aₖ·sin(2π·k·f·t)',
      variables: {
        'f': F0,
        'K': K,
        'active': activeHarmonics,
        'phase': +np.toFixed(2),
        'dominant k': dominant,
        'overshoot': overshoot,
      },
      result: view === 'mel'
        ? `mel axis · peak at k=${dominant} (${Math.round(hzToMel(dominant * F0 * 1000))} mel)`
        : `${activeHarmonics} harmonic${activeHarmonics === 1 ? '' : 's'} · peak at k=${dominant}`,
      mathDetails: {
        params: [
          { label: 'harmonics', info: `${activeHarmonics} non-zero partials. Each aₖ is one bar of the amplitude spectrum on the right.` },
          { label: 'duality', info: 'The same signal is fully described by the waveform (time) OR the bars (frequency) — that is the Fourier duality.' },
          view === 'mel'
            ? { label: 'mel warp', info: 'The frequency axis is compressed logarithmically: a step from 1→2 kHz spans more mel than 7→8 kHz, matching how the cochlea resolves pitch.' }
            : { label: 'phase', info: 'Run sweeps the time offset; the waveform slides but the spectrum (|aₖ|) is unchanged.' },
          { label: 'overshoot', info: `Peak |x(t)| / |a|max ≈ ${overshoot}. Truncated series overshoot sharp edges (the Gibbs phenomenon) — visible as ripples near jumps.` },
        ],
        implication: activeHarmonics <= 1
          ? 'A single harmonic is a pure sine — the simplest possible spectrum.'
          : view === 'mel'
            ? 'On a mel axis the high harmonics crowd together — exactly the bands a log-mel front-end pools into one feature.'
            : 'More harmonics build sharper edges; an ideal square/saw needs infinitely many.',
      },
    });

    // Conceptual audio tutor: ONE spoken explanation per phase. The phase key
    // changes when the chosen voice (preset), the spectrum axis, or the active
    // harmonic count changes — so the tutor re-explains only when the concept on
    // screen actually changes, and stays quiet through the phase animation.
    const voice = preset === 'custom' ? 'this custom spectrum' : `the ${preset} wave`;
    const harmWord = `${activeHarmonics} harmonic${activeHarmonics === 1 ? '' : 's'}`;
    if (view === 'mel') {
      narration.narratePhase(
        `mel:${preset}:${activeHarmonics}`,
        `Now we warp the spectrum onto a mel axis. Mel of f equals 2595 times the log of one plus f over 700, ` +
        `which spaces frequencies the way the ear hears them. ${capitalise(voice)} keeps the same ${harmWord}, ` +
        `but watch the bars on the right crowd together at high frequency. That crowding is exactly what a log mel ` +
        `speech front end pools into a single feature.`,
      );
    } else {
      narration.narratePhase(
        `run:${preset}:${activeHarmonics}`,
        `Fourier synthesis builds a signal by adding sine waves. x of t equals the sum over k of a k times the sine of ` +
        `two pi k f t, so each slider is one harmonic, and harmonic k oscillates k times faster than the fundamental. ` +
        `${capitalise(voice)} sums ${harmWord}, with the strongest at k equals ${dominant}. Watch the waveform on the left ` +
        `and the amplitude bars on the right; they are the same signal seen in time and in frequency. As the run sweeps the ` +
        `phase the waveform slides but the bars never move, because phase does not change which frequencies are present.`,
      );
    }
  };
  const sim = useSimLoop(step, { initialSpeed: 60 });
  const reset = () => { sim.stop(); setPhase(0); setLastLog(null); narration.cancel(); };

  const insight = `${preset === 'custom' ? 'Custom spectrum' : preset} with ${activeHarmonics} active harmonic${activeHarmonics === 1 ? '' : 's'}` +
    `${view === 'mel' ? ', shown on a perceptual mel axis' : ''}. ` +
    'The waveform (left) and the amplitude bars (right) are two views of the SAME signal — time vs frequency. ' +
    'A square wave needs odd harmonics ∝ 1/k; a sawtooth needs all of them. Switching to the mel axis warps the bars to the perceptual spacing a log-mel ASR front-end uses. This decomposition is the foundation of every spectral audio feature.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'HARMONICS', value: activeHarmonics, color: ACCENT },
        { label: 'DOMINANT', value: `k=${dominant}` },
        { label: 'AXIS', value: view },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, fourierPython(amps, F0, preset, view))}
      narration={narration}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'center' }}>
          <FunctionPlot
            width={560} height={300} domain={[0, 1]} range={data.range}
            series={[
              ...(showComponents ? data.comps.map((c) => ({ points: c, color: COMP, width: 1 })) : []),
              { points: data.sum, color: ACCENT, width: 2.8 },
            ]}
            xLabel="time (one period)" yLabel="x(t)"
          />
          <div style={{ width: 360 }}>
            <MonoLabel style={{ marginBottom: 6 }}>
              {view === 'mel' ? 'Mel-warped spectrum (perceptual)' : 'Amplitude spectrum |aₖ|'}
            </MonoLabel>
            <DistributionBars
              bars={spectrumBars}
              width={360}
              accent={view === 'mel' ? MEL_CLR : ACCENT}
              max={Math.max(1, ...spectrumBars.map((b) => b.value))}
              valueFmt={(v) => v.toFixed(2)}
            />
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="WAVEFORM" items={[
          { color: ACCENT, label: 'sum x(t)' },
          { color: COMP, label: 'harmonics' },
          { color: MEL_CLR, label: 'mel bars' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Fourier Synthesis" hint="Build a periodic signal from harmonics." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Preset</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(['sine', 'square', 'sawtooth', 'triangle'] as const).map((p) => (
                <AlgoPill key={p} active={preset === p} accent={ACCENT} onClick={() => applyPreset(p)}>{p}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Try this · instrument voices</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(['pulse', 'organ', 'clarinet'] as const).map((p) => (
                <AlgoPill key={p} active={preset === p} accent={ACCENT} onClick={() => applyPreset(p)}>{p}</AlgoPill>
              ))}
            </div>
            {preset in EXTRA_BLURB && (
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>{EXTRA_BLURB[preset]}</div>
            )}
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Spectrum axis</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {(['linear', 'mel'] as const).map((v) => (
                <AlgoPill key={v} active={view === v} accent={view === 'mel' ? MEL_CLR : ACCENT} onClick={() => setSpectrumView(v)}>{v}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Harmonic amplitudes aₖ</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {amps.map((a, i) => (
                <ParamSlider
                  key={i}
                  name={`a${i + 1} · harmonic ${i + 1}`}
                  value={a.toFixed(2)}
                  min={-1} max={1} step={0.01} current={a}
                  accent={ACCENT}
                  onChange={(v) => setAmp(i, v)}
                  hint={`weight of k=${i + 1} (${i + 1}×f)`}
                />
              ))}
            </div>
          </div>
          <AlgoPill active={showComponents} accent={ACCENT} onClick={() => setShowComponents((s) => !s)}>
            {showComponents ? 'hide component sines' : 'show component sines'}
          </AlgoPill>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Run sweeps the phase to animate the waveform.</div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Fourier synthesis', preset, view, fundamental: F0, harmonics: activeHarmonics, dominant, amplitudes: amps.map((a) => +a.toFixed(3)) }}
      apiPanel={apiPanel}
    />
  );
};

export default FourierLab;
