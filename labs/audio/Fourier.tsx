import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import DistributionBars from '../../components/labkit/viz/DistributionBars';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { fourierPython } from './python';

const ACCENT = '#fb923c';
const COMP = 'rgba(120,160,250,.45)';
const K = 5;            // number of harmonics (= number of amplitude sliders)
const F0 = 1;          // fundamental (one period across the x-axis)
const SAMPLES = 256;   // points over one period

type Preset = 'sine' | 'square' | 'sawtooth' | 'triangle' | 'custom';

// Amplitude presets for harmonics k = 1..K (truncated Fourier series).
const PRESETS: Record<Exclude<Preset, 'custom'>, number[]> = {
  sine: [1, 0, 0, 0, 0],
  square: [1, 0, 1 / 3, 0, 1 / 5],          // odd harmonics ∝ 1/k
  sawtooth: [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5], // all harmonics ∝ 1/k
  triangle: [1, 0, -1 / 9, 0, 1 / 25],       // odd harmonics ∝ 1/k²
};

const FourierLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [amps, setAmps] = useState<number[]>(PRESETS.square);
  const [preset, setPreset] = useState<Preset>('square');
  const [showComponents, setShowComponents] = useState(true);
  const [phase, setPhase] = useState(0); // animated time shift (fraction of a period)
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const setAmp = (i: number, v: number) => {
    setAmps((a) => { const n = [...a]; n[i] = v; return n; });
    setPreset('custom');
  };
  const applyPreset = (p: Exclude<Preset, 'custom'>) => { setAmps(PRESETS[p]); setPreset(p); setPhase(0); setLastLog(null); };

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
  const spectrumBars = amps.map((a, i) => ({
    label: `k=${i + 1}`,
    value: Math.abs(a),
    color: ACCENT,
    highlight: Math.abs(a) === Math.max(...amps.map(Math.abs)) && Math.abs(a) > 1e-4,
  }));

  const step = () => {
    const np = (phase + 0.02) % 1;
    setPhase(np);
    const dominant = amps.reduce((bi, a, i, arr) => (Math.abs(a) > Math.abs(arr[bi]) ? i : bi), 0) + 1;
    setLastLog({
      algorithm: `Fourier Series · ${preset}`,
      stepDescription: 'Sum the harmonics at the current phase to reconstruct x(t)',
      formula: 'x(t) = Σₖ aₖ·sin(2π·k·f·t)',
      variables: {
        'f': F0,
        'K': K,
        'active': activeHarmonics,
        'phase': +np.toFixed(2),
        'dominant k': dominant,
      },
      result: `${activeHarmonics} harmonic${activeHarmonics === 1 ? '' : 's'} · peak at k=${dominant}`,
      mathDetails: {
        params: [
          { label: 'harmonics', info: `${activeHarmonics} non-zero partials. Each aₖ is one bar of the amplitude spectrum on the right.` },
          { label: 'duality', info: 'The same signal is fully described by the waveform (time) OR the bars (frequency) — that is the Fourier duality.' },
          { label: 'phase', info: 'Run sweeps the time offset; the waveform slides but the spectrum (|aₖ|) is unchanged.' },
        ],
        implication: activeHarmonics <= 1
          ? 'A single harmonic is a pure sine — the simplest possible spectrum.'
          : 'More harmonics build sharper edges; an ideal square/saw needs infinitely many.',
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 60 });
  const reset = () => { sim.stop(); setPhase(0); setLastLog(null); };

  const insight = `${preset === 'custom' ? 'Custom spectrum' : preset} with ${activeHarmonics} active harmonic${activeHarmonics === 1 ? '' : 's'}. ` +
    'The waveform (left) and the amplitude bars (right) are two views of the SAME signal — time vs frequency. ' +
    'A square wave needs odd harmonics ∝ 1/k; a sawtooth needs all of them. This decomposition is the foundation of every spectral audio feature.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'HARMONICS', value: activeHarmonics, color: ACCENT },
        { label: 'FUNDAMENTAL', value: `${F0}f` },
        { label: 'PRESET', value: preset },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, fourierPython(amps, F0, preset))}
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
            <MonoLabel style={{ marginBottom: 6 }}>Amplitude spectrum |aₖ|</MonoLabel>
            <DistributionBars
              bars={spectrumBars}
              width={360}
              accent={ACCENT}
              max={Math.max(1, ...amps.map((a) => Math.abs(a)))}
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
      currentParams={{ topic: 'Fourier synthesis', preset, fundamental: F0, harmonics: activeHarmonics, amplitudes: amps.map((a) => +a.toFixed(3)) }}
      apiPanel={apiPanel}
    />
  );
};

export default FourierLab;
