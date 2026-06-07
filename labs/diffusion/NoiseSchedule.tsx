import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotSeries, PlotMarker } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, RunControls, Legend, MonoLabel, ParamSlider } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { noiseSchedulePython } from './python';
import { DenoiseBar, PresetRow } from './viz';

const ACCENT = '#f59e0b';      // cosine
const OTHER = '#38bdf8';       // linear
const SIG_COL = '#fb7185';     // sigmoid
const SNR_COL = '#34d399';
const BETA_COL = '#f87171';
type Schedule = 'cosine' | 'linear' | 'sigmoid';
type View = 'abar' | 'logsnr';

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

// β_t, α_t, ᾱ_t and SNR for a given t in [1, T].
// `shift` shifts log-SNR by 2·ln(shift) — the standard resolution-shift trick:
// larger shift keeps more signal (used for higher-resolution images).
function curves(T: number, schedule: Schedule, shift = 1) {
  const beta: number[] = [], alpha: number[] = [], abar: number[] = [], snr: number[] = [], logsnr: number[] = [];
  const ls = 2 * Math.log(Math.max(1e-3, shift));   // log-SNR shift
  const fromAbar = (raw: number[]) => {
    // apply log-SNR shift to a raw abar list, then rebuild β/α/SNR.
    let prev = 1;
    for (let i = 0; i < raw.length; i++) {
      let ab = raw[i];
      if (shift !== 1) {
        // shift in log-SNR space: SNR' = SNR·e^{ls}; ᾱ' = SNR'/(1+SNR')
        const s0 = ab / (1 - ab + 1e-8);
        const s1 = s0 * Math.exp(ls);
        ab = s1 / (1 + s1);
      }
      ab = Math.max(1e-5, Math.min(1 - 1e-7, ab));
      const a = Math.min(1, ab / prev);
      prev = ab;
      const s = ab / (1 - ab + 1e-8);
      beta.push(Math.min(0.999, 1 - a)); alpha.push(a); abar.push(ab);
      snr.push(s); logsnr.push(Math.log(s + 1e-12));
    }
  };

  if (schedule === 'linear') {
    const b0 = 1e-4, b1 = 0.02;
    let prod = 1; const raw: number[] = [];
    for (let i = 1; i <= T; i++) { prod *= 1 - (b0 + (b1 - b0) * (i / T)); raw.push(prod); }
    fromAbar(raw);
  } else if (schedule === 'sigmoid') {
    // EDM-style: ᾱ_t from a sigmoid ramp in (shifted) log-SNR over [start,end].
    const start = 3, end = -3; const raw: number[] = [];
    for (let i = 1; i <= T; i++) {
      const u = (i - 1) / (T - 1);
      const z = start + (end - start) * u;     // log-SNR sweeps high→low
      raw.push(sigmoid(z));                      // ᾱ = σ(logSNR) = SNR/(1+SNR)
    }
    fromAbar(raw);
  } else {
    const s0 = 0.008;
    const f = (u: number) => Math.cos(((u / T + s0) / (1 + s0)) * (Math.PI / 2)) ** 2;
    const f0 = f(0); const raw: number[] = [];
    for (let i = 1; i <= T; i++) raw.push(Math.max(1e-5, f(i) / f0));
    fromAbar(raw);
  }
  return { beta, alpha, abar, snr, logsnr };
}

const colorOf = (s: Schedule) => (s === 'cosine' ? ACCENT : s === 'linear' ? OTHER : SIG_COL);

interface Preset { name: string; schedule: Schedule; view: View; T: number; shift: number; }
const PRESETS: Preset[] = [
  { name: 'Cosine default', schedule: 'cosine', view: 'abar', T: 200, shift: 1 },
  { name: 'Linear waste', schedule: 'linear', view: 'logsnr', T: 200, shift: 1 },
  { name: 'EDM sigmoid', schedule: 'sigmoid', view: 'logsnr', T: 256, shift: 1 },
  { name: 'Hi-res shift', schedule: 'cosine', view: 'logsnr', T: 256, shift: 2 },
];

const NoiseScheduleLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [schedule, setSchedule] = useState<Schedule>('cosine');
  const [view, setView] = useState<View>('abar');
  const [T, setT] = useState(200);
  const [shift, setShift] = useState(1);
  const [marker, setMarker] = useState(0); // index into [1..T]
  const [preset, setPreset] = useState<string | undefined>();
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const cos = useMemo(() => curves(T, 'cosine', shift), [T, shift]);
  const lin = useMemo(() => curves(T, 'linear', shift), [T, shift]);
  const sig = useMemo(() => curves(T, 'sigmoid', shift), [T, shift]);
  const active = schedule === 'cosine' ? cos : schedule === 'linear' ? lin : sig;

  // First t where SNR crosses below 1 — the hardest-to-denoise crossover.
  const crossIdx = useMemo(() => {
    const i = active.snr.findIndex((s) => s < 1);
    return i < 0 ? active.snr.length - 1 : i;
  }, [active]);

  const mt = Math.min(marker, T - 1);          // index
  const tStep = mt + 1;
  const abM = active.abar[mt];
  const snrM = active.snr[mt];

  const xs = (arr: number[]) => arr.map((y, i) => ({ x: i + 1, y }));

  const data = useMemo(() => {
    if (view === 'logsnr') {
      const all = [...cos.logsnr, ...lin.logsnr, ...sig.logsnr];
      const mn = Math.min(...all), mx = Math.max(...all);
      const pad = (mx - mn) * 0.08 || 1;
      const series: PlotSeries[] = [
        { points: xs(cos.logsnr), color: ACCENT, width: 2.6 },
        { points: xs(lin.logsnr), color: OTHER, width: 2.2, dash: true },
        { points: xs(sig.logsnr), color: SIG_COL, width: 2.2 },
      ];
      return { series, range: [mn - pad, mx + pad] as [number, number] };
    }
    const series: PlotSeries[] = [
      { points: xs(active.abar), color: colorOf(schedule), width: 2.8, area: true },
      { points: xs(active.alpha), color: '#a78bfa', width: 1.8 },
      { points: xs(active.beta.map((b) => b * 25)), color: BETA_COL, width: 1.8, dash: true }, // β×25 to be visible
      { points: xs(active.snr.map((s) => s / (s + 1))), color: SNR_COL, width: 1.6 },          // SNR/(SNR+1) ∈ [0,1]
    ];
    return { series, range: [-0.05, 1.05] as [number, number] };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, schedule, cos, lin, sig, active]);

  const log = (idx: number) => {
    const i = Math.min(idx, T - 1);
    const ab = active.abar[i];
    const s = active.snr[i];
    setLastLog({
      algorithm: `Noise schedule · ${schedule}${shift !== 1 ? ` · shift ${shift}` : ''}`,
      stepDescription: 'Evaluate the schedule curves at the marker step',
      formula: schedule === 'cosine'
        ? 'ᾱₜ = cos²((t/T+s)/(1+s)·π/2)'
        : schedule === 'sigmoid'
          ? 'ᾱₜ = σ(logSNRₜ),  logSNR: +3→−3'
          : 'ᾱₜ = ∏ᵢ (1−βᵢ),  βᵢ linear',
      variables: {
        't': i + 1,
        'β_t': +active.beta[i].toFixed(5),
        'α_t': +active.alpha[i].toFixed(4),
        'ᾱ_t': +ab.toFixed(4),
        'SNR': +s.toFixed(3),
        'logSNR': +Math.log(s + 1e-12).toFixed(2),
        ...(shift !== 1 ? { 'shift': shift, 'Δlog SNR': +(2 * Math.log(shift)).toFixed(2) } : {}),
      },
      result: `t=${i + 1}/${T} · ᾱ=${ab.toFixed(3)} · SNR=${s.toFixed(2)}`,
      mathDetails: {
        params: [
          { label: 'β_t', info: `${active.beta[i].toFixed(5)}. Per-step variance added by the forward process.` },
          { label: 'ᾱ_t', info: `${ab.toFixed(3)} = ∏(1−βᵢ). Cumulative signal retained up to step t.` },
          { label: 'SNR', info: `${s.toFixed(2)} = ᾱ/(1−ᾱ). Crosses 1 at t≈${crossIdx + 1}; cosine keeps it high longer than linear, sigmoid is symmetric in log-SNR.` },
          ...(shift !== 1
            ? [{ label: 'shift', info: `${shift}×. Shifts log-SNR by ${(2 * Math.log(shift)).toFixed(2)} nats — the resolution-shift trick: bigger shift retains more signal for high-res images.` }]
            : []),
        ],
        implication: schedule === 'cosine'
          ? 'Cosine removes information more evenly — fewer wasted steps near pure noise.'
          : schedule === 'sigmoid'
            ? 'Sigmoid (EDM-style) is symmetric in log-SNR — concentrates steps around the SNR≈1 crossover where denoising matters most.'
            : 'Linear pushes ᾱ→0 early at high resolution, wasting late steps on already-noisy data.',
      },
    });
  };

  const speak = (idx: number) => {
    const i = Math.min(idx, T - 1);
    const ab = active.abar[i];
    const s = active.snr[i];
    if (i === crossIdx) {
      narration.narrate(`S N R crosses one at step ${i + 1}. Hardest denoising region.`, { interrupt: true });
    } else if (i + 1 >= T) {
      narration.narrate('Marker reached the end of the chain. Signal gone.', { interrupt: true });
    } else {
      narration.narrate(`Step ${i + 1}. Alpha bar ${ab.toFixed(2)}, log S N R ${Math.log(s + 1e-12).toFixed(1)}.`);
    }
  };

  const step = () => {
    setMarker((m) => {
      const nm = m + 1 >= T ? 0 : m + 1;
      log(nm);
      speak(nm);
      return nm;
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 40 });
  const reset = () => { narration.cancel(); sim.stop(); setMarker(0); setLastLog(null); };

  const applyPreset = (name: string) => {
    const p = PRESETS.find((x) => x.name === name);
    if (!p) return;
    narration.cancel(); sim.stop();
    setSchedule(p.schedule); setView(p.view); setT(p.T); setShift(p.shift);
    setMarker(0); setLastLog(null); setPreset(name);
  };

  const markerX = tStep;
  const markerY = view === 'logsnr' ? active.logsnr[mt] : active.abar[mt];
  const crossX = crossIdx + 1;
  const crossY = view === 'logsnr' ? active.logsnr[crossIdx] : active.abar[crossIdx];
  const markers: PlotMarker[] = [
    { x: markerX, y: markerY, color: colorOf(schedule), label: `t=${tStep}` },
    { x: crossX, y: crossY, color: SNR_COL, label: 'SNR=1' },
  ];

  const insight = `${schedule} schedule${shift !== 1 ? ` (log-SNR shift ${shift}×)` : ''}, viewing ${view === 'logsnr' ? 'log-SNR' : 'ᾱ_t / α_t / β_t / SNR'}. ` +
    `At t=${tStep}: ᾱ=${abM.toFixed(3)}, SNR=${snrM.toFixed(2)}; SNR crosses 1 near t=${crossX}. ` +
    'Cosine keeps ᾱ near 1 longer then drops smoothly; sigmoid (EDM) is symmetric in log-SNR and packs steps near the crossover; linear destroys structure too early at high resolutions.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'SCHED', value: schedule, color: colorOf(schedule) },
        { label: 'T', value: T },
        { label: 't', value: tStep },
        { label: 'ᾱ', value: abM.toFixed(3) },
        { label: 'SNR=1', value: `t≈${crossX}`, color: SNR_COL },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, noiseSchedulePython(schedule, T, shift))}
      grid={(
        <FunctionPlot
          width={560}
          height={460}
          domain={[1, T]}
          range={data.range}
          series={data.series}
          markers={markers}
          xLabel="t (diffusion step)"
          yLabel={view === 'logsnr' ? 'log SNR' : 'ᾱ, α, β×25, SNR/(SNR+1)'}
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Highlight schedule</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={schedule === 'cosine'} accent={ACCENT} onClick={() => { setSchedule('cosine'); setPreset(undefined); log(marker); }}>cosine</AlgoPill>
            <AlgoPill active={schedule === 'linear'} accent={OTHER} onClick={() => { setSchedule('linear'); setPreset(undefined); log(marker); }}>linear</AlgoPill>
            <AlgoPill active={schedule === 'sigmoid'} accent={SIG_COL} onClick={() => { setSchedule('sigmoid'); setPreset(undefined); log(marker); }}>sigmoid · EDM</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>View</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={view === 'abar'} accent={ACCENT} onClick={() => setView('abar')}>ᾱ / α / β / SNR</AlgoPill>
            <AlgoPill active={view === 'logsnr'} accent={ACCENT} onClick={() => setView('logsnr')}>log-SNR compare</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'flex-end' }}>
          <DenoiseBar noiseFrac={Math.sqrt(1 - abM)} dir={1} label={`t=${tStep} · ᾱ=${abM.toFixed(2)}`} />
          {view === 'logsnr' ? (
            <Legend title="log-SNR" items={[
              { color: ACCENT, label: 'cosine' },
              { color: OTHER, label: 'linear' },
              { color: SIG_COL, label: 'sigmoid' },
            ]} />
          ) : (
            <Legend title="CURVES" items={[
              { color: colorOf(schedule), label: 'ᾱ_t (signal)' },
              { color: '#a78bfa', label: 'α_t = 1−β_t' },
              { color: BETA_COL, label: 'β_t ×25' },
              { color: SNR_COL, label: 'SNR/(SNR+1)' },
            ]} />
          )}
        </div>
      )}
      rewardLabel="ᾱ AT MARKER"
      rewardValue={abM.toFixed(3)}
      rewardSeries={active.abar.filter((_, i) => i % Math.max(1, Math.floor(T / 50)) === 0)}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Noise Schedules" hint="Compare linear, cosine and sigmoid; shift the log-SNR." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try these</MonoLabel>
            <PresetRow presets={PRESETS} activeName={preset} accent={ACCENT} onPick={applyPreset} />
          </div>
          <ParamSlider name="T · total steps" value={String(T)} min={40} max={500} step={20} current={T} onChange={(v) => { setT(v); setPreset(undefined); reset(); }} hint="length of the diffusion chain" />
          <ParamSlider name="t · marker" value={`${tStep}`} min={1} max={T} step={1} current={tStep} onChange={(v) => { sim.stop(); setMarker(v - 1); log(v - 1); }} hint="sweep the marker along t" />
          <ParamSlider name="log-SNR shift ×" value={shift.toFixed(2)} min={0.25} max={4} step={0.25} current={shift} onChange={(v) => { setShift(v); setPreset(undefined); reset(); }} hint="resolution shift: >1 keeps more signal" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={10} max={200} step={10} current={sim.speed} onChange={sim.setSpeed} hint="marker sweep interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Diffusion noise schedules', schedule, view, T, shift, t: tStep, alphaBar: +abM.toFixed(4), snr: +snrM.toFixed(3), snrCrossT: crossX }}
      apiPanel={apiPanel}
    />
  );
};

export default NoiseScheduleLab;
