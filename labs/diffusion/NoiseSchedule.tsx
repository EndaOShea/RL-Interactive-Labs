import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotSeries } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, RunControls, Legend, MonoLabel, ParamSlider } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { noiseSchedulePython } from './python';

const ACCENT = '#f59e0b';      // cosine
const OTHER = '#38bdf8';       // linear
const SNR_COL = '#34d399';
const BETA_COL = '#f87171';
type Schedule = 'cosine' | 'linear';
type View = 'abar' | 'logsnr';

// β_t, α_t, ᾱ_t and SNR for a given t in [1, T].
function curves(T: number, schedule: Schedule) {
  const beta: number[] = [], alpha: number[] = [], abar: number[] = [], snr: number[] = [], logsnr: number[] = [];
  if (schedule === 'linear') {
    const b0 = 1e-4, b1 = 0.02;
    let prod = 1;
    for (let i = 1; i <= T; i++) {
      const b = b0 + (b1 - b0) * (i / T);
      const a = 1 - b;
      prod *= a;
      beta.push(b); alpha.push(a); abar.push(prod);
      const s = prod / (1 - prod + 1e-8);
      snr.push(s); logsnr.push(Math.log(s + 1e-12));
    }
  } else {
    const s0 = 0.008;
    const f = (u: number) => Math.cos(((u / T + s0) / (1 + s0)) * (Math.PI / 2)) ** 2;
    const f0 = f(0);
    let prev = 1;
    for (let i = 1; i <= T; i++) {
      const ab = Math.max(1e-5, f(i) / f0);
      const a = Math.min(1, ab / prev);
      prev = ab;
      beta.push(Math.min(0.999, 1 - a)); alpha.push(a); abar.push(ab);
      const s = ab / (1 - ab + 1e-8);
      snr.push(s); logsnr.push(Math.log(s + 1e-12));
    }
  }
  return { beta, alpha, abar, snr, logsnr };
}

const NoiseScheduleLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [schedule, setSchedule] = useState<Schedule>('cosine');
  const [view, setView] = useState<View>('abar');
  const [T, setT] = useState(200);
  const [marker, setMarker] = useState(0); // index into [1..T]
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const cos = useMemo(() => curves(T, 'cosine'), [T]);
  const lin = useMemo(() => curves(T, 'linear'), [T]);
  const active = schedule === 'cosine' ? cos : lin;

  const mt = Math.min(marker, T - 1);          // index
  const tStep = mt + 1;
  const abM = active.abar[mt];
  const snrM = active.snr[mt];

  const xs = (arr: number[]) => arr.map((y, i) => ({ x: i + 1, y }));

  const data = useMemo(() => {
    if (view === 'logsnr') {
      const all = [...cos.logsnr, ...lin.logsnr];
      const mn = Math.min(...all), mx = Math.max(...all);
      const pad = (mx - mn) * 0.08 || 1;
      const series: PlotSeries[] = [
        { points: xs(cos.logsnr), color: ACCENT, width: 2.6 },
        { points: xs(lin.logsnr), color: OTHER, width: 2.2, dash: true },
      ];
      return { series, range: [mn - pad, mx + pad] as [number, number] };
    }
    // ᾱ view: plot β_t, α_t, ᾱ_t for the ACTIVE schedule + SNR (squashed to [0,1] via ᾱ already shown)
    const series: PlotSeries[] = [
      { points: xs(active.abar), color: schedule === 'cosine' ? ACCENT : OTHER, width: 2.8, area: true },
      { points: xs(active.alpha), color: '#a78bfa', width: 1.8 },
      { points: xs(active.beta.map((b) => b * 25)), color: BETA_COL, width: 1.8, dash: true }, // β×25 to be visible
      { points: xs(active.snr.map((s) => s / (s + 1))), color: SNR_COL, width: 1.6 },          // SNR/(SNR+1) ∈ [0,1]
    ];
    return { series, range: [-0.05, 1.05] as [number, number] };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, schedule, cos, lin, active]);

  const log = (idx: number) => {
    const i = Math.min(idx, T - 1);
    const ab = active.abar[i];
    const s = active.snr[i];
    setLastLog({
      algorithm: `Noise schedule · ${schedule}`,
      stepDescription: 'Evaluate the schedule curves at the marker step',
      formula: schedule === 'cosine'
        ? 'ᾱₜ = cos²((t/T+s)/(1+s)·π/2)'
        : 'ᾱₜ = ∏ᵢ (1−βᵢ),  βᵢ linear',
      variables: {
        't': i + 1,
        'β_t': +active.beta[i].toFixed(5),
        'α_t': +active.alpha[i].toFixed(4),
        'ᾱ_t': +ab.toFixed(4),
        'SNR': +s.toFixed(3),
        'logSNR': +Math.log(s + 1e-12).toFixed(2),
      },
      result: `t=${i + 1}/${T} · ᾱ=${ab.toFixed(3)} · SNR=${s.toFixed(2)}`,
      mathDetails: {
        params: [
          { label: 'β_t', info: `${active.beta[i].toFixed(5)}. Per-step variance added by the forward process.` },
          { label: 'ᾱ_t', info: `${ab.toFixed(3)} = ∏(1−βᵢ). Cumulative signal retained up to step t.` },
          { label: 'SNR', info: `${s.toFixed(2)} = ᾱ/(1−ᾱ). Crosses 1 mid-schedule; cosine keeps it high for longer than linear.` },
        ],
        implication: schedule === 'cosine'
          ? 'Cosine removes information more evenly — fewer wasted steps near pure noise.'
          : 'Linear pushes ᾱ→0 early at high resolution, wasting late steps on already-noisy data.',
      },
    });
  };

  const step = () => {
    setMarker((m) => {
      const nm = m + 1 >= T ? 0 : m + 1;
      log(nm);
      return nm;
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 40 });
  const reset = () => { sim.stop(); setMarker(0); setLastLog(null); };

  const markerX = tStep;
  const markerY = view === 'logsnr' ? active.logsnr[mt] : active.abar[mt];

  const insight = `${schedule} schedule, viewing ${view === 'logsnr' ? 'log-SNR' : 'ᾱ_t / α_t / β_t / SNR'}. ` +
    `At t=${tStep}: ᾱ=${abM.toFixed(3)}, SNR=${snrM.toFixed(2)}. ` +
    'Cosine keeps ᾱ near 1 longer then drops smoothly, so the model spends steps where denoising is actually informative; linear destroys structure too early at high resolutions.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'SCHED', value: schedule, color: schedule === 'cosine' ? ACCENT : OTHER },
        { label: 'T', value: T },
        { label: 't', value: tStep },
        { label: 'ᾱ', value: abM.toFixed(3) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, noiseSchedulePython(schedule, T))}
      grid={(
        <FunctionPlot
          width={560}
          height={460}
          domain={[1, T]}
          range={data.range}
          series={data.series}
          markers={[{ x: markerX, y: markerY, color: schedule === 'cosine' ? ACCENT : OTHER, label: `t=${tStep}` }]}
          xLabel="t (diffusion step)"
          yLabel={view === 'logsnr' ? 'log SNR' : 'ᾱ, α, β×25, SNR/(SNR+1)'}
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Highlight schedule</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 16 }}>
            <AlgoPill active={schedule === 'cosine'} accent={ACCENT} onClick={() => { setSchedule('cosine'); log(marker); }}>cosine</AlgoPill>
            <AlgoPill active={schedule === 'linear'} accent={OTHER} onClick={() => { setSchedule('linear'); log(marker); }}>linear</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>View</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={view === 'abar'} accent={ACCENT} onClick={() => setView('abar')}>ᾱ / α / β / SNR</AlgoPill>
            <AlgoPill active={view === 'logsnr'} accent={ACCENT} onClick={() => setView('logsnr')}>log-SNR compare</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={view === 'logsnr' ? (
        <Legend title="log-SNR" items={[
          { color: ACCENT, label: 'cosine' },
          { color: OTHER, label: 'linear' },
        ]} />
      ) : (
        <Legend title="CURVES" items={[
          { color: schedule === 'cosine' ? ACCENT : OTHER, label: 'ᾱ_t (signal)' },
          { color: '#a78bfa', label: 'α_t = 1−β_t' },
          { color: BETA_COL, label: 'β_t ×25' },
          { color: SNR_COL, label: 'SNR/(SNR+1)' },
        ]} />
      )}
      rewardLabel="ᾱ AT MARKER"
      rewardValue={abM.toFixed(3)}
      rewardSeries={active.abar.filter((_, i) => i % Math.max(1, Math.floor(T / 50)) === 0)}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Noise Schedules" hint="Compare how linear vs cosine destroy information over t." />
          <ParamSlider name="T · total steps" value={String(T)} min={40} max={500} step={20} current={T} onChange={(v) => { setT(v); reset(); }} hint="length of the diffusion chain" />
          <ParamSlider name="t · marker" value={`${tStep}`} min={1} max={T} step={1} current={tStep} onChange={(v) => { sim.stop(); setMarker(v - 1); log(v - 1); }} hint="sweep the marker along t" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={10} max={200} step={10} current={sim.speed} onChange={sim.setSpeed} hint="marker sweep interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Diffusion noise schedules', schedule, view, T, t: tStep, alphaBar: +abM.toFixed(4), snr: +snrM.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default NoiseScheduleLab;
