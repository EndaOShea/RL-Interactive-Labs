import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, RunControls, Legend, MonoLabel, ParamSlider } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { forwardReversePython } from './python';

const ACCENT = '#f59e0b';
type Schedule = 'cosine' | 'linear';
type Dataset = 'two-moons' | 'ring' | 'blobs';

interface DPoint { x0: number; y0: number; ex: number; ey: number; cls: number; }

// 2-D data generators in centred coords ~[-1.4, 1.4]. Each point also carries a
// FIXED noise vector ε so scrubbing t replays the exact forward marginal.
function makeData(kind: Dataset, n: number): DPoint[] {
  const pts: DPoint[] = [];
  const push = (x: number, y: number, cls: number) =>
    pts.push({ x0: x, y0: y, ex: randn(), ey: randn(), cls });
  if (kind === 'two-moons') {
    const m = Math.floor(n / 2);
    for (let i = 0; i < m; i++) {
      const t = (i / (m - 1)) * Math.PI;
      push(Math.cos(t) - 0.5 + randn() * 0.06, Math.sin(t) - 0.25 + randn() * 0.06, 0);
      push(1 - Math.cos(t) - 0.5 + randn() * 0.06, 0.25 - Math.sin(t) + randn() * 0.06, 1);
    }
  } else if (kind === 'ring') {
    for (let i = 0; i < n; i++) {
      const t = (i / n) * 2 * Math.PI;
      const r = 1.05 + randn() * 0.05;
      push(r * Math.cos(t), r * Math.sin(t), t > Math.PI ? 0 : 1);
    }
  } else {
    const centers = [{ x: -0.85, y: -0.85 }, { x: 0.85, y: -0.85 }, { x: 0, y: 0.95 }];
    for (let i = 0; i < n; i++) {
      const c = i % 3;
      push(centers[c].x + randn() * 0.17, centers[c].y + randn() * 0.17, c);
    }
  }
  return pts;
}

// alpha-bar at step t in [0, T]: cumulative signal-retention in [0, 1].
function alphaBar(t: number, T: number, schedule: Schedule): number {
  if (t <= 0) return 1;
  if (t >= T) return schedule === 'cosine' ? 1e-4 : 1e-4;
  if (schedule === 'cosine') {
    const s = 0.008;
    const f = (u: number) => Math.cos(((u / T + s) / (1 + s)) * (Math.PI / 2)) ** 2;
    return Math.max(1e-4, f(t) / f(0));
  }
  // linear beta schedule -> cumulative product
  const b0 = 1e-4, b1 = 0.02;
  let prod = 1;
  for (let i = 1; i <= t; i++) {
    const beta = b0 + (b1 - b0) * (i / T);
    prod *= 1 - beta;
  }
  return Math.max(1e-4, prod);
}

const N = 500;

const ForwardReverseLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [schedule, setSchedule] = useState<Schedule>('cosine');
  const [dataset, setDataset] = useState<Dataset>('two-moons');
  const [T, setT] = useState(200);
  const [data, setData] = useState<DPoint[]>(() => makeData('two-moons', N));
  const [t, setTime] = useState(0);
  const [dir, setDir] = useState<1 | -1>(1); // forward (noising) / reverse (denoising)
  const [snrSeries, setSnrSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const ab = useMemo(() => alphaBar(t, T, schedule), [t, T, schedule]);
  const sqrtAb = Math.sqrt(ab);
  const sqrtOneMinus = Math.sqrt(1 - ab);
  const snr = ab / (1 - ab + 1e-8);

  // Current diffused cloud (exact marginal). Colour by original cluster so the
  // structure visibly dissolves and re-forms.
  const points = useMemo(
    () => data.map((p) => ({
      x: sqrtAb * p.x0 + sqrtOneMinus * p.ex,
      y: sqrtAb * p.y0 + sqrtOneMinus * p.ey,
      cls: p.cls,
      size: 3.6,
    })),
    [data, sqrtAb, sqrtOneMinus],
  );

  const log = (tt: number, d: 1 | -1) => {
    const a = alphaBar(tt, T, schedule);
    const s = a / (1 - a + 1e-8);
    setLastLog({
      algorithm: `${d === 1 ? 'Forward' : 'Reverse'} diffusion · ${schedule}`,
      stepDescription: d === 1
        ? 'Add noise toward the prior — signal dissolves into N(0, I)'
        : 'Remove noise toward the data — structure re-forms (here via stored x₀, ε)',
      formula: 'xₜ = √(ᾱₜ)·x₀ + √(1−ᾱₜ)·ε',
      variables: {
        't': tt,
        'ᾱ_t': +a.toFixed(4),
        '√(ᾱ_t)': +Math.sqrt(a).toFixed(3),
        '√(1−ᾱ_t)': +Math.sqrt(1 - a).toFixed(3),
        'SNR': +s.toFixed(3),
      },
      result: `t=${tt}/${T} · ᾱ=${a.toFixed(3)} · SNR=${s.toFixed(2)}`,
      mathDetails: {
        params: [
          { label: 'ᾱ_t', info: `${a.toFixed(3)}. Cumulative signal retained. Near 1 = mostly data, near 0 = mostly noise.` },
          { label: 'SNR', info: `${s.toFixed(2)} = ᾱ/(1−ᾱ). Signal vs noise power; crosses 1 mid-schedule (hardest to denoise).` },
          { label: d === 1 ? 'forward' : 'reverse', info: d === 1 ? 'A fixed, parameter-free Markov chain — no learning needed.' : 'A real model learns εθ(xₜ,t) to predict the noise; here we reuse the stored ε to show the exact marginal.' },
        ],
        implication: a > 0.6 ? 'Structure still clearly visible — early in the chain.'
          : a < 0.05 ? 'Essentially pure Gaussian noise — the prior x_T.'
            : 'Transition region — clusters blurring together around SNR≈1.',
      },
    });
  };

  const step = () => {
    setTime((cur) => {
      let nd = dir;
      let nt = cur + dir * Math.max(1, Math.round(T / 60));
      if (nt >= T) { nt = T; nd = -1; setDir(-1); }
      else if (nt <= 0) { nt = 0; nd = 1; setDir(1); }
      const a = alphaBar(nt, T, schedule);
      setSnrSeries((s) => [...s, a / (1 - a + 1e-8)].slice(-50));
      log(nt, nd);
      return nt;
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 90 });

  const regen = (d = dataset) => {
    setData(makeData(d, N)); setTime(0); setDir(1); setSnrSeries([]); setLastLog(null);
  };
  const reset = () => { sim.stop(); setTime(0); setDir(1); setSnrSeries([]); setLastLog(null); };

  const insight = `t=${t}/${T} on a ${schedule} schedule. ᾱ_t=${ab.toFixed(3)}, SNR=${snr.toFixed(2)}. ` +
    (ab > 0.6 ? 'The data structure is intact — early forward steps barely perturb it.'
      : ab < 0.05 ? 'The cloud has collapsed to the Gaussian prior x_T; all class structure is gone.'
        : 'Around the SNR≈1 crossover the clusters merge — this is where the learned denoiser does the hardest work.');

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 't', value: `${t}/${T}`, color: ACCENT },
        { label: 'ᾱ_t', value: ab.toFixed(3) },
        { label: 'SNR', value: snr.toFixed(2) },
        { label: 'DIR', value: dir === 1 ? 'forward' : 'reverse', color: dir === 1 ? '#f87171' : '#34d399' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, forwardReversePython(schedule, T, dataset))}
      grid={(
        <ScatterPlot
          points={points}
          domain={[-1.6, 1.6]}
          range={[-1.6, 1.6]}
          width={520}
          height={520}
          xLabel="x₁"
          yLabel="x₂"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Schedule</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 16 }}>
            <AlgoPill active={schedule === 'cosine'} accent={ACCENT} onClick={() => { setSchedule('cosine'); reset(); }}>cosine</AlgoPill>
            <AlgoPill active={schedule === 'linear'} accent={ACCENT} onClick={() => { setSchedule('linear'); reset(); }}>linear</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Dataset</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {(['two-moons', 'ring', 'blobs'] as Dataset[]).map((d) => (
              <AlgoPill key={d} active={dataset === d} accent={ACCENT} onClick={() => { setDataset(d); regen(d); }}>{d}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="ORIGINAL CLUSTER" items={[
          { color: '#f59e0b', label: 'Cluster 0' },
          { color: '#34d399', label: 'Cluster 1' },
          { color: '#fbbf24', label: 'Cluster 2' },
        ]} />
      )}
      rewardLabel="SNR = ᾱ/(1−ᾱ)"
      rewardValue={snr.toFixed(2)}
      rewardSeries={snrSeries}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Forward & Reverse" hint="Scrub t to noise then denoise via the exact marginal." />
          <ParamSlider name="T · diffusion steps" value={String(T)} min={40} max={500} step={20} current={T} onChange={(v) => { setT(v); reset(); }} hint="length of the noising chain" />
          <ParamSlider name="t · current step" value={`${t}`} min={0} max={T} step={1} current={t} onChange={(v) => { sim.stop(); setTime(v); log(v, dir); }} hint="drag to scrub forward/reverse" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="animation interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Forward/reverse diffusion', schedule, dataset, T, t, alphaBar: +ab.toFixed(4), snr: +snr.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default ForwardReverseLab;
