import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, RunControls, Legend, MonoLabel, ParamSlider } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { forwardReversePython } from './python';
import { DenoiseBar, PresetRow } from './viz';

const ACCENT = '#f59e0b';
type Schedule = 'cosine' | 'linear';
type Dataset = 'two-moons' | 'ring' | 'blobs';
type Sampler = 'ddpm' | 'ddim';            // stochastic many-step vs deterministic few-step

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

// Per-frame t advance. DDIM strides in big jumps (few-step deterministic);
// DDPM walks the chain finely (many-step stochastic).
const stepStride = (sampler: Sampler, steps: number, T: number) =>
  sampler === 'ddim' ? Math.max(1, Math.round(T / steps)) : Math.max(1, Math.round(T / 60));

interface Preset { name: string; schedule: Schedule; dataset: Dataset; T: number; sampler: Sampler; steps: number; guidance: number; }
const PRESETS: Preset[] = [
  { name: 'DDPM baseline', schedule: 'linear', dataset: 'two-moons', T: 200, sampler: 'ddpm', steps: 50, guidance: 0 },
  { name: 'Fast DDIM', schedule: 'cosine', dataset: 'ring', T: 300, sampler: 'ddim', steps: 20, guidance: 0 },
  { name: 'Guided blobs', schedule: 'cosine', dataset: 'blobs', T: 200, sampler: 'ddim', steps: 30, guidance: 3 },
  { name: 'Coarse 8-step', schedule: 'cosine', dataset: 'two-moons', T: 240, sampler: 'ddim', steps: 8, guidance: 1.5 },
];

const ForwardReverseLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [schedule, setSchedule] = useState<Schedule>('cosine');
  const [dataset, setDataset] = useState<Dataset>('two-moons');
  const [sampler, setSampler] = useState<Sampler>('ddpm');
  const [steps, setSteps] = useState(50);          // DDIM stride count
  const [guidance, setGuidance] = useState(0);     // classifier-free guidance scale w
  const [T, setT] = useState(200);
  const [data, setData] = useState<DPoint[]>(() => makeData('two-moons', N));
  const [t, setTime] = useState(0);
  const [dir, setDir] = useState<1 | -1>(1); // forward (noising) / reverse (denoising)
  const [snrSeries, setSnrSeries] = useState<number[]>([]);
  const [preset, setPreset] = useState<string | undefined>();
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const ab = useMemo(() => alphaBar(t, T, schedule), [t, T, schedule]);
  const sqrtAb = Math.sqrt(ab);
  const sqrtOneMinus = Math.sqrt(1 - ab);
  const snr = ab / (1 - ab + 1e-8);

  // Classifier-free guidance: on the reverse pass, sharpen each point toward its
  // class centroid in proportion to the guidance scale and how much signal has
  // re-formed. Analytic stand-in for w·(εθ_cond − εθ_uncond).
  const centroids = useMemo(() => {
    const acc: Record<number, { x: number; y: number; n: number }> = {};
    data.forEach((p) => {
      const c = acc[p.cls] || (acc[p.cls] = { x: 0, y: 0, n: 0 });
      c.x += p.x0; c.y += p.y0; c.n += 1;
    });
    Object.values(acc).forEach((c) => { c.x /= c.n; c.y /= c.n; });
    return acc;
  }, [data]);

  // Current diffused cloud (exact marginal). Colour by original cluster so the
  // structure visibly dissolves and re-forms. Guidance pulls reverse-pass points
  // toward their class centroid (only meaningful with w>0 while denoising).
  const points = useMemo(() => {
    const gPull = dir === -1 ? guidance * 0.04 * sqrtAb : 0;
    return data.map((p) => {
      const cx = sqrtAb * p.x0 + sqrtOneMinus * p.ex;
      const cy = sqrtAb * p.y0 + sqrtOneMinus * p.ey;
      const c = centroids[p.cls];
      return {
        x: cx + (c ? (c.x - cx) * gPull : 0),
        y: cy + (c ? (c.y - cy) * gPull : 0),
        cls: p.cls,
        size: 3.6,
      };
    });
  }, [data, sqrtAb, sqrtOneMinus, dir, guidance, centroids]);

  const log = (tt: number, d: 1 | -1) => {
    const a = alphaBar(tt, T, schedule);
    const s = a / (1 - a + 1e-8);
    const stride = stepStride(sampler, steps, T);
    setLastLog({
      algorithm: `${d === 1 ? 'Forward' : 'Reverse'} diffusion · ${schedule} · ${sampler.toUpperCase()}`,
      stepDescription: d === 1
        ? 'Add noise toward the prior — signal dissolves into N(0, I)'
        : sampler === 'ddim'
          ? `Deterministic DDIM step (stride ${stride}, w=${guidance}) — structure re-forms in few steps`
          : 'Stochastic DDPM step — structure re-forms via the learned/stored ε',
      formula: d === -1 && sampler === 'ddim'
        ? 'x₀̂ = (xₜ−√(1−ᾱₜ)·ε̂)/√ᾱₜ;  xₜ₋₁ = √ᾱₜ₋₁·x₀̂ + √(1−ᾱₜ₋₁)·ε̂'
        : 'xₜ = √(ᾱₜ)·x₀ + √(1−ᾱₜ)·ε',
      variables: {
        't': tt,
        'ᾱ_t': +a.toFixed(4),
        '√(ᾱ_t)': +Math.sqrt(a).toFixed(3),
        '√(1−ᾱ_t)': +Math.sqrt(1 - a).toFixed(3),
        'SNR': +s.toFixed(3),
        ...(d === -1 ? { 'sampler': sampler, 'w (cfg)': guidance, 'stride': stride } : {}),
      },
      result: `t=${tt}/${T} · ᾱ=${a.toFixed(3)} · SNR=${s.toFixed(2)}` + (d === -1 ? ` · ${sampler}${guidance > 0 ? ` w=${guidance}` : ''}` : ''),
      mathDetails: {
        params: [
          { label: 'ᾱ_t', info: `${a.toFixed(3)}. Cumulative signal retained. Near 1 = mostly data, near 0 = mostly noise.` },
          { label: 'SNR', info: `${s.toFixed(2)} = ᾱ/(1−ᾱ). Signal vs noise power; crosses 1 mid-schedule (hardest to denoise).` },
          d === 1
            ? { label: 'forward', info: 'A fixed, parameter-free Markov chain — no learning needed.' }
            : sampler === 'ddim'
              ? { label: 'DDIM', info: `Deterministic ODE-style step: predict x₀̂ then re-noise to t−1. Skips ${stride}× steps, so ${steps} steps ≈ ${T}-step DDPM quality.` }
              : { label: 'DDPM', info: 'Ancestral stochastic sampling: each step adds fresh noise σ_t·z. Needs many steps for sharp samples.' },
          ...(d === -1 && guidance > 0
            ? [{ label: 'w (cfg)', info: `${guidance}. Classifier-free guidance scale. ε̂ = (1+w)·ε_cond − w·ε_uncond pushes samples toward the class — sharper but less diverse.` }]
            : []),
        ],
        implication: a > 0.6 ? 'Structure still clearly visible — early in the chain.'
          : a < 0.05 ? 'Essentially pure Gaussian noise — the prior x_T.'
            : 'Transition region — clusters blurring together around SNR≈1.',
      },
    });
  };

  // Plain-English narration of the live event on the cloud.
  const speak = (tt: number, d: 1 | -1, a: number, s: number) => {
    if (d === 1) {
      if (tt >= T) narration.narrate('Cloud fully dissolved into Gaussian noise. Prior reached.', { interrupt: true });
      else narration.narrate(`Adding noise at step ${tt}. Signal ${(Math.sqrt(a) * 100).toFixed(0)} percent, S N R ${s.toFixed(1)}.`);
    } else {
      if (tt <= 0) {
        narration.narrate(`Sample fully denoised with ${sampler === 'ddim' ? 'DDIM' : 'DDPM'}. Structure restored.`, { interrupt: true });
      } else {
        const g = guidance > 0 ? `, guidance ${guidance}` : '';
        narration.narrate(`Removing noise at step ${tt}. Signal back to ${((1 - Math.sqrt(1 - a)) * 100).toFixed(0)} percent${g}.`);
      }
    }
  };

  const step = () => {
    setTime((cur) => {
      let nd = dir;
      const stride = stepStride(sampler, steps, T);
      let nt = cur + dir * stride;
      if (nt >= T) { nt = T; nd = -1; setDir(-1); }
      else if (nt <= 0) { nt = 0; nd = 1; setDir(1); }
      const a = alphaBar(nt, T, schedule);
      const s = a / (1 - a + 1e-8);
      setSnrSeries((sr) => [...sr, s].slice(-50));
      log(nt, nd);
      speak(nt, nd, a, s);
      return nt;
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 90 });

  const regen = (d = dataset) => {
    narration.cancel();
    setData(makeData(d, N)); setTime(0); setDir(1); setSnrSeries([]); setLastLog(null);
  };
  const reset = () => { narration.cancel(); sim.stop(); setTime(0); setDir(1); setSnrSeries([]); setLastLog(null); };

  const applyPreset = (name: string) => {
    const p = PRESETS.find((x) => x.name === name);
    if (!p) return;
    narration.cancel(); sim.stop();
    setSchedule(p.schedule); setSampler(p.sampler); setSteps(p.steps); setGuidance(p.guidance); setT(p.T);
    setDataset(p.dataset); setData(makeData(p.dataset, N));
    setTime(0); setDir(1); setSnrSeries([]); setLastLog(null); setPreset(name);
  };

  const noiseFrac = sqrtOneMinus; // ≈ fraction of std from noise

  const insight = `t=${t}/${T} on a ${schedule} schedule, reverse sampler ${sampler.toUpperCase()}${guidance > 0 ? ` (cfg w=${guidance})` : ''}. ᾱ_t=${ab.toFixed(3)}, SNR=${snr.toFixed(2)}. ` +
    (ab > 0.6 ? 'The data structure is intact — early forward steps barely perturb it.'
      : ab < 0.05 ? 'The cloud has collapsed to the Gaussian prior x_T; all class structure is gone.'
        : 'Around the SNR≈1 crossover the clusters merge — this is where the learned denoiser does the hardest work.') +
    (sampler === 'ddim' ? ` DDIM strides ${stepStride(sampler, steps, T)} chain-steps per move, so ~${steps} steps replace ${T}.` : ' DDPM walks the chain finely for maximum diversity.');

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 't', value: `${t}/${T}`, color: ACCENT },
        { label: 'ᾱ_t', value: ab.toFixed(3) },
        { label: 'SNR', value: snr.toFixed(2) },
        { label: 'SAMP', value: sampler.toUpperCase(), color: sampler === 'ddim' ? '#38bdf8' : '#a78bfa' },
        { label: 'DIR', value: dir === 1 ? 'forward' : 'reverse', color: dir === 1 ? '#f87171' : '#34d399' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, forwardReversePython(schedule, T, dataset, sampler, steps, guidance))}
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
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={schedule === 'cosine'} accent={ACCENT} onClick={() => { setSchedule('cosine'); setPreset(undefined); reset(); }}>cosine</AlgoPill>
            <AlgoPill active={schedule === 'linear'} accent={ACCENT} onClick={() => { setSchedule('linear'); setPreset(undefined); reset(); }}>linear</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Reverse sampler</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={sampler === 'ddpm'} accent={ACCENT} onClick={() => { setSampler('ddpm'); setPreset(undefined); reset(); }}>DDPM · stochastic</AlgoPill>
            <AlgoPill active={sampler === 'ddim'} accent={ACCENT} onClick={() => { setSampler('ddim'); setPreset(undefined); reset(); }}>DDIM · deterministic</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Dataset</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {(['two-moons', 'ring', 'blobs'] as Dataset[]).map((d) => (
              <AlgoPill key={d} active={dataset === d} accent={ACCENT} onClick={() => { setDataset(d); setPreset(undefined); regen(d); }}>{d}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'flex-end' }}>
          <DenoiseBar noiseFrac={noiseFrac} dir={dir} label={`${sampler.toUpperCase()}${sampler === 'ddim' ? ` · ${steps} steps` : ''}${guidance > 0 ? ` · w=${guidance}` : ''}`} />
          <Legend title="ORIGINAL CLUSTER" items={[
            { color: '#f59e0b', label: 'Cluster 0' },
            { color: '#34d399', label: 'Cluster 1' },
            { color: '#fbbf24', label: 'Cluster 2' },
          ]} />
        </div>
      )}
      rewardLabel="SNR = ᾱ/(1−ᾱ)"
      rewardValue={snr.toFixed(2)}
      rewardSeries={snrSeries}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Forward & Reverse" hint="Scrub t to noise then denoise; pick DDPM or DDIM." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try these</MonoLabel>
            <PresetRow presets={PRESETS} activeName={preset} accent={ACCENT} onPick={applyPreset} />
          </div>
          <ParamSlider name="T · diffusion steps" value={String(T)} min={40} max={500} step={20} current={T} onChange={(v) => { setT(v); setPreset(undefined); reset(); }} hint="length of the noising chain" />
          <ParamSlider name="t · current step" value={`${t}`} min={0} max={T} step={1} current={t} onChange={(v) => { sim.stop(); setTime(v); log(v, dir); }} hint="drag to scrub forward/reverse" />
          {sampler === 'ddim' && (
            <ParamSlider name="DDIM steps" value={`${steps}`} min={4} max={100} step={2} current={steps} onChange={(v) => { setSteps(v); setPreset(undefined); reset(); }} hint="fewer steps = bigger strides" />
          )}
          <ParamSlider name="w · CFG guidance" value={guidance.toFixed(1)} min={0} max={6} step={0.5} current={guidance} onChange={(v) => { setGuidance(v); setPreset(undefined); }} hint="0 = unguided; higher = sharper, less diverse" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="animation interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Forward/reverse diffusion', schedule, dataset, sampler, ddimSteps: steps, guidance, T, t, alphaBar: +ab.toFixed(4), snr: +snr.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default ForwardReverseLab;
