import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, type ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, RunControls, Legend, MonoLabel, ParamSlider } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { forwardReversePython } from './python';
import { DenoiseBar, PresetRow } from './viz';
import { useTheme } from '../../utils/theme';
import { buildSchedule, snrCrossing } from './schedules';
import {
  makeData, makeEps, makeMixture, ReverseRun, sampleMetrics, idealMetrics, meanNN,
  KDE_S, N_POINTS, type Data, type Dataset, type Sampler, type SampleMetrics,
} from './sampler';

const ACCENT = '#f59e0b';
type Sched = 'cosine' | 'linear';

// Forward view: every frame is the exact marginal xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε, drawn with
// ONE fixed ε per point (so frames are exact samples of q(xₜ), not one Markov path).
function forwardCloud(data: Data, eps: { ex: Float64Array; ey: Float64Array }, ab: number) {
  const sa = Math.sqrt(ab);
  const sn = Math.sqrt(1 - ab);
  const x = new Float64Array(data.n);
  const y = new Float64Array(data.n);
  for (let i = 0; i < data.n; i++) {
    x[i] = sa * (data.x[i] ?? 0) + sn * (eps.ex[i] ?? 0);
    y[i] = sa * (data.y[i] ?? 0) + sn * (eps.ey[i] ?? 0);
  }
  return { x, y };
}

const fmtNum = (v: number) => (!Number.isFinite(v) ? '∞' : v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4) ? v.toExponential(1) : v.toFixed(3));
const snrOf = (a: number) => a / (1 - a);

interface Preset { name: string; schedule: Sched; dataset: Dataset; T: number; sampler: Sampler; steps: number; guidance: number; }
const PRESETS: Preset[] = [
  { name: 'DDPM baseline', schedule: 'linear', dataset: 'two-moons', T: 200, sampler: 'ddpm', steps: 50, guidance: 0 },
  { name: 'Fast DDIM', schedule: 'cosine', dataset: 'ring', T: 300, sampler: 'ddim', steps: 20, guidance: 0 },
  { name: 'Guided blobs', schedule: 'cosine', dataset: 'blobs', T: 200, sampler: 'ddim', steps: 30, guidance: 3 },
  { name: 'Coarse 8-step', schedule: 'cosine', dataset: 'two-moons', T: 240, sampler: 'ddim', steps: 8, guidance: 0 },
];

const ForwardReverseLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [schedule, setSchedule] = useState<Sched>('cosine');
  const [dataset, setDataset] = useState<Dataset>('two-moons');
  const [sampler, setSampler] = useState<Sampler>('ddpm');
  const [steps, setSteps] = useState(50);          // DDIM step count S
  const [guidance, setGuidance] = useState(0);     // classifier-free guidance scale w
  const [T, setT] = useState(200);
  const [seed, setSeed] = useState(1);
  const [t, setTime] = useState(0);
  const [dir, setDir] = useState<1 | -1>(1);        // forward (noising) / reverse (sampling)
  const [series, setSeries] = useState<number[]>([]);
  const [preset, setPreset] = useState<string | undefined>();
  const [logOn, setLogOn] = useState(false);
  const narration = useNarration();

  const data = useMemo(() => makeData(dataset, N_POINTS, seed), [dataset, seed]);
  const eps = useMemo(() => makeEps(data.n, seed), [data.n, seed]);
  const mix = useMemo(() => makeMixture(data), [data]);
  const sched = useMemo(() => buildSchedule(schedule, T), [schedule, T]);
  const cross = useMemo(() => snrCrossing(sched), [sched]);
  const ideal = useMemo(() => idealMetrics(data, seed), [data, seed]);
  const S = Math.min(steps, T);
  // The reverse run: fresh x_T ~ N(0, I), then the chosen sampler driven by the exact
  // KDE denoiser. Frames are computed lazily (and cached) as the view reaches them.
  const run = useMemo(
    () => new ReverseRun({ mix, sched, sampler, steps: S, w: guidance, seed }),
    [mix, sched, sampler, S, guidance, seed],
  );
  const last = run.ts.length - 1;

  // Going forward any t is exact; going back the view sits on a step the sampler visits.
  const k = dir === -1 ? run.indexOf(t) : 0;
  const tShown = dir === -1 ? (run.ts[k] ?? 0) : Math.min(t, T);
  const ab = sched.abar[tShown] ?? 1;
  const snr = snrOf(ab);

  const cloud = useMemo(
    () => (dir === -1 ? run.frame(k) : forwardCloud(data, eps, ab)),
    [dir, run, k, data, eps, ab],
  );
  const metrics: SampleMetrics = useMemo(
    () => (dir === -1
      ? sampleMetrics(data, cloud.x, cloud.y)
      : { toData: meanNN(cloud.x, cloud.y, data.x, data.y), toSample: NaN }),
    [dir, data, cloud],
  );

  const points = useMemo(() => {
    const pts: ScatterPoint[] = [];
    if (dir === -1) {
      for (let i = 0; i < data.n; i++) pts.push({ x: data.x[i] ?? 0, y: data.y[i] ?? 0, cls: -1, size: 2.2, faint: true });
    }
    for (let i = 0; i < data.n; i++) pts.push({ x: cloud.x[i] ?? 0, y: cloud.y[i] ?? 0, cls: data.cls[i] ?? 0, size: 3.4 });
    return pts;
  }, [dir, data, cloud]);

  const crossTxt = cross === null ? 'no crossing (SNR stays above 1)' : `t≈${cross.toFixed(0)}`;
  const schedTxt = schedule === 'cosine'
    ? 'cosine: ᾱₜ = f(t)/f(0), f(t) = cos²((t/T+s)/(1+s)·π/2), s = 0.008, βₜ ≤ 0.999'
    : `linear: βₜ rises linearly from ${(0.1 / T).toExponential(1)} to ${(20 / T).toFixed(3)} (DDPM's 1e-4 → 0.02 scaled by 1000/T)`;
  const cfgF = guidance > 0 ? 'ε̂ = (1+w)·ε_cond − w·ε_uncond' : 'ε̂ = ε_cond (w = 0)';
  const sampTag = sampler === 'ddim' ? `DDIM·${S}` : `DDPM·${T}`;

  // Verdict on the final samples, from the metrics (never asserted).
  const gapRatio = Number.isFinite(metrics.toSample) ? metrics.toSample / ideal.toSample : NaN;
  const verdict = () => {
    const causes: string[] = [];
    if (guidance > 0) causes.push(`guidance w = ${guidance} pushes samples away from the other classes`);
    if (sampler === 'ddim' && S < 20) causes.push(`${S} DDIM steps are coarse — the last step jumps from t=${run.ts[last - 1] ?? T} straight to x̂₀, a posterior mean that averages across nearby data`);
    return gapRatio > 1.25
      ? `Coverage gap is ${gapRatio.toFixed(1)}× the exact-draw value: ${causes.join('; ') || 'the samples miss part of the data'}.`
      : `Coverage gap within ${Math.abs(100 * (gapRatio - 1)).toFixed(0)}% of exact KDE draws — the sampler reproduces the density closely.`;
  };

  // ---- Math panel (derived from the current state, so it is never stale) ----
  const buildLog = (): SimulationUpdate => {
    if (dir === 1) {
      return {
        algorithm: `Forward diffusion · ${schedule} · exact marginal`,
        stepDescription: 'Draw xₜ from q(xₜ | x₀) with each point\'s fixed ε — every frame is an exact sample of the marginal, not one Markov sample path',
        formula: 'xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε,  ᾱₜ = ∏ᵢ₌₁ᵗ (1−βᵢ)',
        variables: {
          t: tShown,
          'ᾱ_t': +ab.toPrecision(4),
          '√ᾱ_t': +Math.sqrt(ab).toPrecision(4),
          '√(1−ᾱ_t)': +Math.sqrt(1 - ab).toPrecision(4),
          SNR: fmtNum(snr),
          'xₜ→data': +metrics.toData.toFixed(4),
        },
        result: `t=${tShown}/${T} · ᾱ=${fmtNum(ab)} · SNR=${fmtNum(snr)}`,
        mathDetails: {
          params: [
            { label: 'ᾱ_t', info: `${fmtNum(ab)}. Cumulative signal retained (${schedTxt}). Near 1 = mostly data, near 0 = mostly noise.` },
            { label: 'SNR', info: `${fmtNum(snr)} = ᾱ/(1−ᾱ). On this schedule SNR = 1 at ${crossTxt} of ${T}.` },
            { label: 'ε', info: 'One N(0, I) draw per point, fixed for the whole run: scrubbing t re-draws xₜ from the same ε, so every frame is an exact sample of q(xₜ) (consecutive frames are not a Markov chain path).' },
            { label: 'xₜ→data', info: `${metrics.toData.toFixed(3)}: mean distance from each noised point to its nearest clean training point.` },
          ],
          implication: ab > 0.6 ? 'Structure still clearly visible — early in the chain.'
            : ab < 0.05 ? 'Essentially pure Gaussian noise — close to the prior N(0, I).'
              : 'Mid-chain: signal and noise powers are comparable (SNR = 1 where ᾱ = 0.5).',
        },
      };
    }
    const common = [
      { label: 'x̂₀ (denoiser)', info: `Exact E[x₀ | xₜ] for a Gaussian kernel density over the ${data.n} training points (bandwidth s = ${KDE_S}): component weights ∝ N(xₜ; √ᾱₜ·μᵢ, (ᾱₜs² + 1 − ᾱₜ)I). It is what a perfectly trained ε-network would output for that density — nothing is trained.` },
      { label: 'ε̂', info: `(xₜ − √ᾱₜ·x̂₀)/√(1−ᾱₜ). ε_cond uses only the components of the sample's own class, ε_uncond all ${data.n}. Each sample is conditioned on one class, in the data's class proportions.` },
      { label: 'w (cfg)', info: guidance > 0 ? `${guidance}. Extrapolates away from the unconditional prediction, which pushes samples away from the other classes: tighter class clusters and a larger coverage gap.` : '0. Plain conditional sampling (ε̂ = ε_cond): each sample is drawn from its own class\'s density.' },
      { label: 'off-data / gap', info: `${metrics.toData.toFixed(3)} / ${metrics.toSample.toFixed(3)}: mean distance from each sample to its nearest training point / from each training point to its nearest sample. Exact draws from the KDE score ${ideal.toData.toFixed(3)} / ${ideal.toSample.toFixed(3)}.` },
    ];
    const metricVars = { 'off-data': +metrics.toData.toFixed(4), gap: +metrics.toSample.toFixed(4) };
    if (k === 0) {
      return {
        algorithm: `Reverse · ${sampler.toUpperCase()} · start`,
        stepDescription: 'Fresh starting noise x_T ~ N(0, I), drawn independently of the forward pass',
        formula: `x_T ~ N(0, I);  x̂₀ = E[x₀ | xₜ] (exact KDE posterior mean);  ${cfgF}`,
        variables: { t: tShown, sampler: sampTag, w: guidance, ...metricVars },
        result: `t=${tShown}/${T} · fresh N(0, I) noise · off-data ${metrics.toData.toFixed(3)}`,
        mathDetails: { params: common, implication: 'Pure noise: nothing of the data is present yet. Every later frame comes from the sampler, not from the forward pass.' },
      };
    }
    const c = run.coeffs(k - 1);
    const implication = k < last
      ? `Samples at ᾱ = ${fmtNum(c.abarNext)}: still carrying noise; each step moves them toward the denoiser's estimate x̂₀.`
      : verdict();
    if (sampler === 'ddpm') {
      return {
        algorithm: `Reverse · DDPM (ancestral, σₜ² = β̃ₜ) · ${schedule}${guidance > 0 ? ` · CFG w=${guidance}` : ''}`,
        stepDescription: `Step t = ${c.t} → ${c.tNext}: the exact denoiser predicts ε̂; DDPM removes it and adds fresh noise σₜ·z`,
        formula: `xₜ₋₁ = (xₜ − βₜ/√(1−ᾱₜ)·ε̂)/√αₜ + σₜ·z,  σₜ² = (1−ᾱₜ₋₁)/(1−ᾱₜ)·βₜ;  ${cfgF}`,
        variables: {
          't → t−1': `${c.t} → ${c.tNext}`,
          'β_t': +c.beta.toPrecision(4),
          'α_t': +c.alpha.toPrecision(4),
          'ᾱ_t': +c.abar.toPrecision(4),
          'σ_t': +c.sigma.toPrecision(4),
          w: guidance,
          ...metricVars,
        },
        result: `t=${c.t}→${c.tNext} · σ=${c.sigma.toFixed(4)} · off-data ${metrics.toData.toFixed(3)} · gap ${metrics.toSample.toFixed(3)}`,
        mathDetails: {
          params: [
            ...common,
            { label: 'σ_t', info: `${c.sigma.toFixed(4)}. σₜ² = β̃ₜ = (1−ᾱₜ₋₁)/(1−ᾱₜ)·βₜ with a fresh seeded z each step; β̃₁ = 0, so the final step adds no noise. DDPM makes ${T} denoiser calls.` },
          ],
          implication,
        },
      };
    }
    return {
      algorithm: `Reverse · DDIM (η = 0) · ${S} steps · ${schedule}${guidance > 0 ? ` · CFG w=${guidance}` : ''}`,
      stepDescription: `Step t = ${c.t} → ${c.tNext} (stride ${c.t - c.tNext}): predict x̂₀, then move deterministically to the next visited step`,
      formula: `x̂₀ = (xₜ − √(1−ᾱₜ)·ε̂)/√ᾱₜ;  xₜ′ = √ᾱₜ′·x̂₀ + √(1−ᾱₜ′)·ε̂;  ${cfgF}`,
      variables: {
        't → t′': `${c.t} → ${c.tNext}`,
        'ᾱ_t': +c.abar.toPrecision(4),
        'ᾱ_t′': +c.abarNext.toPrecision(4),
        S,
        w: guidance,
        ...metricVars,
      },
      result: `t=${c.t}→${c.tNext} · off-data ${metrics.toData.toFixed(3)} · gap ${metrics.toSample.toFixed(3)}`,
      mathDetails: {
        params: [
          ...common,
          { label: 'DDIM', info: `η = 0: no noise is added, so each x_T maps to exactly one sample. ${S} steps at t = round(k·T/S), k = ${S}…0 — ${S} denoiser calls instead of ${T}.` },
        ],
        implication,
      },
    };
  };
  const lastLog = logOn ? buildLog() : null;

  // Conceptual audio-tutor narration: one explanation per phase, paraphrasing this
  // lab's own Context + live math, never a per-step play-by-play.
  const speak = (tt: number, d: 1 | -1, a: number, done: boolean, fm?: SampleMetrics) => {
    const samp = sampler === 'ddim' ? 'D D I M' : 'D D P M';
    const inTransition = a < 0.45 && a > 0.08;
    if (d === 1) {
      if (tt >= T) {
        narration.narratePhase('done:forward',
          'The cloud has dissolved into essentially a standard Gaussian. The forward process is a fixed, parameter-free noising chain, so any starting shape ends as noise. The reverse pass now starts from fresh Gaussian noise, not from this endpoint, and generates new samples.');
      } else if (inTransition) {
        narration.narratePhase('mid:forward',
          'Now near the signal-to-noise crossover, where alpha-bar is around a half and signal and noise have roughly equal power. The clusters blur into each other here.');
      } else {
        narration.narratePhase(`run:forward:${schedule}:${dataset}`,
          `The challenge here: turn structured data into noise in a way we can later reverse. The forward process has a closed-form marginal: x at step t is the square root of alpha-bar times the clean point, plus the square root of one minus alpha-bar times Gaussian noise. Each frame re-draws x t from every point's own fixed noise vector, so each frame is an exact sample of that marginal. On the ${schedule} schedule alpha-bar slides from one toward zero, and signal-to-noise crosses one ${cross === null ? 'nowhere in this chain' : `near step ${cross.toFixed(0)} of ${T}`}. Watch the ${dataset} structure smear into a featureless blob.`);
      }
      return;
    }
    if (done && fm) {
      const g = fm.toSample / ideal.toSample;
      narration.narratePhase(`done:reverse:${sampler}:${S}:${guidance}`,
        `Sampling complete with ${samp}. The coverage gap, the mean distance from each training point to its nearest sample, is ${fm.toSample.toFixed(3)}, against ${ideal.toSample.toFixed(3)} for exact draws from the density. ${g > 1.25 ? `That is ${g.toFixed(1)} times larger, so the samples miss part of the data${guidance > 0 ? ': guidance has pushed them away from the other classes' : sampler === 'ddim' && S < 20 ? ': with so few steps the last step jumps from a still-noisy step straight to the denoiser\'s estimate, an average over nearby data rather than a sample' : ''}.` : 'The sampler reproduces the density closely.'}`);
    } else if (inTransition) {
      narration.narratePhase('mid:reverse',
        'Mid-way back, near signal-to-noise one. The denoiser\'s posterior is still spread over many training points, so its estimate of the clean point is a blend; each step commits the samples a little more to one part of the data.');
    } else {
      const guideClause = guidance > 0
        ? ` Classifier-free guidance is on at scale ${guidance}: the prediction is extrapolated away from the unconditional one, which pushes each sample away from the other classes, giving tighter clusters and less coverage.`
        : ' Guidance is off, so each sample is drawn from its own class\'s density.';
      const sampClause = sampler === 'ddim'
        ? `${samp} is deterministic: it predicts the clean point, then jumps straight to the next of its ${S} visited steps without adding noise, so it needs ${S} denoiser calls instead of ${T}.`
        : `${samp} is ancestral: each of the ${T} steps removes the predicted noise and adds a little fresh noise, with none on the final step.`;
      narration.narratePhase(`run:reverse:${sampler}:${guidance}`,
        `The reverse pass starts from fresh Gaussian noise and generates new samples. The denoiser is exact, not learned: for each noisy point it computes the posterior mean of the clean point under a kernel density over the training data, which is what a perfectly trained noise-prediction network would output. ${sampClause}${guideClause} The grey dots are the training data, for reference.`);
    }
  };

  const fwdStride = Math.max(1, Math.round(T / 60));

  const step = () => {
    if (dir === 1) {
      if (t >= T) {
        // forward chain finished → the reverse pass starts from FRESH noise x_T ~ N(0, I)
        const fr = run.frame(0);
        setDir(-1); setTime(T); setLogOn(true);
        setSeries([meanNN(fr.x, fr.y, data.x, data.y)]);
        speak(T, -1, sched.abar[T] ?? 0, false);
        return;
      }
      const nt = Math.min(T, t + fwdStride);
      const a = sched.abar[nt] ?? 0;
      const fr = forwardCloud(data, eps, a);
      setSeries((s) => [...s, meanNN(fr.x, fr.y, data.x, data.y)].slice(-80));
      setTime(nt); setLogOn(true);
      speak(nt, 1, a, false);
      return;
    }
    if (k >= last) {
      // the reverse pass had finished (the loop paused there) → start a new forward pass
      setDir(1); setTime(0); setSeries([]); setLogOn(true);
      speak(0, 1, 1, false);
      return;
    }
    const nk = sampler === 'ddpm' ? Math.min(last, k + fwdStride) : k + 1;
    const fr = run.frame(nk);
    const nt = run.ts[nk] ?? 0;
    setSeries((s) => [...s, meanNN(fr.x, fr.y, data.x, data.y)].slice(-80));
    setTime(nt); setLogOn(true);
    const done = nk >= last;
    speak(nt, -1, sched.abar[nt] ?? 1, done, done ? sampleMetrics(data, fr.x, fr.y) : undefined);
    if (done) sim.stop();   // hold the final samples on screen
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const reset = () => { narration.cancel(); sim.stop(); setDir(1); setTime(0); setSeries([]); setLogOn(false); };
  // A setting change re-computes the current view in place (same direction and t), so
  // e.g. at t = 0 of the reverse pass you can flip DDPM/DDIM, S or w and compare results.
  const onParam = () => { narration.cancel(); setPreset(undefined); setSeries([]); };
  const goDir = (d: 1 | -1) => { narration.cancel(); sim.stop(); setDir(d); setTime(d === 1 ? 0 : T); setSeries([]); setLogOn(true); };
  const newSeed = () => { onParam(); setSeed((s) => s + 1); };

  const applyPreset = (name: string) => {
    const p = PRESETS.find((x) => x.name === name);
    if (!p) return;
    narration.cancel(); sim.stop();
    setSchedule(p.schedule); setSampler(p.sampler); setSteps(p.steps); setGuidance(p.guidance); setT(p.T);
    setDataset(p.dataset);
    setDir(1); setTime(0); setSeries([]); setLogOn(false); setPreset(name);
  };

  const insight = dir === 1
    ? `Forward, t=${tShown}/${T} on the ${schedule} schedule: ᾱ_t=${fmtNum(ab)}, SNR=${fmtNum(snr)} (SNR = 1 at ${crossTxt}). ` +
      (ab > 0.6 ? 'The data structure is intact — early forward steps barely perturb it.'
        : ab < 0.05 ? 'The cloud is close to the Gaussian prior; the class structure is gone.'
          : 'Mid-chain the clusters merge as signal and noise powers become comparable.') +
      ' Each frame is an exact marginal sample drawn with one fixed ε per point.'
    : `Reverse ${sampTag}${guidance > 0 ? ` with CFG w=${guidance}` : ''}, t=${tShown}/${T}: samples started from fresh N(0, I) noise and are driven by the exact KDE denoiser (s=${KDE_S}). ` +
      `Off-data ${metrics.toData.toFixed(3)}, coverage gap ${metrics.toSample.toFixed(3)} (exact KDE draws: ${ideal.toData.toFixed(3)} / ${ideal.toSample.toFixed(3)}).` +
      (k === last ? ` ${verdict()}` : '');

  const classItems = Array.from({ length: data.nClasses }, (_, c) => ({ color: CLASS_COLORS[c % CLASS_COLORS.length], label: `class ${c}` }));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 't', value: `${tShown}/${T}`, color: ACCENT },
        { label: 'ᾱ_t', value: fmtNum(ab) },
        { label: 'SNR', value: fmtNum(snr) },
        { label: 'SAMP', value: sampTag, color: sampler === 'ddim' ? '#38bdf8' : '#a78bfa' },
        { label: 'DIR', value: dir === 1 ? 'forward' : 'reverse', color: dir === 1 ? (isLight ? 'var(--bad)' : '#f87171') : (isLight ? 'var(--good)' : '#34d399') },
        ...(dir === -1 ? [
          { label: 'off-data', value: metrics.toData.toFixed(3) },
          { label: 'gap', value: metrics.toSample.toFixed(3), color: gapRatio > 1.25 ? (isLight ? 'var(--bad)' : '#f87171') : undefined },
        ] : []),
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, forwardReversePython(schedule, T, dataset, sampler, steps, guidance, seed))}
      grid={(
        <ScatterPlot
          points={points}
          domain={[-3, 3]}
          range={[-3, 3]}
          width={520}
          height={520}
          xLabel="x₁"
          yLabel="x₂"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Direction</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={dir === 1} accent={ACCENT} onClick={() => goDir(1)}>forward · q(xₜ|x₀)</AlgoPill>
            <AlgoPill active={dir === -1} accent={ACCENT} onClick={() => goDir(-1)}>reverse · sampler</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Schedule</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={schedule === 'cosine'} accent={ACCENT} onClick={() => { setSchedule('cosine'); onParam(); }}>cosine</AlgoPill>
            <AlgoPill active={schedule === 'linear'} accent={ACCENT} onClick={() => { setSchedule('linear'); onParam(); }}>linear</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Reverse sampler</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={sampler === 'ddpm'} accent={ACCENT} onClick={() => { setSampler('ddpm'); onParam(); }}>DDPM · stochastic</AlgoPill>
            <AlgoPill active={sampler === 'ddim'} accent={ACCENT} onClick={() => { setSampler('ddim'); onParam(); }}>DDIM · deterministic</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Dataset</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {(['two-moons', 'ring', 'blobs'] as Dataset[]).map((d) => (
              <AlgoPill key={d} active={dataset === d} accent={ACCENT} onClick={() => { setDataset(d); onParam(); }}>{d}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={newSeed} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'flex-end' }}>
          <DenoiseBar abar={ab} dir={dir} label={`${sampTag}${guidance > 0 ? ` · w=${guidance}` : ''} · seed ${seed}`} />
          <Legend
            title={dir === -1 ? 'COLOUR = CONDITIONING CLASS' : 'COLOUR = CLASS'}
            items={dir === -1 ? [...classItems, { color: 'var(--t2)', label: 'training data' }] : classItems}
          />
        </div>
      )}
      rewardLabel={dir === 1 ? 'xₜ → DATA (MEAN NN)' : 'SAMPLES → DATA'}
      rewardValue={metrics.toData.toFixed(3)}
      rewardSeries={series}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Forward & Reverse" hint="Noise the data, then sample new points from fresh noise." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try these</MonoLabel>
            <PresetRow presets={PRESETS} activeName={preset} accent={ACCENT} onPick={applyPreset} />
          </div>
          <ParamSlider name="T · diffusion steps" value={String(T)} min={40} max={500} step={20} current={T} onChange={(v) => { setT(v); setTime((cur) => Math.min(cur, v)); onParam(); }} hint="length of the chain (DDPM makes T denoiser calls)" />
          <ParamSlider
            name="t · current step" value={`${tShown}`} min={0} max={T} step={1} current={tShown}
            onChange={(v) => { sim.stop(); setTime(dir === -1 ? (run.ts[run.indexOf(v)] ?? v) : v); setSeries([]); setLogOn(true); }}
            hint={dir === 1 ? 'scrub the forward marginal' : sampler === 'ddim' ? `snaps to DDIM's ${S + 1} visited steps` : 'scrub the reverse chain'}
          />
          {sampler === 'ddim' && (
            <ParamSlider name="DDIM steps S" value={`${S}${steps > T ? ' (= T)' : ''}`} min={4} max={100} step={2} current={steps} onChange={(v) => { setSteps(v); onParam(); }} hint="fewer steps = bigger strides, coarser samples" />
          )}
          <ParamSlider name="w · CFG guidance" value={guidance.toFixed(1)} min={0} max={6} step={0.5} current={guidance} onChange={(v) => { setGuidance(v); onParam(); }} hint="0 = plain conditional; higher pushes classes apart" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="animation interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Forward/reverse diffusion', schedule, dataset, sampler, ddimSteps: S, guidance, T, t: tShown,
        direction: dir === 1 ? 'forward' : 'reverse', alphaBar: +ab.toPrecision(4), snr: fmtNum(snr), seed,
        denoiser: `exact KDE posterior mean, s=${KDE_S}`,
        offData: +metrics.toData.toFixed(4), coverageGap: Number.isFinite(metrics.toSample) ? +metrics.toSample.toFixed(4) : null,
        idealOffData: +ideal.toData.toFixed(4), idealCoverageGap: +ideal.toSample.toFixed(4),
      }}
      apiPanel={apiPanel}
    />
  );
};

export default ForwardReverseLab;
