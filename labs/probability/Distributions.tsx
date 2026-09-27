import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import DistributionBars, { Bar } from '../../components/labkit/viz/DistributionBars';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { distributionsPython } from './python';
import {
  FAMILIES, DEFAULTS, ORDER, BATCH, makeLayout, newAccum, addSample, drawValue, l1Error, empVar,
} from './families';
import type { Family, Params, Accum } from './families';

const ACCENT = '#c084fc';
const TRUE = '#c084fc';   // analytic / reference curve
const EMP = '#34d399';    // empirical histogram
const MEAN = '#fbbf24';   // analytic mean marker
const K_MAX = 50;

interface Run { sig: string; acc: Accum; err: number[]; }

const DistributionsLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [family, setFamily] = useState<Family>('binomial');
  const [params, setParams] = useState<Params>({ ...DEFAULTS.binomial });
  const [sumMode, setSumMode] = useState(false);
  const [kSum, setKSum] = useState(10);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [, force] = useState(0);

  const k = sumMode ? kSum : 1;
  const def = FAMILIES[family];
  const layout = useMemo(() => makeLayout(def, params, k), [def, params, k]);
  // Accumulators are tied to the exact (family, params, k) they were drawn under:
  // any parameter change starts a fresh run, even mid-play.
  const sig = `${family}|${JSON.stringify(params)}|${k}`;
  const runRef = useRef<Run>({ sig, acc: newAccum(layout.nBins), err: [] });
  const run = runRef.current.sig === sig ? runRef.current : null;
  const acc = run?.acc ?? null;
  const n = acc?.n ?? 0;

  const mean1 = def.mean(params), var1 = def.variance(params);
  const mean = k * mean1, variance = k * var1;          // of X (k = 1) or of the sum S_k
  const entropy = def.entropy(params);                   // of a single X, in nats
  const empMean = acc && n > 0 ? acc.mean : null;
  const empVariance = acc && n > 0 ? empVar(acc) : null;
  const err = run && run.err.length ? run.err[run.err.length - 1] ?? null : null;

  const resetSamples = () => {
    runRef.current = { sig: '', acc: newAccum(0), err: [] };
    setLastLog(null); force((c) => c + 1);
  };

  const hName = def.kind === 'discrete' ? 'entropy' : 'differential entropy';
  const intro = () => {
    const kindLine = def.kind === 'discrete'
      ? 'This is a discrete family, so its probability mass function gives the actual probability of each integer outcome, shown as bars.'
      : 'This is a continuous family, so its probability density function gives a density that integrates to one, shown as a curve.';
    if (k > 1) {
      return `The challenge: see why the bell curve is everywhere. Each draw here is the SUM of ${k} independent ${def.label} values, so its mean is ${k} times ${mean1.toFixed(2)} and its variance ${k} times ${var1.toFixed(2)}. The purple curve is the normal distribution with exactly that mean and variance — the Central Limit Theorem's prediction. Press Run and watch the green histogram of sums: for skewed families it still leans at small k, and the leftover gap to the bell shrinks as you raise k.`;
    }
    return `The challenge: turn a real-world mechanism into the right probability model, then trust it under uncertainty. ${def.note} ${kindLine} Its mean sits at ${mean1.toFixed(2)}, its variance at ${var1.toFixed(2)}, and its ${hName} at ${entropy.toFixed(2)} nats. Press Run to draw samples one batch at a time and watch the green empirical histogram, the sample mean and the sample variance close in on the analytic values — that convergence is the Law of Large Numbers, the bridge from a model on paper to data in the wild.`;
  };

  // one tick: BATCH draws of X (or of S_k), then the L1 distance to the reference
  const step = () => {
    narration.narratePhase(`run:${family}:${k > 1 ? 'sum' : 'single'}`, intro());
    let r = runRef.current;
    if (r.sig !== sig) { r = { sig, acc: newAccum(layout.nBins), err: [] }; runRef.current = r; }
    for (let i = 0; i < BATCH; i++) addSample(r.acc, layout, drawValue(def, params, k));
    const e = l1Error(r.acc, layout);
    r.err = [...r.err, e].slice(-80);
    force((c) => c + 1);

    const nn = r.acc.n;
    const outPct = (100 * r.acc.out) / nn;
    setLastLog({
      algorithm: k > 1 ? `${def.label} · sum of ${k} draws (CLT)` : `${def.label} · ${def.kind === 'discrete' ? 'PMF' : 'PDF'} sampling`,
      stepDescription: `Drew ${BATCH} ${k > 1 ? `sums of ${k} draws` : 'samples'} (total ${nn}) and updated the empirical histogram`,
      formula: k > 1 ? 'Sₖ = X₁+…+Xₖ  ≈  N(kμ, kσ²)' : def.formula,
      variables: {
        ...Object.fromEntries(def.knobs.map((kn) => [kn.key === 'lam' ? 'λ' : kn.key === 'a' && family === 'beta' ? 'α' : kn.key === 'b' && family === 'beta' ? 'β' : kn.key, params[kn.key] ?? 0])),
        ...(k > 1 ? { k } : {}),
        mean: +mean.toFixed(4),
        'x̄ (sample)': +r.acc.mean.toFixed(4),
        var: +variance.toFixed(4),
        's² (sample)': +empVar(r.acc).toFixed(4),
        ...(k === 1 ? { 'H (nats)': +entropy.toFixed(4) } : {}),
        samples: nn,
        'out of window': `${r.acc.out} (${outPct.toFixed(2)}%)`,
        'L1 err': +e.toFixed(4),
      },
      result: k > 1
        ? `n=${nn} sums · L1 to N(${mean.toFixed(2)}, ${variance.toFixed(2)}) = ${e.toFixed(4)}`
        : `n=${nn} samples · x̄=${r.acc.mean.toFixed(3)} (μ=${mean.toFixed(3)}) · L1 = ${e.toFixed(4)}`,
      mathDetails: k > 1
        ? {
          params: [
            { label: 'mean & variance of a sum', info: `Independent draws add: E[Sₖ] = kμ = ${mean.toFixed(3)} and Var(Sₖ) = kσ² = ${variance.toFixed(3)} — exactly, for any k. The sample values are x̄ = ${r.acc.mean.toFixed(3)} and s² = ${empVar(r.acc).toFixed(3)}.` },
            { label: 'CLT', info: 'The shape of Sₖ approaches the normal N(kμ, kσ²) as k grows (bins of integer sums are compared with the normal over [j−½, j+½]).' },
            { label: 'L1 floor', info: 'The reference is the normal approximation, not the exact law of Sₖ, so the L1 levels off at the CLT gap for this k (plus sampling noise) instead of reaching 0 — raise k to shrink it.' },
          ],
          implication: e < 0.05
            ? `The histogram of sums is now within L1 ${e.toFixed(3)} of the bell — the CLT at work for k = ${k}.`
            : `Still visibly non-normal at k = ${k} (L1 ${e.toFixed(3)}): skewed families such as the Exponential and Geometric converge slowly — raise k.`,
        }
        : {
          params: [
            { label: 'mean E[X]', info: `Analytic μ = ${mean.toFixed(4)}; the sample mean x̄ = ${r.acc.mean.toFixed(4)} converges to it (Law of Large Numbers).` },
            { label: 'variance', info: `Analytic σ² = ${variance.toFixed(4)}; the sample variance s² = ${empVar(r.acc).toFixed(4)} converges to it too.` },
            { label: 'entropy', info: `${def.kind === 'discrete' ? 'H = −Σ p ln p' : 'h = −∫ f ln f'} = ${entropy.toFixed(4)} nats (natural log${def.kind === 'continuous' ? '; a differential entropy, which can be negative' : ''}).` },
            { label: 'LLN', info: 'L1 = Σ|empirical − exact bin probability| over the window, plus the out-of-window mass difference; it → 0 as n grows.' },
          ],
          implication: e < 0.05
            ? 'The empirical histogram now closely matches the analytic law — the LLN has kicked in.'
            : 'Still noisy: with few samples the histogram wobbles around the true shape. Keep running.',
        },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 120 });

  const reset = () => { sim.stop(); narration.cancel(); resetSamples(); };
  const switchFamily = (f: Family) => {
    sim.stop(); narration.cancel(); setFamily(f); setParams({ ...DEFAULTS[f] }); resetSamples();
  };
  // Any parameter change restarts the accumulators (running or not) — never mix
  // samples drawn under different parameters in one histogram.
  const setKnob = (key: string, v: number) => {
    setParams((p) => {
      const next = { ...p, [key]: v };
      if (family === 'uniform') {   // keep the Uniform's a < b (Beta's a, b are free shapes)
        if (key === 'a' && (next.b ?? 0) <= v) next.b = v + 0.1;
        if (key === 'b' && v <= (next.a ?? 0)) return p;
      }
      return next;
    });
    resetSamples();
  };
  const setSampling = (on: boolean) => { setSumMode(on); narration.cancel(); resetSamples(); };
  const setK = (v: number) => { setKSum(v); resetSamples(); };

  // ---- build viz ----
  const counts = acc?.counts ?? [];
  const out = acc?.out ?? 0;
  let grid: React.ReactNode;
  let legend: React.ReactNode;
  if (layout.type === 'integer' && k === 1) {
    // single discrete draws: interleaved analytic / empirical bars per k
    const interleaved: Bar[] = [];
    for (let b = 0; b < layout.nBins; b++) {
      interleaved.push({ label: String(layout.center(b)), value: layout.mass[b] ?? 0, color: TRUE });
      interleaved.push({ label: '↳', value: n > 0 ? (counts[b] ?? 0) / n : 0, color: EMP, muted: n === 0 });
    }
    if (layout.outMass > 1e-4) {
      interleaved.push({ label: `>${layout.hi}`, value: layout.outMass, color: TRUE });
      interleaved.push({ label: '↳', value: n > 0 ? out / n : 0, color: EMP, muted: n === 0 });
    }
    const yMax = Math.max(...interleaved.map((bar) => bar.value), 0.05);
    grid = (
      <div style={{ width: 520, maxHeight: '74vh', overflowY: 'auto' }} className="custom-scrollbar">
        <DistributionBars bars={interleaved} width={500} max={yMax} accent={TRUE} valueFmt={(v) => v.toFixed(3)} rowH={20} />
      </div>
    );
    legend = <Legend title="PMF" items={[{ color: TRUE, label: 'analytic P(k)' }, { color: EMP, label: 'empirical (↳)' }]} />;
  } else {
    // continuous draws, or sums of k draws: a density histogram over the reference curve
    const integer = layout.type === 'integer';
    const edge0 = integer ? layout.lo - 0.5 : layout.lo;
    const width = integer ? 1 : layout.binW;
    const histPts: { x: number; y: number }[] = [];
    for (let b = 0; b < layout.nBins; b++) {
      const dens = n > 0 ? (counts[b] ?? 0) / n / width : 0;
      histPts.push({ x: edge0 + b * width, y: dens });
      histPts.push({ x: edge0 + (b + 1) * width, y: dens });
    }
    const x0 = edge0, x1 = edge0 + layout.nBins * width;
    const curve: { x: number; y: number }[] = [];
    let cMax = 0;
    for (let i = 0; i <= 240; i++) {
      const x = x0 + (i / 240) * (x1 - x0);
      const y = layout.density ? layout.density(x) : 0;
      curve.push({ x, y }); if (Number.isFinite(y)) cMax = Math.max(cMax, y);
    }
    const yMax = Math.max(cMax, ...histPts.map((p) => p.y), 0.05) * 1.08;
    const meanY = layout.density ? layout.density(mean) : 0;
    grid = (
      <FunctionPlot
        width={580} height={440} domain={[x0, x1]} range={[0, yMax]}
        series={[
          { points: histPts, color: EMP, width: 1.6, area: true },
          { points: curve, color: TRUE, width: 2.6 },
        ]}
        markers={[{ x: mean, y: meanY, color: MEAN, label: `${k > 1 ? 'kμ' : 'μ'}=${mean.toFixed(2)}` }]}
        xLabel={k > 1 ? `Sₖ = sum of ${k} draws` : 'x'} yLabel="density"
      />
    );
    legend = <Legend title={k > 1 ? 'CLT' : 'PDF'} items={[
      { color: TRUE, label: k > 1 ? `normal N(kμ, kσ²)` : 'analytic f(x)' },
      { color: EMP, label: k > 1 ? 'histogram of sums' : 'empirical hist' },
      { color: MEAN, label: k > 1 ? 'mean kμ' : 'mean μ' },
    ]} />;
  }
  const outNote = n > 0
    ? `${out} of ${n} draws (${((100 * out) / n).toFixed(2)}%) fell outside the plotted window — counted in n, not drawn; exact share ${(100 * layout.outMass).toFixed(2)}%.`
    : `Out-of-window draws are counted in n but not drawn (exact share ${(100 * layout.outMass).toFixed(2)}%).`;
  const stage = (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      {grid}
      <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', maxWidth: 560, textAlign: 'center', lineHeight: 1.5 }}>{outNote}</div>
    </div>
  );

  const errPlot = run?.err ?? [];
  const fmt = (v: number | null, d = 3) => (v == null ? '—' : v.toFixed(d));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'family', value: k > 1 ? `${def.label} Σ${k}` : def.label, color: TRUE },
        { label: 'mean', value: `${mean.toFixed(3)} · x̄ ${fmt(empMean)}` },
        { label: 'var', value: `${variance.toFixed(3)} · s² ${fmt(empVariance)}` },
        ...(k === 1 ? [{ label: 'H nats', value: entropy.toFixed(3) }] : []),
        { label: 'n', value: n },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, distributionsPython(family, params, k))}
      grid={stage}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={legend}
      rewardLabel={k > 1 ? 'L1 to normal' : 'L1 error'}
      rewardValue={err != null ? err.toFixed(4) : '—'}
      rewardSeries={errPlot}
      lastLog={lastLog}
      contextInsight={k > 1
        ? `Each draw is the sum of k=${k} independent ${def.label} values: mean kμ=${mean.toFixed(3)} (sample ${fmt(empMean)}), variance kσ²=${variance.toFixed(3)} (sample ${fmt(empVariance)}). The purple curve is the Central Limit Theorem's normal N(kμ, kσ²); the L1 distance to it${err != null ? ` (now ${err.toFixed(3)})` : ''} settles at the CLT gap for this k rather than at zero — raise k to watch the gap close.`
        : `${def.note} ${def.kind === 'discrete' ? 'PMF' : 'PDF'} ${def.formula}. Mean=${mean.toFixed(3)} (sample ${fmt(empMean)}), variance=${variance.toFixed(3)} (sample ${fmt(empVariance)}), ${hName}=${entropy.toFixed(3)} nats. Running draws samples; by the Law of Large Numbers the green empirical histogram converges to the purple analytic ${def.kind === 'discrete' ? 'mass' : 'density'} and the L1 distance to the exact bin probabilities → 0.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Probability Distributions" hint="Common families · PMF/PDF, mean, variance, entropy" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Family</MonoLabel>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 7 }}>
              {ORDER.map((f) => (
                <AlgoPill key={f} active={family === f} accent={ACCENT} onClick={() => switchFamily(f)}>{FAMILIES[f].label}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Parameters</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
              {def.knobs.map((kn) => (
                <ParamSlider
                  key={kn.key} name={kn.name}
                  value={kn.step >= 1 ? String(params[kn.key] ?? 0) : (params[kn.key] ?? 0).toFixed(2)}
                  min={kn.min} max={kn.max} step={kn.step} current={params[kn.key] ?? 0}
                  onChange={(v) => setKnob(kn.key, v)} hint={kn.hint} accent={TRUE}
                />
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>What each draw is</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              <AlgoPill active={!sumMode} accent={ACCENT} onClick={() => setSampling(false)}>single draw X</AlgoPill>
              <AlgoPill active={sumMode} accent={ACCENT} onClick={() => setSampling(true)}>sum of k draws (CLT)</AlgoPill>
            </div>
          </div>
          {sumMode && (
            <ParamSlider name="k · draws per sum" value={String(kSum)} min={2} max={K_MAX} step={1} current={kSum}
              onChange={setK} hint="Sₖ = X₁+…+Xₖ — compared with N(kμ, kσ²)" accent={TRUE} />
          )}
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint={`${BATCH} draws per tick`} accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5 }}>
            Bernoulli→Binomial (a sum of n), Binomial→Poisson (rare-event limit), sums→Normal (the CLT — pick “sum of k draws”). Entropy is in nats (natural log). Changing any parameter restarts the sample.
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: `Probability distribution: ${def.label}${k > 1 ? ` (sum of ${k} draws, CLT)` : ''}`, family, ...params, k, mean: +mean.toFixed(4), variance: +variance.toFixed(4), sampleMean: empMean != null ? +empMean.toFixed(4) : null, sampleVariance: empVariance != null ? +empVariance.toFixed(4) : null, entropyNats: +entropy.toFixed(4), nSamples: n, l1: err != null ? +err.toFixed(4) : null }}
      apiPanel={apiPanel}
    />
  );
};

export default DistributionsLab;
