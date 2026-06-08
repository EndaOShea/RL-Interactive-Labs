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
import { randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { distributionsPython, Family } from './python';

const ACCENT = '#c084fc';
const TRUE = '#c084fc';   // analytic curve
const EMP = '#34d399';    // empirical histogram

// ---- numeric helpers (no scipy) ----
function lgamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
const lfact = (n: number) => lgamma(n + 1);
const logChoose = (n: number, k: number) => lfact(n) - lfact(k) - lfact(n - k);

interface FamilyDef {
  label: string;
  kind: 'discrete' | 'continuous';
  formula: string;
  // parameter knobs to show (key, name, min, max, step)
  knobs: { key: string; name: string; min: number; max: number; step: number; hint: string }[];
  // discrete: support 0..kMax ; continuous: plotting domain
  support: (p: Record<string, number>) => [number, number];
  pmf?: (k: number, p: Record<string, number>) => number;
  pdf?: (x: number, p: Record<string, number>) => number;
  sample: (p: Record<string, number>) => number;
  mean: (p: Record<string, number>) => number;
  variance: (p: Record<string, number>) => number;
  entropy?: (p: Record<string, number>) => number;
  note: string;
}

const FAMILIES: Record<Family, FamilyDef> = {
  bernoulli: {
    label: 'Bernoulli', kind: 'discrete', formula: 'P(k)=pᵏ(1−p)¹⁻ᵏ',
    knobs: [{ key: 'p', name: 'p · success prob', min: 0.01, max: 0.99, step: 0.01, hint: 'P(success)' }],
    support: () => [0, 1],
    pmf: (k, p) => (k === 1 ? p.p : k === 0 ? 1 - p.p : 0),
    sample: (p) => (Math.random() < p.p ? 1 : 0),
    mean: (p) => p.p, variance: (p) => p.p * (1 - p.p),
    entropy: (p) => { const q = 1 - p.p; const h = (a: number) => (a > 0 ? -a * Math.log2(a) : 0); return h(p.p) + h(q); },
    note: 'A single yes/no trial — the atom every other count distribution is built from.',
  },
  binomial: {
    label: 'Binomial', kind: 'discrete', formula: 'P(k)=C(n,k)pᵏ(1−p)ⁿ⁻ᵏ',
    knobs: [
      { key: 'n', name: 'n · trials', min: 1, max: 40, step: 1, hint: 'number of trials' },
      { key: 'p', name: 'p · success prob', min: 0.01, max: 0.99, step: 0.01, hint: 'per-trial P(success)' },
    ],
    support: (p) => [0, p.n],
    pmf: (k, p) => (k < 0 || k > p.n ? 0 : Math.exp(logChoose(p.n, k) + k * Math.log(p.p) + (p.n - k) * Math.log(1 - p.p))),
    sample: (p) => { let s = 0; for (let i = 0; i < p.n; i++) if (Math.random() < p.p) s++; return s; },
    mean: (p) => p.n * p.p, variance: (p) => p.n * p.p * (1 - p.p),
    note: 'Successes in n independent Bernoulli trials — sum of n coin flips.',
  },
  poisson: {
    label: 'Poisson', kind: 'discrete', formula: 'P(k)=e⁻λ λᵏ/k!',
    knobs: [{ key: 'lam', name: 'λ · rate', min: 0.2, max: 18, step: 0.2, hint: 'expected count' }],
    support: (p) => [0, Math.max(8, Math.ceil(p.lam + 4 * Math.sqrt(p.lam)))],
    pmf: (k, p) => (k < 0 ? 0 : Math.exp(-p.lam + k * Math.log(p.lam) - lfact(k))),
    sample: (p) => { const L = Math.exp(-p.lam); let k = 0, prod = 1; do { k++; prod *= Math.random(); } while (prod > L); return k - 1; },
    mean: (p) => p.lam, variance: (p) => p.lam,
    note: 'Counts of rare events at rate λ — the n→∞, p→0 limit of the Binomial.',
  },
  geometric: {
    label: 'Geometric', kind: 'discrete', formula: 'P(k)=(1−p)ᵏ⁻¹p',
    knobs: [{ key: 'p', name: 'p · success prob', min: 0.05, max: 0.95, step: 0.01, hint: 'P(success) per trial' }],
    support: (p) => [1, Math.max(6, Math.ceil(Math.log(0.02) / Math.log(1 - p.p)))],
    pmf: (k, p) => (k < 1 ? 0 : Math.pow(1 - p.p, k - 1) * p.p),
    sample: (p) => Math.ceil(Math.log(1 - Math.random()) / Math.log(1 - p.p)),
    mean: (p) => 1 / p.p, variance: (p) => (1 - p.p) / (p.p * p.p),
    note: 'Trials until the first success — memoryless: past failures do not help.',
  },
  uniform: {
    label: 'Uniform', kind: 'continuous', formula: 'f(x)=1/(b−a)',
    knobs: [
      { key: 'a', name: 'a · lower', min: -3, max: 0, step: 0.1, hint: 'support start' },
      { key: 'b', name: 'b · upper', min: 0.5, max: 4, step: 0.1, hint: 'support end' },
    ],
    support: (p) => [p.a - 0.4 * (p.b - p.a), p.b + 0.4 * (p.b - p.a)],
    pdf: (x, p) => (x >= p.a && x <= p.b ? 1 / (p.b - p.a) : 0),
    sample: (p) => p.a + (p.b - p.a) * Math.random(),
    mean: (p) => (p.a + p.b) / 2, variance: (p) => (p.b - p.a) ** 2 / 12,
    entropy: (p) => Math.log(p.b - p.a),
    note: 'No preference on [a,b] — maximum entropy given a bounded support.',
  },
  normal: {
    label: 'Normal', kind: 'continuous', formula: 'f(x)=e^(−z²/2)/(σ√2π)',
    knobs: [
      { key: 'mu', name: 'μ · mean', min: -3, max: 3, step: 0.1, hint: 'centre' },
      { key: 'sigma', name: 'σ · std-dev', min: 0.2, max: 2.5, step: 0.05, hint: 'spread' },
    ],
    support: (p) => [p.mu - 4 * p.sigma, p.mu + 4 * p.sigma],
    pdf: (x, p) => { const z = (x - p.mu) / p.sigma; return Math.exp(-0.5 * z * z) / (p.sigma * Math.sqrt(2 * Math.PI)); },
    sample: (p) => p.mu + p.sigma * randn(),
    mean: (p) => p.mu, variance: (p) => p.sigma * p.sigma,
    entropy: (p) => 0.5 * Math.log(2 * Math.PI * Math.E * p.sigma * p.sigma),
    note: 'The bell curve — the CLT attractor for sums of many small effects.',
  },
  exponential: {
    label: 'Exponential', kind: 'continuous', formula: 'f(x)=λe^(−λx)',
    knobs: [{ key: 'lam', name: 'λ · rate', min: 0.3, max: 4, step: 0.05, hint: 'rate (mean=1/λ)' }],
    support: (p) => [0, Math.max(3, 6 / p.lam)],
    pdf: (x, p) => (x >= 0 ? p.lam * Math.exp(-p.lam * x) : 0),
    sample: (p) => -Math.log(1 - Math.random()) / p.lam,
    mean: (p) => 1 / p.lam, variance: (p) => 1 / (p.lam * p.lam),
    entropy: (p) => 1 - Math.log(p.lam),
    note: 'Waiting time between Poisson events — the continuous memoryless law.',
  },
  beta: {
    label: 'Beta', kind: 'continuous', formula: 'f(x)∝xᵃ⁻¹(1−x)ᵇ⁻¹',
    knobs: [
      { key: 'a', name: 'α · shape', min: 0.5, max: 8, step: 0.1, hint: 'pulls toward 1' },
      { key: 'b', name: 'β · shape', min: 0.5, max: 8, step: 0.1, hint: 'pulls toward 0' },
    ],
    support: () => [0, 1],
    pdf: (x, p) => {
      if (x <= 0 || x >= 1) return 0;
      const logB = lgamma(p.a) + lgamma(p.b) - lgamma(p.a + p.b);
      return Math.exp((p.a - 1) * Math.log(x) + (p.b - 1) * Math.log(1 - x) - logB);
    },
    sample: (p) => {
      // ratio of two Gamma draws (Marsaglia–Tsang) → Beta(a,b)
      const ga = gamma(p.a), gb = gamma(p.b);
      return ga / (ga + gb);
    },
    mean: (p) => p.a / (p.a + p.b),
    variance: (p) => (p.a * p.b) / ((p.a + p.b) ** 2 * (p.a + p.b + 1)),
    note: 'A flexible law over a probability in [0,1] — the Bernoulli’s conjugate prior.',
  },
};

// Marsaglia–Tsang Gamma sampler (shape k, scale 1) — for Beta draws.
function gamma(k: number): number {
  if (k < 1) { const u = Math.random(); return gamma(1 + k) * Math.pow(u, 1 / k); }
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x = randn(); let v = 1 + c * x; if (v <= 0) continue;
    v = v * v * v; const u = Math.random();
    if (u < 1 - 0.0331 * x ** 4) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

const DEFAULTS: Record<Family, Record<string, number>> = {
  bernoulli: { p: 0.4 },
  binomial: { n: 20, p: 0.4 },
  poisson: { lam: 4 },
  geometric: { p: 0.35 },
  uniform: { a: 0, b: 1 },
  normal: { mu: 0, sigma: 1 },
  exponential: { lam: 1 },
  beta: { a: 2, b: 5 },
};

const ORDER: Family[] = ['bernoulli', 'binomial', 'poisson', 'geometric', 'uniform', 'normal', 'exponential', 'beta'];
const N_BINS = 36;   // continuous histogram bins

const DistributionsLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [family, setFamily] = useState<Family>('binomial');
  const [params, setParams] = useState<Record<string, number>>({ ...DEFAULTS.binomial });
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // sampling accumulators
  const histRef = useRef<Map<number, number>>(new Map());   // discrete: value→count ; continuous: bin→count
  const [nSamples, setNSamples] = useState(0);
  const [, force] = useState(0);
  const errSeries = useRef<number[]>([]);
  const [errPlot, setErrPlot] = useState<number[]>([]);

  const def = FAMILIES[family];
  const support = def.support(params);
  const mean = def.mean(params);
  const variance = def.variance(params);
  const entropy = def.entropy ? def.entropy(params) : null;

  // analytic curve points
  const analytic = useMemo(() => {
    if (def.kind === 'discrete') {
      const [lo, hi] = support;
      const bars: { k: number; p: number }[] = [];
      for (let k = lo; k <= hi; k++) bars.push({ k, p: def.pmf!(k, params) });
      return { bars, pts: [] as { x: number; y: number }[], yMax: Math.max(...bars.map((b) => b.p), 1e-6) };
    }
    const [lo, hi] = support;
    const N = 201;
    const pts: { x: number; y: number }[] = [];
    let yMax = 0;
    for (let i = 0; i < N; i++) { const x = lo + (i / (N - 1)) * (hi - lo); const y = def.pdf!(x, params); pts.push({ x, y }); yMax = Math.max(yMax, y); }
    return { bars: [], pts, yMax: yMax * 1.15 || 1 };
  }, [family, params]);

  const resetSamples = () => {
    histRef.current = new Map(); setNSamples(0); errSeries.current = []; setErrPlot([]); setLastLog(null);
  };

  const intro = (f: Family) => {
    const d = FAMILIES[f];
    const kindLine = d.kind === 'discrete'
      ? 'This is a discrete family, so its probability mass function gives the actual probability of each integer outcome, shown as bars.'
      : 'This is a continuous family, so its probability density function gives a density that integrates to one, shown as a curve.';
    return `The challenge: turn a real-world mechanism into the right probability model, then trust it under uncertainty. ${d.note} ${kindLine} Its mean sits at ${d.mean(params).toFixed(2)} and its variance at ${d.variance(params).toFixed(2)}. Press Run to draw samples one batch at a time and watch the green empirical histogram climb toward the purple analytic shape — that convergence is the Law of Large Numbers, the bridge from a model on paper to data in the wild.`;
  };

  // batch of samples per step → accumulate histogram, measure total-variation-ish error to analytic.
  const step = () => {
    narration.narratePhase(`run:${family}`, intro(family));
    const BATCH = 40;
    const h = histRef.current;
    const [lo, hi] = support;
    for (let i = 0; i < BATCH; i++) {
      const x = def.sample(params);
      let key: number;
      if (def.kind === 'discrete') key = Math.round(x);
      else { const t = (x - lo) / (hi - lo); key = Math.max(0, Math.min(N_BINS - 1, Math.floor(t * N_BINS))); }
      h.set(key, (h.get(key) || 0) + 1);
    }
    const n = nSamples + BATCH;
    setNSamples(n);

    // L1 error between empirical and analytic (probability mass)
    let err = 0;
    if (def.kind === 'discrete') {
      for (let k = lo; k <= hi; k++) {
        const emp = (h.get(k) || 0) / n;
        err += Math.abs(emp - def.pmf!(k, params));
      }
    } else {
      const binW = (hi - lo) / N_BINS;
      for (let b = 0; b < N_BINS; b++) {
        const emp = (h.get(b) || 0) / n / binW;             // empirical density
        const xc = lo + (b + 0.5) * binW;
        err += Math.abs(emp - def.pdf!(xc, params)) * binW;  // ∫|emp−pdf|
      }
    }
    errSeries.current = [...errSeries.current, err].slice(-80);
    setErrPlot(errSeries.current);
    force((c) => c + 1);

    setLastLog({
      algorithm: `${def.label} · ${def.kind === 'discrete' ? 'PMF' : 'PDF'} sampling`,
      stepDescription: `Drew ${BATCH} samples (total ${n}) and updated the empirical histogram`,
      formula: def.formula,
      variables: {
        ...Object.fromEntries(def.knobs.map((kn) => [kn.key === 'lam' ? 'λ' : kn.key, params[kn.key]])),
        mean: +mean.toFixed(4),
        var: +variance.toFixed(4),
        ...(entropy != null ? { 'H (nats/bits)': +entropy.toFixed(3) } : {}),
        n, 'L1 err': +err.toFixed(4),
      },
      result: `n=${n} samples · empirical→analytic L1 error = ${err.toFixed(4)}`,
      mathDetails: {
        params: [
          { label: 'mean E[X]', info: 'Centre of mass of the distribution — the long-run average of samples.' },
          { label: 'variance', info: 'E[(X−μ)²], the spread; the empirical variance converges to it too.' },
          { label: 'LLN', info: 'As n grows the histogram converges to the true pmf/pdf and the L1 error → 0.' },
        ],
        implication: err < 0.05
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

  const setKnob = (key: string, v: number) => {
    setParams((p) => {
      const next = { ...p, [key]: v };
      if (key === 'a' && next.b <= v) next.b = v + 0.1;     // keep uniform a<b
      if (key === 'b' && v <= next.a) return p;
      return next;
    });
    if (!sim.isPlaying) resetSamples();
  };

  // ---- build viz ----
  const n = nSamples;
  let grid: React.ReactNode;
  let legend: React.ReactNode;
  if (def.kind === 'discrete') {
    const [lo, hi] = support;
    const h = histRef.current;
    // interleaved analytic/empirical pairs — render as a grouped list (k then its ↳ empirical)
    const interleaved: Bar[] = [];
    for (let k = lo; k <= hi; k++) {
      interleaved.push({ label: String(k), value: def.pmf!(k, params), color: TRUE });
      interleaved.push({ label: '↳', value: n > 0 ? (h.get(k) || 0) / n : 0, color: EMP, muted: n === 0 });
    }
    const yMax = Math.max(analytic.yMax, ...interleaved.map((b) => b.value), 0.05);
    grid = (
      <div style={{ width: 520, maxHeight: '74vh', overflowY: 'auto' }} className="custom-scrollbar">
        <DistributionBars bars={interleaved} width={500} max={yMax} accent={TRUE} valueFmt={(v) => v.toFixed(3)} rowH={20} />
      </div>
    );
    legend = <Legend title="PMF" items={[{ color: TRUE, label: 'analytic P(k)' }, { color: EMP, label: 'empirical (↳)' }]} />;
  } else {
    const [lo, hi] = support;
    const binW = (hi - lo) / N_BINS;
    const h = histRef.current;
    // empirical histogram as a step series (density), overlaid on the analytic pdf.
    const histPts: { x: number; y: number }[] = [];
    for (let b = 0; b < N_BINS; b++) {
      const dens = n > 0 ? (h.get(b) || 0) / n / binW : 0;
      const x0 = lo + b * binW, x1 = lo + (b + 1) * binW;
      histPts.push({ x: x0, y: dens });
      histPts.push({ x: x1, y: dens });
    }
    const yMax = Math.max(analytic.yMax, ...histPts.map((p) => p.y), 0.05) * 1.05;
    grid = (
      <FunctionPlot
        width={580} height={440} domain={[lo, hi]} range={[0, yMax]}
        series={[
          { points: histPts, color: EMP, width: 1.6, area: true },
          { points: analytic.pts, color: TRUE, width: 2.6 },
        ]}
        markers={[{ x: mean, y: def.pdf!(mean, params), color: '#fbbf24', label: `μ=${mean.toFixed(2)}` }]}
        xLabel="x" yLabel="density"
      />
    );
    legend = <Legend title="PDF" items={[{ color: TRUE, label: 'analytic f(x)' }, { color: EMP, label: 'empirical hist' }, { color: '#fbbf24', label: 'mean μ' }]} />;
  }

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'family', value: def.label, color: TRUE },
        { label: 'mean', value: mean.toFixed(3) },
        { label: 'var', value: variance.toFixed(3) },
        ...(entropy != null ? [{ label: 'H', value: entropy.toFixed(3) }] : []),
        { label: 'n', value: n },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, distributionsPython(family))}
      grid={grid}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={legend}
      rewardLabel="L1 error"
      rewardValue={errPlot.length ? errPlot[errPlot.length - 1].toFixed(4) : '—'}
      rewardSeries={errPlot.map((e) => -e)}
      lastLog={lastLog}
      contextInsight={`${def.note} ${def.kind === 'discrete' ? 'PMF' : 'PDF'} ${def.formula}. Mean=${mean.toFixed(3)}, variance=${variance.toFixed(3)}${entropy != null ? `, entropy=${entropy.toFixed(3)}` : ''}. Running draws samples; by the Law of Large Numbers the green empirical histogram converges to the purple analytic ${def.kind === 'discrete' ? 'mass' : 'density'} as n grows (L1 error → 0).`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Probability Distributions" hint="Common families · PMF/PDF, mean, variance" />
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
                  value={kn.step >= 1 ? String(params[kn.key]) : params[kn.key].toFixed(2)}
                  min={kn.min} max={kn.max} step={kn.step} current={params[kn.key]}
                  onChange={(v) => setKnob(kn.key, v)} hint={kn.hint} accent={TRUE}
                />
              ))}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="samples per tick = 40 / interval" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5 }}>
            Bernoulli→Binomial (sum of n), Binomial→Poisson (rare-event limit), Binomial/Poisson→Normal (CLT). Run to watch sampling converge to the curve.
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: `Probability distribution: ${def.label}`, family, ...params, mean: +mean.toFixed(4), variance: +variance.toFixed(4), entropy: entropy != null ? +entropy.toFixed(4) : null, nSamples: n }}
      apiPanel={apiPanel}
    />
  );
};

export default DistributionsLab;
