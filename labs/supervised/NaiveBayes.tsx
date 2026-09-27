import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint, ScatterMarker, ScatterEllipse, ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { clamp01, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { naiveBayesPython } from './python';
import { useTheme } from '../../utils/theme';
import { makeNbData, NB_CENTERS } from './supData';
import { NB_BINS, NbVariant, NbOpts, fitNb, posteriors, predictNb, accuracyNb, disagreement } from './nbCore';

const ACCENT = '#fbbf24';
const ALPHA_VALUES = [0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5];

interface Preset { id: string; name: string; variant: NbVariant; spread: number; unequal: boolean; imbalance: number; alpha: number; learnedPrior: boolean; tip: string; }
const PRESETS: Preset[] = [
  { id: 'default', name: 'Unequal shapes · 3× class 1', variant: 'gaussian', spread: 0.1, unequal: true, imbalance: 3, alpha: 1, learnedPrior: true,
    tip: 'Each class has its own variances (tight / wide / tall), so the borders curve (quadratic); class 1 has 3× the points, so its prior P(1) = 0.6 widens its region. Run, then open the Math tab: it reports how much of the plane each effect changes.' },
  { id: 'tight', name: 'Tight Gaussians', variant: 'gaussian', spread: 0.06, unequal: true, imbalance: 1, alpha: 1, learnedPrior: true,
    tip: 'Well-separated blobs: ≈ 99.8% train accuracy. The per-class variances still curve the borders — ≈ 17% of the plane is labelled differently from a shared-variance (straight-line) model.' },
  { id: 'equal', name: 'Equal shapes (≈ linear)', variant: 'gaussian', spread: 0.09, unequal: false, imbalance: 1, alpha: 1, learnedPrior: true,
    tip: 'All three classes share one round spread, so the quadratic terms nearly cancel: the borders are almost straight (only ≈ 4% of the plane differs from the shared-variance linear model) and equal counts make the priors equal.' },
  { id: 'overlap', name: 'Overlapping', variant: 'gaussian', spread: 0.14, unequal: true, imbalance: 1, alpha: 1, learnedPrior: true,
    tip: 'Heavy overlap: ≈ 88% train accuracy, and in the contested middle the top posterior falls toward a tie.' },
  { id: 'priors', name: 'Imbalanced priors (4×)', variant: 'gaussian', spread: 0.12, unequal: true, imbalance: 4, alpha: 1, learnedPrior: true,
    tip: 'Class 1 has 4× the points, so P(1) = 4/6. Switch Priors → Uniform: ≈ 9% of the plane changes class as the boundary retreats toward class 1 — and train accuracy drops ≈ 2 points.' },
  { id: 'multinomial', name: 'Multinomial · binned', variant: 'multinomial', spread: 0.09, unequal: true, imbalance: 1, alpha: 1, learnedPrior: true,
    tip: `Counts per ${NB_BINS} cells on each axis with Laplace α = 1: one decision per ${NB_BINS}×${NB_BINS} grid cell (dashed), so the regions are blocky, not elliptical.` },
  { id: 'high-smoothing', name: 'High smoothing · α = 4', variant: 'multinomial', spread: 0.09, unequal: true, imbalance: 1, alpha: 4, learnedPrior: true,
    tip: 'α = 4 floods every cell with pseudo-counts: posteriors flatten (mean top posterior over the plane ≈ 58% vs ≈ 71% at α = 1) and ≈ 6% of the plane changes class.' },
  { id: 'no-smoothing', name: 'No smoothing · α = 0.001', variant: 'multinomial', spread: 0.09, unequal: true, imbalance: 1, alpha: 0.001, learnedPrior: true,
    tip: 'α → 0: a cell a class never visited gives that class likelihood ≈ 0, so posteriors turn near-certain (mean top posterior over the plane ≈ 86% vs ≈ 71% at α = 1) and a single stray point can flip a whole cell.' },
];

const NaiveBayesLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [perClass, setPerClass] = useState(28);
  const [spread, setSpread] = useState(0.1);
  const [unequal, setUnequal] = useState(true);
  const [imbalance, setImbalance] = useState(3);
  const [seed, setSeed] = useState(1);
  const [variant, setVariant] = useState<NbVariant>('gaussian');
  const [alpha, setAlpha] = useState(1);
  const [learnedPrior, setLearnedPrior] = useState(true);
  const [activePreset, setActivePreset] = useState<string | null>('default');
  const [query, setQuery] = useState({ x: 0.5, y: 0.5 });
  const [confSeries, setConfSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const K = NB_CENTERS.length;
  const data = useMemo(() => makeNbData(perClass, spread, unequal, imbalance, seed), [perClass, spread, unequal, imbalance, seed]);
  const model = useMemo(() => fitNb(data, K), [data, K]);
  const opts: NbOpts = useMemo(() => ({ variant, alpha, learnedPrior }), [variant, alpha, learnedPrior]);
  const acc = useMemo(() => accuracyNb(model, opts, data), [model, opts, data]);
  // How much each ingredient matters on THIS data (fraction of the unit square whose label changes).
  const curveShare = useMemo(() => disagreement((x, y) => predictNb(model, { ...opts, variant: 'gaussian' }, x, y), (x, y) => predictNb(model, { ...opts, variant: 'gaussian', pooledVar: true }, x, y), 60), [model, opts]);
  const priorShare = useMemo(() => disagreement((x, y) => predictNb(model, { ...opts, learnedPrior: true }, x, y), (x, y) => predictNb(model, { ...opts, learnedPrior: false }, x, y), 60), [model, opts]);
  const priors = model.s.map((c) => (learnedPrior ? c.n / model.n : 1 / K));

  const step = () => {
    const nx = clamp01(query.x + (Math.random() - 0.5) * 0.14);
    const ny = clamp01(query.y + (Math.random() - 0.5) * 0.14);
    const post = posteriors(model, opts, nx, ny);
    const pred = post.indexOf(Math.max(...post));
    setQuery({ x: nx, y: ny });
    setConfSeries((s) => [...s, post[pred] ?? 0].slice(-60));
    const conf = Math.round((post[pred] ?? 0) * 100);

    if (variant === 'gaussian') {
      narration.narratePhase(
        `run:gaussian:${learnedPrior}`,
        `The challenge here: given a new point of unknown class, decide which of three overlapping clusters it most likely came from, and how sure to be. Gaussian naive bayes applies Bayes rule with one strong shortcut, that the two features are independent given the class, so the posterior of a class is its prior times a one dimensional Gaussian likelihood for each feature, normalised over the classes. Each ellipse is one class's fitted axis aligned Gaussian at plus or minus two sigma; because every class has its own variances, the boundaries are curved, quadratic rather than straight. ${learnedPrior ? 'The prior is each class\'s share of the training points, so the larger class claims more of the contested ground.' : 'The prior is uniform here, one third each, so only the likelihoods decide.'} Naive bayes powers spam filtering, sentiment analysis and medical screening.`,
      );
    } else {
      narration.narratePhase(
        `run:multinomial:${alpha}`,
        `The challenge here: classify a point among three clusters using only counts, while coping with cells where a class was never seen. Multinomial naive bayes cuts each axis into ${NB_BINS} cells and learns how often each class lands in each cell; the likelihood of a feature is the smoothed frequency, the count plus alpha over the class total plus alpha times ${NB_BINS}. That gives blocky decision regions that follow the grid. Alpha, now ${alpha}, decides how unseen cells are handled: large alpha smooths and blurs, alpha near zero makes an unseen cell almost rule a class out. This is the classic text classifier behind spam filtering and document categorisation from word counts.`,
      );
    }
    if (conf >= 90) narration.narratePhase(`conf:${variant}:high`, 'The query now sits firmly inside one class, so the winning posterior is very high. The naive independence assumption tends to make these probabilities over confident, so trust the ranking more than the exact number.');
    else if (conf < 50) narration.narratePhase(`conf:${variant}:tie`, 'Here the query lands where the class likelihoods overlap, so the top posterior barely beats the others. This contested region is exactly where the prior can tip the decision.');

    setLastLog({
      algorithm: variant === 'gaussian' ? 'Gaussian Naive Bayes' : 'Multinomial Naive Bayes',
      stepDescription: `Classify the query (${nx.toFixed(2)}, ${ny.toFixed(2)}) by its posterior P(class | x)`,
      formula: variant === 'gaussian'
        ? 'P(c|x) ∝ P(c) · 𝒩(x₁; μ_c1, σ²_c1) · 𝒩(x₂; μ_c2, σ²_c2)'
        : `P(c|x) ∝ P(c) · Π_f (count_{c,f}(bin) + α)/(N_c + ${NB_BINS}α)`,
      variables: { 'P(0|x)': +(post[0] ?? 0).toFixed(4), 'P(1|x)': +(post[1] ?? 0).toFixed(4), 'P(2|x)': +(post[2] ?? 0).toFixed(4), 'ŷ': pred, 'P(0), P(1), P(2)': priors.map((p) => p.toFixed(3)).join(', ') },
      result: `class ${pred} · ${conf}%`,
      mathDetails: {
        params: [
          { label: 'independence', info: 'Naive assumption: x₁ and x₂ are independent given the class, so the likelihood is a product of one-dimensional terms (log-space sum in code).' },
          variant === 'gaussian'
            ? { label: 'Gaussian', info: `Per class and feature, MLE mean and variance (floored at 1e-3); ellipses show ±2σ. Per-class variances make the log-posterior quadratic: ${(curveShare * 100).toFixed(1)}% of the plane is labelled differently from the same model with one shared (pooled) variance, whose borders would be straight.` }
            : { label: 'multinomial', info: `Each axis is cut into ${NB_BINS} equal cells on [0, 1]; likelihood = smoothed cell frequency, so the decision is constant on each of the ${NB_BINS}×${NB_BINS} grid cells.` },
          { label: 'prior', info: `${learnedPrior ? 'Learned from class frequencies' : 'Uniform (1/3 each)'}: P(c) = ${priors.map((p) => p.toFixed(3)).join(' / ')}. Learned vs uniform priors label ${(priorShare * 100).toFixed(1)}% of the plane differently.` },
          ...(variant === 'multinomial' ? [{ label: 'Laplace α', info: `${alpha}. Adds α pseudo-counts to every cell so an unseen cell is not exactly zero. Large α → flatter posteriors; α → 0 → brittle, over-confident.` }] : []),
        ],
        implication: (post[pred] ?? 0) > 0.8 ? 'Confident — the query sits firmly inside one class.' : 'Uncertain — the query lies where class likelihoods overlap, so the prior matters most here.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });
  const stopAll = () => { sim.stop(); narration.cancel(); };
  const clearRun = () => { setConfSeries([]); setLastLog(null); };
  const custom = () => setActivePreset(null);
  const change = (fn: () => void) => { stopAll(); fn(); custom(); clearRun(); };
  const reset = () => { stopAll(); setQuery({ x: 0.5, y: 0.5 }); clearRun(); };
  const applyPreset = (p: Preset) => {
    stopAll(); setVariant(p.variant); setSpread(p.spread); setUnequal(p.unequal); setImbalance(p.imbalance); setAlpha(p.alpha); setLearnedPrior(p.learnedPrior);
    setActivePreset(p.id); clearRun();
  };

  const points: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.cls }));
  const ellipses: ScatterEllipse[] = variant === 'gaussian'
    ? model.s.map((c, k) => ({ cx: c.mx, cy: c.my, rx: Math.sqrt(c.vx) * 2, ry: Math.sqrt(c.vy) * 2, angle: 0, color: CLASS_COLORS[k % CLASS_COLORS.length] }))
    : [];
  const gridCol = isLight ? 'rgba(30,40,70,.35)' : 'rgba(230,236,255,.3)';
  const binLines: ScatterLine[] = variant === 'multinomial'
    ? Array.from({ length: NB_BINS - 1 }, (_, k) => (k + 1) / NB_BINS).flatMap((v) => [
      { x1: v, y1: 0, x2: v, y2: 1, dash: true, width: 1, color: gridCol },
      { x1: 0, y1: v, x2: 1, y2: v, dash: true, width: 1, color: gridCol },
    ])
    : [];
  const markers: ScatterMarker[] = [{ x: query.x, y: query.y, color: isLight ? 'var(--t0)' : '#fff', r: 6 }];
  const post = posteriors(model, opts, query.x, query.y);
  const pred = post.indexOf(Math.max(...post));
  const fieldKey = `${variant}-${alpha}-${learnedPrior}-${seed}-${perClass}-${spread}-${unequal}-${imbalance}`;
  const tip = PRESETS.find((p) => p.id === activePreset)?.tip;
  const alphaIdx = Math.max(0, ALPHA_VALUES.indexOf(alpha));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'MODEL', value: variant === 'gaussian' ? 'GNB' : 'MNB', color: ACCENT },
        { label: 'PRED', value: pred, color: CLASS_COLORS[pred] },
        { label: 'P', value: `${((post[pred] ?? 0) * 100).toFixed(0)}%` },
        { label: 'ACC', value: `${(acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, naiveBayesPython({ data, variant, alpha, learnedPrior, query, spread, unequal, imbalance, seed }))}
      grid={(
        // 64 raster cells per axis = 8 per multinomial bin, so the blocky regions are drawn exactly.
        <ScatterPlot width={460} height={460} points={points} classify={(x, y) => predictNb(model, opts, x, y)} fieldKey={fieldKey} fieldResolution={64} ellipses={ellipses} lines={binLines} markers={markers} xLabel="x₁" yLabel="x₂" />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => change(() => setSeed((s) => s + 1))} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      legend={(
        <Legend title="NAIVE BAYES" items={[
          ...NB_CENTERS.map((_, k) => ({ color: CLASS_COLORS[k], label: `Class ${k} · P=${(priors[k] ?? 0).toFixed(2)}` })),
          ...(variant === 'gaussian'
            ? [{ node: <span style={{ width: 12, height: 8, borderRadius: 6, border: `1px solid ${isLight ? 'var(--t1)' : '#fff'}`, display: 'inline-block' }} />, label: '±2σ' }]
            : [{ node: <span style={{ width: 10, height: 10, border: `1px dashed ${isLight ? 'var(--t1)' : '#fff'}`, display: 'inline-block' }} />, label: `${NB_BINS}×${NB_BINS} bins` }]),
        ]} />
      )}
      rewardLabel="MAX POSTERIOR"
      rewardValue={`${((post[pred] ?? 0) * 100).toFixed(0)}%`}
      rewardSeries={confSeries}
      lastLog={lastLog}
      contextInsight={variant === 'gaussian'
        ? `Gaussian Naive Bayes fits one axis-aligned Gaussian per class (ellipses, ±2σ) and picks the largest posterior. ${unequal ? 'The classes have different variances, so the borders curve' : 'All classes share one spread, so the borders are nearly straight'} (${(curveShare * 100).toFixed(0)}% of the plane differs from a shared-variance model); priors ${learnedPrior ? 'come from the class counts' : 'are uniform'} (learned vs uniform changes ${(priorShare * 100).toFixed(0)}% of the plane).`
        : `Multinomial Naive Bayes bins each axis into ${NB_BINS} cells and models per-class cell frequencies with Laplace α = ${alpha}: one decision per grid cell (dashed). Priors ${learnedPrior ? 'come from the class counts' : 'are uniform'}.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Naive Bayes" hint="Press Run to roam the query point." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Likelihood model</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={variant === 'gaussian'} accent={ACCENT} onClick={() => change(() => setVariant('gaussian'))}>Gaussian</AlgoPill>
              <AlgoPill active={variant === 'multinomial'} accent={ACCENT} onClick={() => change(() => setVariant('multinomial'))}>Multinomial</AlgoPill>
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Priors P(c)</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={learnedPrior} accent={ACCENT} onClick={() => change(() => setLearnedPrior(true))}>Learned (counts)</AlgoPill>
              <AlgoPill active={!learnedPrior} accent={ACCENT} onClick={() => change(() => setLearnedPrior(false))}>Uniform</AlgoPill>
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Class shapes</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={!unequal} accent={ACCENT} onClick={() => change(() => setUnequal(false))}>Equal (round)</AlgoPill>
              <AlgoPill active={unequal} accent={ACCENT} onClick={() => change(() => setUnequal(true))}>Unequal</AlgoPill>
            </div>
          </div>
          {variant === 'multinomial' && (
            <ParamSlider name="Laplace α" value={String(alpha)} min={0} max={ALPHA_VALUES.length - 1} step={1} current={alphaIdx} onChange={(v) => change(() => setAlpha(ALPHA_VALUES[v] ?? 1))} hint="smoothing pseudo-counts per cell" />
          )}
          <ParamSlider name="Class-1 imbalance" value={`×${imbalance}`} min={1} max={4} step={0.5} current={imbalance} onChange={(v) => change(() => setImbalance(v))} hint="class 1 gets this many × the points of classes 0 and 2" />
          <ParamSlider name="Spread" value={spread.toFixed(2)} min={0.05} max={0.16} step={0.01} current={spread} onChange={(v) => change(() => setSpread(Math.round(v * 100) / 100))} hint="base σ (class overlap)" />
          <ParamSlider name="Points / class" value={String(perClass)} min={12} max={50} step={2} current={perClass} onChange={(v) => change(() => setPerClass(v))} hint="classes 0 and 2 (class 1 × imbalance)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={30} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="query-walk interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.id} active={activePreset === p.id} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {tip ?? 'Custom settings — press Run to roam the query and read its posterior.'}
            </p>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        algorithm: variant === 'gaussian' ? 'Gaussian Naive Bayes' : 'Multinomial Naive Bayes', classes: K, priors: priors.map((p) => +p.toFixed(3)), learnedPrior,
        unequalShapes: unequal, imbalance, spread, alpha: variant === 'multinomial' ? alpha : undefined, trainAcc: +acc.toFixed(3),
        curvedShare: +curveShare.toFixed(3), priorShiftShare: +priorShare.toFixed(3),
      }}
      apiPanel={apiPanel}
    />
  );
};

export default NaiveBayesLab;
