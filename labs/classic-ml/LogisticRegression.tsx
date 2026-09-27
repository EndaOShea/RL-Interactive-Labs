import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from './shared';
import { mulberry32 } from './rng';
import { seededTwoClass, twoClassBayesError, TWO_CLASS_SIGMA, TWO_CLASS_SPAN } from './datasets';
import type { LPt } from './datasets';
import { sigmoid, logregMetrics, logregStep, linearlySeparable } from './logregCore';
import { logregPython } from './python';
import { PresetChips, Preset } from './presets';
import { useTheme } from '../../utils/theme';

interface Cfg { alpha: number; separation: number; l2: number; }
// Presets use data seed 1; the claims were also checked on 300 other draws.
const PRESETS: Preset<Cfg>[] = [
  { id: 'easy', label: 'Well-separated', hint: 'Far-apart classes: every draw is linearly separable and the line classifies every point within a few dozen epochs.', values: { alpha: 0.6, separation: 0.9, l2: 0 } },
  { id: 'overlap', label: 'Overlapping', hint: 'The classes genuinely overlap (Bayes error ≈ 11.5%): no draw is linearly separable, so accuracy plateaus well below 100% (≈ 89% on average).', values: { alpha: 0.5, separation: 0.3, l2: 0 } },
  { id: 'blowup', label: 'Weight blow-up', hint: 'Separable data, no penalty: ‖w‖ keeps growing (≈ 6 → 10 → 14 over 100 / 500 / 2000 epochs) and the band keeps narrowing.', values: { alpha: 1.2, separation: 0.9, l2: 0 } },
  { id: 'reg', label: 'L2-regularised', hint: 'The same data with λ = 0.05: ‖w‖ settles at ≈ 2.3 — a finite, calmer boundary.', values: { alpha: 1.2, separation: 0.9, l2: 0.05 } },
];

const LogisticRegressionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [perClass, setPerClass] = useState(40);
  const [separation, setSeparation] = useState(0.6);
  const [seed, setSeed] = useState(1);
  const [alpha, setAlpha] = useState(0.5);
  const [l2, setL2] = useState(0);
  const [w1, setW1] = useState(0);
  const [w2, setW2] = useState(0);
  const [b, setB] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [accSeries, setAccSeries] = useState<number[]>([]);
  const [presetId, setPresetId] = useState<string | undefined>();
  const [milestone, setMilestone] = useState(false);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const data: LPt[] = useMemo(() => seededTwoClass(mulberry32(seed), perClass, separation), [seed, perClass, separation]);
  const separable = useMemo(() => linearlySeparable(data), [data]);
  const bayes = twoClassBayesError(separation);
  const metrics = useMemo(() => logregMetrics(data, w1, w2, b), [data, w1, w2, b]);
  const wNorm = Math.hypot(w1, w2);

  const step = () => {
    const m = data.length;
    if (!m) return;
    const st = logregStep(data, w1, w2, b, alpha, l2);
    setW1(st.w1); setW2(st.w2); setB(st.b);
    setEpoch((e) => e + 1);
    setAccSeries((a) => [...a, st.acc].slice(-60));

    const intro = l2 > 0
      ? `The challenge here: find the straight line that best separates the two classes and report a calibrated probability for each side. Logistic regression passes a weighted sum of the features through the sigmoid, then minimises the mean cross-entropy plus half lambda times the squared weights. On separable data that penalty matters: without it the weights run off to infinity, so the penalty caps the weight norm and keeps the probabilities sane. Watch the solid line, where the probability is one half, and the dashed quarter and three-quarter band. This is a staple of credit scoring, medical diagnosis and spam filtering.`
      : `The challenge here: find the straight line that best separates the two classes and report a probability for each side. Logistic regression passes a weighted sum of the features through the sigmoid — one over one plus e to the minus z — and trains by minimising the mean cross-entropy. Each step moves the weights along its gradient, the average of prediction minus label times the input. Watch the solid line, where the probability is one half, tilt to split the classes, with the dashed quarter and three-quarter contours as the confidence band. This is a staple of credit scoring, medical diagnosis and spam filtering.`;
    narration.narratePhase(`run:${l2 > 0}`, intro);
    if (st.correct === m && !milestone) {
      setMilestone(true);
      narration.narratePhase('done:separated', `Every training point is now on the correct side — 100 percent accuracy, so this draw is linearly separable. On separable data the cross-entropy has no finite minimum, so ${l2 > 0 ? 'the L2 penalty is what holds the weights, and the confidence band, in check' : 'the weight norm will keep growing and the confidence band keep narrowing unless you add an L2 penalty'}.`);
    } else if (st.correct < m && milestone) setMilestone(false);

    setLastLog({
      algorithm: `Logistic Regression · Cross-Entropy GD${l2 > 0 ? ' · L2' : ''}`,
      stepDescription: `Epoch ${epoch + 1} — gradient = mean((p − y)·x)${l2 > 0 ? ' + λw' : ''}`,
      formula: l2 > 0 ? 'p = σ(w·x + b),   w ← w − α(mean((p−y)x) + λw)' : 'p = σ(w·x + b),   w ← w − α·mean((p − y)·x)',
      variables: { 'w₁': +w1.toFixed(4), 'w₂': +w2.toFixed(4), 'b': +b.toFixed(4), 'BCE': +st.loss.toFixed(4), 'acc': +st.acc.toFixed(3) },
      result: `acc = ${(st.acc * 100).toFixed(1)}% (${st.correct}/${m})`,
      mathDetails: {
        params: [
          { label: 'α', info: `${alpha}. Step size for the weight update.` },
          { label: 'λ', info: l2 > 0 ? `${l2}. L2 penalty ½λ‖w‖² (bias not penalised) pulls the weights toward 0 each step, capping ‖w‖.` : 'No penalty — on separable data ‖w‖ can grow without bound.' },
          { label: '‖w‖', info: `${wNorm.toFixed(2)}. Weight norm; the sigmoid sharpens as it grows, so the p = 0.25–0.75 band narrows.` },
          { label: 'separable', info: separable ? 'Yes (exact test: the two classes\' convex hulls do not overlap) — 100% training accuracy is reachable.' : `No (the classes' convex hulls overlap) — no line classifies every point; the generator's Bayes error is ≈ ${(bayes * 100).toFixed(1)}%.` },
        ],
        implication: st.correct === m ? 'Every training point is on the correct side of the line.' : separable ? 'Still adjusting — a perfect line exists for this draw.' : 'The classes overlap — accuracy will plateau below 100%.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const zero = () => { setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setMilestone(false); setLastLog(null); narration.cancel(); };
  const regen = () => { sim.stop(); setSeed((s) => s + 1); zero(); };
  const reset = () => { sim.stop(); zero(); };
  const applyPreset = (p: Preset<Cfg>) => {
    sim.stop();
    setAlpha(p.values.alpha); setSeparation(p.values.separation); setL2(p.values.l2); setSeed(1);
    zero(); setPresetId(p.id);
    narration.narratePhase(`preset:${p.id}`, p.hint);
  };

  // p = t  ⇒  w·x + b = logit(t). Solve y as a function of x for each contour.
  const contour = (t: number, color: string, dash: boolean): ScatterLine => {
    const k = Math.log(t / (1 - t));
    if (Math.abs(w2) < 1e-6) { const x = (k - b) / (w1 || 1e-6); return { x1: x, y1: 0, x2: x, y2: 1, color, width: dash ? 1.6 : 2.4, dash }; }
    return { x1: 0, y1: (k - b) / w2, x2: 1, y2: (k - (w1 + b)) / w2, color, width: dash ? 1.6 : 2.4, dash };
  };
  const lines: ScatterLine[] = [
    contour(0.25, isLight ? 'rgba(18,23,42,.4)' : 'rgba(255,255,255,.4)', true),
    contour(0.75, isLight ? 'rgba(18,23,42,.4)' : 'rgba(255,255,255,.4)', true),
    contour(0.5, isLight ? 'var(--t0)' : '#fff', false),
  ];

  const classify = (x: number, y: number) => (sigmoid(w1 * x + w2 * y + b) > 0.5 ? 1 : 0);
  const offset = TWO_CLASS_SPAN * separation;

  const insight = `α = ${alpha}, separation ${separation.toFixed(1)} (class means ±${offset.toFixed(3)} from the centre on both axes, σ = ${TWO_CLASS_SIGMA}; Bayes error ≈ ${(bayes * 100).toFixed(1)}%)${l2 > 0 ? `, L2 λ = ${l2}` : ''}. ` +
    (separable ? 'This draw IS linearly separable, so a line can reach 100% — and without a penalty ‖w‖ never stops growing. '
      : 'This draw is NOT linearly separable: no line classifies every point, so accuracy plateaus. ') +
    'Dashed lines mark the p = 0.25 / 0.75 band; it narrows as ‖w‖ grows.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'EPOCH', value: epoch },
        { label: 'BCE', value: metrics.loss.toFixed(3) },
        { label: '‖w‖', value: wNorm.toFixed(1) },
        { label: 'ACC', value: `${(metrics.acc * 100).toFixed(1)}%`, color: GOOD },
        { label: 'SEPARABLE', value: separable ? 'yes' : 'no' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, logregPython({ data, alpha, l2, separation, labEpochs: epoch }))}
      grid={(
        <ScatterPlot
          points={data}
          classify={classify}
          fieldKey={`${seed}-${perClass}-${separation}-${epoch}-${w1}-${w2}-${b}`}
          lines={lines}
          xLabel="x₁"
          yLabel="x₂"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Regularisation</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={l2 === 0} onClick={() => { setL2(0); setPresetId(undefined); }}>None</AlgoPill>
            <AlgoPill active={l2 > 0} onClick={() => { setL2(l2 > 0 ? l2 : 0.05); setPresetId(undefined); }}>L2 · weight decay</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={regen} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="CLASSES" items={[
          { color: CLASS_COLORS[0], label: 'Class 0' },
          { color: CLASS_COLORS[1], label: 'Class 1' },
          { node: <span style={{ width: 12, height: 2, background: isLight ? 'var(--t0)' : '#fff', display: 'inline-block' }} />, label: 'p = 0.5' },
          { node: <span style={{ width: 12, height: 2, background: isLight ? 'rgba(18,23,42,.4)' : 'rgba(255,255,255,.4)', display: 'inline-block' }} />, label: 'p = .25/.75' },
        ]} />
      )}
      rewardLabel="ACCURACY"
      rewardValue={`${(metrics.acc * 100).toFixed(1)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Classifier Parameters" hint="Tune α, separation and L2; press Run to train." />
          <PresetChips presets={PRESETS} activeId={presetId} onApply={applyPreset} />
          <ParamSlider name="α · learning rate" value={alpha.toFixed(2)} min={0.05} max={2} step={0.05} current={alpha} onChange={(v) => { setAlpha(v); setPresetId(undefined); }} hint="gradient-descent step size" />
          <ParamSlider name="L2 λ · weight decay" value={l2.toFixed(3)} min={0} max={0.2} step={0.005} current={l2} onChange={(v) => { setL2(v); setPresetId(undefined); }} hint="penalty ½λ‖w‖² caps the weight norm" />
          <ParamSlider name="Class separation" value={separation.toFixed(1)} min={0} max={1} step={0.1} current={separation} onChange={(v) => { sim.stop(); setSeparation(v); zero(); setPresetId(undefined); }} hint={`0 = same centre · Bayes error ≈ ${(bayes * 100).toFixed(1)}%`} />
          <ParamSlider name="Points / class" value={String(perClass)} min={10} max={80} step={5} current={perClass} onChange={(v) => { sim.stop(); setPerClass(v); zero(); setPresetId(undefined); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="epoch interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Logistic Regression', alpha, separation, l2, separable, bayesError: +bayes.toFixed(4), w: [+w1.toFixed(2), +w2.toFixed(2)], b: +b.toFixed(2), epoch }}
      apiPanel={apiPanel}
    />
  );
};

export default LogisticRegressionLab;
