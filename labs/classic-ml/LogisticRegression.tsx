import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { Pt, makeTwoClass, ParamsWrap, ParamsHead } from './shared';
import { logregPython } from './python';
import { PresetChips, Preset } from './presets';

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

interface Cfg { alpha: number; separation: number; l2: number; }
const PRESETS: Preset<Cfg>[] = [
  { id: 'easy', label: 'Well-separated', hint: 'Far-apart clusters — the line locks in fast at ~100% accuracy.', values: { alpha: 0.6, separation: 0.9, l2: 0 } },
  { id: 'overlap', label: 'Overlapping', hint: 'Classes overlap — no line is perfect; accuracy plateaus below 100%.', values: { alpha: 0.5, separation: 0.2, l2: 0 } },
  { id: 'blowup', label: 'Weight blow-up', hint: 'Separable data with no penalty lets the weights grow without bound.', values: { alpha: 1.2, separation: 0.9, l2: 0 } },
  { id: 'reg', label: 'L2-regularised', hint: 'A penalty keeps the weights finite — a calmer, better-calibrated boundary.', values: { alpha: 1.2, separation: 0.9, l2: 0.05 } },
];

const LogisticRegressionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [perClass, setPerClass] = useState(40);
  const [separation, setSeparation] = useState(0.6);
  const [alpha, setAlpha] = useState(0.5);
  const [l2, setL2] = useState(0);
  const [data, setData] = useState<Pt[]>(() => makeTwoClass(40, 0.6));
  const [w1, setW1] = useState(0);
  const [w2, setW2] = useState(0);
  const [b, setB] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [accSeries, setAccSeries] = useState<number[]>([]);
  const [presetId, setPresetId] = useState<string | undefined>();
  const [milestone, setMilestone] = useState(false);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const metrics = useMemo(() => {
    const m = data.length || 1;
    let loss = 0, correct = 0;
    data.forEach((p) => {
      const pr = sigmoid(w1 * p.x + w2 * p.y + b);
      const c = Math.min(1 - 1e-9, Math.max(1e-9, pr));
      loss += -(p.cls * Math.log(c) + (1 - p.cls) * Math.log(1 - c));
      if ((pr > 0.5 ? 1 : 0) === p.cls) correct++;
    });
    return { loss: loss / m, acc: correct / m };
  }, [w1, w2, b, data]);

  const wNorm = Math.hypot(w1, w2);

  const step = () => {
    const m = data.length;
    if (!m) return;
    let gw1 = 0, gw2 = 0, gb = 0, loss = 0, correct = 0;
    data.forEach((p) => {
      const pr = sigmoid(w1 * p.x + w2 * p.y + b);
      const e = pr - p.cls;
      gw1 += e * p.x; gw2 += e * p.y; gb += e;
      const c = Math.min(1 - 1e-9, Math.max(1e-9, pr));
      loss += -(p.cls * Math.log(c) + (1 - p.cls) * Math.log(1 - c));
      if ((pr > 0.5 ? 1 : 0) === p.cls) correct++;
    });
    gw1 = gw1 / m + l2 * w1; gw2 = gw2 / m + l2 * w2; gb /= m; loss /= m;
    const acc = correct / m;
    setW1((v) => v - alpha * gw1); setW2((v) => v - alpha * gw2); setB((v) => v - alpha * gb);
    setEpoch((e) => e + 1);
    setAccSeries((a) => [...a, acc].slice(-60));

    const intro = l2 > 0
      ? `Logistic regression draws a straight decision boundary, but with L2 regularisation. It passes a weighted sum of the features through the sigmoid to get a probability, then minimises cross-entropy plus a penalty lambda times the squared weights. On separable data that penalty matters: without it the weights run off to infinity, so the penalty caps the weight norm and keeps the probabilities calibrated. The solid white line is where the probability is one half; the dashed lines are the quarter and three-quarter band, and it narrows as the weight norm grows.`
      : `Logistic regression draws a straight decision boundary. It passes a weighted sum of the features through the sigmoid — one over one plus e to the minus z — turning the score into a probability, and trains by minimising cross-entropy. Each step moves the weights along the clean gradient, the average of the prediction minus the label times the input. Watch the solid white line, where the probability is one half, tilt to split the two classes, with the dashed quarter and three-quarter contours forming the confidence band.`;
    narration.narratePhase(`run:${l2 > 0}`, intro);
    if (acc >= 0.99 && !milestone) {
      setMilestone(true);
      narration.narratePhase('done:separated', `The boundary now separates every point — accuracy has reached one hundred percent. On perfectly separable data the cross-entropy loss has no finite minimum, so ${l2 > 0 ? 'the L2 penalty is what holds the weights, and the confidence band, in check' : 'the weight norm will keep growing and the confidence band keep narrowing unless you add an L2 penalty'}.`);
    } else if (acc < 0.99 && milestone) setMilestone(false);

    setLastLog({
      algorithm: `Logistic Regression · Cross-Entropy GD${l2 > 0 ? ' · L2' : ''}`,
      stepDescription: `Epoch ${epoch + 1} — gradient = (p − y)·x${l2 > 0 ? ' + λw' : ''}`,
      formula: l2 > 0 ? 'p = σ(w·x + b),   w ← w − α(mean((p−y)x) + λw)' : 'p = σ(w·x + b),   w ← w − α·mean((p − y)·x)',
      variables: { 'w₁': w1, 'w₂': w2, 'b': b, 'BCE': loss, 'acc': acc },
      result: `acc = ${(acc * 100).toFixed(1)}%`,
      mathDetails: {
        params: [
          { label: 'α', info: `${alpha}. Step size for the weight update.` },
          { label: 'λ', info: l2 > 0 ? `${l2}. L2 penalty pulls the weights toward 0 each step, capping ‖w‖ and keeping probabilities calibrated.` : 'No penalty — on separable data ‖w‖ can grow without bound.' },
          { label: '‖w‖', info: `${wNorm.toFixed(2)}. Weight norm; the sigmoid sharpens as it grows, so the boundary band narrows.` },
          { label: 'boundary', info: 'The white line is where p = 0.5; the dashed lines are the p = 0.25 / 0.75 margin.' },
        ],
        implication: acc >= 0.95 ? 'Classes are well separated by a line.' : 'Still adjusting — or the data may not be linearly separable.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 60 });

  const regen = (pc = perClass, sep = separation) => { setData(makeTwoClass(pc, sep)); setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setMilestone(false); setLastLog(null); narration.cancel(); };
  const reset = () => { sim.stop(); setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setMilestone(false); setLastLog(null); narration.cancel(); };
  const applyPreset = (p: Preset<Cfg>) => {
    setAlpha(p.values.alpha); setSeparation(p.values.separation); setL2(p.values.l2);
    setData(makeTwoClass(perClass, p.values.separation));
    setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setMilestone(false); setPresetId(p.id);
    narration.cancel(); narration.narratePhase(`preset:${p.id}`, p.hint);
  };

  // p = t  ⇒  w·x + b = logit(t). Solve y as a function of x for each contour.
  const contour = (t: number, color: string, dash: boolean): ScatterLine => {
    const k = Math.log(t / (1 - t));
    if (Math.abs(w2) < 1e-6) { const x = (k - b) / (w1 || 1e-6); return { x1: x, y1: 0, x2: x, y2: 1, color, width: dash ? 1.6 : 2.4, dash }; }
    return { x1: 0, y1: (k - b) / w2, x2: 1, y2: (k - (w1 + b)) / w2, color, width: dash ? 1.6 : 2.4, dash };
  };
  const lines: ScatterLine[] = [
    contour(0.25, 'rgba(255,255,255,.4)', true),
    contour(0.75, 'rgba(255,255,255,.4)', true),
    contour(0.5, '#fff', false),
  ];

  const classify = (x: number, y: number) => (sigmoid(w1 * x + w2 * y + b) > 0.5 ? 1 : 0);

  const insight = `α = ${alpha}, separation ${separation.toFixed(1)}${l2 > 0 ? `, L2 λ=${l2}` : ''}. ` +
    (metrics.acc >= 0.97 ? 'A straight line cleanly separates the two classes here. '
      : 'The boundary is still moving — raise separation or run more epochs; a single line can never split interleaved classes. ') +
    'Dashed lines mark the p = 0.25 / 0.75 confidence band; it narrows as ‖w‖ grows.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'EPOCH', value: epoch },
        { label: 'BCE', value: metrics.loss.toFixed(3) },
        { label: '‖w‖', value: wNorm.toFixed(1) },
        { label: 'ACC', value: `${(metrics.acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, logregPython(alpha, l2))}
      grid={(
        <ScatterPlot
          points={data}
          classify={classify}
          fieldKey={`${epoch}-${l2}`}
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
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="CLASSES" items={[
          { color: CLASS_COLORS[0], label: 'Class 0' },
          { color: CLASS_COLORS[1], label: 'Class 1' },
          { node: <span style={{ width: 12, height: 2, background: '#fff', display: 'inline-block' }} />, label: 'p = 0.5' },
          { node: <span style={{ width: 12, height: 2, background: 'rgba(255,255,255,.4)', display: 'inline-block' }} />, label: 'p = .25/.75' },
        ]} />
      )}
      rewardLabel="ACCURACY"
      rewardValue={`${(metrics.acc * 100).toFixed(0)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Classifier Parameters" hint="Tune α, separation and L2; press Run to train." />
          <PresetChips presets={PRESETS} activeId={presetId} onApply={applyPreset} />
          <ParamSlider name="α · learning rate" value={alpha.toFixed(2)} min={0.05} max={2} step={0.05} current={alpha} onChange={(v) => { setAlpha(v); setPresetId(undefined); }} hint="gradient-descent step size" />
          <ParamSlider name="L2 λ · weight decay" value={l2.toFixed(3)} min={0} max={0.2} step={0.005} current={l2} onChange={(v) => { setL2(v); setPresetId(undefined); }} hint="caps the weight norm" />
          <ParamSlider name="Class separation" value={separation.toFixed(1)} min={0} max={1} step={0.1} current={separation} onChange={(v) => { setSeparation(v); regen(perClass, v); }} hint="how far apart the clusters sit" />
          <ParamSlider name="Points / class" value={String(perClass)} min={10} max={80} step={5} current={perClass} onChange={(v) => { setPerClass(v); regen(v, separation); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="epoch interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Logistic Regression', alpha, separation, l2, w: [+w1.toFixed(2), +w2.toFixed(2)], b: +b.toFixed(2), epoch }}
      apiPanel={apiPanel}
    />
  );
};

export default LogisticRegressionLab;
