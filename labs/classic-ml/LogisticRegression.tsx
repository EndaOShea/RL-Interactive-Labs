import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, RunControls, Legend, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { Pt, makeTwoClass, ParamsWrap, ParamsHead } from './shared';
import { logregPython } from './python';

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

const LogisticRegressionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [perClass, setPerClass] = useState(40);
  const [separation, setSeparation] = useState(0.6);
  const [alpha, setAlpha] = useState(0.5);
  const [data, setData] = useState<Pt[]>(() => makeTwoClass(40, 0.6));
  const [w1, setW1] = useState(0);
  const [w2, setW2] = useState(0);
  const [b, setB] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [accSeries, setAccSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

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
    gw1 /= m; gw2 /= m; gb /= m; loss /= m;
    const acc = correct / m;
    setW1((v) => v - alpha * gw1); setW2((v) => v - alpha * gw2); setB((v) => v - alpha * gb);
    setEpoch((e) => e + 1);
    setAccSeries((a) => [...a, acc].slice(-60));
    setLastLog({
      algorithm: 'Logistic Regression · Cross-Entropy GD',
      stepDescription: `Epoch ${epoch + 1} — gradient = (p − y)·x`,
      formula: 'p = σ(w·x + b),   w ← w − α·mean((p − y)·x)',
      variables: { 'w₁': w1, 'w₂': w2, 'b': b, 'BCE': loss, 'acc': acc },
      result: `acc = ${(acc * 100).toFixed(1)}%`,
      mathDetails: {
        params: [
          { label: 'α', info: `${alpha}. Step size for the weight update.` },
          { label: 'BCE', info: `${loss.toFixed(4)}. Binary cross-entropy — punishes confident wrong predictions hardest.` },
          { label: 'boundary', info: 'The white line is where p = 0.5 (w·x + b = 0); it slides until the classes are split.' },
        ],
        implication: acc >= 0.95 ? 'Classes are well separated by a line.' : 'Still adjusting — or the data may not be linearly separable.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 60 });

  const regen = (pc = perClass, sep = separation) => { setData(makeTwoClass(pc, sep)); setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setLastLog(null); };
  const reset = () => { sim.stop(); setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setLastLog(null); };

  const boundary: ScatterLine = Math.abs(w2) < 1e-6
    ? { x1: -b / (w1 || 1e-6), y1: 0, x2: -b / (w1 || 1e-6), y2: 1, color: '#fff', width: 2.4 }
    : { x1: 0, y1: -(b) / w2, x2: 1, y2: -(w1 + b) / w2, color: '#fff', width: 2.4 };

  const classify = (x: number, y: number) => (sigmoid(w1 * x + w2 * y + b) > 0.5 ? 1 : 0);

  const insight = `α = ${alpha}, separation ${separation.toFixed(1)}. ` +
    (metrics.acc >= 0.97 ? 'A straight line cleanly separates the two classes here.'
      : 'The boundary is still moving — raise separation or run more epochs; remember a single line can never split interleaved classes.');

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'EPOCH', value: epoch },
        { label: 'BCE', value: metrics.loss.toFixed(3) },
        { label: 'ACC', value: `${(metrics.acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, logregPython(alpha))}
      grid={(
        <ScatterPlot
          points={data}
          classify={classify}
          fieldKey={epoch}
          lines={[boundary]}
          xLabel="x₁"
          yLabel="x₂"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="CLASSES" items={[
          { color: CLASS_COLORS[0], label: 'Class 0' },
          { color: CLASS_COLORS[1], label: 'Class 1' },
          { node: <span style={{ width: 12, height: 2, background: '#fff', display: 'inline-block' }} />, label: 'p = 0.5' },
        ]} />
      )}
      rewardLabel="ACCURACY"
      rewardValue={`${(metrics.acc * 100).toFixed(0)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Classifier Parameters" hint="Tune α and class separation; press Run to train." />
          <ParamSlider name="α · learning rate" value={alpha.toFixed(2)} min={0.05} max={2} step={0.05} current={alpha} onChange={setAlpha} hint="gradient-descent step size" />
          <ParamSlider name="Class separation" value={separation.toFixed(1)} min={0} max={1} step={0.1} current={separation} onChange={(v) => { setSeparation(v); regen(perClass, v); }} hint="how far apart the clusters sit" />
          <ParamSlider name="Points / class" value={String(perClass)} min={10} max={80} step={5} current={perClass} onChange={(v) => { setPerClass(v); regen(v, separation); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="epoch interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Logistic Regression', alpha, separation, w: [+w1.toFixed(2), +w2.toFixed(2)], b: +b.toFixed(2), epoch }}
      apiPanel={apiPanel}
    />
  );
};

export default LogisticRegressionLab;
