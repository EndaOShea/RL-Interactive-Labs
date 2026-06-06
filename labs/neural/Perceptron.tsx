import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterLine, ScatterMarker, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, RunControls, Legend, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { perceptronPython } from './python';

const ACCENT = '#2dd4bf';
const DOM: [number, number] = [-1.2, 1.2];
interface PPt { x: number; y: number; yy: number; }

const makeData = (perClass: number, sep: number): PPt[] => {
  const off = 0.2 + sep * 0.45, out: PPt[] = [];
  for (let i = 0; i < perClass; i++) {
    out.push({ x: -off + randn() * 0.13, y: -off + randn() * 0.13, yy: -1 });
    out.push({ x: off + randn() * 0.13, y: off + randn() * 0.13, yy: 1 });
  }
  return out;
};

const PerceptronLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [perClass, setPerClass] = useState(20);
  const [sep, setSep] = useState(0.5);
  const [data, setData] = useState<PPt[]>(() => makeData(20, 0.5));
  const [w1, setW1] = useState(0.4);
  const [w2, setW2] = useState(-0.6);
  const [b, setB] = useState(0);
  const [idx, setIdx] = useState(0);
  const [updates, setUpdates] = useState(0);
  const [passErrors, setPassErrors] = useState(0);
  const [mistakes, setMistakes] = useState(0);
  const [converged, setConverged] = useState(false);
  const [accSeries, setAccSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const acc = useMemo(() => { let ok = 0; data.forEach((p) => { if ((w1 * p.x + w2 * p.y + b > 0 ? 1 : -1) === p.yy) ok++; }); return ok / (data.length || 1); }, [w1, w2, b, data]);

  const step = () => {
    const lr = 0.5; const p = data[idx]; const n = data.length;
    const score = w1 * p.x + w2 * p.y + b;
    const wrong = p.yy * score <= 0;
    let m = mistakes;
    if (wrong) { setW1((v) => v + lr * p.yy * p.x); setW2((v) => v + lr * p.yy * p.y); setB((v) => v + lr * p.yy); setUpdates((u) => u + 1); m += 1; setMistakes(m); }
    const next = (idx + 1) % n;
    setIdx(next);
    setAccSeries((s) => [...s, acc].slice(-60));
    if (next === 0) { setPassErrors(m); setMistakes(0); if (m === 0) { setConverged(true); sim.pause(); } }
    setLastLog({
      algorithm: 'Perceptron · learning rule',
      stepDescription: wrong ? `Point ${idx + 1} misclassified — update weights` : `Point ${idx + 1} correct — no change`,
      formula: 'if y(w·x+b) ≤ 0:  w ← w + η·y·x,  b ← b + η·y',
      variables: { 'y': p.yy, 'score': score, 'updates': updates + (wrong ? 1 : 0) },
      result: wrong ? 'updated' : 'ok',
      mathDetails: {
        params: [
          { label: 'rule', info: 'Only misclassified points move the boundary — toward correctly classifying them.' },
          { label: 'convergence', info: 'If the data is linearly separable, the perceptron is guaranteed to converge in finite updates.' },
          { label: 'vs logistic', info: 'Perceptron gives a hard label and any separating line; logistic/SVM optimise a smooth/margin objective.' },
        ],
        implication: 'A single neuron can only draw a straight boundary — not separable ⇒ it never settles (needs hidden layers).',
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 120 });

  const regen = (pc = perClass, s = sep) => { setData(makeData(pc, s)); reset(); };
  const reset = () => { sim.stop(); setW1(0.4); setW2(-0.6); setB(0); setIdx(0); setUpdates(0); setPassErrors(0); setMistakes(0); setConverged(false); setAccSeries([]); setLastLog(null); };

  const classify = (x: number, y: number) => (w1 * x + w2 * y + b > 0 ? 1 : 0);
  const yAt = (x: number) => (Math.abs(w2) < 1e-6 ? 0 : -(w1 * x + b) / w2);
  const lines: ScatterLine[] = [{ x1: DOM[0], y1: yAt(DOM[0]), x2: DOM[1], y2: yAt(DOM[1]), color: '#fff', width: 2.4 }];
  const points: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.yy > 0 ? 1 : 0 }));
  const cur = data[idx];
  const markers: ScatterMarker[] = cur ? [{ x: cur.x, y: cur.y, color: '#fff', r: 8, ring: true }] : [];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'UPDATES', value: updates },
        { label: 'PASS ERR', value: passErrors },
        { label: 'ACC', value: `${(acc * 100).toFixed(0)}%`, color: GOOD },
        { label: 'STATUS', value: converged ? 'CONVERGED' : 'learning', color: converged ? GOOD : ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, perceptronPython())}
      grid={<ScatterPlot width={460} height={460} domain={DOM} range={DOM} points={points} classify={classify} fieldKey={`${updates}-${idx}`} lines={lines} markers={markers} xLabel="x₁" yLabel="x₂" />}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="PERCEPTRON" items={[
          { color: CLASS_COLORS[0], label: 'Class −1' },
          { color: CLASS_COLORS[1], label: 'Class +1' },
          { node: <span style={{ width: 12, height: 2, background: '#fff', display: 'inline-block' }} />, label: 'Boundary' },
        ]} />
      )}
      rewardLabel="ACCURACY"
      rewardValue={`${(acc * 100).toFixed(0)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={`The perceptron — the original neuron (1958). It cycles through points, nudging its weights only on mistakes; on linearly separable data it provably converges. Stack these with non-linear activations and you get the MLP next door.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Perceptron" hint="Single neuron, online learning rule." />
          <ParamSlider name="Separation" value={sep.toFixed(1)} min={0} max={1} step={0.1} current={sep} onChange={(v) => { setSep(v); regen(perClass, v); }} hint="class gap (low = may not converge)" />
          <ParamSlider name="Points / class" value={String(perClass)} min={8} max={50} step={2} current={perClass} onChange={(v) => { setPerClass(v); regen(v, sep); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="one point / tick" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Perceptron', separation: sep, updates, acc: +acc.toFixed(3), converged }}
      apiPanel={apiPanel}
    />
  );
};

export default PerceptronLab;
