import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterLine, ScatterMarker, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, RunControls, Legend, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { svmPython } from './python';

const ACCENT = '#fbbf24';
const DOM: [number, number] = [-1.2, 1.2];
interface SPt { x: number; y: number; yy: number; } // yy ∈ {-1,+1}

const makeData = (perClass: number, sep: number): SPt[] => {
  const off = 0.22 + sep * 0.3;
  const out: SPt[] = [];
  for (let i = 0; i < perClass; i++) {
    out.push({ x: -off + randn() * 0.16, y: -off + randn() * 0.16, yy: -1 });
    out.push({ x: off + randn() * 0.16, y: off + randn() * 0.16, yy: 1 });
  }
  return out;
};

const SvmLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [perClass, setPerClass] = useState(35);
  const [sep, setSep] = useState(0.5);
  const [C, setC] = useState(5);
  const [data, setData] = useState<SPt[]>(() => makeData(35, 0.5));
  const [w1, setW1] = useState(0);
  const [w2, setW2] = useState(0);
  const [b, setB] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [accSeries, setAccSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const lambda = 1 / C;
  const metrics = useMemo(() => {
    const norm = Math.hypot(w1, w2) || 1e-9;
    let ok = 0, sv = 0;
    data.forEach((p) => {
      const m = p.yy * (w1 * p.x + w2 * p.y + b);
      if ((w1 * p.x + w2 * p.y + b > 0 ? 1 : -1) === p.yy) ok++;
      if (m <= 1 + 1e-6) sv++;
    });
    return { acc: ok / (data.length || 1), margin: 2 / norm, sv };
  }, [w1, w2, b, data]);

  const step = () => {
    const n = data.length; const lr = 0.1;
    let gw1 = lambda * w1, gw2 = lambda * w2, gb = 0;
    data.forEach((p) => {
      const m = p.yy * (w1 * p.x + w2 * p.y + b);
      if (m < 1) { gw1 -= (p.yy * p.x) / n; gw2 -= (p.yy * p.y) / n; gb -= p.yy / n; }
    });
    setW1((v) => v - lr * gw1); setW2((v) => v - lr * gw2); setB((v) => v - lr * gb);
    setEpoch((e) => e + 1);
    setAccSeries((s) => [...s, metrics.acc].slice(-60));
    setLastLog({
      algorithm: 'Support Vector Machine · soft margin',
      stepDescription: `Epoch ${epoch + 1} — subgradient step on hinge loss`,
      formula: 'min ½‖w‖² + C·Σ max(0, 1 − yᵢ(w·xᵢ+b))',
      variables: { 'w₁': w1, 'w₂': w2, 'b': b, 'margin': metrics.margin, 'SV': metrics.sv },
      result: `margin ${metrics.margin.toFixed(2)} · ${metrics.sv} SV`,
      mathDetails: {
        params: [
          { label: 'C', info: `${C}. Penalty for margin violations. Large C → hard margin (few violations); small C → wider, softer margin.` },
          { label: 'margin', info: `${metrics.margin.toFixed(3)} = 2/‖w‖. SVM maximises this gap between the classes.` },
          { label: 'support vectors', info: `${metrics.sv}. Only points on/inside the margin shape the boundary.` },
        ],
        implication: 'The boundary is placed to be as far as possible from both classes — defined solely by the support vectors.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 50 });
  const regen = (pc = perClass, s = sep) => { setData(makeData(pc, s)); setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setLastLog(null); };
  const reset = () => { sim.stop(); setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setLastLog(null); };

  const classify = (x: number, y: number) => (w1 * x + w2 * y + b > 0 ? 1 : 0);
  const yAt = (x: number, off: number) => Math.abs(w2) < 1e-6 ? 0 : (off - w1 * x - b) / w2;
  const lines: ScatterLine[] = (Math.abs(w1) + Math.abs(w2) > 1e-6) ? [
    { x1: DOM[0], y1: yAt(DOM[0], 0), x2: DOM[1], y2: yAt(DOM[1], 0), color: '#fff', width: 2.4 },
    { x1: DOM[0], y1: yAt(DOM[0], 1), x2: DOM[1], y2: yAt(DOM[1], 1), color: ACCENT, width: 1.4, dash: true },
    { x1: DOM[0], y1: yAt(DOM[0], -1), x2: DOM[1], y2: yAt(DOM[1], -1), color: ACCENT, width: 1.4, dash: true },
  ] : [];
  const points: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.yy > 0 ? 1 : 0 }));
  const svMarkers: ScatterMarker[] = (Math.abs(w1) + Math.abs(w2) > 1e-6)
    ? data.filter((p) => p.yy * (w1 * p.x + w2 * p.y + b) <= 1 + 1e-6).map((p) => ({ x: p.x, y: p.y, color: '#fff', r: 9, ring: true }))
    : [];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'EPOCH', value: epoch },
        { label: 'MARGIN', value: metrics.margin.toFixed(2), color: ACCENT },
        { label: 'SV', value: metrics.sv },
        { label: 'ACC', value: `${(metrics.acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, svmPython(C))}
      grid={(
        <ScatterPlot width={460} height={460} domain={DOM} range={DOM} points={points} classify={classify} fieldKey={`${epoch}`} lines={lines} markers={svMarkers} xLabel="x₁" yLabel="x₂" />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="SVM" items={[
          { color: CLASS_COLORS[0], label: 'Class −1' },
          { color: CLASS_COLORS[1], label: 'Class +1' },
          { node: <span style={{ width: 12, height: 2, background: '#fff', display: 'inline-block' }} />, label: 'Boundary' },
          { node: <span style={{ width: 11, height: 11, borderRadius: '50%', border: '1.5px solid #fff', display: 'inline-block' }} />, label: 'Support vec.' },
        ]} />
      )}
      rewardLabel="ACCURACY"
      rewardValue={`${(metrics.acc * 100).toFixed(0)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={`C=${C}. SVM finds the maximum-margin boundary — the one with the widest empty street between classes. Only the support vectors (ringed) touch the margin and determine the line; everything else could move freely without changing it.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="SVM Parameters" hint="Soft-margin linear SVM." />
          <ParamSlider name="C · penalty" value={String(C)} min={0.5} max={50} step={0.5} current={C} onChange={(v) => { setC(v); reset(); }} hint="margin-violation cost (1/λ)" />
          <ParamSlider name="Separation" value={sep.toFixed(1)} min={0} max={1} step={0.1} current={sep} onChange={(v) => { setSep(v); regen(perClass, v); }} hint="class gap" />
          <ParamSlider name="Points / class" value={String(perClass)} min={10} max={70} step={5} current={perClass} onChange={(v) => { setPerClass(v); regen(v, sep); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="epoch interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'SVM (soft-margin)', C, margin: +metrics.margin.toFixed(3), supportVectors: metrics.sv, epoch }}
      apiPanel={apiPanel}
    />
  );
};

export default SvmLab;
