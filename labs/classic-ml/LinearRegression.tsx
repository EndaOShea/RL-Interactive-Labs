import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, RunControls, Legend, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { clamp01, randn, ParamsWrap, ParamsHead } from './shared';
import { linregPython } from './python';

const TRUE_W = 0.7, TRUE_B = 0.12;
const ACCENT = '#34d399';

const makeData = (n: number, noise: number) =>
  Array.from({ length: n }, () => { const x = Math.random(); return { x, y: clamp01(TRUE_W * x + TRUE_B + randn() * noise) }; });

const LinearRegressionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [n, setN] = useState(45);
  const [noise, setNoise] = useState(0.05);
  const [alpha, setAlpha] = useState(0.3);
  const [data, setData] = useState(() => makeData(45, 0.05));
  const [w, setW] = useState(0);
  const [b, setB] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const curJ = useMemo(() => {
    if (!data.length) return 0;
    let s = 0; data.forEach((p) => { const e = (w * p.x + b) - p.y; s += e * e; });
    return 0.5 * s / data.length;
  }, [w, b, data]);

  const step = () => {
    const m = data.length;
    if (!m) return;
    let dw = 0, db = 0, J = 0;
    data.forEach((p) => { const e = (w * p.x + b) - p.y; dw += e * p.x; db += e; J += e * e; });
    dw /= m; db /= m; J = 0.5 * J / m;
    const nw = w - alpha * dw, nb = b - alpha * db;
    setW(nw); setB(nb); setEpoch((e) => e + 1);
    setLoss((L) => [...L, J].slice(-60));
    setLastLog({
      algorithm: 'Linear Regression · Gradient Descent',
      stepDescription: `Epoch ${epoch + 1} — step parameters downhill`,
      formula: 'θ ← θ − α ∇J,   J = ½·mean((ŷ − y)²)',
      variables: { 'w': nw, 'b': nb, '∂J/∂w': dw, '∂J/∂b': db, 'J': J },
      result: `J = ${J.toFixed(4)}`,
      mathDetails: {
        params: [
          { label: 'α', info: `${alpha}. Step size. Too small = slow; too large = the loss oscillates or diverges.` },
          { label: '∂J/∂w', info: `${dw.toFixed(3)}. Slope of the loss w.r.t. the weight; we move against it.` },
          { label: 'J', info: `${J.toFixed(4)}. Mean squared error — should fall smoothly toward a floor set by the noise.` },
        ],
        implication: J > (loss[loss.length - 1] ?? Infinity) ? 'Loss rose — α is too large for this surface.' : 'Loss is decreasing — the fit is improving.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 70 });

  const regen = (count = n, ns = noise) => { setData(makeData(count, ns)); setW(0); setB(0); setEpoch(0); setLoss([]); setLastLog(null); };
  const reset = () => { sim.stop(); setW(0); setB(0); setEpoch(0); setLoss([]); setLastLog(null); };

  const insight = `α = ${alpha}. ` +
    (alpha >= 0.8 ? 'High learning rate — watch for the loss bouncing or exploding instead of settling.'
      : alpha <= 0.05 ? 'Low learning rate — stable but it will take many epochs to reach the fit.'
        : 'A balanced learning rate: the line should ease onto the data over a few dozen epochs.');

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'EPOCH', value: epoch },
        { label: 'LOSS', value: curJ.toFixed(4), color: GOOD },
        { label: 'w', value: w.toFixed(2) },
        { label: 'b', value: b.toFixed(2) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, linregPython(alpha))}
      grid={(
        <FunctionPlot
          domain={[0, 1]}
          range={[0, 1]}
          scatter={data.map((p) => ({ x: p.x, y: p.y, color: 'var(--t1)' }))}
          series={[{ points: [{ x: 0, y: b }, { x: 1, y: w + b }], color: ACCENT, width: 2.6 }]}
          xLabel="x"
          yLabel="y"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="FIT" items={[
          { color: 'var(--t1)', label: 'Data' },
          { color: ACCENT, label: 'ŷ = w·x + b' },
        ]} />
      )}
      rewardLabel="LOSS (MSE)"
      rewardValue={curJ.toFixed(4)}
      rewardSeries={loss}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Regression Parameters" hint="Tune α and the data; press Run to descend." />
          <ParamSlider name="α · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={setAlpha} hint="gradient-descent step size" />
          <ParamSlider name="Noise" value={noise.toFixed(3)} min={0} max={0.15} step={0.005} current={noise} onChange={(v) => { setNoise(v); regen(n, v); }} hint="scatter around the true line" />
          <ParamSlider name="Points" value={String(n)} min={10} max={80} step={5} current={n} onChange={(v) => { setN(v); regen(v, noise); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="epoch interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Linear Regression (GD)', alpha, w: +w.toFixed(3), b: +b.toFixed(3), epoch }}
      apiPanel={apiPanel}
    />
  );
};

export default LinearRegressionLab;
