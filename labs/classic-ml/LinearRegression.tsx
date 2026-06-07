import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { clamp01, randn, ParamsWrap, ParamsHead } from './shared';
import { linregPython } from './python';
import { PresetChips, Preset } from './presets';

const ACCENT = '#34d399';
const RESID = 'rgba(248,113,113,.5)';

// Curved ground truth so polynomial degree actually matters.
const truth = (x: number) => 0.18 + 0.95 * x - 0.55 * x * x + 0.42 * Math.pow(x - 0.5, 3) * 4;
const makeData = (n: number, noise: number) =>
  Array.from({ length: n }, () => { const x = Math.random(); return { x, y: clamp01(truth(x) + randn() * noise) }; });

const features = (x: number, degree: number) => Array.from({ length: degree }, (_, d) => Math.pow(x, d + 1));

interface Cfg { degree: number; ridge: number; alpha: number; }
const PRESETS: Preset<Cfg>[] = [
  { id: 'line', label: 'Straight line', hint: 'Degree 1 underfits the curved truth — residuals stay large (high bias).', values: { degree: 1, ridge: 0, alpha: 0.4 } },
  { id: 'cubic', label: 'Cubic fit', hint: 'Degree 3 captures the bend — the loss floor drops to the noise level.', values: { degree: 3, ridge: 0, alpha: 0.3 } },
  { id: 'overfit', label: 'Overfit (deg 6)', hint: 'High degree wiggles through noise. Add ridge to tame it.', values: { degree: 6, ridge: 0, alpha: 0.25 } },
  { id: 'ridge', label: 'Ridge-tamed', hint: 'λ shrinks the high-degree weights — a smoother curve that generalises.', values: { degree: 6, ridge: 0.05, alpha: 0.25 } },
];

const LinearRegressionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [n, setN] = useState(45);
  const [noise, setNoise] = useState(0.05);
  const [alpha, setAlpha] = useState(0.3);
  const [degree, setDegree] = useState(1);
  const [ridge, setRidge] = useState(0);
  const [data, setData] = useState(() => makeData(45, 0.05));
  const [w, setW] = useState<number[]>([0]);
  const [b, setB] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState<number[]>([]);
  const [presetId, setPresetId] = useState<string | undefined>();
  const [converged, setConverged] = useState(false);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const predict = (x: number, ww: number[], bb: number) => features(x, ww.length).reduce((s, f, i) => s + ww[i] * f, bb);

  const curJ = useMemo(() => {
    if (!data.length) return 0;
    let s = 0; data.forEach((p) => { const e = predict(p.x, w, b) - p.y; s += e * e; });
    return 0.5 * s / data.length;
  }, [w, b, data]);

  const step = () => {
    const m = data.length;
    if (!m) return;
    const dw = new Array(degree).fill(0);
    let db = 0, J = 0;
    data.forEach((p) => {
      const e = predict(p.x, w, b) - p.y;
      const f = features(p.x, degree);
      for (let i = 0; i < degree; i++) dw[i] += e * f[i];
      db += e; J += e * e;
    });
    for (let i = 0; i < degree; i++) dw[i] = dw[i] / m + ridge * w[i]; // ridge weight decay
    db /= m; J = 0.5 * J / m;
    const nw = w.map((wi, i) => wi - alpha * dw[i]);
    const nb = b - alpha * db;
    const prev = loss[loss.length - 1] ?? Infinity;
    setW(nw); setB(nb); setEpoch((e) => e + 1);
    setLoss((L) => [...L, J].slice(-60));

    const settled = Math.abs(prev - J) < 1e-6 && epoch > 8;
    const modelWord = degree === 1 ? 'a straight line' : `a degree ${degree} polynomial`;
    const intro = ridge > 0
      ? `This fits ${modelWord} by gradient descent, but with ridge regularisation. The loss is the mean squared error plus lambda times the squared weights, so each step both follows the data gradient and shrinks every weight a little toward zero — gentle weight decay. That trades a touch of bias for much less variance, keeping a high-degree curve from wiggling through the noise. Watch the green curve and the red residual sticks, and the loss curve settling toward the noise floor.`
      : `This fits ${modelWord} by gradient descent: we minimise the mean squared error, the average squared vertical gap between the curve and the points. Each epoch nudges the weights downhill, theta moves a small step alpha against the gradient of the loss. Watch the green fit bend toward the data, the red sticks shrink as residuals fall, and the loss curve decay toward the noise floor.`;
    narration.narratePhase(`run:${degree}:${ridge > 0}`, intro);
    if (J > prev * 1.05) {
      narration.narratePhase('diverge', `The loss is climbing instead of falling, which means alpha is too large: each step overshoots the minimum and bounces up the far wall of the loss surface. Lower the learning rate to make gradient descent stable.`);
    } else if (settled && !converged) {
      setConverged(true);
      narration.narratePhase(`done:${degree}:${ridge > 0}`, `The fit has converged — the loss has flattened near ${J.toFixed(4)}. That floor reflects the irreducible noise in the data, not a poor model${ridge > 0 ? ', and ridge has kept the weights small and the curve smooth' : ''}. Compare the curve against the truth: that is the bias-variance balance you chose with the degree.`);
    }

    setLastLog({
      algorithm: `${degree > 1 ? `Polynomial (deg ${degree})` : 'Linear'} Regression · GD${ridge > 0 ? ' · Ridge' : ''}`,
      stepDescription: `Epoch ${epoch + 1} — step weights downhill`,
      formula: ridge > 0 ? 'w ← w − α(∇J + λw),   ŷ = Σ wⱼ xʲ + b' : 'θ ← θ − α ∇J,   J = ½·mean((ŷ − y)²)',
      variables: { 'w₁': nw[0], 'b': nb, '∂J/∂w₁': dw[0], 'J': J, 'λ': ridge },
      result: `J = ${J.toFixed(4)}`,
      mathDetails: {
        params: [
          { label: 'α', info: `${alpha}. Step size. Too small = slow; too large = the loss oscillates or diverges.` },
          { label: 'degree', info: `${degree}. Polynomial order — degree 1 is a line; higher degrees bend to fit curvature (and can overfit).` },
          { label: 'λ', info: ridge > 0 ? `${ridge}. Ridge penalty shrinks the weights toward 0, trading a little bias for much less variance.` : 'No regularisation — weights are unconstrained.' },
          { label: 'J', info: `${J.toFixed(4)}. Mean squared error — should fall toward a floor set by the noise.` },
        ],
        implication: J > prev ? 'Loss rose — α is too large for this surface.' : 'Loss is decreasing — the fit is improving.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const resetWeights = (deg: number) => { setW(new Array(deg).fill(0)); setB(0); setEpoch(0); setLoss([]); setConverged(false); setLastLog(null); narration.cancel(); };
  const regen = (count = n, ns = noise) => { setData(makeData(count, ns)); resetWeights(degree); };
  const reset = () => { sim.stop(); resetWeights(degree); };
  const changeDegree = (d: number) => { setDegree(d); resetWeights(d); setPresetId(undefined); };
  const applyPreset = (p: Preset<Cfg>) => {
    setDegree(p.values.degree); setRidge(p.values.ridge); setAlpha(p.values.alpha);
    resetWeights(p.values.degree); setPresetId(p.id);
    narration.narratePhase(`preset:${p.id}`, p.hint);
  };

  // Richer visuals: sampled curve for the (possibly polynomial) fit + residual sticks.
  const curve = Array.from({ length: 60 }, (_, i) => { const x = i / 59; return { x, y: predict(x, w, b) }; });
  const residuals = data.map((p) => ({ points: [{ x: p.x, y: p.y }, { x: p.x, y: predict(p.x, w, b) }], color: RESID, width: 1 }));

  const insight = `α = ${alpha}, degree ${degree}${ridge > 0 ? `, ridge λ=${ridge}` : ''}. ` +
    (degree === 1 ? 'A straight line cannot follow the curved truth — expect a high loss floor (bias). '
      : ridge > 0 ? 'Ridge keeps the high-degree weights small, so the curve stays smooth. '
        : degree >= 5 ? 'A high-degree polynomial can chase noise — watch for wiggles between points. '
          : 'The polynomial bends to follow the data; the loss floor drops toward the noise. ') +
    'Red sticks show each residual (gap between fit and point).';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'EPOCH', value: epoch },
        { label: 'LOSS', value: curJ.toFixed(4), color: GOOD },
        { label: 'DEG', value: degree },
        { label: 'λ', value: ridge.toFixed(2) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, linregPython(alpha, degree, ridge))}
      grid={(
        <FunctionPlot
          domain={[0, 1]}
          range={[0, 1]}
          scatter={data.map((p) => ({ x: p.x, y: p.y, color: 'var(--t1)' }))}
          series={[...residuals, { points: curve, color: ACCENT, width: 2.6 }]}
          xLabel="x"
          yLabel="y"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Model</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={degree === 1} onClick={() => changeDegree(1)}>Linear · deg 1</AlgoPill>
            <AlgoPill active={degree > 1 && ridge === 0} onClick={() => { setRidge(0); changeDegree(degree > 1 ? degree : 3); }}>Polynomial</AlgoPill>
            <AlgoPill active={ridge > 0} onClick={() => { if (degree < 2) changeDegree(6); setRidge(ridge > 0 ? ridge : 0.05); setPresetId(undefined); }}>Ridge (L2)</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="FIT" items={[
          { color: 'var(--t1)', label: 'Data' },
          { color: ACCENT, label: 'ŷ = Σ wⱼxʲ + b' },
          { color: '#f87171', label: 'Residual' },
        ]} />
      )}
      rewardLabel="LOSS (MSE)"
      rewardValue={curJ.toFixed(4)}
      rewardSeries={loss}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Regression Parameters" hint="Tune α, degree and ridge; press Run to descend." />
          <PresetChips presets={PRESETS} activeId={presetId} onApply={applyPreset} />
          <ParamSlider name="α · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={(v) => { setAlpha(v); setPresetId(undefined); }} hint="gradient-descent step size" />
          <ParamSlider name="Polynomial degree" value={String(degree)} min={1} max={8} step={1} current={degree} onChange={changeDegree} hint="model flexibility (1 = line)" />
          <ParamSlider name="Ridge λ" value={ridge.toFixed(3)} min={0} max={0.2} step={0.005} current={ridge} onChange={(v) => { setRidge(v); setPresetId(undefined); }} hint="L2 weight-shrinkage penalty" />
          <ParamSlider name="Noise" value={noise.toFixed(3)} min={0} max={0.15} step={0.005} current={noise} onChange={(v) => { setNoise(v); regen(n, v); }} hint="scatter around the true curve" />
          <ParamSlider name="Points" value={String(n)} min={10} max={80} step={5} current={n} onChange={(v) => { setN(v); regen(v, noise); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="epoch interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Polynomial/Ridge Regression (GD)', alpha, degree, ridge, b: +b.toFixed(3), epoch }}
      apiPanel={apiPanel}
    />
  );
};

export default LinearRegressionLab;
