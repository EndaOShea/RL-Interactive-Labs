import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from './shared';
import { mulberry32 } from './rng';
import { makeRegData, gdStep, closedForm, objective, halfMse, hessianEigs, truthGap, truth, predict } from './linregCore';
import type { Model } from './linregCore';
import { linregPython } from './python';
import { PresetChips, Preset } from './presets';
import { useTheme } from '../../utils/theme';

const ACCENT = '#34d399';
const RESID = 'rgba(248,113,113,.5)';
const N_TEST = 60;
const DIVERGED = 1e6;

type Status = 'run' | 'converged' | 'diverged';
const zeroModel = (d: number): Model => ({ w: new Array(d).fill(0), b: 0 });

interface Cfg { degree: number; ridge: number; alpha: number; n: number; noise: number; perTick: number; }
// Every preset uses data seed 1; the numbers in the hints were measured on it and on 80 other seeds.
const PRESETS: Preset<Cfg>[] = [
  { id: 'line', label: 'Straight line', hint: 'Degree 1 cannot bend: J settles far above the noise floor ½σ², and most of it is bias.', values: { degree: 1, ridge: 0, alpha: 0.5, n: 45, noise: 0.1, perTick: 1 } },
  { id: 'cubic', label: 'Cubic fit', hint: 'Degree 3 matches the true curve: GD converges in ~30 epochs to a curve within ≈0.001 of the truth, and held-out test J lands on the noise floor ½σ².', values: { degree: 3, ridge: 0, alpha: 1.0, n: 45, noise: 0.1, perTick: 1 } },
  { id: 'overfit', label: 'Overfit (deg 12)', hint: '15 points, degree 12: train J drops below the noise floor while test J keeps rising — the curve is fitting the noise.', values: { degree: 12, ridge: 0, alpha: 1.5, n: 15, noise: 0.1, perTick: 20 } },
  { id: 'ridge', label: 'Ridge-tamed', hint: 'Same 15 points and degree 12 with λ = 0.01: the penalty keeps the curve smooth and test J near 2× the floor.', values: { degree: 12, ridge: 0.01, alpha: 1.5, n: 15, noise: 0.1, perTick: 20 } },
  { id: 'diverge', label: 'α too large', hint: 'α = 2.4 is above this data\'s stability limit 2/λmax (≤ 2 here): every step overshoots and the loss explodes.', values: { degree: 3, ridge: 0, alpha: 2.4, n: 45, noise: 0.1, perTick: 1 } },
];

const LinearRegressionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [n, setN] = useState(45);
  const [noise, setNoise] = useState(0.1);
  const [seed, setSeed] = useState(1);
  const [alpha, setAlpha] = useState(0.5);
  const [degree, setDegree] = useState(1);
  const [ridge, setRidge] = useState(0);
  const [perTick, setPerTick] = useState(1);
  const [model, setModel] = useState<Model>(() => zeroModel(1));
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState<number[]>([]);
  const [status, setStatus] = useState<Status>('run');
  const [presetId, setPresetId] = useState<string | undefined>();
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const data = useMemo(() => makeRegData(mulberry32(seed), n, N_TEST, noise), [seed, n, noise]);
  const { train, test } = data;
  const cf = useMemo(() => closedForm(train, degree, ridge), [train, degree, ridge]);
  const objStar = useMemo(() => objective(train, cf, ridge), [train, cf, ridge]);
  const eig = useMemo(() => hessianEigs(train, degree, ridge), [train, degree, ridge]);
  const bound = 2 / eig.max;
  const floor = 0.5 * noise * noise;
  const trainJ = halfMse(train, model);
  const testJ = halfMse(test, model);
  const finite = Number.isFinite(trainJ) && trainJ < DIVERGED;

  const describeFit = (m: Model) => {
    const tJ = halfMse(train, m), vJ = halfMse(test, m), gap = truthGap(m);
    if (tJ < 0.8 * floor && vJ > 1.5 * floor) {
      return `Train J is ${tJ.toFixed(4)}, BELOW the noise floor one half sigma squared of ${floor.toFixed(4)}: the curve is fitting the noise. Held-out test J is ${vJ.toFixed(4)}, which is the real verdict — this is overfitting.`;
    }
    if (gap > Math.max(floor, 1e-5)) {
      return `Train J is ${tJ.toFixed(4)}. Noise alone would give about ${floor.toFixed(4)}; the fitted curve is on average ${gap.toFixed(4)} away from the true curve, in the same half-mean-square units. That gap is bias: degree ${degree} cannot bend enough. Test J is ${vJ.toFixed(4)}.`;
    }
    return `Train J is ${tJ.toFixed(4)} and test J ${vJ.toFixed(4)}, both close to the noise floor of ${floor.toFixed(4)}; the curve is only ${gap.toFixed(4)} from the true one. The model has captured the signal and what remains is irreducible noise.`;
  };

  const step = () => {
    if (status !== 'run' || !train.length) { sim.pause(); return; }
    let m = model;
    let e = epoch;
    let st: ReturnType<typeof gdStep> | null = null;
    const Js: number[] = [];
    let nextStatus: Status = 'run';
    for (let t = 0; t < Math.max(1, perTick); t++) {
      st = gdStep(train, m, alpha, ridge);
      m = st.next; e += 1; Js.push(st.J);
      const obj = objective(train, m, ridge);
      if (!Number.isFinite(obj) || obj > DIVERGED) { nextStatus = 'diverged'; break; }
      if (obj - objStar < 1e-6) { nextStatus = 'converged'; break; }
    }
    setModel(m); setEpoch(e);
    setLoss((L) => [...L, ...Js].slice(-120));
    setStatus(nextStatus);
    const modelWord = degree === 1 ? 'a straight line' : `a degree ${degree} polynomial`;
    const intro = ridge > 0
      ? `The challenge here: fit ${modelWord} to the points without letting a flexible curve chase the noise. Gradient descent minimises half the mean squared error plus half lambda times the squared weights, so each step follows the data gradient and also shrinks every weight toward zero — weight decay. The features are Legendre polynomials on minus one to one, which keeps the loss surface well conditioned; a high-degree Legendre term costs more weight per unit of wiggle, so the penalty prefers smooth curves. Watch the train and held-out test losses in the header. Regularised regression like this underpins forecasting, pricing models and genomics.`
      : `The challenge here: find the curve that best predicts y from x. We fit ${modelWord} by gradient descent on half the mean squared error: every epoch moves the weights a step alpha against the gradient. The features are Legendre polynomials on minus one to one, which keeps the loss surface well conditioned, and the dashed curve is the exact least-squares answer gradient descent is heading for. Watch the red residuals shrink, and compare the training loss with the held-out test loss. Regression like this is the workhorse of forecasting and trend estimation.`;
    narration.narratePhase(`run:${degree}:${ridge > 0}`, intro);
    if (nextStatus === 'diverged') {
      sim.pause();
      narration.narratePhase(`diverge:${alpha}`, `The loss is exploding. For this data gradient descent is only stable when alpha is below two over the largest eigenvalue of the loss curvature, which is ${bound.toFixed(3)}; alpha is ${alpha}, so every step overshoots the minimum and lands higher on the far wall. Lower alpha below ${bound.toFixed(2)}.`);
    } else if (nextStatus === 'converged') {
      sim.pause();
      narration.narratePhase(`done:${degree}:${ridge}:${seed}`, `Gradient descent has reached the closed-form optimum after ${e} epochs. ${describeFit(m)}`);
    }
    const J = Js[Js.length - 1] ?? 0;
    const dw1 = st?.dw[0] ?? 0;
    setLastLog({
      algorithm: `${degree > 1 ? `Polynomial (deg ${degree})` : 'Linear'} Regression · GD${ridge > 0 ? ' · Ridge' : ''}`,
      stepDescription: `Epoch${Js.length > 1 ? `s ${epoch + 1}–${e}` : ` ${e}`} — step the weights downhill`,
      formula: ridge > 0 ? 'w ← w − α(∇J + λw),  b ← b − α·∂J/∂b,  ŷ = b + Σⱼ wⱼPⱼ(x)' : 'θ ← θ − α∇J,  J = ½·mean((ŷ − y)²),  ŷ = b + Σⱼ wⱼPⱼ(x)',
      variables: { 'J (train)': +J.toFixed(5), 'w₁': +(m.w[0] ?? 0).toFixed(4), 'b': +m.b.toFixed(4), '∂J/∂w₁': +dw1.toFixed(4), 'λ': ridge, 'α': alpha },
      result: nextStatus === 'diverged' ? `diverged — α > 2/λmax = ${bound.toFixed(3)}` : `J = ${J.toFixed(5)}${nextStatus === 'converged' ? ' · converged' : ''}`,
      mathDetails: {
        params: [
          { label: 'α', info: `${alpha}. Step size. Stable only when α < 2/λmax = ${bound.toFixed(3)} for this data (λmax = largest eigenvalue of the loss curvature).` },
          { label: 'degree', info: `${degree}. Legendre polynomials P₁..P_${degree} on x ∈ [−1, 1] — the same curves as x, x², …, but orthogonal, so GD converges fast.` },
          { label: 'λ', info: ridge > 0 ? `${ridge}. Ridge penalty ½λ‖w‖² (bias not penalised): its gradient λw shrinks every weight toward 0 each step.` : 'No regularisation — weights are unconstrained.' },
          { label: 'J', info: `${J.toFixed(5)} = ½·mean((ŷ−y)²) on the ${train.length} training points; noise floor ½σ² = ${floor.toFixed(5)}; closed-form optimum ${halfMse(train, cf).toFixed(5)}.` },
        ],
        implication: nextStatus === 'diverged' ? 'α is past the stability limit — the loss grows every step.' : nextStatus === 'converged' ? 'At the optimum — further steps change nothing.' : 'Loss is decreasing toward the closed-form optimum.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const resetWeights = (deg: number) => { setModel(zeroModel(deg)); setEpoch(0); setLoss([]); setStatus('run'); setLastLog(null); narration.cancel(); };
  const reset = () => { sim.stop(); resetWeights(degree); };
  const regen = () => { sim.stop(); setSeed((s) => s + 1); resetWeights(degree); };
  const changeDegree = (d: number) => { sim.stop(); setDegree(d); resetWeights(d); setPresetId(undefined); };
  const applyPreset = (p: Preset<Cfg>) => {
    sim.stop();
    const v = p.values;
    setDegree(v.degree); setRidge(v.ridge); setAlpha(v.alpha); setN(v.n); setNoise(v.noise); setPerTick(v.perTick); setSeed(1);
    resetWeights(v.degree); setPresetId(p.id);
    narration.narratePhase(`preset:${p.id}`, p.hint);
  };

  const grid = Array.from({ length: 121 }, (_, i) => -1 + (2 * i) / 120);
  const fitCurve = grid.map((x) => ({ x, y: finite ? predict(x, model) : NaN }));
  const cfCurve = grid.map((x) => ({ x, y: predict(x, cf) }));
  const truthCurve = grid.map((x) => ({ x, y: truth(x) }));
  const residuals = finite ? train.map((p) => ({ points: [{ x: p.x, y: p.y }, { x: p.x, y: predict(p.x, model) }], color: RESID, width: 1 })) : [];

  const insight = `α = ${alpha} (stable below 2/λmax = ${bound.toFixed(3)}), degree ${degree}${ridge > 0 ? `, ridge λ = ${ridge}` : ''}, ${train.length} training / ${test.length} held-out points, noise σ = ${noise} → floor ½σ² = ${floor.toFixed(4)}. ` +
    (finite ? `Now: train J ${trainJ.toFixed(4)}, test J ${testJ.toFixed(4)}, closed-form train J ${halfMse(train, cf).toFixed(4)} / test J ${halfMse(test, cf).toFixed(4)}. ` : 'The run diverged. ') +
    'Dashed: the closed-form optimum GD is heading for; faint: the true curve; red sticks: residuals.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'EPOCH', value: epoch },
        { label: 'TRAIN J', value: finite ? trainJ.toFixed(4) : '∞', color: GOOD },
        { label: 'TEST J', value: finite ? testJ.toFixed(4) : '∞', color: finite && testJ > 1.5 * floor ? BAD : undefined },
        { label: '½σ²', value: floor.toFixed(4) },
        { label: '2/λmax', value: bound.toFixed(2), color: alpha >= bound ? BAD : undefined },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, linregPython({ train, test, degree, alpha, ridge, noise, labEpochs: epoch }))}
      grid={(
        <FunctionPlot
          domain={[-1, 1]}
          range={[-2, 1.5]}
          scatter={[
            ...test.map((p) => ({ x: p.x, y: p.y, color: 'var(--t2)', r: 2.2 })),
            ...train.map((p) => ({ x: p.x, y: p.y, color: 'var(--t1)' })),
          ]}
          series={[
            { points: truthCurve, color: 'var(--t2)', width: 1.2 },
            ...residuals,
            { points: cfCurve, color: isLight ? 'var(--good)' : ACCENT, width: 1.4, dash: true },
            { points: fitCurve, color: isLight ? 'var(--good)' : ACCENT, width: 2.6 },
          ]}
          xLabel="x"
          yLabel="y"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Model</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={degree === 1} onClick={() => { setRidge(0); changeDegree(1); }}>Linear · deg 1</AlgoPill>
            <AlgoPill active={degree > 1 && ridge === 0} onClick={() => { setRidge(0); changeDegree(degree > 1 ? degree : 3); }}>Polynomial</AlgoPill>
            <AlgoPill active={ridge > 0} onClick={() => { if (degree < 2) changeDegree(12); setRidge(ridge > 0 ? ridge : 0.01); setPresetId(undefined); }}>Ridge (L2)</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={regen} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="FIT" items={[
          { color: 'var(--t1)', label: 'Train' },
          { color: 'var(--t2)', label: 'Held-out test' },
          { color: ACCENT, label: 'GD fit ŷ = b + Σ wⱼPⱼ(x)' },
          { node: <span style={{ width: 12, height: 0, borderTop: `2px dashed ${ACCENT}`, display: 'inline-block' }} />, label: 'Closed-form optimum' },
          { color: '#f87171', label: 'Residual' },
        ]} />
      )}
      rewardLabel="TRAIN LOSS J = ½·MSE"
      rewardValue={finite ? trainJ.toFixed(4) : '∞'}
      rewardSeries={loss}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Regression Parameters" hint="Tune α, degree and ridge; press Run to descend." />
          <PresetChips presets={PRESETS} activeId={presetId} onApply={applyPreset} />
          <ParamSlider name="α · learning rate" value={alpha.toFixed(2)} min={0.01} max={2.5} step={0.01} current={alpha} onChange={(v) => { setAlpha(v); setPresetId(undefined); if (status === 'diverged') resetWeights(degree); }} hint={`step size · stable below 2/λmax = ${bound.toFixed(2)}`} />
          <ParamSlider name="Polynomial degree" value={String(degree)} min={1} max={12} step={1} current={degree} onChange={changeDegree} hint="Legendre P₁…P_d (1 = line)" />
          <ParamSlider name="Ridge λ" value={ridge.toFixed(3)} min={0} max={0.2} step={0.001} current={ridge} onChange={(v) => { setRidge(v); setPresetId(undefined); if (status !== 'run') resetWeights(degree); }} hint="penalty ½λ‖w‖²" />
          <ParamSlider name="Noise σ" value={noise.toFixed(2)} min={0} max={0.3} step={0.01} current={noise} onChange={(v) => { sim.stop(); setNoise(v); resetWeights(degree); setPresetId(undefined); }} hint="scatter around the true curve" />
          <ParamSlider name="Training points" value={String(n)} min={10} max={80} step={5} current={n} onChange={(v) => { sim.stop(); setN(v); resetWeights(degree); setPresetId(undefined); }} hint={`+ ${N_TEST} held-out test points`} />
          <ParamSlider name="Epochs per tick" value={String(perTick)} min={1} max={50} step={1} current={perTick} onChange={setPerTick} hint="fast-forward gradient descent" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="tick interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Polynomial/Ridge Regression (GD, Legendre basis)', alpha, stableAlphaBelow: +bound.toFixed(3), degree, ridge, noise, trainPoints: n, epoch, trainJ: finite ? +trainJ.toFixed(5) : 'diverged', testJ: finite ? +testJ.toFixed(5) : 'diverged' }}
      apiPanel={apiPanel}
    />
  );
};

export default LinearRegressionLab;
