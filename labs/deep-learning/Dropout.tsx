import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { clamp01, randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { dropoutPython } from './python';

const ACCENT = '#f43f5e';
const MAX_EPOCHS = 150;
const H = 48;            // fixed random hidden ReLU features
const LR = 0.5;          // readout learning rate

// XOR-style 4 clusters, deliberately NOISY so a flexible model can overfit and a
// jagged boundary forms with no dropout. Classes [0,1,1,0] make it non-linear.
const CENTERS = [{ x: 0.3, y: 0.32 }, { x: 0.7, y: 0.32 }, { x: 0.3, y: 0.7 }, { x: 0.7, y: 0.7 }];
const CLS = [0, 1, 1, 0];

interface DPt { x: number; y: number; y01: number; train: boolean; }
const makeData = (perCluster: number): DPt[] =>
  CENTERS.flatMap((c, ci) =>
    Array.from({ length: perCluster }, () => ({
      x: clamp01(c.x + randn() * 0.11),
      y: clamp01(c.y + randn() * 0.11),
      y01: CLS[ci],
      train: Math.random() < 0.6,   // ~60% train / 40% validation
    })),
  );

// Fixed random hidden layer: h_j(x,y) = max(0, a_j·x + b_j·y + c_j).
interface Feat { a: number; b: number; c: number; }
const makeFeatures = (): Feat[] =>
  Array.from({ length: H }, () => ({ a: randn() * 2.5, b: randn() * 2.5, c: randn() }));

const hidden = (feats: Feat[], x: number, y: number): number[] =>
  feats.map((f) => Math.max(0, f.a * x + f.b * y + f.c));

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

const DropoutLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [perCluster, setPerCluster] = useState(28);
  const [dropoutRate, setDropoutRate] = useState(0.3);   // p ∈ [0, 0.6]
  const [data, setData] = useState<DPt[]>(() => makeData(28));

  // Fixed random features (regenerated only on "New data"); trainable readout.
  const featsRef = useRef<Feat[]>(makeFeatures());
  const [weights, setWeights] = useState<number[]>(() => new Array(H).fill(0));
  const [bias, setBias] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [valSeries, setValSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [version, setVersion] = useState(0);

  // EVAL probability — all features, NO dropout mask (inference behaviour).
  const probAt = (w: number[], b: number, x: number, y: number): number => {
    const h = hidden(featsRef.current, x, y);
    let z = b;
    for (let j = 0; j < H; j++) z += w[j] * h[j];
    return sigmoid(z);
  };

  const accuracy = (w: number[], b: number, train: boolean): number => {
    const pts = data.filter((p) => p.train === train);
    if (!pts.length) return 0;
    let ok = 0;
    for (const p of pts) if ((probAt(w, b, p.x, p.y) >= 0.5 ? 1 : 0) === p.y01) ok++;
    return ok / pts.length;
  };

  const metrics = useMemo(() => {
    const tr = accuracy(weights, bias, true);
    const va = accuracy(weights, bias, false);
    return { tr, va, gap: tr - va };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [weights, bias, data, version]);

  const step = () => {
    if (epoch >= MAX_EPOCHS) { sim.pause(); narrateDone(); return; }
    const feats = featsRef.current;
    const keep = 1 - dropoutRate;
    const w = weights.slice();
    let b = bias;
    const train = data.filter((p) => p.train);

    // One epoch of logistic-loss gradient descent over the training split, with
    // inverted dropout applied to the hidden features on each forward pass.
    for (const p of train) {
      const h = hidden(feats, p.x, p.y);
      // Per-example Bernoulli(keep) mask / keep (inverted dropout). p=0 → identity.
      const hd = h.map((hj) => (dropoutRate > 0 && Math.random() >= keep ? 0 : hj / keep));
      let z = b;
      for (let j = 0; j < H; j++) z += w[j] * hd[j];
      const err = sigmoid(z) - p.y01;        // ∂(logistic loss)/∂z
      for (let j = 0; j < H; j++) w[j] -= LR * err * hd[j] / train.length;
      b -= LR * err / train.length;
    }

    setWeights(w);
    setBias(b);
    const e = epoch + 1;
    setEpoch(e);

    const tr = accuracy(w, b, true);
    const va = accuracy(w, b, false);
    const gap = tr - va;
    setValSeries((s) => [...s, va].slice(-60));

    narration.narratePhase(
      `run:${dropoutRate > 0 ? 'dropout' : 'plain'}`,
      dropoutRate > 0
        ? `The challenge here: fit this noisy, non-linear data without memorising the noise — without the boundary contorting around individual points. Dropout randomly zeroes hidden units on every training step, so the readout can't rely on any single feature. That is like averaging a huge ensemble of thinned sub-networks, and it pushes the decision boundary smoother. At test time dropout is switched off and all units are used. Dropout is a staple regulariser in deep nets for vision, speech and language.`
        : `The challenge here: fit this noisy, non-linear data without memorising the noise — without the boundary contorting around individual points. With dropout at zero the network trains every unit on every step, so it is free to memorise individual noisy points and the boundary becomes jagged, overfitting the training set. Watch the gap between training and validation accuracy: turn dropout up and that gap should shrink. Dropout is a staple regulariser in deep nets for vision, speech and language.`,
    );

    setLastLog({
      algorithm: 'Dropout',
      stepDescription: `Epoch ${e}: forward pass with per-unit Bernoulli mask (p=${dropoutRate}), logistic-loss gradient step on the readout`,
      formula: 'hᵢ ← hᵢ · Bernoulli(1−p)/(1−p)   (train only)',
      variables: { epoch: e, p: dropoutRate, 'train acc': +tr.toFixed(3), 'val acc': +va.toFixed(3), gap: +gap.toFixed(3) },
      result: `train ${(tr * 100).toFixed(0)}% · val ${(va * 100).toFixed(0)}% · gap ${(gap * 100).toFixed(0)}%`,
      mathDetails: {
        params: [
          { label: 'dropout rate p', info: `${dropoutRate}. Each hidden unit is kept with probability 1−p and zeroed otherwise, then surviving units are scaled by 1/(1−p) so the expected activation is unchanged (inverted dropout).` },
          { label: 'ensemble view', info: 'Every training step samples a different thinned sub-network. Over many steps the readout learns weights that work across all of them — effectively averaging an exponential ensemble of networks, which regularises.' },
          { label: 'eval / inference', info: 'At evaluation (the decision field and validation accuracy here) dropout is disabled: all units are active and unscaled, giving the deterministic average prediction.' },
          { label: 'train–val gap', info: `${(gap * 100).toFixed(0)}%. A large gap signals overfitting. Increasing p shrinks the gap by stopping the network from depending on individual noisy points.` },
        ],
        implication: dropoutRate > 0
          ? 'Dropout active — the boundary stays smooth and the train–validation gap stays small.'
          : 'No dropout — the boundary can wrap around individual noisy points, widening the train–validation gap.',
      },
    });
  };

  const narrateDone = () => {
    narration.narratePhase(
      `done:${dropoutRate > 0 ? 'dropout' : 'plain'}`,
      dropoutRate > 0
        ? `Training has converged. With dropout on, the decision boundary stayed smooth and the gap between training and validation accuracy is small — the model generalised rather than memorised.`
        : `Training has converged. With no dropout the boundary hugged individual noisy points and training accuracy ran ahead of validation accuracy. Raise the dropout rate and run again to close that gap.`,
    );
  };

  const sim = useSimLoop(step, { initialSpeed: 80 });

  const resetTraining = () => {
    sim.stop();
    narration.cancel();
    setWeights(new Array(H).fill(0));
    setBias(0);
    setEpoch(0);
    setValSeries([]);
    setLastLog(null);
    setVersion((v) => v + 1);
  };

  const regen = (n = perCluster) => {
    sim.stop();
    narration.cancel();
    featsRef.current = makeFeatures();
    setData(makeData(n));
    setWeights(new Array(H).fill(0));
    setBias(0);
    setEpoch(0);
    setValSeries([]);
    setLastLog(null);
    setVersion((v) => v + 1);
  };

  const setPreset = (p: number) => { setDropoutRate(p); resetTraining(); };

  const fieldKey = `${epoch}-${dropoutRate}-${version}`;
  // Show TRAIN points (the set the model fits / can overfit).
  const plotPoints: ScatterPoint[] = data
    .filter((p) => p.train)
    .map((p) => ({ x: p.x, y: p.y, cls: p.y01 }));

  const gapColor = metrics.gap >= 0.18 ? BAD : 'var(--t0)';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'EPOCH', value: epoch },
        { label: 'TRAIN ACC', value: `${(metrics.tr * 100).toFixed(0)}%`, color: GOOD },
        { label: 'VAL ACC', value: `${(metrics.va * 100).toFixed(0)}%` },
        { label: 'GAP', value: `${(metrics.gap * 100).toFixed(0)}%`, color: gapColor },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, dropoutPython(dropoutRate, epoch))}
      grid={(
        <ScatterPlot
          width={440} height={440}
          points={plotPoints}
          classify={(x, y) => (probAt(weights, bias, x, y) >= 0.5 ? 1 : 0)}
          fieldKey={fieldKey}
          xLabel="x₁" yLabel="x₂"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={resetTraining} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel="VALIDATION ACC"
      rewardValue={`${(metrics.va * 100).toFixed(0)}%`}
      rewardSeries={valSeries}
      lastLog={lastLog}
      contextInsight={`A small classifier with ${H} fixed random ReLU features is trained on noisy XOR-style data. Dropout randomly zeroes hidden units during training so no single feature can dominate — smoothing the boundary and shrinking the train–validation accuracy gap. At p=0 the model overfits; turn p up and the gap closes.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Dropout" hint="Run = one training epoch; dropout active on train only." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={dropoutRate === 0} accent={ACCENT} onClick={() => setPreset(0)}>No dropout (p=0)</AlgoPill>
              <AlgoPill active={dropoutRate === 0.3} accent={ACCENT} onClick={() => setPreset(0.3)}>Dropout 0.3</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {dropoutRate > 0
                ? 'Units are randomly dropped each step → the boundary stays smooth and the train–val gap stays small.'
                : 'Every unit trains every step → the boundary can wrap noisy points and overfit (large train–val gap).'}
            </p>
          </div>
          <ParamSlider name="Dropout rate p" value={dropoutRate.toFixed(2)} min={0} max={0.6} step={0.05} current={dropoutRate} onChange={(v) => { resetTraining(); setDropoutRate(v); }} hint="fraction of units zeroed per train step" />
          <ParamSlider name="Points / cluster" value={String(perCluster)} min={16} max={40} step={2} current={perCluster} onChange={(v) => { setPerCluster(v); regen(v); }} hint="dataset size / noise density" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="epoch interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Dropout', dropoutRate, hiddenFeatures: H, epoch, trainAcc: +metrics.tr.toFixed(3), valAcc: +metrics.va.toFixed(3), gap: +metrics.gap.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default DropoutLab;
