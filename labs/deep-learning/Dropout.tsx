import React, { useEffect, useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import FunctionPlot, { PlotSeries } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, Legend, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { dropoutPython } from './python';
import { createDropoutRun, stepDropoutRun, probAt, DROPOUT_DEFAULTS } from './dropoutSim';
import type { DropoutConfig, DropoutRun, Arm, Metrics } from './dropoutSim';

const ACCENT = '#f43f5e';
const EPOCHS_PER_TICK = 2;
const GAP_BAD = 0.15;          // chip: a train–val accuracy gap of 15 points or more is flagged
const GRID = 40;               // boundary-length measurement grid (GRID × GRID cells)

/** Number of adjacent cell pairs on a GRID×GRID lattice whose predicted class differs — a boundary-length proxy. */
function boundaryEdges(arm: Arm): number {
  const cls: number[] = [];
  for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) cls.push(probAt(arm.net, (i + 0.5) / GRID, (j + 0.5) / GRID) >= 0.5 ? 1 : 0);
  let e = 0;
  for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) {
    const c = cls[j * GRID + i];
    if (i + 1 < GRID && cls[j * GRID + i + 1] !== c) e++;
    if (j + 1 < GRID && cls[(j + 1) * GRID + i] !== c) e++;
  }
  return e;
}

const fmtPct = (v: number) => `${(v * 100).toFixed(0)}%`;
const last = (a: Arm): Metrics => a.history[a.history.length - 1]!;

const DropoutLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [perCluster, setPerCluster] = useState(DROPOUT_DEFAULTS.perCluster);
  const [dropoutRate, setDropoutRate] = useState(0.3);
  const [seed, setSeed] = useState(DROPOUT_DEFAULTS.seed);
  const [epoch, setEpoch] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [version, setVersion] = useState(0);
  const runRef = useRef<DropoutRun | null>(null);

  const cfg: DropoutConfig = useMemo(() => ({
    perCluster, noise: DROPOUT_DEFAULTS.noise, hidden: DROPOUT_DEFAULTS.hidden, lr: DROPOUT_DEFAULTS.lr, seed,
  }), [perCluster, seed]);

  // Rebuild both arms (same data, same initial weights) whenever the setup changes.
  useEffect(() => {
    runRef.current = createDropoutRun(cfg, dropoutRate);
    setEpoch(0);
    setLastLog(null);
    setVersion((v) => v + 1);
  }, [cfg, dropoutRate]);

  const run = runRef.current;

  const summary = (r: DropoutRun): string => {
    const a = last(r.plain), b = last(r.drop);
    const gapA = a.trainAcc - a.valAcc, gapB = b.trainAcc - b.valAcc;
    const minA = Math.min(...r.plain.history.map((h) => h.valLoss));
    const argA = r.plain.history.findIndex((h) => h.valLoss === minA);
    const eA = boundaryEdges(r.plain), eB = boundaryEdges(r.drop);
    const loss = b.valLoss < a.valLoss
      ? `Validation loss ends at ${a.valLoss.toFixed(2)} without dropout versus ${b.valLoss.toFixed(2)} with p = ${r.drop.p}.`
      : `This time dropout did not lower the validation loss (${a.valLoss.toFixed(2)} without, ${b.valLoss.toFixed(2)} with p = ${r.drop.p}).`;
    const rise = a.valLoss - minA > 0.05
      ? ` The no-dropout net's validation loss bottomed out at ${minA.toFixed(2)} around epoch ${argA} and has climbed since, while its training loss kept falling to ${a.trainLoss.toFixed(2)} — it is memorising the training points.`
      : '';
    const gap = gapB < gapA
      ? ` The train–validation accuracy gap is ${fmtPct(gapA)} without dropout and ${fmtPct(gapB)} with it.`
      : ` The accuracy gap did not shrink on this data (${fmtPct(gapA)} without, ${fmtPct(gapB)} with) — with only ${r.data.Nva} validation points, accuracy is a noisy measure; the loss is the steadier signal.`;
    const edges = ` The no-dropout decision boundary is ${eA > eB ? `longer and more contorted (${eA} vs ${eB} boundary cells on a ${GRID}×${GRID} grid)` : `not longer this time (${eA} vs ${eB} boundary cells)`}.`;
    return `${loss}${rise}${gap}${edges}`;
  };

  const step = () => {
    const r = runRef.current;
    if (!r) return;
    if (r.epoch >= DROPOUT_DEFAULTS.maxEpochs) { sim.pause(); return; }
    for (let k = 0; k < EPOCHS_PER_TICK && r.epoch < DROPOUT_DEFAULTS.maxEpochs; k++) stepDropoutRun(r);
    setEpoch(r.epoch);
    const a = last(r.plain), b = last(r.drop);

    narration.narratePhase(
      `run:${r.drop.p}`,
      `The challenge here: fit noisy, overlapping data without memorising the noise. Two copies of the same 2-64-64-1 network, with the same starting weights, train on the same points: the left one plainly, the right one with dropout at rate ${r.drop.p} — every training step zeroes each hidden unit with that probability and scales the survivors up by one over one minus p, so no unit can rely on particular partners. At evaluation dropout is off and every unit is used. Watch the loss curves: solid lines are validation loss, dashed are training loss.`,
    );
    if (r.epoch >= DROPOUT_DEFAULTS.maxEpochs) {
      sim.pause();
      narration.narratePhase(`done:${r.drop.p}`, `Training is complete after ${r.epoch} epochs. ${summary(r)}`);
    }

    setLastLog({
      algorithm: 'Dropout',
      stepDescription: `Epoch ${r.epoch}: one full-batch Adam step for each net; the right net samples a fresh Bernoulli mask per hidden unit and example (p = ${r.drop.p})`,
      formula: 'train: h = ReLU(Wx + b) ⊙ m,  m ∈ {0, 1/(1−p)};   eval: h = ReLU(Wx + b)',
      variables: {
        epoch: r.epoch, p: r.drop.p,
        'val loss p=0': +a.valLoss.toFixed(3), [`val loss p=${r.drop.p}`]: +b.valLoss.toFixed(3),
        'gap p=0': +(a.trainAcc - a.valAcc).toFixed(3), [`gap p=${r.drop.p}`]: +(b.trainAcc - b.valAcc).toFixed(3),
      },
      result: `p=0: train ${fmtPct(a.trainAcc)} · val ${fmtPct(a.valAcc)} · val loss ${a.valLoss.toFixed(3)}  |  p=${r.drop.p}: train ${fmtPct(b.trainAcc)} · val ${fmtPct(b.valAcc)} · val loss ${b.valLoss.toFixed(3)}`,
      mathDetails: {
        params: [
          { label: 'inverted dropout', info: `During training each hidden unit is kept with probability 1−p = ${(1 - r.drop.p).toFixed(2)} and zeroed otherwise; survivors are scaled by 1/(1−p) so the expected activation is unchanged. At evaluation (decision fields, accuracies, losses shown) nothing is dropped or rescaled.` },
          { label: 'setup', info: `Both nets: 2 → 64 → 64 → 1 (ReLU, sigmoid output, binary cross-entropy), identical seeded initial weights, full-batch Adam (η = ${DROPOUT_DEFAULTS.lr}). Data: 4 XOR clusters × ${perCluster} points, std ${DROPOUT_DEFAULTS.noise} (overlapping), ${r.data.Ntr} train / ${r.data.Nva} validation (seed ${seed}).` },
          { label: 'ensemble view', info: 'Each step trains a different thinned sub-network that shares weights with the rest; the full net at evaluation behaves like an average over those sub-networks.' },
          { label: 'train–val gap', info: `p=0: ${fmtPct(a.trainAcc - a.valAcc)} · p=${r.drop.p}: ${fmtPct(b.trainAcc - b.valAcc)} (flagged at ${GAP_BAD * 100}%). Validation loss is the steadier overfitting signal on ${r.data.Nva} validation points.` },
        ],
        implication: b.valLoss < a.valLoss
          ? `At epoch ${r.epoch} dropout's validation loss is lower (${b.valLoss.toFixed(3)} vs ${a.valLoss.toFixed(3)}).`
          : `At epoch ${r.epoch} dropout's validation loss is not lower yet (${b.valLoss.toFixed(3)} vs ${a.valLoss.toFixed(3)}).`,
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 40 });

  const halt = () => { sim.stop(); narration.cancel(); };
  const resetTraining = () => {
    halt();
    runRef.current = createDropoutRun(cfg, dropoutRate);
    setEpoch(0);
    setLastLog(null);
    setVersion((v) => v + 1);
  };

  // ---- visuals ------------------------------------------------------------
  const points: ScatterPoint[] = run
    ? [
      ...Array.from({ length: run.data.Nva }, (_, i) => ({ x: run.data.Xva[2 * i]!, y: run.data.Xva[2 * i + 1]!, cls: run.data.Yva[i]!, faint: true })),
      ...Array.from({ length: run.data.Ntr }, (_, i) => ({ x: run.data.Xtr[2 * i]!, y: run.data.Xtr[2 * i + 1]!, cls: run.data.Ytr[i]! })),
    ]
    : [];
  const fieldKey = `${version}-${epoch}`;
  const hist = (a: Arm | undefined, key: 'trainLoss' | 'valLoss') => (a ? a.history.map((h, i) => ({ x: i, y: h[key] })) : []);
  const maxLoss = run ? Math.max(0.8, ...run.plain.history.map((h) => h.valLoss), ...run.drop.history.map((h) => h.valLoss)) : 1;
  const lossTop = Math.ceil(maxLoss * 5) / 5;
  const lossSeries: PlotSeries[] = [
    { points: hist(run?.plain, 'trainLoss'), color: BAD, width: 1.4, dash: true },
    { points: hist(run?.plain, 'valLoss'), color: BAD, width: 2.2 },
    { points: hist(run?.drop, 'trainLoss'), color: GOOD, width: 1.4, dash: true },
    { points: hist(run?.drop, 'valLoss'), color: GOOD, width: 2.2 },
  ];
  const a = run ? last(run.plain) : null;
  const b = run ? last(run.drop) : null;
  const gapA = a ? a.trainAcc - a.valAcc : 0;
  const gapB = b ? b.trainAcc - b.valAcc : 0;
  const caption = (txt: string, color: string) => (
    <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color, margin: '0 0 4px 4px' }}>{txt}</div>
  );

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'EPOCH', value: `${epoch}/${DROPOUT_DEFAULTS.maxEpochs}` },
        { label: 'GAP p=0', value: a ? fmtPct(gapA) : '—', color: a && gapA >= GAP_BAD ? BAD : undefined },
        { label: `GAP p=${dropoutRate}`, value: b ? fmtPct(gapB) : '—', color: b && gapB >= GAP_BAD ? BAD : undefined },
        { label: 'VAL LOSS 0 | p', value: a && b ? `${a.valLoss.toFixed(2)} | ${b.valLoss.toFixed(2)}` : '—', color: a && b ? (b.valLoss < a.valLoss ? GOOD : BAD) : undefined },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, dropoutPython({ p: dropoutRate, perCluster, seed }))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: 12 }}>
            <div>
              {caption(`p = 0 · no dropout · val ${a ? fmtPct(a.valAcc) : '—'}`, 'var(--bad)')}
              <ScatterPlot width={292} height={262} points={points} fieldKey={`${fieldKey}-a`} fieldResolution={30}
                classify={run ? (x, y) => (probAt(run.plain.net, x, y) >= 0.5 ? 1 : 0) : undefined} xLabel="x₁" yLabel="x₂" />
            </div>
            <div>
              {caption(`p = ${dropoutRate} · dropout · val ${b ? fmtPct(b.valAcc) : '—'}`, 'var(--good)')}
              <ScatterPlot width={292} height={262} points={points} fieldKey={`${fieldKey}-b`} fieldResolution={30}
                classify={run ? (x, y) => (probAt(run.drop.net, x, y) >= 0.5 ? 1 : 0) : undefined} xLabel="x₁" yLabel="x₂" />
            </div>
          </div>
          <FunctionPlot width={596} height={196} series={lossSeries} domain={[0, DROPOUT_DEFAULTS.maxEpochs]} range={[0, lossTop]} xLabel="epoch" yLabel="BCE loss" />
        </div>
      )}
      legend={<Legend title="LOSS CURVES" items={[{ color: BAD, label: 'p = 0' }, { color: GOOD, label: `p = ${dropoutRate}` }, { color: 'var(--t2)', label: 'solid val · dashed train · faint dots = val points' }]} />}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={resetTraining} onNewMap={() => { halt(); setSeed((s) => s + 1); }} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel={`VAL LOSS · p = ${dropoutRate}`}
      rewardValue={b ? b.valLoss.toFixed(3) : '—'}
      rewardSeries={run ? run.drop.history.map((h) => h.valLoss) : undefined}
      lastLog={lastLog}
      contextInsight={`The same 2-64-64-1 network, from the same starting weights, trained on the same noisy XOR points — left without dropout, right with inverted dropout at p = ${dropoutRate} on both hidden layers. Dots are training points (faint = validation). On the default data the no-dropout net's validation loss turns back up while its training loss keeps falling (memorisation), and its boundary grows more contorted; dropout keeps the validation loss lower. Accuracy on ${run ? run.data.Nva : 'a few dozen'} validation points is noisy — read the loss curves.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Dropout" hint="Run = full-batch epochs for both nets; dropout acts during training only." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Right-hand net: dropout rate</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={dropoutRate === 0.3} accent={ACCENT} onClick={() => { halt(); setDropoutRate(0.3); }}>p = 0.3</AlgoPill>
              <AlgoPill active={dropoutRate === 0.5} accent={ACCENT} onClick={() => { halt(); setDropoutRate(0.5); }}>p = 0.5</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              The left net always trains with p = 0, from the same initial weights on the same data.
            </p>
          </div>
          <ParamSlider name="Dropout rate p" value={dropoutRate.toFixed(2)} min={0.05} max={0.6} step={0.05} current={dropoutRate} onChange={(v) => { halt(); setDropoutRate(Math.round(v * 100) / 100); }} hint="fraction of hidden units zeroed per training step" />
          <ParamSlider name="Points / cluster" value={String(perCluster)} min={16} max={40} step={2} current={perCluster} onChange={(v) => { halt(); setPerCluster(v); }} hint="dataset size (60% train / 40% validation)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint={`interval per ${EPOCHS_PER_TICK} epochs`} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Dropout (side by side)', dropoutRate, network: '2-64-64-1 ReLU', seed, perCluster, epoch, noDropout: a ? { trainAcc: +a.trainAcc.toFixed(3), valAcc: +a.valAcc.toFixed(3), valLoss: +a.valLoss.toFixed(3) } : null, withDropout: b ? { trainAcc: +b.trainAcc.toFixed(3), valAcc: +b.valAcc.toFixed(3), valLoss: +b.valLoss.toFixed(3) } : null }}
      apiPanel={apiPanel}
    />
  );
};

export default DropoutLab;
