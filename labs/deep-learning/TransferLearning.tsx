import React, { useEffect, useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotSeries, PlotMarker } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, Legend, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { transferPython } from './python';
import {
  pretrain, createExperiment, evalSweepPoint, TL_SWEEP, TL_DEFAULTS, TL_H, TL_TAGS, TL_EPOCHS, TL_LR, TL_TRIALS,
  TL_SOURCE_PER, TL_POOL_PER, TL_VAL_PER, FT_BACKBONE_LR_SCALE,
} from './transferSim';
import type { SweepPoint } from './transferSim';

const ACCENT = '#f43f5e';
const FROZEN_COLOR = 'var(--acc)';
type Mode = 'frozen' | 'finetune';
const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
const N_MAX = TL_SWEEP[TL_SWEEP.length - 1]!;

const TransferLearningLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [seed, setSeed] = useState(TL_DEFAULTS.seed);
  const [rotationDeg, setRotationDeg] = useState(TL_DEFAULTS.rotationDeg);
  const [nIndex, setNIndex] = useState(TL_DEFAULTS.nIndex);
  const [mode, setMode] = useState<Mode>('frozen');
  const [points, setPoints] = useState<SweepPoint[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // Computed sweep points live in a ref (the source of truth) so a slow step can
  // never be repeated by an overdue interval tick that still sees stale state.
  const pointsRef = useRef<SweepPoint[]>([]);

  // Pretraining depends on the seed only; the target data on the seed and θ.
  const pre = useMemo(() => pretrain(seed), [seed]);
  const exp = useMemo(() => createExperiment({ seed, rotationDeg }, pre), [pre, seed, rotationDeg]);
  useEffect(() => { pointsRef.current = []; setPoints([]); setLastLog(null); }, [exp]);

  const at = points[nIndex] ?? null;
  const n = TL_SWEEP[nIndex] ?? TL_SWEEP[0]!;
  const done = points.length === TL_SWEEP.length;
  const modeAcc = (p: SweepPoint) => (mode === 'frozen' ? p.frozen : p.finetune);
  const modeName = mode === 'frozen' ? 'frozen φ' : 'fine-tuned φ';

  const logFor = (p: SweepPoint): SimulationUpdate => ({
    algorithm: 'Transfer Learning',
    stepDescription: `n = ${p.n} labelled target points · ${TL_TRIALS} random subsets · ${TL_EPOCHS} full-batch Adam steps per learner`,
    formula: 'φ(x) = σ(W₂·ReLU(W₁x + b₁) + b₂) (pretrained tagger);  ŷ = σ(wᵀφ(x) + c)',
    variables: {
      n: p.n,
      'scratch acc': +p.scratch.toFixed(3),
      'frozen acc': +p.frozen.toFixed(3),
      'fine-tune acc': +p.finetune.toFixed(3),
      θ: rotationDeg,
    },
    result: `n=${p.n}: scratch ${pct(p.scratch)} · frozen ${pct(p.frozen)} · fine-tune ${pct(p.finetune)}`,
    mathDetails: {
      params: [
        { label: 'pretraining (source task)', info: `The backbone φ (2 → ${TL_H} ReLU → ${TL_TAGS} sigmoid tags) was trained with Adam on ${4 * TL_SOURCE_PER} labelled source points to answer "is it cluster k?" for each of the four clusters (${pct(exp.sourceAcc)} of source points get their own cluster's tag highest). Its four tag outputs are the features handed to the new task.` },
        { label: 'target task', info: `A different labelling — XOR of the clusters — on a domain rotated by θ = ${rotationDeg}° about the centre. n labelled points are drawn at random from a ${4 * TL_POOL_PER}-point pool; accuracy is measured on ${4 * TL_VAL_PER} separate validation points.` },
        { label: 'three learners', info: `Same architecture (2 → ${TL_H} → ${TL_TAGS} → 1) and the same ${TL_EPOCHS} Adam steps (η = ${TL_LR}): scratch = random He init, all layers train; frozen φ = pretrained backbone fixed, only the 5-parameter head trains; fine-tune = backbone trains at η × ${FT_BACKBONE_LR_SCALE}, head at η. Heads start at zero.` },
        { label: 'sample efficiency', info: `At n = ${p.n}: frozen ${pct(p.frozen)}, fine-tune ${pct(p.finetune)}, scratch ${pct(p.scratch)} (mean of ${TL_TRIALS} subsets, the same subsets for every learner).` },
      ],
      implication: Math.max(p.frozen, p.finetune) - p.scratch > 0.05
        ? `Transfer leads scratch by ${((Math.max(p.frozen, p.finetune) - p.scratch) * 100).toFixed(0)} points at n = ${p.n}.`
        : p.scratch - Math.max(p.frozen, p.finetune) > 0.02
          ? `At n = ${p.n} scratch is ahead — with enough labels (or a large domain shift) reuse stops paying off.`
          : `At n = ${p.n} the learners are within 5 points of each other.`,
    },
  });

  const summary = (pts: SweepPoint[]): string => {
    const small = pts[TL_DEFAULTS.nIndex] ?? pts[0]!;
    const big = pts[pts.length - 1]!;
    const lead = Math.max(small.frozen, small.finetune) - small.scratch;
    const s1 = lead > 0.05
      ? `With ${small.n} labels, frozen transfer reaches ${pct(small.frozen)} and fine-tuning ${pct(small.finetune)}, while the same network trained from scratch reaches ${pct(small.scratch)}.`
      : `With ${small.n} labels the three learners are close: frozen ${pct(small.frozen)}, fine-tune ${pct(small.finetune)}, scratch ${pct(small.scratch)}.`;
    const s2 = big.scratch >= big.finetune - 0.03
      ? ` With all ${big.n} labels, scratch reaches ${pct(big.scratch)} — it has caught up with fine-tuning (${pct(big.finetune)}).`
      : ` Even with all ${big.n} labels scratch reaches only ${pct(big.scratch)} against fine-tuning's ${pct(big.finetune)}.`;
    const s3 = big.finetune - big.frozen > 0.03
      ? ` The frozen backbone plateaus at ${pct(big.frozen)}: the target domain is rotated ${rotationDeg}° away from the pretraining data, so fixed features fit it imperfectly, and only fine-tuning can adapt them.`
      : ` Frozen and fine-tuned transfer end within a few points (${pct(big.frozen)} vs ${pct(big.finetune)}) — at a ${rotationDeg}° shift the pretrained features still fit.`;
    return s1 + s2 + s3;
  };

  const step = () => {
    const i = pointsRef.current.length;
    if (i >= TL_SWEEP.length) { sim.pause(); return; }
    if (i === 0) {
      narration.narratePhase('run', `The challenge here: get high accuracy on a new task from only a handful of labelled examples. A backbone pretrained on six hundred labelled points of a related task — tagging which of four clusters a point came from — already produces useful features, so a tiny head can learn the new labelling from few examples. Training the same network from scratch has to learn everything from those few points. For each label budget, the lab trains all three learners on the same random subsets and plots their real validation accuracy.`);
    }
    const p = evalSweepPoint(exp, i);
    const next = [...pointsRef.current, p];
    pointsRef.current = next;
    setPoints(next);
    setNIndex(i);
    setLastLog(logFor(p));
    if (next.length >= TL_SWEEP.length) {
      sim.pause();
      narration.narratePhase(`done:${seed}:${rotationDeg}`, `All three curves are complete. ${summary(next)}`);
    }
  };

  const sim = useSimLoop(step, { initialSpeed: 450 });

  const halt = () => { sim.stop(); narration.cancel(); };
  const reset = () => { halt(); pointsRef.current = []; setPoints([]); setLastLog(null); };
  const pickN = (idx: number) => { setNIndex(idx); const p = points[idx]; if (p) setLastLog(logFor(p)); };

  // ---- plot -------------------------------------------------------------
  const line = (key: 'scratch' | 'frozen' | 'finetune') => points.map((p) => ({ x: p.n, y: p[key] }));
  const series: PlotSeries[] = [
    { points: line('scratch'), color: BAD, width: 2.4 },
    { points: line('frozen'), color: FROZEN_COLOR, width: mode === 'frozen' ? 2.6 : 1.4, dash: mode !== 'frozen' },
    { points: line('finetune'), color: GOOD, width: mode === 'finetune' ? 2.6 : 1.4, dash: mode !== 'finetune' },
  ];
  const markers: PlotMarker[] = at ? [
    { x: at.n, y: modeAcc(at), color: mode === 'frozen' ? FROZEN_COLOR : GOOD, label: pct(modeAcc(at)) },
    { x: at.n, y: at.scratch, color: BAD, label: pct(at.scratch) },
  ] : [];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'LABELS n', value: n },
        { label: 'SCRATCH', value: at ? pct(at.scratch) : '—', color: BAD },
        { label: 'FROZEN φ', value: at ? pct(at.frozen) : '—', color: FROZEN_COLOR },
        { label: 'FINE-TUNE', value: at ? pct(at.finetune) : '—', color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, transferPython({ seed, rotationDeg, n, mode }))}
      grid={(
        <FunctionPlot
          width={460}
          height={440}
          domain={[0, N_MAX]}
          range={[0.4, 1]}
          series={series}
          markers={markers}
          xLabel="labelled target examples n"
          yLabel="validation accuracy"
        />
      )}
      legend={<Legend title="LEARNERS" items={[{ color: BAD, label: 'from scratch' }, { color: FROZEN_COLOR, label: 'frozen φ' }, { color: GOOD, label: 'fine-tune' }]} />}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={() => { if (done) { reset(); } sim.toggle(); }} onReset={reset} onNewMap={() => { halt(); setSeed((s) => s + 1); }} speed={sim.speed} onSpeed={sim.setSpeed} />}
      rewardLabel={`${mode === 'frozen' ? 'FROZEN' : 'FINE-TUNE'} VAL ACC`}
      rewardValue={at ? modeAcc(at).toFixed(2) : '—'}
      rewardSeries={points.map(modeAcc)}
      lastLog={lastLog}
      contextInsight={`Three learners with the same 2 → ${TL_H} → ${TL_TAGS} → 1 architecture face the same target task (XOR of four clusters, domain rotated ${rotationDeg}°) with the same n labels: from scratch (red) starts from random weights; frozen φ reuses a backbone pretrained on a related source task and trains only a 5-parameter head; fine-tune also updates the backbone at a tenth of the learning rate. Run trains all three at each n (${TL_TRIALS} random label subsets each) and plots real validation accuracy.${done ? ` ${summary(points)}` : ''}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Transfer Learning" hint="Run trains all three learners at each label budget, left to right." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Highlighted transfer mode</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={mode === 'frozen'} accent={ACCENT} onClick={() => setMode('frozen')}>Frozen φ</AlgoPill>
              <AlgoPill active={mode === 'finetune'} accent={ACCENT} onClick={() => setMode('finetune')}>Fine-tune</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {mode === 'frozen'
                ? 'Frozen: the pretrained backbone is fixed; only the head (4 weights + bias) trains — little to overfit with few labels.'
                : `Fine-tune: the backbone trains too, at ${FT_BACKBONE_LR_SCALE}× the head's learning rate, so it can adapt to the shifted target domain.`}
              {' '}All three curves are always computed on the same label subsets; this only changes the highlight.
            </p>
          </div>
          <ParamSlider name="Labelled examples n" value={String(n)} min={0} max={TL_SWEEP.length - 1} step={1} current={nIndex} onChange={(v) => pickN(Math.round(v))} hint={`label budget (${TL_SWEEP.join(', ')})`} />
          <ParamSlider name="Domain shift θ" value={`${rotationDeg}°`} min={0} max={45} step={5} current={rotationDeg} onChange={(v) => { halt(); setRotationDeg(v); }} hint="target clusters rotated from the pretraining data" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={150} max={1000} step={50} current={sim.speed} onChange={sim.setSpeed} hint="interval per label budget" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Transfer Learning', seed, rotationDeg, labelledExamples: n, highlighted: modeName, scratchAcc: at ? +at.scratch.toFixed(3) : null, frozenAcc: at ? +at.frozen.toFixed(3) : null, finetuneAcc: at ? +at.finetune.toFixed(3) : null, sourceAcc: +exp.sourceAcc.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default TransferLearningLab;
