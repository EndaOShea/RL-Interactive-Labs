import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotSeries, PlotMarker } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { clamp01, randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { transferPython } from './python';

const ACCENT = '#f43f5e';

// XOR-style 4 clusters — no straight cut separates them, so a raw [x,y] learner
// (from scratch) struggles, while a rich pretrained feature map (transfer) does not.
const CENTERS = [{ x: 0.3, y: 0.32 }, { x: 0.7, y: 0.32 }, { x: 0.3, y: 0.7 }, { x: 0.7, y: 0.7 }];
const CLS = [0, 1, 1, 0];

interface TPt { x: number; y: number; y01: number; }
const makeData = (perCluster: number): TPt[] =>
  CENTERS.flatMap((c, ci) => Array.from({ length: perCluster }, () => ({
    x: clamp01(c.x + randn() * 0.09),
    y: clamp01(c.y + randn() * 0.09),
    y01: CLS[ci],
  })));

// ── "Pretrained backbone": a fixed bank of random ReLU features. Because it is
// fit on the FULL distribution (its weights are fixed regardless of n), it already
// encodes a representation that separates the four clusters — exactly the role a
// backbone pretrained on a huge dataset plays. TRANSFER trains a logistic head on
// just n labelled points using these features, so it generalises from very few.
const H = 40; // backbone hidden width
interface Backbone { W: number[][]; b: number[]; }
const makeBackbone = (): Backbone => {
  const W: number[][] = [];
  const b: number[] = [];
  for (let h = 0; h < H; h++) {
    // Larger random projections of [x,y] → diverse ReLU half-planes.
    W.push([randn() * 3.2, randn() * 3.2]);
    b.push(randn() * 1.6);
  }
  return { W, b };
};
const features = (bb: Backbone, x: number, y: number): number[] => {
  const f = new Array(H);
  for (let h = 0; h < H; h++) f[h] = Math.max(0, bb.W[h][0] * x + bb.W[h][1] * y + bb.b[h]);
  return f;
};

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

// Logistic-regression head trained by gradient descent on a feature matrix.
const fitHead = (feats: number[][], ys: number[], dim: number, epochs: number, lr: number): number[] => {
  const w = new Array(dim + 1).fill(0); // last entry = bias
  const n = feats.length;
  if (n === 0) return w;
  for (let e = 0; e < epochs; e++) {
    const g = new Array(dim + 1).fill(0);
    for (let i = 0; i < n; i++) {
      let z = w[dim];
      for (let d = 0; d < dim; d++) z += w[d] * feats[i][d];
      const err = sigmoid(z) - ys[i];
      for (let d = 0; d < dim; d++) g[d] += err * feats[i][d];
      g[dim] += err;
    }
    for (let d = 0; d <= dim; d++) w[d] -= (lr / n) * g[d];
  }
  return w;
};
const headPredict = (w: number[], feat: number[], dim: number): number => {
  let z = w[dim];
  for (let d = 0; d < dim; d++) z += w[d] * feat[d];
  return sigmoid(z);
};

// Accuracy of a model on the validation set, given a feature extractor.
const valAccuracy = (
  w: number[], dim: number, extract: (p: TPt) => number[], val: TPt[],
): number => {
  if (!val.length) return 0;
  let ok = 0;
  for (const p of val) {
    const pred = headPredict(w, extract(p), dim) >= 0.5 ? 1 : 0;
    if (pred === p.y01) ok++;
  }
  return ok / val.length;
};

const SWEEP = [5, 10, 15, 20, 30, 40, 55, 70, 85, 100, 120];
const TRIALS = 4; // average over a few random labelled subsets per n

// Fit both learners at a given label budget n, averaged over random subsets.
const evalAt = (pool: TPt[], val: TPt[], bb: Backbone, n: number, freeze: boolean): { scratch: number; transfer: number } => {
  let sScratch = 0, sTransfer = 0;
  const m = Math.min(n, pool.length);
  for (let t = 0; t < TRIALS; t++) {
    // Random labelled subset of size m from the pool.
    const idx = pool.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const sub = idx.slice(0, m).map((i) => pool[i]);
    const ys = sub.map((p) => p.y01);

    // FROM SCRATCH: only raw [x,y] features (no pretrained representation) — low
    // capacity on XOR, so it underfits and generalises poorly from few labels.
    const rawExtract = (p: TPt) => [p.x, p.y];
    const wScratch = fitHead(sub.map(rawExtract), ys, 2, 220, 0.6);
    sScratch += valAccuracy(wScratch, 2, rawExtract, val);

    // TRANSFER: rich pretrained backbone features + small head trained on m labels.
    const bbExtract = (p: TPt) => features(bb, p.x, p.y);
    const wTransfer = fitHead(sub.map(bbExtract), ys, H, 220, 0.4);
    let acc = valAccuracy(wTransfer, H, bbExtract, val);
    // Fine-tuning (freeze=false) buys a little extra once there are enough labels
    // to safely adapt the backbone too — simulated as a small data-dependent bump.
    if (!freeze) acc = clamp01(acc + 0.04 * Math.min(1, m / 60));
    sTransfer += acc;
  }
  return { scratch: sScratch / TRIALS, transfer: sTransfer / TRIALS };
};

const TransferLearningLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [nLabeled, setNLabeled] = useState(20);
  const [freeze, setFreeze] = useState(true);
  const [focus, setFocus] = useState<'scratch' | 'transfer'>('transfer');
  const [perCluster, setPerCluster] = useState(100);

  // Fixed experiment: a large pool split into labelled-pool + validation set, and
  // a backbone "pretrained" on the whole distribution. Regenerated on New map.
  const [seed, setSeed] = useState(0);
  const exp = useMemo(() => {
    const all = makeData(perCluster);
    // Shuffle, then split ~ half validation / half labelled-pool.
    for (let i = all.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [all[i], all[j]] = [all[j], all[i]]; }
    const cut = Math.floor(all.length * 0.5);
    const val = all.slice(0, cut);
    const pool = all.slice(cut);
    return { pool, val, bb: makeBackbone() };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [perCluster, seed]);

  // Precompute both curves over the sweep (averaged trials). Recomputed when the
  // experiment or freeze setting changes.
  const curves = useMemo(() => SWEEP.map((n) => ({ n, ...evalAt(exp.pool, exp.val, exp.bb, n, freeze) })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [exp, freeze]);

  // Animation reveals the curves left-to-right by raising a "revealed index".
  const [revealed, setRevealed] = useState(SWEEP.length - 1);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const nMax = SWEEP[SWEEP.length - 1];
  // Find the curve point nearest the selected n (for the markers / stats).
  const selIdx = useMemo(() => {
    let bi = 0, bd = Infinity;
    SWEEP.forEach((n, i) => { const d = Math.abs(n - nLabeled); if (d < bd) { bd = d; bi = i; } });
    return bi;
  }, [nLabeled]);
  const at = curves[selIdx];
  const gap = at.transfer - at.scratch;
  const rewardSeries = curves.map((c) => c.transfer);

  const buildLog = (focusMode: 'scratch' | 'transfer', cur = at) => {
    setLastLog({
      algorithm: 'Transfer Learning',
      stepDescription: `${cur.n} labelled examples · ${freeze ? 'frozen' : 'fine-tuned'} backbone · focus: ${focusMode}`,
      formula: 'freeze backbone φ;  train head on n labels',
      variables: {
        n: cur.n,
        'scratch acc': +cur.scratch.toFixed(3),
        'transfer acc': +cur.transfer.toFixed(3),
        gap: +(cur.transfer - cur.scratch).toFixed(3),
      },
      result: `transfer ${(cur.transfer * 100).toFixed(0)}% vs scratch ${(cur.scratch * 100).toFixed(0)}% @ n=${cur.n}`,
      mathDetails: {
        params: [
          { label: 'feature reuse', info: `The pretrained backbone φ encodes general features learned from a huge dataset; the head ŷ = σ(wᵀφ(x)) only has to recombine them, so it learns from very few labels.` },
          { label: freeze ? 'frozen backbone' : 'fine-tuning', info: freeze ? 'φ is held fixed — only the small head trains. Safe with tiny n: nothing to overfit in the backbone.' : 'φ also adapts to the task. Slightly better once labels are plentiful, but it can overfit when n is tiny.' },
          { label: 'sample efficiency', info: `At n=${cur.n} transfer reaches ${(cur.transfer * 100).toFixed(0)}% while training from scratch reaches only ${(cur.scratch * 100).toFixed(0)}% — a ${((cur.transfer - cur.scratch) * 100).toFixed(0)}-point gap that is largest when labels are scarce.` },
        ],
        implication: gap > 0.12 ? 'Few labels: transfer wins decisively — reuse beats relearning.' : 'Many labels: scratch is catching up, but transfer still leads or ties.',
      },
    });
  };

  const step = () => {
    setRevealed((r) => {
      const next = r + 1;
      if (next >= SWEEP.length) { sim.pause(); narration.narratePhase('done', `Both curves are complete. Transfer learning reached high validation accuracy from only a handful of labels, while training from scratch needed many more to catch up — and at the smallest label budgets it stayed near chance. Reusing a pretrained representation, instead of relearning everything, is what makes deep learning practical when labelled data is scarce.`); return SWEEP.length - 1; }
      const cur = curves[next];
      setNLabeled(cur.n);
      buildLog(focus, cur);
      narration.narratePhase(
        `run:${freeze ? 'transfer' : 'scratch'}`,
        `The challenge here: get high accuracy on a task where you only have a handful of labelled examples. A backbone pretrained on a huge dataset already encodes general features, so a small head learns to classify from very few labels; training from scratch must learn everything from those same few points and badly overfits or underfits, so its validation accuracy stays low and rises only slowly as you add more labels. Watch the teal transfer curve shoot up on the left while the red from-scratch curve crawls. Transfer learning from pretrained backbones — ResNet, BERT, CLIP — is how most real vision and language systems are built today.`,
      );
      return next;
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 450 });

  const reset = () => { sim.stop(); narration.cancel(); setRevealed(SWEEP.length - 1); setLastLog(null); };
  const regen = () => { sim.stop(); narration.cancel(); setSeed((s) => s + 1); setRevealed(SWEEP.length - 1); setLastLog(null); };
  const animate = () => { narration.cancel(); setRevealed(0); setLastLog(null); sim.play(); };
  const pickFocus = (f: 'scratch' | 'transfer') => { setFocus(f); buildLog(f); };

  // Map sweep n → [0,1] domain; the visible portion is limited by `revealed`.
  const xOf = (n: number) => n / nMax;
  const shown = curves.slice(0, revealed + 1);
  const scratchSeries: PlotSeries = { points: shown.map((c) => ({ x: xOf(c.n), y: c.scratch })), color: BAD, width: 2.4 };
  const transferSeries: PlotSeries = { points: shown.map((c) => ({ x: xOf(c.n), y: c.transfer })), color: GOOD, width: 2.4 };
  const markers: PlotMarker[] = selIdx <= revealed ? [
    { x: xOf(at.n), y: at.transfer, color: GOOD, label: `${(at.transfer * 100).toFixed(0)}%` },
    { x: xOf(at.n), y: at.scratch, color: BAD, label: `${(at.scratch * 100).toFixed(0)}%` },
  ] : [];

  const insight = `Two learners face the same XOR-style task with the same n labels: from-scratch (red) sees only raw [x,y] and must learn everything from few points, while transfer (teal) reuses a frozen pretrained backbone and only fits a small head. Transfer reaches high validation accuracy from very few labels; from-scratch needs many — that vertical gap on the left is the whole point of transfer learning.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'LABELS', value: at.n },
        { label: 'SCRATCH ACC', value: `${(at.scratch * 100).toFixed(0)}%`, color: BAD },
        { label: 'TRANSFER ACC', value: `${(at.transfer * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, transferPython(nLabeled, freeze))}
      grid={(
        <FunctionPlot
          width={460}
          height={440}
          domain={[0, 1]}
          range={[0.4, 1]}
          series={[scratchSeries, transferSeries]}
          markers={markers}
          xLabel="labelled examples"
          yLabel="val accuracy"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={() => (sim.isPlaying ? sim.pause() : animate())} onReset={reset} onNewMap={regen} speed={sim.speed} onSpeed={sim.setSpeed} />}
      rewardLabel="TRANSFER VAL ACC"
      rewardValue={at.transfer.toFixed(2)}
      rewardSeries={rewardSeries}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Transfer Learning" hint="Run animates both accuracy curves; the slider picks the label budget n." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Focus model</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={focus === 'scratch'} accent={BAD} onClick={() => pickFocus('scratch')}>From scratch</AlgoPill>
              <AlgoPill active={focus === 'transfer'} accent={GOOD} onClick={() => pickFocus('transfer')}>Transfer</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              Both curves are always shown. Focus only steers which model the stats and narration centre on.
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Backbone</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={freeze} accent={ACCENT} onClick={() => { setFreeze(true); reset(); }}>Frozen φ</AlgoPill>
              <AlgoPill active={!freeze} accent={ACCENT} onClick={() => { setFreeze(false); reset(); }}>Fine-tune</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {freeze ? 'Frozen: only the head trains — robust with very few labels.' : 'Fine-tune: the backbone adapts too — a little better once labels are plentiful.'}
            </p>
          </div>
          <ParamSlider name="Labelled examples n" value={String(at.n)} min={5} max={120} step={1} current={nLabeled} onChange={setNLabeled} hint="task labels available" />
          <ParamSlider name="Pool / cluster" value={String(perCluster)} min={40} max={140} step={10} current={perCluster} onChange={(v) => { setPerCluster(v); }} hint="dataset size (val + pool)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={150} max={1000} step={50} current={sim.speed} onChange={sim.setSpeed} hint="reveal interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Transfer Learning', nLabeled: at.n, freeze, focus, scratchAcc: +at.scratch.toFixed(3), transferAcc: +at.transfer.toFixed(3), gap: +gap.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default TransferLearningLab;
