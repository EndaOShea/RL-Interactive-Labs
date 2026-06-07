import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { clamp01, randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { gradientBoostingPython } from './python';

const ACCENT = '#fbbf24';

// XOR-style 4-cluster data: no single straight cut separates it, so boosting
// must stack several shallow trees to carve the four quadrants.
const CENTERS = [{ x: 0.28, y: 0.3 }, { x: 0.72, y: 0.3 }, { x: 0.28, y: 0.72 }, { x: 0.72, y: 0.72 }];
const CLS = [0, 1, 1, 0];
interface GPt { x: number; y: number; y01: number; }
const makeData = (perCluster: number): GPt[] =>
  CENTERS.flatMap((c, ci) => Array.from({ length: perCluster }, () => ({ x: clamp01(c.x + randn() * 0.085), y: clamp01(c.y + randn() * 0.085), y01: CLS[ci] })));

type Variant = 'xgboost' | 'lightgbm' | 'catboost';
const VARIANT_LABEL: Record<Variant, string> = { xgboost: 'XGBoost', lightgbm: 'LightGBM', catboost: 'CatBoost' };

// A boosted regression tree: predicts a Newton leaf weight at (x,y).
type TNode =
  | { leaf: true; val: number }
  | { leaf: false; feat: 0 | 1; thr: number; left: TNode; right: TNode };

const featVal = (p: GPt, f: 0 | 1) => (f === 0 ? p.x : p.y);
const predict = (n: TNode, x: number, y: number): number =>
  n.leaf ? n.val : predict((n.feat === 0 ? x : y) <= n.thr ? n.left : n.right, x, y);

interface Grad { g: number; h: number; } // gradient & hessian of logistic loss
// Newton leaf weight  w* = −ΣG / (ΣH + λ)
const leafWeight = (idx: number[], gr: Grad[], lambda: number) => {
  let G = 0, H = 0;
  for (const i of idx) { G += gr[i].g; H += gr[i].h; }
  return -G / (H + lambda);
};
// Split gain  ½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ
const splitGain = (L: number[], R: number[], gr: Grad[], lambda: number, gamma: number) => {
  let GL = 0, HL = 0, GR = 0, HR = 0;
  for (const i of L) { GL += gr[i].g; HL += gr[i].h; }
  for (const i of R) { GR += gr[i].g; HR += gr[i].h; }
  const G = GL + GR, H = HL + HR;
  return 0.5 * (GL * GL / (HL + lambda) + GR * GR / (HR + lambda) - G * G / (H + lambda)) - gamma;
};

interface Split { feat: 0 | 1; thr: number; L: number[]; R: number[]; gain: number; }
const bestSplit = (pts: GPt[], idx: number[], gr: Grad[], lambda: number, gamma: number, minLeaf: number): Split | null => {
  let best: Split | null = null;
  for (const feat of [0, 1] as const) {
    const vals = [...new Set(idx.map((i) => featVal(pts[i], feat)))].sort((a, b) => a - b);
    for (let v = 0; v < vals.length - 1; v++) {
      const thr = (vals[v] + vals[v + 1]) / 2;
      const L: number[] = [], R: number[] = [];
      for (const i of idx) (featVal(pts[i], feat) <= thr ? L : R).push(i);
      if (L.length < minLeaf || R.length < minLeaf) continue;
      const gain = splitGain(L, R, gr, lambda, gamma);
      if (!best || gain > best.gain) best = { feat, thr, L, R, gain };
    }
  }
  return best;
};

// XGBoost — level-wise (depth-first to a fixed max depth): split every node.
const buildLevelWise = (pts: GPt[], idx: number[], gr: Grad[], depth: number, maxDepth: number, lambda: number, gamma: number, minLeaf: number): TNode => {
  if (depth >= maxDepth || idx.length < 2 * minLeaf) return { leaf: true, val: leafWeight(idx, gr, lambda) };
  const s = bestSplit(pts, idx, gr, lambda, gamma, minLeaf);
  if (!s || s.gain <= 0) return { leaf: true, val: leafWeight(idx, gr, lambda) };
  return { leaf: false, feat: s.feat, thr: s.thr, left: buildLevelWise(pts, s.L, gr, depth + 1, maxDepth, lambda, gamma, minLeaf), right: buildLevelWise(pts, s.R, gr, depth + 1, maxDepth, lambda, gamma, minLeaf) };
};

// LightGBM — leaf-wise (best-first): repeatedly split the leaf with the biggest
// gain until num_leaves is reached. Grows deep, unbalanced trees.
const buildLeafWise = (pts: GPt[], root: number[], gr: Grad[], numLeaves: number, lambda: number, gamma: number, minLeaf: number): TNode => {
  interface Leaf { idx: number[]; node: { leaf: true; val: number }; split: Split | null; }
  const makeLeaf = (idx: number[]): Leaf => ({ idx, node: { leaf: true, val: leafWeight(idx, gr, lambda) }, split: bestSplit(pts, idx, gr, lambda, gamma, minLeaf) });
  // Mutable tree: we keep references so we can replace a leaf node in place.
  let rootRef: TNode = { leaf: true, val: leafWeight(root, gr, lambda) };
  const leaves: { leaf: Leaf; set: (n: TNode) => void }[] = [{ leaf: makeLeaf(root), set: (n) => { rootRef = n; } }];
  for (let count = 1; count < numLeaves; count++) {
    let bi = -1, bg = 0;
    leaves.forEach((l, i) => { if (l.leaf.split && l.leaf.split.gain > bg) { bg = l.leaf.split.gain; bi = i; } });
    if (bi < 0) break;
    const { leaf, set } = leaves[bi];
    const s = leaf.split!;
    const lNode: { leaf: true; val: number } = { leaf: true, val: leafWeight(s.L, gr, lambda) };
    const rNode: { leaf: true; val: number } = { leaf: true, val: leafWeight(s.R, gr, lambda) };
    const branch: TNode = { leaf: false, feat: s.feat, thr: s.thr, left: lNode, right: rNode };
    set(branch);
    leaves.splice(bi, 1);
    leaves.push({ leaf: makeLeaf(s.L), set: (n) => { branch.left = n; } });
    leaves.push({ leaf: makeLeaf(s.R), set: (n) => { branch.right = n; } });
  }
  return rootRef;
};

// CatBoost — symmetric / oblivious: pick ONE (feature, threshold) per level and
// apply the same test to every node on that level. Balanced, regularised trees.
const buildSymmetric = (pts: GPt[], root: number[], gr: Grad[], maxDepth: number, lambda: number, gamma: number, minLeaf: number): TNode => {
  let groups: number[][] = [root];
  const tests: { feat: 0 | 1; thr: number }[] = [];
  for (let d = 0; d < maxDepth; d++) {
    // Choose the single split that maximises summed gain across all groups.
    let best: { feat: 0 | 1; thr: number; gain: number } | null = null;
    const cand = new Set<string>();
    for (const feat of [0, 1] as const) {
      const vals = [...new Set(root.map((i) => featVal(pts[i], feat)))].sort((a, b) => a - b);
      for (let v = 0; v < vals.length - 1; v++) cand.add(`${feat}:${(vals[v] + vals[v + 1]) / 2}`);
    }
    cand.forEach((c) => {
      const [fs, ts] = c.split(':'); const feat = +fs as 0 | 1; const thr = +ts;
      let total = 0, ok = false;
      for (const grp of groups) {
        const L: number[] = [], R: number[] = [];
        for (const i of grp) (featVal(pts[i], feat) <= thr ? L : R).push(i);
        if (L.length >= minLeaf && R.length >= minLeaf) { total += splitGain(L, R, gr, lambda, gamma); ok = true; }
      }
      if (ok && (!best || total > best.gain)) best = { feat, thr, gain: total };
    });
    if (!best || best.gain <= 0) break;
    tests.push({ feat: best.feat, thr: best.thr });
    groups = groups.flatMap((grp) => {
      const L: number[] = [], R: number[] = [];
      for (const i of grp) (featVal(pts[i], best!.feat) <= best!.thr ? L : R).push(i);
      return [L, R];
    });
  }
  // Materialise the oblivious tree from the per-level tests.
  const build = (idx: number[], level: number): TNode => {
    if (level >= tests.length) return { leaf: true, val: leafWeight(idx, gr, lambda) };
    const { feat, thr } = tests[level];
    const L: number[] = [], R: number[] = [];
    for (const i of idx) (featVal(pts[i], feat) <= thr ? L : R).push(i);
    return { leaf: false, feat, thr, left: build(L, level + 1), right: build(R, level + 1) };
  };
  return build(root, 0);
};

const buildTree = (variant: Variant, pts: GPt[], gr: Grad[], maxDepth: number, numLeaves: number, lambda: number, gamma: number, minLeaf: number): TNode => {
  const idx = pts.map((_, i) => i);
  if (variant === 'xgboost') return buildLevelWise(pts, idx, gr, 0, maxDepth, lambda, gamma, minLeaf);
  if (variant === 'lightgbm') return buildLeafWise(pts, idx, gr, numLeaves, lambda, gamma, minLeaf);
  return buildSymmetric(pts, idx, gr, maxDepth, lambda, gamma, minLeaf);
};

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const MAX_ROUNDS = 40;

const GradientBoostingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [perCluster, setPerCluster] = useState(24);
  const [variant, setVariant] = useState<Variant>('xgboost');
  const [lr, setLr] = useState(0.3);           // shrinkage / learning rate
  const [maxDepth, setMaxDepth] = useState(3);
  const [numLeaves, setNumLeaves] = useState(8);
  const [lambda, setLambda] = useState(1);     // L2 leaf regularisation
  const [data, setData] = useState<GPt[]>(() => makeData(24));
  const [trees, setTrees] = useState<TNode[]>([]);
  const [lossSeries, setLossSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [version, setVersion] = useState(0);

  const gamma = 0; // min split gain — kept at 0 here; lambda is the visible knob
  const minLeaf = 1;

  // Ensemble raw score F(x,y) = lr · Σ_t treeₜ(x,y).
  const scoreAt = (x: number, y: number) => lr * trees.reduce((s, t) => s + predict(t, x, y), 0);
  const metrics = useMemo(() => {
    if (!data.length) return { loss: 0.6931, acc: 0 };
    let loss = 0, ok = 0;
    for (const p of data) {
      const F = lr * trees.reduce((s, t) => s + predict(t, p.x, p.y), 0);
      const prob = sigmoid(F);
      loss += -(p.y01 * Math.log(prob + 1e-9) + (1 - p.y01) * Math.log(1 - prob + 1e-9));
      if ((prob >= 0.5 ? 1 : 0) === p.y01) ok++;
    }
    return { loss: loss / data.length, acc: ok / data.length };
  }, [trees, data, lr]);

  const step = () => {
    if (trees.length >= MAX_ROUNDS) { sim.pause(); return; }
    // Newton boosting: gradient g = p − y, hessian h = p(1−p) of logistic loss.
    const gr: Grad[] = data.map((p) => {
      const prob = sigmoid(lr * trees.reduce((s, t) => s + predict(t, p.x, p.y), 0));
      return { g: prob - p.y01, h: Math.max(1e-6, prob * (1 - prob)) };
    });
    const tree = buildTree(variant, data, gr, maxDepth, numLeaves, lambda, gamma, minLeaf);
    const next = [...trees, tree];
    setTrees(next);

    let loss = 0, ok = 0;
    for (const p of data) {
      const F = lr * next.reduce((s, t) => s + predict(t, p.x, p.y), 0);
      const prob = sigmoid(F);
      loss += -(p.y01 * Math.log(prob + 1e-9) + (1 - p.y01) * Math.log(1 - prob + 1e-9));
      if ((prob >= 0.5 ? 1 : 0) === p.y01) ok++;
    }
    loss /= data.length; const acc = ok / data.length;
    setLossSeries((s) => [...s, loss].slice(-60));

    const growth = variant === 'xgboost'
      ? `level-wise, splitting every node down to depth ${maxDepth}`
      : variant === 'lightgbm'
        ? `leaf-wise, repeatedly splitting the single highest-gain leaf up to ${numLeaves} leaves`
        : `symmetric, applying one shared split test per level down to depth ${maxDepth}`;
    narration.narratePhase(
      `run:${variant}`,
      `The challenge here: classify these four XOR-style clusters, which no single tree can separate, by stacking many shallow trees. Gradient boosting solves it by fitting each new tree to the errors of the ones before it — using the gradient and curvature of the loss to compute Newton leaf weights, w star equals minus sum of gradients over sum of hessians plus lambda. ${VARIANT_LABEL[variant]} grows its trees ${growth}, then adds them shrunk by the learning rate. Watch the boundary sharpen tree by tree as the training loss falls. Gradient boosting like this wins most tabular machine-learning competitions and powers ranking, fraud detection, credit scoring and forecasting in industry.`
    );
    if (acc >= 0.99) {
      narration.narratePhase(
        `done:${variant}`,
        `${VARIANT_LABEL[variant]} now classifies every training point, with the loss driven low after ${next.length} trees. Each tree was weak on its own, but boosting their corrections together solved a pattern no single shallow tree could. A smaller learning rate with more trees usually generalises better than a few aggressive ones.`
      );
    }

    setLastLog({
      algorithm: `Gradient Boosting · ${VARIANT_LABEL[variant]}`,
      stepDescription: `Round ${next.length}: fit a tree to the negative gradient, add it with shrinkage ${lr}`,
      formula: 'w* = −Σg / (Σh + λ)   ·   F ← F + η·tree',
      variables: { 'round': next.length, 'η': lr, 'λ': lambda, 'loss': +loss.toFixed(4), 'acc': acc },
      result: `loss ${loss.toFixed(3)} · acc ${(acc * 100).toFixed(0)}%`,
      mathDetails: {
        params: [
          { label: VARIANT_LABEL[variant], info: growth + '.' },
          { label: 'gradient/hessian', info: 'Logistic loss: g = p − y, h = p(1−p). Trees fit −g; leaves use the Newton step w* = −Σg/(Σh+λ).' },
          { label: 'shrinkage η', info: `${lr}. Each tree is scaled by the learning rate before being added — smaller η needs more trees but generalises better.` },
          { label: 'λ (L2)', info: `${lambda}. Regularises leaf weights, shrinking them toward zero and damping noisy splits.` },
        ],
        implication: acc >= 0.99 ? 'Training data fully fit — lower η / fewer trees to avoid overfitting.' : 'Residual errors remain — the next tree will target the points still misclassified.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 400 });
  const regen = (n = perCluster) => { sim.stop(); narration.cancel(); setData(makeData(n)); setTrees([]); setLossSeries([]); setLastLog(null); setVersion((v) => v + 1); };
  const reset = () => { sim.stop(); narration.cancel(); setTrees([]); setLossSeries([]); setLastLog(null); };
  const pickVariant = (v: Variant) => { setVariant(v); reset(); };

  const fieldKey = `${variant}-${trees.length}-${lr}-${maxDepth}-${numLeaves}-${lambda}-${version}`;
  const plotPoints: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.y01 }));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'TREES', value: trees.length },
        { label: 'LOSS', value: metrics.loss.toFixed(3) },
        { label: 'ACC', value: `${(metrics.acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, gradientBoostingPython(variant, lr, maxDepth, numLeaves, lambda))}
      grid={(
        <ScatterPlot
          width={440} height={440}
          points={plotPoints}
          classify={(x, y) => (scoreAt(x, y) >= 0 ? 1 : 0)}
          fieldKey={fieldKey}
          xLabel="x₁" yLabel="x₂"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel="TRAINING LOSS"
      rewardValue={metrics.loss.toFixed(3)}
      rewardSeries={lossSeries}
      lastLog={lastLog}
      contextInsight={`${VARIANT_LABEL[variant]} gradient boosting. Each Run adds one shallow tree fit to the negative gradient of the logistic loss; the trees stack into the decision field on the left, and the boundary sharpens as the loss falls. The three variants share this engine but grow each tree differently — level-wise (XGBoost), leaf-wise (LightGBM) or symmetric (CatBoost).`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Gradient Boosting" hint="Run adds one boosted tree per step." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Framework / tree growth</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {(['xgboost', 'lightgbm', 'catboost'] as Variant[]).map((v) => (
                <AlgoPill key={v} active={variant === v} accent={ACCENT} onClick={() => pickVariant(v)}>{VARIANT_LABEL[v]}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {variant === 'xgboost' ? 'Level-wise: split every node down to max depth — balanced, regularised trees.'
                : variant === 'lightgbm' ? 'Leaf-wise: always split the highest-gain leaf — deep, fast, accurate, can overfit.'
                  : 'Symmetric (oblivious): one shared split per level — balanced, fast to score, robust.'}
            </p>
          </div>
          <ParamSlider name="Learning rate η" value={lr.toFixed(2)} min={0.05} max={1} step={0.05} current={lr} onChange={(v) => { reset(); setLr(v); }} hint="shrinkage per tree" />
          {variant === 'lightgbm'
            ? <ParamSlider name="Num leaves" value={String(numLeaves)} min={2} max={32} step={1} current={numLeaves} onChange={(v) => { reset(); setNumLeaves(v); }} hint="leaf-wise growth budget" />
            : <ParamSlider name="Max depth" value={String(maxDepth)} min={1} max={6} step={1} current={maxDepth} onChange={(v) => { reset(); setMaxDepth(v); }} hint="per-tree depth" />}
          <ParamSlider name="L2 reg λ" value={lambda.toFixed(1)} min={0} max={10} step={0.5} current={lambda} onChange={(v) => { reset(); setLambda(v); }} hint="leaf-weight regularisation" />
          <ParamSlider name="Points / cluster" value={String(perCluster)} min={12} max={40} step={2} current={perCluster} onChange={(v) => { setPerCluster(v); regen(v); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={150} max={1000} step={50} current={sim.speed} onChange={sim.setSpeed} hint="boosting interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: `Gradient Boosting (${VARIANT_LABEL[variant]})`, learningRate: lr, maxDepth, numLeaves, lambda, trees: trees.length, trainLoss: +metrics.loss.toFixed(4), trainAcc: +metrics.acc.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default GradientBoostingLab;
