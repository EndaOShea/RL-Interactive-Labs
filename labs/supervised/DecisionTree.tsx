import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { clamp01, randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { decisionTreePython } from './python';

const ACCENT = '#fbbf24';
const CENTERS = [{ x: 0.28, y: 0.3 }, { x: 0.72, y: 0.3 }, { x: 0.28, y: 0.72 }, { x: 0.72, y: 0.72 }];
const CLS = [0, 1, 1, 0]; // XOR-style: needs ≥2 splits
type Crit = 'gini' | 'entropy';
interface DPt { x: number; y: number; cls: number; }
type TNode =
  | { leaf: true; cls: number; n: number; imp: number }
  | { leaf: false; feat: 0 | 1; thr: number; imp: number; n: number; gain: number; left: TNode; right: TNode };

const makeData = (perCluster: number): DPt[] =>
  CENTERS.flatMap((c, ci) => Array.from({ length: perCluster }, () => ({ x: clamp01(c.x + randn() * 0.08), y: clamp01(c.y + randn() * 0.08), cls: CLS[ci] })));

const counts = (pts: DPt[]) => { const m: Record<number, number> = {}; pts.forEach((p) => { m[p.cls] = (m[p.cls] || 0) + 1; }); return m; };
const impurity = (pts: DPt[], crit: Crit) => {
  if (!pts.length) return 0;
  const c = counts(pts); let v = crit === 'gini' ? 1 : 0;
  Object.values(c).forEach((k) => { const p = k / pts.length; if (crit === 'gini') v -= p * p; else v -= p * Math.log2(p); });
  return v;
};
const majority = (pts: DPt[]) => { const c = counts(pts); let best = 0, bv = -1; Object.entries(c).forEach(([k, v]) => { if (v > bv) { bv = v; best = +k; } }); return best; };

function buildTree(pts: DPt[], depth: number, maxDepth: number, crit: Crit, minLeaf: number): TNode {
  const imp = impurity(pts, crit), maj = majority(pts);
  if (depth >= maxDepth || imp < 1e-9 || pts.length < Math.max(4, 2 * minLeaf)) return { leaf: true, cls: maj, n: pts.length, imp };
  let best: { feat: 0 | 1; thr: number; score: number; L: DPt[]; R: DPt[] } | null = null;
  for (const feat of [0, 1] as const) {
    const vals = [...new Set(pts.map((p) => (feat === 0 ? p.x : p.y)))].sort((a, b) => a - b);
    for (let i = 0; i < vals.length - 1; i++) {
      const thr = (vals[i] + vals[i + 1]) / 2;
      const L = pts.filter((p) => (feat === 0 ? p.x : p.y) <= thr), R = pts.filter((p) => (feat === 0 ? p.x : p.y) > thr);
      if (L.length < minLeaf || R.length < minLeaf) continue;
      const score = (L.length * impurity(L, crit) + R.length * impurity(R, crit)) / pts.length;
      if (!best || score < best.score) best = { feat, thr, score, L, R };
    }
  }
  if (!best || best.score >= imp - 1e-9) return { leaf: true, cls: maj, n: pts.length, imp };
  return { leaf: false, feat: best.feat, thr: best.thr, imp, n: pts.length, gain: imp - best.score, left: buildTree(best.L, depth + 1, maxDepth, crit, minLeaf), right: buildTree(best.R, depth + 1, maxDepth, crit, minLeaf) };
}
const classify = (n: TNode, x: number, y: number): number => n.leaf ? n.cls : classify((n.feat === 0 ? x : y) <= n.thr ? n.left : n.right, x, y);
function countNodes(n: TNode): [number, number] {
  if (n.leaf) return [1, 1];
  const [ln, ll] = countNodes(n.left), [rn, rl] = countNodes(n.right);
  return [1 + ln + rn, ll + rl];
}
/** Best split + info-gain at the root of the newest level (for narration). */
function newestSplits(n: TNode, targetDepth: number, depth = 0, acc: { feat: 0 | 1; thr: number; gain: number }[] = []) {
  if (n.leaf) return acc;
  if (depth === targetDepth - 1) acc.push({ feat: n.feat, thr: n.thr, gain: n.gain });
  newestSplits(n.left, targetDepth, depth + 1, acc);
  newestSplits(n.right, targetDepth, depth + 1, acc);
  return acc;
}

function layoutTree(root: TNode, newDepth: number) {
  const raw: { id: string; depth: number; node: TNode; x: number }[] = [];
  const edges: GEdge[] = []; let leaf = 0, maxD = 0, idc = 0;
  const rec = (node: TNode, depth: number, parent: string | null): { id: string; x: number } => {
    const id = 'n' + (idc++); maxD = Math.max(maxD, depth);
    let x: number;
    if (node.leaf) { x = leaf++; } else { const l = rec(node.left, depth + 1, id), r = rec(node.right, depth + 1, id); x = (l.x + r.x) / 2; }
    raw.push({ id, depth, node, x });
    if (parent) edges.push({ from: parent, to: id });
    return { id, x };
  };
  rec(root, 0, null);
  const lc = Math.max(1, leaf);
  const nodes: GNode[] = raw.map((m) => ({
    id: m.id,
    x: lc <= 1 ? 0.5 : m.x / (lc - 1),
    y: maxD === 0 ? 0.5 : m.depth / maxD,
    label: m.node.leaf ? String(m.node.cls) : (m.node.feat === 0 ? 'x₁' : 'x₂') + '≤' + m.node.thr.toFixed(2),
    sub: m.node.leaf ? `n=${m.node.n}` : undefined,
    // highlight the freshly-grown level in accent
    color: m.node.leaf ? CLASS_COLORS[m.node.cls % CLASS_COLORS.length] : (m.depth === newDepth - 1 ? ACCENT : '#2a3350'),
  }));
  return { nodes, edges };
}

interface Preset { name: string; crit: Crit; depth: number; minLeaf: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'Stump (depth 1)', crit: 'gini', depth: 1, minLeaf: 1, tip: 'A single split can never solve XOR — accuracy stays near 50%.' },
  { name: 'Just enough (d=2)', crit: 'gini', depth: 2, minLeaf: 1, tip: 'Two levels carve the four XOR quadrants — the minimal correct tree.' },
  { name: 'Entropy · gain', crit: 'entropy', depth: 3, minLeaf: 2, tip: 'Information gain (entropy) tends to pick the same cuts here as Gini.' },
  { name: 'Pruned (min-leaf 6)', crit: 'gini', depth: 6, minLeaf: 6, tip: 'A large min-samples-leaf prunes noisy splits → simpler, more robust tree.' },
  { name: 'Overfit (deep)', crit: 'gini', depth: 8, minLeaf: 1, tip: 'Deep + min-leaf 1 memorises noise: many tiny pure leaves.' },
];

const DecisionTreeLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [perCluster, setPerCluster] = useState(22);
  const [crit, setCrit] = useState<Crit>('gini');
  const [minLeaf, setMinLeaf] = useState(1);
  const [depth, setDepth] = useState(0);
  const [data, setData] = useState<DPt[]>(() => makeData(22));
  const [accSeries, setAccSeries] = useState<number[]>([]);
  const [version, setVersion] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const tree = useMemo(() => buildTree(data, 0, depth, crit, minLeaf), [data, depth, crit, minLeaf]);
  const [nNodes, nLeaves] = useMemo(() => countNodes(tree), [tree]);
  const acc = useMemo(() => { if (!data.length) return 0; let ok = 0; data.forEach((p) => { if (classify(tree, p.x, p.y) === p.cls) ok++; }); return ok / data.length; }, [tree, data]);

  const TARGET = 5;
  const step = () => {
    if (depth >= TARGET) { sim.pause(); return; }
    const nd = depth + 1; setDepth(nd);
    const t = buildTree(data, 0, nd, crit, minLeaf);
    let ok = 0; data.forEach((p) => { if (classify(t, p.x, p.y) === p.cls) ok++; });
    const a = ok / data.length;
    setAccSeries((s) => [...s, a].slice(-60));

    // best split + info-gain at this newest level (feeds the live-math payload)
    const splits = newestSplits(t, nd);

    // Conceptual audio tutor — one explanation per phase (keyed, so it speaks once).
    const critWord = crit === 'gini' ? 'Gini impurity' : 'entropy';
    const measure = crit === 'gini'
      ? 'Gini is one minus the sum of the squared class proportions, and reaches zero when a node holds a single class'
      : 'entropy measures the bits of uncertainty in a node, and information gain is the parent entropy minus the weighted entropy of the children';
    narration.narratePhase(
      `run:${crit}:${minLeaf}`,
      `This is a decision tree using ${critWord} to choose its splits. At every node it asks one yes or no question about a feature, and it greedily picks the cut that most reduces impurity. ${measure[0].toUpperCase() + measure.slice(1)}. Because this data is X O R like, a single straight cut can never separate it. Watch the left panel carve into rectangular regions while the tree on the right grows level by level, with the newest split shown in amber.`
    );
    if (a >= 0.99) {
      narration.narratePhase(
        `done:${crit}:${minLeaf}`,
        `The tree now separates the data perfectly, reaching about ${Math.round(a * 100)} percent on the training points. It took at least two levels, since one split alone cannot solve this pattern. But a tree this deep with a small minimum leaf size starts to memorise noise, so watch for overfitting.`
      );
    } else if (nd >= 2) {
      narration.narratePhase(
        `mid:${crit}:${minLeaf}`,
        `Two levels are now in place, which is the minimum needed to carve the four quadrants of this pattern. Each extra level fits finer detail, so notice when the accuracy stops improving meaningfully and the tree only adds tiny leaves.`
      );
    }

    const [tn, tl] = countNodes(t);
    setLastLog({
      algorithm: `Decision Tree · ${crit}`,
      stepDescription: `Grow to depth ${nd} — split on the feature that most reduces impurity`,
      formula: crit === 'gini' ? 'Gini = 1 − Σ pₖ²   →   minimise weighted child impurity' : 'IG = H(parent) − Σ (n_child/n)·H(child)   →   maximise',
      variables: { 'depth': nd, 'nodes': tn, 'leaves': tl, 'gain': splits.length ? +splits.reduce((m, s) => (s.gain > m.gain ? s : m), splits[0]).gain.toFixed(3) : 0, 'acc': a },
      result: `train acc ${(a * 100).toFixed(0)}%`,
      mathDetails: {
        params: [
          { label: crit, info: crit === 'gini' ? 'Gini impurity: 0 when a node is pure (one class).' : 'Entropy & information gain: gain = parent entropy − weighted child entropy; bigger gain = better split.' },
          { label: 'min-leaf', info: `${minLeaf}. A split is only kept if both children hold ≥ ${minLeaf} samples — a pre-pruning knob that limits overfitting.` },
          { label: 'depth', info: `${nd}. Deeper trees fit more detail but overfit — note when train acc stops improving meaningfully.` },
        ],
        implication: a >= 0.99 ? 'Perfectly separates the training data — watch for overfitting on this XOR-like set.' : 'Still impure regions remain — another split level (or smaller min-leaf) may help.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 700 });
  const regen = (n = perCluster) => { narration.cancel(); setData(makeData(n)); setDepth(0); setAccSeries([]); setLastLog(null); setVersion((v) => v + 1); };
  const reset = () => { sim.stop(); narration.cancel(); setDepth(0); setAccSeries([]); setLastLog(null); };
  const applyPreset = (p: Preset) => { sim.stop(); narration.cancel(); setCrit(p.crit); setMinLeaf(p.minLeaf); setDepth(p.depth); setAccSeries([]); setLastLog(null); };

  const fieldKey = `${depth}-${crit}-${minLeaf}-${version}`;
  const plotPoints: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.cls }));
  const { nodes, edges } = useMemo(() => layoutTree(tree, depth), [tree, depth]);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'DEPTH', value: depth },
        { label: 'LEAVES', value: nLeaves },
        { label: 'NODES', value: nNodes },
        { label: 'ACC', value: `${(acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, decisionTreePython(depth, crit, minLeaf))}
      grid={(
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <ScatterPlot width={400} height={400} points={plotPoints} classify={(x, y) => classify(tree, x, y)} fieldKey={fieldKey} xLabel="x₁" yLabel="x₂" />
          <GraphCanvas width={440} height={400} radius={15} nodes={nodes} edges={edges} />
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel="TRAIN ACCURACY"
      rewardValue={`${(acc * 100).toFixed(0)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={`${crit} splitting, min-leaf ${minLeaf}. The data is XOR-like, so a single straight cut can't separate it — the tree needs ≥2 levels. Each node thresholds one feature, carving the rectangular regions on the left; the tree on the right shows the splits, with the newest level highlighted in amber.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Decision Tree" hint="Run grows the tree one level at a time." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Split criterion</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={crit === 'gini'} accent={ACCENT} onClick={() => { setCrit('gini'); reset(); }}>Gini</AlgoPill>
              <AlgoPill active={crit === 'entropy'} accent={ACCENT} onClick={() => { setCrit('entropy'); reset(); }}>Entropy / gain</AlgoPill>
            </div>
          </div>
          <ParamSlider name="Max depth" value={String(depth)} min={0} max={8} step={1} current={depth} onChange={(v) => { sim.stop(); narration.cancel(); setDepth(v); }} hint="tree depth (also via Run)" />
          <ParamSlider name="Min samples / leaf" value={String(minLeaf)} min={1} max={10} step={1} current={minLeaf} onChange={(v) => { sim.stop(); narration.cancel(); setMinLeaf(v); }} hint="pre-pruning — bigger = simpler tree" />
          <ParamSlider name="Points / cluster" value={String(perCluster)} min={10} max={40} step={2} current={perCluster} onChange={(v) => { setPerCluster(v); regen(v); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={150} max={1200} step={50} current={sim.speed} onChange={sim.setSpeed} hint="grow interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {PRESETS.find((p) => p.crit === crit && p.depth === depth && p.minLeaf === minLeaf)?.tip || 'Pick a preset, then press Run to grow level by level.'}
            </p>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Decision Tree', criterion: crit, depth, minLeaf, leaves: nLeaves, trainAcc: +acc.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default DecisionTreeLab;
